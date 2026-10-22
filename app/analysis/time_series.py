"""Time Series: a measure in time order - its memory, its parts and its future (AN-05).

As JMP's Time Series platform, for one measure ordered by a date, a time or
a number (the rows' order when there is none):

* Autocorrelation (ACF) and partial autocorrelation (PACF) up to a lag,
  with the Ljung-Box test of "no autocorrelation up to here" at every lag
  and the +/-2 standard error band under which a bar says nothing.
* Seasonal decomposition (STL): trend, seasonal part and remainder, for a
  season length given or found from the dates (12 for months, 7 for days,
  4 for quarters, 24 for hours, 52 for weeks).
* Stationarity: ADF (null: a unit root) and KPSS (null: stationary), as
  the series is and differenced, and the differences it needs.
* Cross-correlation with an input series, at lags both ways.
* A forecast: seasonal ARIMA (p, d, q)(P, D, Q), its order given or chosen
  by AIC (d from the tests, D from the season's strength), or exponential smoothing (ETS: additive
  errors, an additive trend, and an additive season when there is one),
  with its 95% prediction interval, the fitted values and the residuals.
  AIC, BIC and the residuals' RMSE say how well it fits.

statsmodels does the fitting: acf, pacf, acorr_ljungbox, STL, ARIMA and
ETSModel; dev/tests/test_time_series.py checks the numbers against them and
the forecast on a known series.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from app.analysis.randomness import all_diagnostics
from app.logs.logger import applogger

ROW_ORDER = ""
ARIMA = "arima"
AUTO = "auto"
ETS = "ets"
NONE = "none"
MODELS: tuple[tuple[str, str], ...] = (
    (ETS, "Exponential smoothing (ETS)"), (AUTO, "ARIMA, order chosen by AIC"),
    (ARIMA, "ARIMA (p, d, q)(P, D, Q)"), (NONE, "No forecast"),
)
#: The orders the automatic search tries: p and q up to this; P and Q up to 1.
AUTO_MAX_ORDER = 2
#: A season stronger than this (STL's seasonal strength, 0 to 1) is differenced once (Hyndman's rule).
SEASONAL_STRENGTH_LIMIT = 0.64
#: The prediction interval's level.
CONFIDENCE = 0.95


@dataclass(slots=True)
class TimeSeriesResult:
    measure: str
    time: str
    temporal: bool
    model: str
    season: int
    #: Time, the measure, and - when computed - Fitted, Residual, Trend, Seasonal, Remainder.
    rows: pd.DataFrame
    #: Lag, ACF, PACF, Ljung-Box Q, Prob>Q, and the band.
    correlations: pd.DataFrame
    #: Time, Forecast, Lower 95%, Upper 95%.
    forecast: pd.DataFrame
    parameters: pd.DataFrame
    #: Randomness and serial-independence diagnostics of fitted residuals.
    residual_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: ADF and KPSS on the series as it is and differenced: Test, Series, Statistic, Prob, Says.
    stationarity: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The automatic search's models, best AIC first: Model, AIC, BIC.
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Cross-correlation with the input: Lag, CCF, Upper band, Lower band (empty without an input).
    cross: pd.DataFrame = field(default_factory=pd.DataFrame)
    input: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def stationarity(y: np.ndarray) -> tuple[pd.DataFrame, int]:
    """ADF (null: a unit root) and KPSS (null: stationary) on *y*, then differenced; and the differences
    it needs - the first of 0, 1, 2 where ADF rejects and KPSS does not."""
    from statsmodels.tsa.stattools import adfuller, kpss

    rows = []
    needed = None
    series = np.asarray(y, float)
    for d in range(3):
        if series.size < 8:
            break
        if np.ptp(series) == 0:  # a straight line, differenced: nothing left to test - it is stationary
            needed = d if needed is None else needed
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS's p is read from a table: clipped to 0.01-0.10, said so
            adf_stat, adf_p, *_rest = adfuller(series, autolag="AIC")
            kpss_stat, kpss_p, *_rest = kpss(series, regression="c", nlags="auto")
        label = {0: "As it is", 1: "Differenced once", 2: "Differenced twice"}[d]
        rows.append({"Test": "ADF (unit root)", "Series": label, "Statistic": float(adf_stat),
                     "Prob": float(adf_p), "Says": "stationary" if adf_p < 0.05 else "not stationary"})
        rows.append({"Test": "KPSS (stationary)", "Series": label, "Statistic": float(kpss_stat),
                     "Prob": float(kpss_p), "Says": "not stationary" if kpss_p < 0.05 else "stationary"})
        if needed is None and adf_p < 0.05 and kpss_p >= 0.05:
            needed = d
            break
        series = np.diff(series)
    return pd.DataFrame(rows), 1 if needed is None else needed


def seasonal_strength(y: np.ndarray, season: int) -> float:
    """STL's seasonal strength, 0 (none) to 1: how much of the detrended variation the season is."""
    from statsmodels.tsa.seasonal import STL

    parts = STL(y, period=season, robust=True).fit()
    detrended = parts.seasonal + parts.resid
    variance = float(np.var(detrended))
    return max(0.0, 1.0 - float(np.var(parts.resid)) / variance) if variance > 0 else 0.0


def _arima_label(order: tuple[int, int, int], seasonal: tuple[int, int, int], season: int) -> str:
    label = f"ARIMA({', '.join(str(int(v)) for v in order)})"
    if season >= 2 and any(seasonal):
        label += f"({', '.join(str(int(v)) for v in seasonal)}){season}"
    return label


def search_arima(y: np.ndarray, d: int, season: int, seasonal_d: int,
                 exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every ARIMA(p, d, q)(P, D, Q) with p, q up to AUTO_MAX_ORDER and P, Q up to 1, by AIC (with *exog*)."""
    from statsmodels.tsa.arima.model import ARIMA as ArimaModel

    seasonals = [(0, seasonal_d, 0)]
    if season >= 2:
        seasonals += [(1, seasonal_d, 0), (0, seasonal_d, 1)]
    rows = []
    for p in range(AUTO_MAX_ORDER + 1):
        for q in range(AUTO_MAX_ORDER + 1):
            for seasonal in seasonals:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        fitted = ArimaModel(y, exog=exog, order=(p, d, q), seasonal_order=(*seasonal, season)
                                            if season >= 2 else (0, 0, 0, 0)).fit()
                    except Exception:  # noqa: BLE001 - statsmodels' own failures: a model left out of the search
                        continue
                if np.isfinite(fitted.aic):
                    rows.append({"Model": _arima_label((p, d, q), seasonal, season), "AIC": float(fitted.aic),
                                 "BIC": float(fitted.bic), "order": (p, d, q), "seasonal": seasonal})
    if not rows:
        raise ValueError("No ARIMA model could be fitted to these values.")
    return pd.DataFrame(rows).sort_values("AIC", kind="stable").reset_index(drop=True)


def prewhiten(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """X and Y filtered by X's own autoregression (its order by AIC, up to 10): (x, y, order).

    What is left of X is noise; a correlation of Y with it is Y answering X,
    not two series that each wander (Box and Jenkins' prewhitening).
    """
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order

    xc, yc = x - x.mean(), y - y.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chosen = ar_select_order(xc, maxlag=max(1, min(10, x.size // 5)), ic="aic", trend="n").ar_lags
        order = int(max(chosen)) if chosen else 0
        if order == 0:
            return xc, yc, 0
        phi = np.asarray(AutoReg(xc, lags=order, trend="n").fit().params, float)

    def filtered(z: np.ndarray) -> np.ndarray:
        return z[order:] - sum(phi[i] * z[order - 1 - i: z.size - 1 - i] for i in range(order))

    return filtered(xc), filtered(yc), order


def cross_correlation(y: np.ndarray, x: np.ndarray, lags: int) -> pd.DataFrame:
    """The correlation of y now with x *k* steps earlier, k from -lags to lags (positive: x leads)."""
    yc, xc = y - y.mean(), x - x.mean()
    scale = float(np.sqrt(np.sum(yc ** 2) * np.sum(xc ** 2)))
    n = y.size
    values = []
    for k in range(-lags, lags + 1):
        if k >= 0:
            values.append(float(np.sum(yc[k:] * xc[: n - k])) / scale)
        else:
            values.append(float(np.sum(yc[: n + k] * xc[-k:])) / scale)
    band = 2.0 / np.sqrt(n)
    return pd.DataFrame({"Lag": np.arange(-lags, lags + 1), "CCF": values, "Upper band": band, "Lower band": -band})


def _order(data: pd.DataFrame, time: str) -> tuple[pd.Series, bool]:
    """The time column as numbers or dates, and whether it is dates."""
    values = data[time]
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().all():
        return numbers, False
    dates = pd.to_datetime(values, errors="coerce", format="mixed")
    if dates.notna().all():
        return dates, True
    raise ValueError(f"{time} is neither numbers nor dates.")


#: (typical spacing in days, season length, months per step): a year of steps, by the spacing.
_SPACINGS: tuple[tuple[float, int, int], ...] = (
    (1 / 24, 24, 0), (1.0, 7, 0), (7.0, 52, 0), (30.44, 12, 1), (91.31, 4, 3),
)


def _spacing(dates: pd.Series) -> tuple[int, int]:
    """(season length, months per step) of dates spaced about regularly - monthly ones on the 15th
    too, which pandas' own inference refuses; (0, 0) when the spacing is none of the usual ones."""
    days = pd.Series(pd.DatetimeIndex(dates).sort_values()).diff().dt.total_seconds().dropna() / 86400
    if days.empty:
        return 0, 0
    typical = float(days.median())
    for spacing, season, months in _SPACINGS:
        if abs(typical - spacing) <= 0.12 * spacing:
            return season, months
    return 0, 0


def season_from_dates(dates: pd.Series) -> int:
    """The usual season length of dates spaced regularly: 12 for months, 7 for days...; 0 when unknown."""
    if len(dates) < 3:
        return 0
    return _spacing(dates)[0]


def _future(times: pd.Series, temporal: bool, steps: int) -> pd.Series:
    """The *steps* times after the last, at the series' own spacing."""
    if steps <= 0:
        return pd.Series([], dtype=object)
    if temporal:
        index = pd.DatetimeIndex(times)
        months = _spacing(times)[1] if len(index) >= 3 else 0
        if months:  # whole months or quarters: the 15th stays the 15th
            return pd.Series([index[-1] + pd.DateOffset(months=months * (k + 1)) for k in range(steps)])
        step = pd.Series(index).diff().median()
        return pd.Series([index[-1] + step * (k + 1) for k in range(steps)])
    numbers = times.to_numpy(float)
    step = float(np.median(np.diff(numbers))) if numbers.size > 1 else 1.0
    return pd.Series(numbers[-1] + step * np.arange(1, steps + 1))


def _forecast_input(x: np.ndarray, steps: int) -> tuple[np.ndarray, str]:
    """*steps* values of the input ahead, by its own ARIMA chosen by AIC (no season): (values, label)."""
    from statsmodels.tsa.arima.model import ARIMA as ArimaModel

    _tests, d = stationarity(x)
    best = search_arima(x, d, 0, 0).iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = ArimaModel(x, order=best["order"]).fit().get_forecast(steps).predicted_mean
    return np.asarray(values, float), str(best["Model"])


def _exogenous(data: pd.DataFrame, times: pd.Series, temporal: bool, input: str, lead: int, intervention: str,
               kind: str, notes: list[str]) -> tuple[pd.DataFrame | None, Any, int]:
    """The ARIMA's regressors - the input at its lead, an intervention's step or pulse - their values
    ahead (a function of the horizon), and how many first rows the lead leaves without one."""
    columns: dict[str, np.ndarray] = {}
    future: dict[str, Any] = {}
    skip = 0
    n = len(data)
    if input:
        x = data[input].to_numpy(float)
        name = f"{input} (lag {lead})"
        columns[name] = np.concatenate([np.full(lead, np.nan), x[: n - lead]]) if lead else x
        skip = lead

        def ahead(steps: int, x: np.ndarray = x, lead: int = lead) -> np.ndarray:
            known = x[n - lead: n][:steps] if lead else np.empty(0)
            if steps <= known.size:
                return known
            more, label = _forecast_input(x, steps - known.size)
            notes.append(f"Beyond {known.size} step(s) ahead the forecast uses {input}'s own forecast "
                         f"({label}): its prediction interval does not include that forecast's uncertainty.")
            return np.concatenate([known, more])

        future[name] = ahead
        notes.append(f"Transfer function: {input} enters the model {lead} step(s) earlier, the lag where "
                     "its prewhitened cross-correlation is strongest.")
    if intervention.strip():
        if temporal:
            when = pd.to_datetime(intervention.strip(), errors="coerce", dayfirst=True)
            moments = pd.DatetimeIndex(times)
        else:
            when = pd.to_numeric(pd.Series([intervention.strip()]), errors="coerce").iloc[0]
            moments = pd.Index(times.to_numpy(float))
        if pd.isna(when):
            raise ValueError(f"The intervention '{intervention}' is not a {'date' if temporal else 'number'} "
                             "of the time column.")
        after = np.asarray(moments >= when, dtype=float)
        if not after.any() or after.all():
            raise ValueError(f"The intervention '{intervention}' is not inside the series' time span.")
        if kind == "pulse":
            dummy = np.zeros(n)
            dummy[int(np.argmax(after))] = 1.0
            future["Intervention (pulse)"] = lambda steps: np.zeros(steps)
            columns["Intervention (pulse)"] = dummy
        else:
            future["Intervention (step)"] = lambda steps: np.ones(steps)
            columns["Intervention (step)"] = after
        notes.append(f"Intervention at {intervention}: its estimate is the change the series took then "
                     f"({'a lasting step' if kind != 'pulse' else 'a one-time pulse'}).")
    if not columns:
        return None, None, 0
    exog = pd.DataFrame(columns)

    def future_values(steps: int) -> pd.DataFrame:
        return pd.DataFrame({name: make(steps) for name, make in future.items()})

    return exog, future_values, skip


def analyse(frame: pd.DataFrame, measure: str, time: str = ROW_ORDER, *, model: str = ARIMA,
            order: tuple[int, int, int] = (1, 0, 0), seasonal_order: tuple[int, int, int] = (0, 0, 0),
            season: int = -1, horizon: int = 12, lags: int = 25, input: str = "", input_in_model: bool = False,
            intervention: str = "", intervention_kind: str = "step") -> TimeSeriesResult:
    """*measure* in the order of *time*: autocorrelations, its decomposition and a forecast.

    *season* -1 finds it from the dates (none for numbers), 0 means none. *seasonal_order* (P, D, Q)
    applies with a season. *input*, another column, gives the cross-correlation with the measure and,
    *input_in_model*, enters an ARIMA at the lag its prewhitened cross-correlation peaks (a transfer
    function). *intervention*, a time, adds a step (or pulse) from then on to an ARIMA.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf

    columns = [measure, time] if time else [measure]
    if input:
        columns.append(input)
    data = frame[columns].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    if input:
        data[input] = pd.to_numeric(data[input], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(f"{len(data)} values are too few for a time series: at least 10 are needed.")
    notes: list[str] = []
    if time:
        times, temporal = _order(data, time)
        sort = np.argsort(times.to_numpy(), kind="stable")
        data, times = data.iloc[sort].reset_index(drop=True), times.iloc[sort].reset_index(drop=True)
        if times.duplicated().any():
            notes.append(f"Some times repeat in {time}: the values are taken in row order.")
    else:
        times, temporal = pd.Series(np.arange(1, len(data) + 1, dtype=float)), False
    y = data[measure].to_numpy(float)
    n = y.size
    if season < 0:
        season = season_from_dates(times) if temporal else 0
    if season and n < 2 * season + 1:
        notes.append(f"A season of {season} needs at least {2 * season + 1} values: no seasonal part.")
        season = 0

    lags = int(max(1, min(lags, n // 2 - 1)))
    acf_values = acf(y, nlags=lags, fft=True)
    pacf_values = pacf(y, nlags=lags, method="ywm")
    ljung = acorr_ljungbox(y, lags=list(range(1, lags + 1)))
    band = 2.0 / np.sqrt(n)
    correlations = pd.DataFrame({
        "Lag": np.arange(1, lags + 1), "ACF": acf_values[1:], "PACF": pacf_values[1:],
        "Ljung-Box Q": ljung["lb_stat"].to_numpy(), "Prob>Q": ljung["lb_pvalue"].to_numpy(),
        "Upper band": band, "Lower band": -band,
    })

    rows = pd.DataFrame({"Time": times.dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "")
                         if temporal else times, measure: y})
    if season >= 2:
        from statsmodels.tsa.seasonal import STL

        parts = STL(y, period=season, robust=True).fit()
        rows["Trend"], rows["Seasonal"], rows["Remainder"] = parts.trend, parts.seasonal, parts.resid

    summary: dict[str, Any] = {"Values": n, "Mean": float(y.mean()), "Std Dev": float(y.std(ddof=1)),
                               "Season length": season or "none"}
    tests, differences = stationarity(y)
    summary["Differences suggested (ADF, KPSS)"] = differences
    cross = pd.DataFrame()
    lead = 0
    if input:
        x_values = data[input].to_numpy(float)
        cross = cross_correlation(y, x_values, lags)
        white_x, white_y, ar_order = prewhiten(x_values, y)
        cross["Prewhitened CCF"] = cross_correlation(white_y, white_x, lags)["CCF"].to_numpy()
        cross["Prewhitened band"] = 2.0 / np.sqrt(white_y.size)
        strongest = cross.loc[cross["Prewhitened CCF"].abs().idxmax()]
        summary[f"Strongest cross-correlation with {input}"] = (
            f"{strongest['Prewhitened CCF']:.3f} at lag {int(strongest['Lag'])} (prewhitened)")
        ahead = cross[cross["Lag"] >= 0]
        lead = int(ahead.loc[ahead["Prewhitened CCF"].abs().idxmax(), "Lag"])
        notes.append(f"Lag k > 0: {measure} follows {input} by k steps; k < 0: it leads. Prewhitened: both "
                     f"filtered by {input}'s own AR({ar_order}), so that two series which merely wander do "
                     "not look related.")
    exog, exog_future_parts, skip = _exogenous(data, times, temporal, input if input_in_model else "", lead,
                                               intervention, intervention_kind, notes)
    y_fit = y[skip:]
    exog_fit = exog.iloc[skip:].reset_index(drop=True) if exog is not None else None
    candidates = pd.DataFrame()
    seasonal = tuple(int(v) for v in seasonal_order) if season >= 2 else (0, 0, 0)
    if model == AUTO:
        seasonal_d = 0
        if season >= 2:
            strength = seasonal_strength(y, season)
            summary["Seasonal strength"] = strength
            seasonal_d = int(strength > SEASONAL_STRENGTH_LIMIT)
        # The season's differencing first: what is left is tested again for the ordinary one.
        rest = y[season:] - y[:-season] if seasonal_d else y
        _tests, d = stationarity(rest)
        candidates = search_arima(y_fit, d, season, seasonal_d, exog_fit)
        best = candidates.iloc[0]
        order, seasonal = best["order"], best["seasonal"]
        notes.append(f"{best['Model']}: the lowest AIC of {len(candidates)} models, with d = {d} from the "
                     f"ADF and KPSS tests" + (f" and D = {seasonal_d} from the season's strength."
                                              if season >= 2 else "."))
        candidates = candidates.drop(columns=["order", "seasonal"])
    forecast = pd.DataFrame(columns=["Time", "Forecast", f"Lower {CONFIDENCE:.0%}", f"Upper {CONFIDENCE:.0%}"])
    parameters = pd.DataFrame(columns=["Term", "Estimate", "Std Error", "Prob>|z|"])
    residual_diagnostics = pd.DataFrame(columns=["Test", "n", "Statistic", "Prob", "Note"])
    if model != NONE:
        alpha = 1 - CONFIDENCE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter: the AIC and residuals say how it went
            if model in (ARIMA, AUTO):
                from statsmodels.tsa.arima.model import ARIMA as ArimaModel

                fitted = ArimaModel(y_fit, exog=exog_fit, order=tuple(int(v) for v in order),
                                    seasonal_order=(*seasonal, season) if season >= 2 and any(seasonal)
                                    else (0, 0, 0, 0)).fit()
                future_exog = exog_future_parts(horizon) if exog is not None else None
                prediction = fitted.get_forecast(horizon, exog=future_exog).summary_frame(alpha=alpha)
                mean, low, high = prediction["mean"], prediction["mean_ci_lower"], prediction["mean_ci_upper"]
                fitted_values = np.concatenate([np.full(skip, np.nan), np.asarray(fitted.fittedvalues, float)])
                names = list(fitted.param_names)
                standard_errors, p_values = fitted.bse, fitted.pvalues
                label = _arima_label(tuple(order), seasonal, season)
            elif model == ETS:
                from statsmodels.tsa.exponential_smoothing.ets import ETSModel

                fitted = ETSModel(pd.Series(y), error="add", trend="add", seasonal="add" if season >= 2 else None,
                                  seasonal_periods=season if season >= 2 else None).fit(disp=False)
                prediction = fitted.get_prediction(start=n, end=n + horizon - 1).summary_frame(alpha=alpha)
                mean, low, high = prediction["mean"], prediction["pi_lower"], prediction["pi_upper"]
                fitted_values = fitted.fittedvalues
                names = list(fitted.param_names)
                standard_errors = getattr(fitted, "bse", np.full(len(names), np.nan))
                p_values = getattr(fitted, "pvalues", np.full(len(names), np.nan))
                label = "ETS (A, A" + (", A)" if season >= 2 else ", N)")
            else:
                raise ValueError(f"'{model}' is not one of {', '.join(key for key, _label in MODELS)}")
        residuals = y - np.asarray(fitted_values, dtype=float)
        # ARIMA's first fitted values only start the differencing (the first season's, too): left empty.
        start = (skip + max(order[1], 1) + (season * seasonal[1] if season >= 2 else 0)) if model in (ARIMA, AUTO) else 0
        shown_fitted, shown_residuals = np.asarray(fitted_values, dtype=float).copy(), residuals.copy()
        shown_fitted[:start], shown_residuals[:start] = np.nan, np.nan
        rows["Fitted"], rows["Residual"] = shown_fitted, shown_residuals
        future = _future(times, temporal, horizon)
        forecast = pd.DataFrame({
            "Time": future.dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "") if temporal else future,
            "Forecast": np.asarray(mean, float), f"Lower {CONFIDENCE:.0%}": np.asarray(low, float),
            f"Upper {CONFIDENCE:.0%}": np.asarray(high, float),
        })
        parameters = pd.DataFrame({"Term": names, "Estimate": np.asarray(fitted.params, float),
                                   "Std Error": np.asarray(standard_errors, float),
                                   "Prob>|z|": np.asarray(p_values, float)})
        burn = start  # ARIMA's first fitted values are the start, not a fit
        summary.update({"Model": label, "AIC": float(fitted.aic), "BIC": float(fitted.bic),
                        "RMSE of residuals": float(np.sqrt(np.mean(residuals[burn:] ** 2))),
                        "Forecast periods": horizon})
        residual_values = residuals[burn:]
        diagnostic_lag = min(10, max(1, residual_values.size // 5))
        try:
            diagnostics = all_diagnostics(
                residual_values,
                lags=tuple(dict.fromkeys((1, 5, diagnostic_lag))),
                model_df=0,
                centre=0.0,
            )
            residual_diagnostics = pd.DataFrame([
                {"Test": item.test, "n": item.n, "Statistic": item.statistic,
                 "Prob": item.pvalue, "Note": item.note}
                for item in diagnostics
            ])
            for item in diagnostics:
                if not item.available:
                    applogger.warning(f"Residual diagnostic {item.test}: {item.note}")
            ljung_rows = [item for item in diagnostics if item.test == f"Ljung-Box (lag {diagnostic_lag})"]
            if ljung_rows and np.isfinite(ljung_rows[0].pvalue):
                ljung_p = float(ljung_rows[0].pvalue)
                summary["Residuals: Ljung-Box Prob>Q"] = ljung_p
                if ljung_p < 0.05:
                    notes.append("The residuals are still autocorrelated (Ljung-Box p < 0.05): the model leaves "
                                 "a pattern behind - try another order, or a season.")
        except Exception as exc:  # noqa: BLE001 - the fitted model remains usable
            applogger.error(f"Residual randomness diagnostics failed: {exc}")
            notes.append(f"Residual randomness diagnostics could not be computed: {exc}")
    return TimeSeriesResult(measure=measure, time=time, temporal=temporal, model=model, season=season,
                            rows=rows, correlations=correlations, forecast=forecast, parameters=parameters,
                            residual_diagnostics=residual_diagnostics,
                            stationarity=tests, candidates=candidates, cross=cross, input=input,
                            summary=summary, notes=notes)
