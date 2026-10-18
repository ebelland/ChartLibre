"""Time Series: a measure in time order - its memory, its parts and its future (AN-05).

As JMP's Time Series platform, for one measure ordered by a date, a time or
a number (the rows' order when there is none):

* Autocorrelation (ACF) and partial autocorrelation (PACF) up to a lag,
  with the Ljung-Box test of "no autocorrelation up to here" at every lag
  and the +/-2 standard error band under which a bar says nothing.
* Seasonal decomposition (STL): trend, seasonal part and remainder, for a
  season length given or found from the dates (12 for months, 7 for days,
  4 for quarters, 24 for hours, 52 for weeks).
* A forecast: ARIMA (p, d, q) or exponential smoothing (ETS: additive
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

ROW_ORDER = ""
ARIMA = "arima"
ETS = "ets"
NONE = "none"
MODELS: tuple[tuple[str, str], ...] = (
    (ETS, "Exponential smoothing (ETS)"), (ARIMA, "ARIMA (p, d, q)"), (NONE, "No forecast"),
)
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
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


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


def analyse(frame: pd.DataFrame, measure: str, time: str = ROW_ORDER, *, model: str = ARIMA,
            order: tuple[int, int, int] = (1, 0, 0), season: int = -1, horizon: int = 12,
            lags: int = 25) -> TimeSeriesResult:
    """*measure* in the order of *time*: autocorrelations, its decomposition and a forecast.

    *season* -1 finds it from the dates (none for numbers), 0 means none.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf

    columns = [measure, time] if time else [measure]
    data = frame[columns].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
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
    forecast = pd.DataFrame(columns=["Time", "Forecast", f"Lower {CONFIDENCE:.0%}", f"Upper {CONFIDENCE:.0%}"])
    parameters = pd.DataFrame(columns=["Term", "Estimate", "Std Error", "Prob>|z|"])
    if model != NONE:
        alpha = 1 - CONFIDENCE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter: the AIC and residuals say how it went
            if model == ARIMA:
                from statsmodels.tsa.arima.model import ARIMA as ArimaModel

                fitted = ArimaModel(y, order=tuple(int(v) for v in order)).fit()
                prediction = fitted.get_forecast(horizon).summary_frame(alpha=alpha)
                mean, low, high = prediction["mean"], prediction["mean_ci_lower"], prediction["mean_ci_upper"]
                fitted_values = fitted.fittedvalues
                names = list(fitted.param_names)
                standard_errors, p_values = fitted.bse, fitted.pvalues
                label = f"ARIMA({', '.join(str(int(v)) for v in order)})"
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
        rows["Fitted"], rows["Residual"] = np.asarray(fitted_values, dtype=float), residuals
        future = _future(times, temporal, horizon)
        forecast = pd.DataFrame({
            "Time": future.dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "") if temporal else future,
            "Forecast": np.asarray(mean, float), f"Lower {CONFIDENCE:.0%}": np.asarray(low, float),
            f"Upper {CONFIDENCE:.0%}": np.asarray(high, float),
        })
        parameters = pd.DataFrame({"Term": names, "Estimate": np.asarray(fitted.params, float),
                                   "Std Error": np.asarray(standard_errors, float),
                                   "Prob>|z|": np.asarray(p_values, float)})
        burn = max(order[1], 1) if model == ARIMA else 0  # ARIMA's first fitted values are the start, not a fit
        summary.update({"Model": label, "AIC": float(fitted.aic), "BIC": float(fitted.bic),
                        "RMSE of residuals": float(np.sqrt(np.mean(residuals[burn:] ** 2))),
                        "Forecast periods": horizon})
        white = acorr_ljungbox(residuals[burn:], lags=[min(10, max(1, (n - burn) // 5))])
        summary["Residuals: Ljung-Box Prob>Q"] = float(white["lb_pvalue"].iloc[0])
        if float(white["lb_pvalue"].iloc[0]) < 0.05:
            notes.append("The residuals are still autocorrelated (Ljung-Box p < 0.05): the model leaves "
                         "a pattern behind - try another order, or a season.")
    return TimeSeriesResult(measure=measure, time=time, temporal=temporal, model=model, season=season,
                            rows=rows, correlations=correlations, forecast=forecast, parameters=parameters,
                            summary=summary, notes=notes)
