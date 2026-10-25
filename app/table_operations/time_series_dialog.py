"""Time Series (Analysis): a measure in time order - its autocorrelation, its parts, its forecast (AN-05).

As JMP's Time Series platform: Y is the measure, the time column orders it
(the rows' order without one); an input X gives the cross-correlation. The report gives the autocorrelation and
partial autocorrelation with the Ljung-Box test, the ADF and KPSS tests of
stationarity, the models tried when the order is chosen by AIC, the model's
parameters and fit, and the forecast with its 95% prediction interval. OK writes the rows
(fitted values, residuals, the seasonal decomposition), the correlations
and the forecast as tables, and draws the series with its forecast, the
ACF, the PACF and - with a season - the trend, seasonal part and remainder.
The engine is app.analysis.time_series.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import time_series as tsa
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, IntParam, TextParam
from app.plugin_api import reading_guide
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class TimeSeriesDialog(TableOperationDialogBase):
    """Autocorrelation, seasonal decomposition and an ARIMA or ETS forecast of one measure."""

    Name = "Time Series"
    Description = (
        "A measure in time order: its autocorrelation (ACF, PACF, Ljung-Box), its trend and season "
        "(STL), and an ARIMA or exponential-smoothing forecast with its prediction interval."
    )
    Category = "Predictive and Specialized Modeling"
    CHART_PANELS = ('Time Series', 'Stem Plot')
    Order = 36
    Icon = """
    <path d="M3 16l3-4 3 2 3-6 3 3"/><path d="M15 11l3-2 3 1" stroke-dasharray="2 2"/>
    <path d="M15 8l6-2M15 14l6 2" stroke-dasharray="1 2"/><path d="M3 21h18"/>
    """
    Keywords: str = (
        "time series serie temporale serie storica forecast previsione arima ets exponential smoothing "
        "livellamento esponenziale holt winters acf pacf autocorrelation autocorrelazione ljung-box "
        "stl seasonal decomposition decomposizione stagionale season stagione trend"
    )
    Goals: list[str] = ["signal", "predict"]

    ROLES = (
        ColumnRole("response", "Y, time series", "The measure, one value per time.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1, maximum=1),
        ColumnRole("time", "Time or order", "Optional: dates, times or a run number. Without it, the rows' order.",
                   maximum=1),
        ColumnRole("input", "X, input", "Optional: another series measured at the same times - the "
                                        "cross-correlation says whether it leads or follows Y, and by how much.",
                   kinds=(CONTINUOUS, ORDINAL), maximum=1),
    )
    DISTINCT_ROLES = (
        ("response", "time", "The measure and the time must be two different columns."),
        ("input", "response", "The input must be another column than the measure and the time."),
        ("input", "time", "The input must be another column than the measure and the time."),
    )
    PARAMETERS_TAB = "Time Series"
    PARAMS = (
        ChoiceParam("model", "Forecast:", choices=tuple((label, key) for key, label in tsa.MODELS),
                    tooltip="ARIMA: the next value from the last ones (p), the changes (d) and the last "
                            "errors (q), with a season (P, D, Q) of its own. ARIMA, order chosen by AIC: the "
                            "differences from the stationarity tests, then the p, q, P, Q with the lowest AIC. "
                            "Exponential smoothing: a level, a trend and a season that each follow the newest "
                            "values most."),
        IntParam("p", "ARIMA p (autoregressive):", tooltip="How many past values the next depends on: the "
                                                         "lags where the PACF stands out.",
                 default_value=1, minimum=0, maximum=10),
        IntParam("d", "ARIMA d (differences):", tooltip="1 for a series that wanders or has a trend; 0 for one "
                                                       "that comes back to its mean.",
                 default_value=0, minimum=0, maximum=2),
        IntParam("q", "ARIMA q (moving average):", tooltip="How many past errors the next value depends on: the "
                                                          "lags where the ACF stands out.",
                 default_value=0, minimum=0, maximum=10),
        IntParam("P", "Seasonal P:", tooltip="ARIMA with a season: how many past seasons the next value "
                                             "depends on (the ACF's spikes at the season's multiples).",
                 default_value=0, minimum=0, maximum=3),
        IntParam("D", "Seasonal D:", tooltip="1 takes away last season's value: for a season that repeats "
                                             "steadily.",
                 default_value=0, minimum=0, maximum=1),
        IntParam("Q", "Seasonal Q:", tooltip="How many past seasons' errors the next value depends on.",
                 default_value=0, minimum=0, maximum=3),
        IntParam("season", "Season length:", tooltip="0 finds it from the dates (12 for months, 7 for days, "
                                                     "4 for quarters); 1 means none.",
                 default_value=0, minimum=0, maximum=400),
        IntParam("horizon", "Forecast periods:", tooltip="How many times ahead to forecast.",
                 default_value=12, minimum=1, maximum=1000),
        BoolParam("input_in_model", "X, input in the model (transfer function)",
                  tooltip="ARIMA only: the input enters the model at the lag where its prewhitened "
                          "cross-correlation is strongest; its coefficient says how much Y moves per unit of X. "
                          "Ahead of that lag the forecast uses X's own forecast.",
                  default_value=False, advanced=True),
        TextParam("intervention", "Intervention at:", placeholder="2020-03-01",
                  tooltip="ARIMA only: a date (or a number of the time column) when something changed - a new "
                          "process, a policy, a fault. Its estimate is the change it made.", advanced=True),
        ChoiceParam("intervention_kind", "Intervention:", choices=(("Step (lasting)", "step"),
                                                                   ("Pulse (one time)", "pulse")),
                    tooltip="A step lasts from then on; a pulse affects that one time only.", advanced=True),
        IntParam("lags", "Lags:", tooltip="How far back the autocorrelations look.",
                 default_value=25, minimum=1, maximum=500, advanced=True),
        BoolParam(
            "show_residual_runs",
            "Show residual runs test",
            tooltip=("Wald-Wolfowitz test of whether positive and negative residuals occur in a random "
                     "order around zero. Ties at zero are omitted."),
            default_value=True,
            advanced=True,
        ),
        BoolParam(
            "show_residual_turning_points",
            "Show residual turning-point test",
            tooltip=("Tests whether the number of local peaks and troughs in the residuals is consistent "
                     "with a random sequence."),
            default_value=True,
            advanced=True,
        ),
        BoolParam(
            "show_residual_durbin_watson",
            "Show residual Durbin-Watson statistic",
            tooltip=("Reports the Durbin-Watson diagnostic for lag-one residual correlation. Values near "
                     "2 indicate little lag-one correlation; no generic p-value is reported."),
            default_value=True,
            advanced=True,
        ),
        BoolParam(
            "show_residual_ljung_box",
            "Show residual Ljung-Box tests",
            tooltip=("Tests whether residual autocorrelations through the selected lags are jointly zero."),
            default_value=True,
            advanced=True,
        ),
    )

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        time = casting.columns("time")
        given = casting.columns("input")
        season = int(params.get("season", 0) or 0)
        result = tsa.analyse(
            frame, casting.columns("response")[0], time[0] if time else tsa.ROW_ORDER,
            model=str(params.get("model") or tsa.ARIMA),
            order=(int(params.get("p", 1)), int(params.get("d", 0)), int(params.get("q", 0))),
            seasonal_order=(int(params.get("P", 0)), int(params.get("D", 0)), int(params.get("Q", 0))),
            input=given[0] if given else "",
            input_in_model=bool(params.get("input_in_model", False)),
            intervention=str(params.get("intervention") or ""),
            intervention_kind=str(params.get("intervention_kind") or "step"),
            season=-1 if season == 0 else (0 if season == 1 else season),
            horizon=int(params.get("horizon", 12)), lags=int(params.get("lags", 25)),
        )
        # BoolParam values are restored and saved by the operation framework with
        # the other declared PARAMS. Filter only the presentation table: the
        # engine still computes every diagnostic, so model warnings and summary
        # values remain reproducible independently of the UI selection.
        if not result.residual_diagnostics.empty:
            enabled = {
                "Wald-Wolfowitz runs": bool(params.get("show_residual_runs", True)),
                "Turning-point randomness": bool(params.get("show_residual_turning_points", True)),
                "Durbin-Watson": bool(params.get("show_residual_durbin_watson", True)),
                "Ljung-Box": bool(params.get("show_residual_ljung_box", True)),
            }
            names = result.residual_diagnostics["Test"].astype(str)
            keep = names.map(lambda name: enabled["Ljung-Box"] if name.startswith("Ljung-Box")
                             else enabled.get(name, True))
            result.residual_diagnostics = result.residual_diagnostics.loc[keep].reset_index(drop=True)
        return result

    def report_body(self, result: tsa.TimeSeriesResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), str(value) if isinstance(value, str) else report_html.format_number(value))
            for key, value in result.summary.items())))
        if not result.parameters.empty:
            blocks.append(report_html.section(_("Parameter estimates"),
                                              self.report_table(result.parameters, p_columns=("Prob>|z|",))))
        if not result.candidates.empty:
            blocks.append(report_html.section(_("Models tried, best AIC first"),
                                              self.report_table(result.candidates.head(10), digits=5)))
        if not result.forecast.empty:
            blocks.append(report_html.section(_("Forecast"), self.report_table(result.forecast, digits=5)))
        if not result.stationarity.empty:
            blocks.append(report_html.section(_("Stationarity tests"),
                                              self.report_table(result.stationarity, p_columns=("Prob",), digits=4)))
        if not result.residual_diagnostics.empty:
            blocks.append(report_html.section(
                _("Residual randomness and independence"),
                self.report_table(result.residual_diagnostics, p_columns=("Prob",), digits=4),
            ))
        if not result.cross.empty:
            order = result.cross["Prewhitened CCF"].abs().sort_values(ascending=False).index
            strongest = result.cross.reindex(order).head(8)
            blocks.append(report_html.section(_("Cross-correlation with {input}, strongest lags").format(
                input=result.input), self.report_table(strongest[["Lag", "Prewhitened CCF", "CCF"]], digits=4)))
        shown = result.correlations[["Lag", "ACF", "PACF", "Ljung-Box Q", "Prob>Q"]].head(12)
        blocks.append(report_html.section(_("Autocorrelation, first 12 lags"),
                                          self.report_table(shown, p_columns=("Prob>Q",), digits=4)))
        return "".join(blocks)

    def how_to_read(self, result: tsa.TimeSeriesResult) -> list[str]:
        lines = []
        tests = result.stationarity
        if not tests.empty and "ADF (unit root)" in set(tests["Test"]):
            adf = tests[tests["Test"] == "ADF (unit root)"].iloc[0]
            lines.append(reading_guide.p_value(
                adf["Prob"], null=_("{measure} wandered with no level to come back to").format(measure=result.measure),
                shown=_("{measure} comes back to a stable level").format(measure=result.measure)))
        if not result.candidates.empty:
            runner_up = result.candidates["AIC"].iloc[1] if len(result.candidates) > 1 else None
            lines.append(reading_guide.aic(result.candidates["AIC"].iloc[0], model=str(result.candidates["Model"].iloc[0]),
                                           runner_up=runner_up))
        return lines

    def apply_results(self, result: tsa.TimeSeriesResult) -> list[dict[str, Any]]:
        frames = {"time_series": result.rows, "time_series_acf": result.correlations}
        if not result.residual_diagnostics.empty:
            frames["time_series_residual_diagnostics"] = result.residual_diagnostics
        if not result.cross.empty:
            frames["time_series_ccf"] = result.cross
        if not result.forecast.empty:
            frames["time_series_forecast"] = result.forecast
        tables = self.write_result_tables(frames)
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]
        measure, rows = result.measure, tables["time_series"]
        x_label = result.time or _("Row")
        options: dict[str, Any] = {"show_rolling": False}
        if result.temporal:
            # A gap is five typical steps, not the renderer's 2 h - which cut daily values into dots.
            moments = pd.to_datetime(result.rows["Time"], errors="coerce", format="mixed").dropna()
            step = moments.diff().median() if len(moments) > 1 else pd.NaT
            if pd.notna(step) and step > pd.Timedelta(0):
                options["gap_threshold"] = f"{int((5 * step).total_seconds())}s"

        sql, roles = self.series_select(rows, {"x": "Time", "y": measure})
        series = [(measure, sql, roles, {"linestyle": "-", "marker": "o", "markersize": 3})]
        if "Fitted" in result.rows:
            sql, roles = self.series_select(rows, {"x": "Time", "y": "Fitted"})
            series.append((_("Fitted"), sql, roles, {"linestyle": "--", "marker": "", "color": "#7c3aed"}))
        if "time_series_forecast" in tables:
            forecast = tables["time_series_forecast"]
            sql, roles = self.series_select(forecast, {"x": "Time", "y": "Forecast"})
            series.append((_("Forecast"), sql, roles, {"linestyle": "-", "marker": "", "color": "#d62728"}))
            for bound in ("Lower 95%", "Upper 95%"):
                sql, roles = self.series_select(forecast, {"x": "Time", "y": bound})
                series.append((_("95% prediction interval"), sql, roles,
                               {"linestyle": ":", "marker": "", "color": "#d62728",
                                "show_in_legend": bound == "Lower 95%"}))
        if "Trend" in result.rows:
            sql, roles = self.series_select(rows, {"x": "Time", "y": "Trend"})
            series.append((_("Trend"), sql, roles, {"linestyle": "-", "marker": "", "color": "#16a34a"}))
        title = (_("{measure} and its forecast") if "time_series_forecast" in tables else "{measure}").format(
            measure=measure)
        panels: list[tuple] = [("Time Series", title,
                                x_label, measure, series, options)]
        band = float(result.correlations["Upper band"].iloc[0])
        lines = {"lines": [{"orientation": "horizontal", "value": value,
                            "kwargs": {"color": "0.5", "linestyle": "--", "linewidth": 0.8}}
                           for value in (band, -band, 0.0)]}
        for column, title in (("ACF", _("Autocorrelation (ACF)")), ("PACF", _("Partial autocorrelation (PACF)"))):
            sql, roles = self.series_select(tables["time_series_acf"], {"x": "Lag", "y": column})
            panels.append(("Stem Plot", title, _("Lag"), column, [(column, sql, roles, {})], lines))
        if "time_series_ccf" in tables:
            sql, roles = self.series_select(tables["time_series_ccf"], {"x": "Lag", "y": "Prewhitened CCF"})
            white = float(result.cross["Prewhitened band"].iloc[0])
            white_lines = {"lines": [{"orientation": "horizontal", "value": value,
                                      "kwargs": {"color": "0.5", "linestyle": "--", "linewidth": 0.8}}
                                     for value in (white, -white, 0.0)]}
            panels.append(("Stem Plot", _("Cross-correlation with {input}, prewhitened").format(input=result.input),
                           _("Lag"), "CCF", [(_("Prewhitened CCF"), sql, roles, {})], white_lines))
        if "Trend" in result.rows:
            # Without the trend, whose scale would flatten them: it is on the first chart.
            parts = []
            for column in ("Seasonal", "Remainder"):
                sql, roles = self.series_select(rows, {"x": "Time", "y": column})
                parts.append((_(column), sql, roles, {"linestyle": "-", "marker": ""}))
            panels.append(("Time Series", _("Season ({n}) and remainder").format(n=result.season),
                           x_label, measure, parts, options))
        return self.draw_report(result, measure, panels, written=written)
