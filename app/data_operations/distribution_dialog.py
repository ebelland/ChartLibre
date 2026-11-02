"""Distribution (Analysis): one series' quantiles, moments and normality (AN-02).

Each checked series gets its own report: the quantile table, the moments, the normality tests and the
distribution fitted to it, with its parameters and goodness of fit. OK draws the series' histogram beside
its Q-Q plot, in a figure of its own. The report is the result: nothing is written to the project's
tables. The engine is app.analysis.distribution, the fits app.utils.distribution_fit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.analysis import distribution as dist
from app.data.data_source import row_value
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase
from app.plugin_api.data_operations.dialog_base import xy_columns
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam
from app.plugin_api.data_operations.results import OperationResult
from app.utils import report_html
from app.utils.distribution_fit import DistributionFit, fit_distributions
from app.utils.i18n import _

#: The distributions offered, scipy names: the fitted one is drawn and reported.
FITTED = (("Normal", "norm"), ("Lognormal", "lognorm"), ("Weibull", "weibull_min"), ("Gamma", "gamma"),
          ("Exponential", "expon"))
BEST = "best"


def distribution_label(name: str) -> str:
    """The offered distribution's English name for a scipy name ("lognorm" -> "Lognormal")."""
    return next((label for label, key in FITTED if key == name), name)


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def series_values(dialog: DataOperationDialogBase, row: Any, name: str) -> tuple[Any, str, str]:
    """A series' x and y, its query, and the column its y values come from."""
    frame, roles = dialog._series_frame(row)
    _x, y_column = xy_columns(frame, roles)
    query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
    return dialog.series_xy(row, name), query, y_column


def value_series(source_name: str, query: str, column: str) -> list[tuple]:
    """The series of one column's values, as the figure's panels read them."""
    value_query = f'SELECT {quoted(column)} AS value FROM ({query})'
    return [(source_name, value_query, {"value": "value"}, {})]


def qq_panel(source_name: str, series: list[tuple], key: str) -> tuple:
    """The Q-Q plot of *series* against the distribution *key* (scipy name)."""
    label = _(distribution_label(key))
    return ("Q-Q Plot", _("{distribution} quantile plot").format(distribution=label),
            _("{distribution} quantiles").format(distribution=label), source_name, series, {"distribution": key})


@dataclass(slots=True)
class DistributionSeriesResult(OperationResult):
    """One series' distribution report, and the figure that draws it.

    ``query`` is the series' own SQL, ``column`` the column of its y values.
    """

    source_name: str
    query: str
    column: str
    report: dist.DistributionResult
    fit: DistributionFit | None

    def to_df(self) -> pd.DataFrame:
        return self.report.quantiles

    def preview(self, dialog: DistributionDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def apply(self, dialog: DistributionDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def draw(self, dialog: DistributionDialog) -> None:
        """The histogram with the fitted curve, and the Q-Q plot against it, in a figure of their own."""
        key = self.fit.name if self.fit is not None else "norm"
        series = value_series(self.source_name, self.query, self.column)
        panels = [
            ("Histogram", self.source_name, self.source_name, _("Density"), series, {"distribution_fit": key}),
            qq_panel(self.source_name, series, key),
        ]
        dialog.create_report_figure(f"{_(dialog.Name)} · {self.source_name}",
                                    dialog.results_document(self.source_name, dialog.report_for(self)), panels)


class DistributionDialog(DataOperationDialogBase):
    """Each series' distribution: quantiles, moments, normality, and its histogram beside its Q-Q plot."""

    Name = "Distribution"
    Description = "One series at a time: the histogram, quantiles, moments and normality tests."
    Category = "Basic Analysis"
    Order = 10
    Icon = """
    <path d="M4 20V14h3v6M8.5 20V8h3v12M13 20V5h3v15M17.5 20v-9h3v9"/><path d="M3 20h18"/>
    """
    Keywords: str = (
        "distribution distribuzione quantiles quantili percentiles percentili moments momenti "
        "normality normalità histogram istogramma"
    )
    Goals: list[str] = ["describe"]

    DESTINATIONS = (DataOperationDialogBase.DEST_NONE, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_NEW_FIGURE
    SELECTS_ALL_SERIES = True

    PARAMS = (
        ChoiceParam("distribution", "Distribution:",
                    choices=(*FITTED, ("Best fit (AIC)", BEST)), default_value="norm",
                    tooltip="Fitted to each series: its curve on the histogram, the Q-Q plot against it, its "
                            "parameters and goodness of fit in the report. Best fit: the one of these with the "
                            "lowest AIC."),
        BoolParam("Histogram", "Histogram", default_value=True, tab="charts"),
        BoolParam("Q-Q Plot", "Q-Q Plot", default_value=True, tab="charts"),
    )

    def read_series(self, row: Any, name: str) -> Any:
        """The series' x and y, its query, and the column its y values come from."""
        return series_values(self, row, name)

    def compute_series(self, name: str, data: Any, settings: Mapping[str, Any]) -> DistributionSeriesResult:
        (_x, y), query, column = data
        values = np.asarray(y, dtype=float)
        chosen = str(settings.get("distribution") or "norm")
        names = tuple(key for _label_, key in FITTED) if chosen == BEST else (chosen,)
        report = dist.distribution(pd.DataFrame({name: values}), name)
        found = fit_distributions(values[np.isfinite(values)], names)
        return DistributionSeriesResult(source_name=name, query=query, column=column, report=report,
                                        fit=found[0] if found else None)

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """The report draws its own figure: nothing is added to the series' axis."""
        del results
        return selected_axis_id

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_for(self, result: DistributionSeriesResult) -> str:
        """One series' report: its notes, quantiles, moments, normality and fitted distribution."""
        report = result.report
        parts = [report_html.note(_(note)) for note in report.notes]
        parts.append(report_html.section(_("Quantiles"), self.report_table(report.quantiles)))
        parts.append(report_html.section(_("Moments"), self.report_table(report.moments)))
        parts.append(report_html.section(_("Normality"), self.report_table(report.normality, p_columns=("pvalue",))))
        if result.fit is not None:
            fit = result.fit
            parts.append(report_html.section(
                _("Fitted {distribution}").format(distribution=_(distribution_label(fit.name))),
                report_html.summary_table([
                    *((f"{_('Parameter')} {index + 1}", value) for index, value in enumerate(fit.params)),
                    (_("Kolmogorov-Smirnov D"), fit.ks_statistic), (_("Prob > D"), fit.pvalue),
                    (_("AIC"), fit.aic)])))
        body = "".join(parts)
        box = ""
        if reading_guide.enabled():
            box = report_html.reading_guide(_("How to read this"), self.reading_for(result))
        return report_html.section(f"{result.source_name} (N = {report.n})", report_html.with_reading_guide(body, box))

    def reading_for(self, result: DistributionSeriesResult) -> list[str]:
        """One series' "How to read this" lines, in its own section (the base's box stays empty)."""
        report = result.report
        moments, column = report.moments, result.source_name
        tested = report.normality[pd.to_numeric(report.normality["pvalue"], errors="coerce").notna()]
        return [
            reading_guide.interval(reading_guide.lookup(moments, "Lower 95% Mean", by="Moment"),
                                   reading_guide.lookup(moments, "Upper 95% Mean", by="Moment"),
                                   what=_("the mean of {column}").format(column=column)),
            reading_guide.spread(reading_guide.lookup(moments, "Std Dev", by="Moment"),
                                 reading_guide.lookup(moments, "Mean", by="Moment"), column=column),
            reading_guide.normality(tested["pvalue"].iloc[0], test=str(tested["test"].iloc[0]), column=column)
            if not tested.empty else "",
        ]

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        subtitle = ", ".join(result.source_name for result in results)
        return self.results_document(subtitle, *(self.report_for(result) for result in results))

    def report_table(self, frame: pd.DataFrame, *, p_columns: Sequence[str] = (), digits: int = 6) -> str:
        return report_html.frame_table(frame, _, p_columns=p_columns, digits=digits)
