"""Distribution (Analysis): one column's quantiles, moments and normality (AN-02).

Cast one or more columns as Y. The report gives, per column, the quantile
table, the moments and the normality tests; OK writes them as tables and
draws, per column, the histogram with the chosen distribution's curve beside
its Q-Q plot. The distribution (Parameters: normal by default, lognormal,
Weibull, gamma, exponential, or the best of them by AIC) is fitted to each
column and reported with its goodness of fit. The engine is
app.analysis.distribution, the fits app.utils.distribution_fit. Process capability is in
the Control Chart operation, which knows whether the process is stable.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.analysis import distribution as dist
from app.plugin_api.data_operations.parameter_spec import ChoiceParam
from app.utils.distribution_fit import DistributionFit, fit_distributions
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

#: The distributions offered, scipy names: the fitted one is drawn and reported.
FITTED = (("Normal", "norm"), ("Lognormal", "lognorm"), ("Weibull", "weibull_min"), ("Gamma", "gamma"),
          ("Exponential", "expon"))
BEST = "best"


@dataclass(slots=True)
class DistributionReport:
    results: list[dist.DistributionResult]
    #: The distribution fitted to each column, in the same order (None when it could not be fitted).
    fits: list[DistributionFit | None]


class DistributionDialog(TableOperationDialogBase):
    """Each column's distribution: quantiles, moments, normality."""

    Name = "Distribution"
    Description = "One column at a time: the histogram, quantiles, moments and normality tests."
    Category = "Basic Analysis"
    CHART_PANELS = ('Histogram', 'Q-Q Plot')
    Order = 10
    Icon = """
    <path d="M4 20V14h3v6M8.5 20V8h3v12M13 20V5h3v15M17.5 20v-9h3v9"/><path d="M3 20h18"/>
    """
    Keywords: str = (
        "distribution distribuzione quantiles quantili percentiles percentili moments momenti "
        "normality normalità histogram istogramma"
    )
    Goals: list[str] = ["describe"]

    ROLES = (
        ColumnRole("response", "Y", "The columns to describe, one report each.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1),
    )

    PARAMETERS_TAB = "Distribution"
    PARAMS = (
        ChoiceParam("distribution", "Distribution:",
                    choices=(*FITTED, ("Best fit (AIC)", BEST)), default_value="norm",
                    tooltip="Fitted to each column: its curve on the histogram, the Q-Q plot against it, its "
                            "parameters and goodness of fit in the report. Best fit: the one of these with the "
                            "lowest AIC."),
    )

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        chosen = str(params.get("distribution") or "norm")
        names = tuple(name for _label, name in FITTED) if chosen == BEST else (chosen,)
        results, fits = [], []
        for column in casting.columns("response"):
            results.append(dist.distribution(frame, column))
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            found = fit_distributions(values[np.isfinite(values)], names)
            fits.append(found[0] if found else None)
        return DistributionReport(results, fits)

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_body(self, report: DistributionReport) -> str:
        blocks = []
        for result in report.results:
            parts = [report_html.note(_(note)) for note in result.notes]
            parts.append(report_html.section(_("Quantiles"), self.report_table(result.quantiles)))
            parts.append(report_html.section(_("Moments"), self.report_table(result.moments)))
            parts.append(report_html.section(_("Normality"), self.report_table(result.normality,
                                                                                p_columns=("pvalue",))))
            fit = report.fits[report.results.index(result)]
            if fit is not None:
                parts.append(report_html.section(
                    _("Fitted {distribution}").format(distribution=_(_label(fit.name))),
                    report_html.summary_table([
                        *((f"{_('Parameter')} {index + 1}", value) for index, value in enumerate(fit.params)),
                        (_("Kolmogorov-Smirnov D"), fit.ks_statistic), (_("Prob > D"), fit.pvalue),
                        (_("AIC"), fit.aic)])))
            blocks.append(report_html.section(f"{result.column} (N = {result.n})", "".join(parts)))
        return "".join(blocks)

    # ------------------------------------------------------------------
    # OK: tables and a figure
    # ------------------------------------------------------------------

    def apply_results(self, report: DistributionReport) -> list[dict[str, Any]]:
        table = self.table
        frames: dict[str, list[pd.DataFrame]] = {"quantiles": [], "moments": [], "normality": []}
        for result in report.results:
            for key, frame in (("quantiles", result.quantiles), ("moments", result.moments),
                               ("normality", result.normality)):
                frames[key].append(frame.assign(Column=result.column))
        tables = {}
        for key, parts in frames.items():
            frame = pd.concat(parts, ignore_index=True)
            tables[f"distribution_{key}"] = frame[["Column", *[c for c in frame.columns if c != "Column"]]]
        written: list[dict[str, Any]] = [{"table": name} for name in self.write_result_tables(tables).values()]

        panels = []
        for result, fit in zip(report.results, report.fits):
            column = result.column
            name = fit.name if fit is not None else "norm"
            label = _(_label(name))
            sql, roles = self.series_select(table, {"value": column})
            panels.append(("Histogram", column, column, _("Density"),
                           [(column, sql, roles, {})], {"distribution_fit": name}))
            panels.append(("Q-Q Plot", _("{distribution} quantile plot").format(distribution=label),
                           _("{distribution} quantiles").format(distribution=label), column,
                           [(column, sql, roles, {})], {"distribution": name}))
        columns = ", ".join(result.column for result in report.results)
        return self.draw_report(report, columns, panels, written=written)


def _label(name: str) -> str:
    """The offered distribution's English name for a scipy name ("lognorm" -> "Lognormal")."""
    return next((label for label, key in FITTED if key == name), name)
