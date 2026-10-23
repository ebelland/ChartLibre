"""Distribution (Analysis): one column's quantiles, moments and normality (AN-02).

Cast one or more columns as Y. The report gives, per column, the quantile
table, the moments and the normality tests; OK writes them as tables and
draws, per column, the histogram with its normal curve beside the normal
Q-Q plot. The engine is app.analysis.distribution. Process capability is in
the Control Chart operation, which knows whether the process is stable.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.analysis import distribution as dist
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

@dataclass(slots=True)
class DistributionReport:
    results: list[dist.DistributionResult]


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

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        return DistributionReport([dist.distribution(frame, column) for column in casting.columns("response")])

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_subtitle(self, result: DistributionReport) -> str:
        return f"{self.table} · {', '.join(r.column for r in result.results)}"

    def report_body(self, report: DistributionReport) -> str:
        blocks = []
        for result in report.results:
            parts = [report_html.note(_(note)) for note in result.notes]
            parts.append(report_html.section(_("Quantiles"), self.report_table(result.quantiles)))
            parts.append(report_html.section(_("Moments"), self.report_table(result.moments)))
            parts.append(report_html.section(_("Normality"), self.report_table(result.normality,
                                                                                p_columns=("pvalue",))))
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
        for result in report.results:
            column = result.column
            sql, roles = self.series_select(table, {"value": column})
            panels.append(("Histogram", column, column, _("Density"),
                           [(column, sql, roles, {})], {"distribution_fit": "norm"}))
            panels.append(("Q-Q Plot", _("Normal quantile plot"), _("Normal quantiles"), column,
                           [(column, sql, roles, {})]))
        columns = ", ".join(result.column for result in report.results)
        notes = self.report_document(report)
        figure_id = self.create_report_figure(f"{_(self.Name)} · {columns}", notes, panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": columns})
        return written
