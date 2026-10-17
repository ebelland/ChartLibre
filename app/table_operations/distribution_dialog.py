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

    def format_results(self, report: DistributionReport) -> str:
        return report_html.document(_(self.Name), str(self.table), self._report(report))

    def _report(self, report: DistributionReport) -> str:
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
        repo, table = self._repo, self.table
        frames: dict[str, list[pd.DataFrame]] = {"quantiles": [], "moments": [], "normality": []}
        for result in report.results:
            for key, frame in (("quantiles", result.quantiles), ("moments", result.moments),
                               ("normality", result.normality)):
                frames[key].append(frame.assign(Column=result.column))
        names = {key: repo.free_table_name(f"{table}_distribution_{key}") for key, parts in frames.items() if parts}
        repo.snapshot_for_undo([*names.values(), *repo.DESCRIPTOR_TABLES], label=f"Distribution on '{table}'")
        written: list[dict[str, Any]] = []
        for key, name in names.items():
            frame = pd.concat(frames[key], ignore_index=True)
            frame = frame[["Column", *[c for c in frame.columns if c != "Column"]]]
            repo.import_dataframe(frame, table_name=name, normalize_columns=False)
            written.append({"table": name})

        panels = []
        for result in report.results:
            column = result.column
            sql, roles = self.series_select(table, {"value": column})
            panels.append(("Histogram", column, column, _("Density"),
                           [(column, sql, roles, {})], {"distribution_fit": "norm"}))
            panels.append(("Q-Q Plot", _("Normal quantile plot"), _("Normal quantiles"), column,
                           [(column, sql, roles, {})]))
        columns = ", ".join(result.column for result in report.results)
        notes = report_html.document(_(self.Name), f"{table} · {columns}", self._report(report))
        figure_id = self.create_report_figure(f"{_(self.Name)} · {columns}", notes, panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": columns})
        return written
