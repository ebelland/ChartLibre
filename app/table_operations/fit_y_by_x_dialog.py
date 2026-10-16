"""Fit Y by X (Analysis): one Y against each X, the analysis chosen by their types (AN-03).

Continuous by continuous is a Bivariate fit, continuous by groups a Oneway
(means, ANOVA, Tukey / Games-Howell / Dunn), groups by groups a Contingency
table, groups by continuous a Logistic fit - see app.analysis.fit_y_by_x.
OK writes each report's tables and draws, per X, the scatter with its line,
the box plot by group, the mosaic or the probability curves.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.analysis import fit_y_by_x as fyx
from app.table_operations.column_roles import CONTINUOUS, ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


def _slug(title: str) -> str:
    """A report title as part of a table name: "Tukey HSD (equal variances)" -> "tukey_hsd_equal_variances"."""
    return re.sub(r"\W+", "_", title).strip("_").lower()


@dataclass(slots=True)
class FitYByXReport:
    results: list[fyx.FitYByXResult]


class FitYByXDialog(TableOperationDialogBase):
    """One response against one or more factors, each pair analysed as its types call for."""

    Name = "Fit Y by X"
    Description = ("Y against each X, the analysis chosen by their types: bivariate fit, oneway ANOVA "
                   "with pairwise comparisons, contingency table, or logistic fit.")
    Category = "Distributions"
    Icon = """
    <path d="M4 4v16h16"/><circle cx="8" cy="15" r="1.2"/><circle cx="11" cy="12" r="1.2"/>
    <circle cx="14" cy="11" r="1.2"/><circle cx="17" cy="7" r="1.2"/><path d="M6 17L19 6"/>
    """
    Keywords: str = (
        "fit y by x bivariate bivariata oneway one-way anova post hoc tukey games-howell dunn holm "
        "pairwise comparisons confronti multipli contingency contingenza crosstabs chi-squared "
        "logistic logistica regression line retta correlation correlazione"
    )
    Goals: list[str] = ["compare", "relate"]

    ROLES = (
        ColumnRole("response", "Y", "The response: its type and X's choose the analysis.", minimum=1, maximum=1),
        ColumnRole("factor", "X", "One report per column: continuous, or the groups.", minimum=1),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        if not found and set(casting.columns("response")) & set(casting.columns("factor")):
            found.append(_("Y cannot also be an X."))
        return found

    @staticmethod
    def _continuous(casting: ColumnCasting, column: str, *, as_response: bool) -> bool:
        kind = casting.kinds.get(column)
        # Ordinal: a measure when it is the response, groups when it is the factor.
        return kind == CONTINUOUS or (as_response and kind == "ordinal")

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        y = casting.columns("response")[0]
        y_continuous = self._continuous(casting, y, as_response=True)
        return FitYByXReport([
            fyx.fit_y_by_x(frame, y, x, y_continuous=y_continuous,
                           x_continuous=self._continuous(casting, x, as_response=False))
            for x in casting.columns("factor")
        ])

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def format_results(self, report: FitYByXReport) -> str:
        return report_html.document(_(self.Name), str(self.table), self._report(report))

    def _report(self, report: FitYByXReport) -> str:
        blocks = []
        for result in report.results:
            parts = [report_html.note(_(note)) for note in result.notes]
            for title, table in result.tables.items():
                p_columns = [c for c in table.columns if str(c).startswith("pvalue") or c == "Prob>ChiSq"]
                parts.append(report_html.section(_(title), self.report_table(table, p_columns=p_columns)))
            blocks.append(report_html.section(f"{_(result.kind)}: {result.y} by {result.x}", "".join(parts)))
        return "".join(blocks)

    # ------------------------------------------------------------------
    # OK: tables and a figure
    # ------------------------------------------------------------------

    def apply_results(self, report: FitYByXReport) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        planned: list[tuple[str, pd.DataFrame]] = []
        for result in report.results:
            stem = f"{table}_{result.y}_by_{result.x}"
            for title, frame in result.tables.items():
                planned.append((f"{stem}_{_slug(title)}", frame))
            if not result.curve.empty:
                planned.append((f"{stem}_curve", result.curve))
        names = [repo.free_table_name(name) for name, _frame in planned]
        repo.snapshot_for_undo([*names, *repo.DESCRIPTOR_TABLES], label=f"Fit Y by X on '{table}'")
        written: list[dict[str, Any]] = []
        curves: dict[int, str] = {}
        for name, (planned_name, frame) in zip(names, planned):
            repo.import_dataframe(frame, table_name=name, normalize_columns=False)
            written.append({"table": name})
            if planned_name.endswith("_curve"):
                curves[len(written) - 1] = name

        curve_tables = iter(curves.values())
        panels = [self._panel(result, next(curve_tables) if not result.curve.empty else None)
                  for result in report.results]
        y = report.results[0].y
        notes = report_html.document(_(self.Name), f"{table} · {y}", self._report(report))
        figure_id = self.create_report_figure(f"{_(self.Name)} · {y}", notes, panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": y})
        return written

    def _panel(self, result: fyx.FitYByXResult, curve_table: str | None) -> tuple:
        table, y, x = self.table, result.y, result.x
        title = _("{y} by {x}").format(y=y, x=x)
        if result.kind == fyx.BIVARIATE:
            sql, roles = self.series_select(table, {"x": x, "y": y})
            series = [(y, sql, roles, {"linestyle": "", "marker": "o"})]
            sql, roles = self.series_select(str(curve_table), {"x": x, "y": "Fit"})
            series.append((_("Linear fit"), sql, roles, {"linestyle": "-", "marker": "", "color": "#d62728"}))
            return ("Scatter Plot", title, x, y, series)
        if result.kind == fyx.ONEWAY:
            sql, roles = self.series_select(table, {"value": y, "group": x})
            return ("Box Plot", title, x, y, [(y, sql, roles, {})])
        if result.kind == fyx.CONTINGENCY:
            sql, roles = self.series_select(table, {"x": x, "y": y})
            return ("Mosaic Plot", title, x, y, [(y, sql, roles, {})])
        series = []
        for column in result.curve.columns[1:]:
            sql, roles = self.series_select(str(curve_table), {"x": x, "y": column})
            series.append((column, sql, roles, {"linestyle": "-", "marker": ""}))
        return ("Scatter Plot", title, x, _("Probability"), series)
