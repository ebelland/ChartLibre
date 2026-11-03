"""Multivariate (Analysis): how continuous columns move together, two by two (AN-04).

As JMP's Multivariate platform: the correlation matrix, and a row per pair -
its correlation, confidence interval, p-value and count, each pair on the
rows where both columns have a value. Pearson, Spearman or Kendall. OK writes
the matrix and the pairs as tables and draws the scatterplot matrix. The
engine is app.analysis.multivariate.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import multivariate as mv
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.column_roles import (
    CONTINUOUS,
    ORDINAL,
    ColumnCasting,
    ColumnRole,
)
from app.plugin_api.data_operations.parameter_spec import ChoiceParam
from app.plugin_api.data_operations.roles_base import RolesOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

#: The most columns the scatterplot matrix draws (the Pair Plot's slots).
PAIR_PLOT_COLUMNS = 6


class MultivariateDialog(RolesOperationDialogBase):
    """Correlations of every pair of columns, with their p-values, and the scatterplot matrix."""

    Name = "Multivariate"
    Description = (
        "How continuous columns move together: the correlation matrix, each pair's correlation "
        "with its confidence interval and p-value, and the scatterplot matrix."
    )
    Category = "Multivariate Methods"
    Order = 39
    Icon = """
    <rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/>
    <circle cx="16" cy="6" r="1.2"/><circle cx="18" cy="8" r="1.2"/><circle cx="6" cy="17" r="1.2"/><circle cx="8" cy="15" r="1.2"/>
    """
    Keywords: str = (
        "multivariate multivariata correlation correlazione correlations correlazioni pearson spearman kendall "
        "correlation matrix matrice di correlazione scatterplot matrix matrice di dispersione p-value"
    )
    Goals: list[str] = ["relate", "explain"]

    ROLES = (
        ColumnRole("response", "Y, columns", "The continuous columns to correlate, two by two.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=2),
    )
    PARAMETERS_TAB = "Multivariate"
    PARAMS = (
        ChoiceParam("method", "Correlation:", choices=tuple((label, key) for key, label in mv.METHODS),
                    tooltip="Pearson measures a straight-line relation; Spearman and Kendall any relation "
                            "that keeps going one way, and are not swayed by outliers."),
    )

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        return mv.correlations(frame, casting.columns("response"), method=str(params.get("method") or mv.PEARSON))

    def report_subtitle(self, result: mv.CorrelationResult) -> str:
        return f"{self.table} · {_(dict(mv.METHODS)[result.method])}"

    def report_body(self, result: mv.CorrelationResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), report_html.format_number(value)) for key, value in result.summary.items())))
        blocks.append(report_html.section(_("Correlations"), self.report_table(result.matrix, digits=4)))
        blocks.append(report_html.section(_("Pairwise correlations, strongest evidence first"),
                                          self.report_table(mv.strongest_pairs(result.pairs),
                                                            p_columns=("Prob>|t|",), digits=4)))
        return "".join(blocks)

    def how_to_read(self, result: mv.CorrelationResult) -> list[str]:
        pairs = mv.strongest_pairs(result.pairs)
        if pairs.empty:
            return []
        top = pairs.iloc[0]
        x, y = str(top["Variable"]), str(top["by Variable"])
        return [reading_guide.correlation(top["Correlation"], x=x, y=y),
                reading_guide.p_value(top["Prob>|t|"], null=_("{x} and {y} were not correlated").format(x=x, y=y),
                                      shown=_("{x} and {y} are correlated").format(x=x, y=y))]

    def apply_results(self, result: mv.CorrelationResult) -> list[dict[str, Any]]:
        tables = self.write_result_tables({"correlations": result.matrix, "correlation_pairs": result.pairs})
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]
        columns = result.columns[:PAIR_PLOT_COLUMNS]
        sql = f"SELECT {', '.join(self.quoted(c) for c in columns)} FROM {self.quoted(self.table)}"
        panels = [("Pair Plot", _("Scatterplot matrix"), "", "", [(self.table, sql, {}, {})])]
        return self.draw_report(result, self.table, panels, written=written)
