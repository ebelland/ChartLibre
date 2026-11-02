"""Method Comparison (Quality and Process): do two methods measuring the same samples agree? (ID12)

Y is the method tested, X the reference, one row per sample. The report gives the Bland-Altman bias and
limits of agreement with their intervals, and the Deming and Passing-Bablok lines with theirs - whether
slope 1 and intercept 0 are inside. OK writes the samples with their means and differences and the lines,
and draws the Bland-Altman plot and the two methods against each other with the identity and both lines.
The engine is app.analysis.method_comparison.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pandas as pd

from app.analysis import method_comparison as mc
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.column_roles import (
    CONTINUOUS,
    ColumnCasting,
    ColumnRole,
)
from app.plugin_api.data_operations.parameter_spec import FloatParam
from app.plugin_api.data_operations.roles_base import RolesOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


def _number(value: float) -> str:
    return report_html.format_number(value)


class MethodComparisonDialog(RolesOperationDialogBase):
    """Bland-Altman, Deming and Passing-Bablok: a method tested against a reference on the same samples."""

    Name = "Method Comparison"
    Description = ("Do two methods agree on the same samples? Bland-Altman bias and limits of agreement, Deming "
                   "and Passing-Bablok regression with their intervals.")
    Category = "Quality and Process"
    Order = 65
    Icon = """
    <path d="M4 20L20 4" stroke-dasharray="2 2"/><circle cx="7" cy="16" r="1.3"/><circle cx="11" cy="12.5" r="1.3"/>
    <circle cx="14" cy="10" r="1.3"/><circle cx="17" cy="6.5" r="1.3"/><path d="M3 21h18M3 3v18"/>
    """
    Keywords: str = (
        "method comparison confronto metodi bland altman agreement accordo limits of agreement deming "
        "passing bablok regression bias laboratory laboratorio clsi ep09 validation validazione"
    )
    Goals: ClassVar[list[str]] = ["relate", "process"]

    ROLES = (
        ColumnRole("response", "Y, method tested", "The results of the method being tested.",
                   kinds=(CONTINUOUS,), minimum=1, maximum=1),
        ColumnRole("factor", "X, reference", "The same samples' results by the reference method.",
                   kinds=(CONTINUOUS,), minimum=1, maximum=1),
    )
    DISTINCT_ROLES = (("response", "factor", "The method tested cannot also be the reference."),)
    PARAMETERS_TAB = "Method Comparison"
    PARAMS = (
        FloatParam("ratio", "Error variance ratio:", default_value=1.0, minimum=0.001, maximum=1000.0, decimals=3,
                   tooltip="Deming: the method tested's error variance over the reference's - 1 when they are "
                           "alike (each one's repeatability, squared)."),
        FloatParam("confidence", "Confidence:", default_value=0.95, minimum=0.5, maximum=0.999, decimals=3,
                   step=0.01, tooltip="The level of every interval."),
    )

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        return mc.compare(frame, casting.columns("response")[0], casting.columns("factor")[0],
                          ratio=float(params.get("ratio", 1.0)), confidence=float(params.get("confidence", 0.95)))

    def report_subtitle(self, result: mc.Comparison) -> str:
        return f"{result.y} vs {result.x}"

    def report_body(self, result: mc.Comparison) -> str:
        a = result.agreement
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Bland-Altman agreement"), report_html.summary_table([
            (_("Samples"), str(a.n)),
            (_("Bias (mean difference)"), f"{_number(a.bias)}  [{_number(a.bias_ci[0])}, {_number(a.bias_ci[1])}]"),
            (_("SD of the differences"), _number(a.sd)),
            (_("Lower limit of agreement"), f"{_number(a.lower)}  [{_number(a.lower_ci[0])}, {_number(a.lower_ci[1])}]"),
            (_("Upper limit of agreement"), f"{_number(a.upper)}  [{_number(a.upper_ci[0])}, {_number(a.upper_ci[1])}]"),
            (_("Difference vs mean: slope, p"), f"{_number(a.trend_slope)}, {report_html.format_p_value(a.trend_p)}"),
        ])))
        lines = pd.DataFrame([line.row() for line in result.lines])
        blocks.append(report_html.section(_("Regression"), self.report_table(lines, digits=5)))
        return "".join(blocks)

    def how_to_read(self, result: mc.Comparison) -> list[str]:
        a = result.agreement
        return [_("Bias {bias}: {y} reads on average {bias} more than {x}. 95% of the differences fall between "
                  "{lower} and {upper} - whether that is close enough depends on what the results are used for.")
                .format(bias=_number(a.bias), y=result.y, x=result.x, lower=_number(a.lower), upper=_number(a.upper)),
                reading_guide.p_value(a.trend_p, null=_("the difference did not change with the level"),
                                      shown=_("the difference grows with the level (a proportional bias)"))]

    def apply_results(self, result: mc.Comparison) -> list[dict[str, Any]]:
        lines = pd.DataFrame([line.row() for line in result.lines])
        tables = self.write_result_tables({"comparison": result.rows, "comparison_lines": lines})
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]
        rows, a = tables["comparison"], result.agreement
        sql, roles = self.series_select(rows, {"x": "Mean", "y": "Difference"})
        reference = {"lines": [{"orientation": "horizontal", "value": value,
                                "kwargs": {"color": color, "linestyle": style, "linewidth": 1.0}}
                               for value, color, style in ((a.bias, "#2e7d32", "-"), (a.upper, "#c62828", "--"),
                                                           (a.lower, "#c62828", "--"), (0.0, "0.5", ":"))]}
        panels: list[tuple] = [("Scatter Plot", _("Bland-Altman"), _("Mean of the two methods"),
                                _("{y} - {x}").format(y=result.y, x=result.x),
                                [(_("Samples"), sql, roles, {"marker": "o", "linestyle": ""})], reference)]
        sql, roles = self.series_select(rows, {"x": result.x, "y": result.y})
        low, high = float(result.rows[result.x].min()), float(result.rows[result.x].max())
        series = [(_("Samples"), sql, roles, {"marker": "o", "linestyle": ""}),
                  (_("Identity"), f"SELECT {low} AS x, {low} AS y UNION ALL SELECT {high}, {high}",
                   {"x": "x", "y": "y"}, {"linestyle": ":", "marker": "", "color": "0.5"})]
        for line, color in zip(result.lines, ("#1565c0", "#ef6c00"), strict=True):
            ends = " UNION ALL ".join(f"SELECT {value} AS x, {line.intercept + line.slope * value} AS y"
                                      for value in (low, high))
            series.append((line.method, ends, {"x": "x", "y": "y"}, {"linestyle": "-", "marker": "", "color": color}))
        panels.append(("Scatter Plot", _("{y} against {x}").format(y=result.y, x=result.x), result.x, result.y, series))
        return self.draw_report(result, self.table, panels, written=written)
