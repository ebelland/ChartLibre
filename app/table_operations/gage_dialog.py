"""Gage R&R (AN-06): how much of the variation the measuring system adds - JMP's Variability / Gauge Chart.

Choose the measurements, the parts and (for reproducibility) the operators; a
tolerance (USL - LSL) adds % Tolerance. app.analysis.gage gives the variance
components by the ANOVA method and the number of distinct categories; OK draws
the study on a Multi-Vari chart, with the report in the notes.
"""
from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import gage
from app.analysis.capability import parse_limit
from app.plugin_api.data_operations.parameter_spec import TextParam
from app.plugin_api import reading_guide
from app.plugin_api.table_operations.column_roles import CONTINUOUS, NOMINAL, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class GageDialog(TableOperationDialogBase):
    Name = "Gage R&R"
    Description = ("How much of the variation the measuring system adds: repeatability and reproducibility by "
                   "the ANOVA method, % of the study variation and of the tolerance, distinct categories.")
    Category = "Quality and Process"
    Order = 61
    Icon = """
    <path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/><path d="M8 18h8"/>
    """
    Keywords: str = (
        "gage r&r gauge r&r msa measurement system analysis analisi del sistema di misura ripetibilità "
        "riproducibilità repeatability reproducibility variability chart operatori parti ndc"
    )
    Goals: list[str] = ["process"]

    ONE_PANEL = True
    ROLES = (
        ColumnRole("response", "Measurement", "The measured values.", kinds=(CONTINUOUS, ORDINAL),
                   minimum=1, maximum=1),
        ColumnRole("part", "Part", "Which part each measurement is of.", kinds=(NOMINAL, ORDINAL),
                   minimum=1, maximum=1),
        ColumnRole("operator", "Operator", "Optional: who measured. Without it, repeatability alone.",
                   kinds=(NOMINAL, ORDINAL), maximum=1),
    )
    DISTINCT_ROLES = tuple((role, other, "The measurement, the part and the operator must be different columns.")
                           for role, other in (("response", "part"), ("response", "operator"), ("part", "operator")))
    PARAMETERS_TAB = "Gage R&R"
    PARAMS = (
        TextParam("tolerance", "Tolerance (USL - LSL):", placeholder="optional",
                  tooltip="The width of the specification: adds each source's share of it (% Tolerance)."),
    )

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        operator = casting.columns("operator")
        return gage.gage_rr(frame, casting.columns("response")[0], casting.columns("part")[0],
                            operator[0] if operator else "", tolerance=parse_limit(params.get("tolerance")))

    def report_body(self, result: gage.GageStudy) -> str:
        blocks = [_verdict_html(result)]
        blocks.append(report_html.section(_("Variance components"), self.report_table(result.components, digits=4)))
        blocks.append(report_html.section(_("Analysis of variance"),
                                          self.report_table(result.anova, p_columns=("Prob > F",), digits=4)))
        blocks.append(report_html.section(_("Study"), report_html.summary_table((
            (_("Parts"), str(result.parts)), (_("Operators"), str(result.operators)),
            (_("Measurements per part and operator"), report_html.format_number(result.replicates)),
            (_("Distinct categories"), str(result.ndc))))))
        return "".join(blocks)

    def how_to_read(self, result: gage.GageStudy) -> list[str]:
        return [reading_guide.gage(reading_guide.lookup(result.components, "Gage R&R", by="Source", value="% Study Var"),
                                   categories=result.ndc)]

    def apply_results(self, result: gage.GageStudy) -> list[dict[str, Any]]:
        name = self.write_result_tables({"gage": result.components})["gage"]
        roles = {"y": result.measure, "x": result.part}
        if result.operator:
            roles["group"] = result.operator
        sql, series_roles = self.series_select(str(self.table), roles)
        panel = ("Multi-Vari Chart", _("{measure} by part and operator").format(measure=result.measure),
                 result.part, result.measure, [(result.measure, sql, series_roles, {})])
        return self.draw_report(result, result.measure, [panel], written=[{"table": name}])


def _verdict_html(result: gage.GageStudy) -> str:
    """The sentences, the verdict first and larger, as the quick answers write theirs."""
    lines = [_(template).format(**values) for template, values in result.sentences]
    colour = "#1a7f37" if result.grr_percent < 10 else ("#9a6700" if result.grr_percent <= 30 else "#a40000")
    head = f"<p style='font-size:125%; color:{colour}'><b>{html.escape(lines[0])}</b></p>"
    return head + "".join(f"<p>{html.escape(line)}</p>" for line in lines[1:])
