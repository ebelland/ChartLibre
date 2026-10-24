"""Quick answer: "Are there differences?" - a measure across the groups of another column.

For someone who does not know which test to run: choose what to compare and
what the groups are; app.analysis.answers.differences picks the test, and
the report opens with the answer in a sentence, then what it rests on.
OK writes the groups' summary as a table and draws the comparison - box
plots for numbers, a mosaic for categories - with the report in its notes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import answers
from app.plugin_api.table_operations.column_roles import NOMINAL, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.quick_answer import QuickAnswerDialogBase
from app.plugin_api.table_operations.next_steps import NextStep, other_columns
from app.utils import report_html
from app.utils.i18n import _


class DifferencesDialog(QuickAnswerDialogBase):
    Name = "Are there differences?"
    Description = ("Is a measure different from one group to another? The right test is chosen for you, "
                   "and the answer is given in words.")
    Category = "Quick answers"
    CHART_PANELS = ('Box Plot', 'Mosaic Plot')
    Icon = """
    <rect x="3" y="8" width="5" height="9" rx="1"/><rect x="10" y="4" width="5" height="9" rx="1"/>
    <rect x="17" y="11" width="4" height="8" rx="1"/><path d="M2 21h20"/>
    """
    Keywords: str = (
        "ci sono differenze differenza differences compare confronta confronto gruppi groups t-test "
        "anova kruskal mann-whitney chi-squared chi quadrato significant significativo quick answer "
        "risposta rapida"
    )
    Goals: list[str] = ["compare"]

    ROLES = (
        ColumnRole("response", "Measure", "What to compare: numbers (a weight, a time) or categories (pass/fail).",
                   minimum=1, maximum=1),
        ColumnRole("group", "Groups", "The column that says which group each row is in.",
                   kinds=(NOMINAL, ORDINAL), minimum=1, maximum=1),
    )
    DISTINCT_ROLES = (("response", "group", "The measure and the groups must be two different columns."),)
    SUBTITLE_ROLES = ("response", "group")
    RESULT_TABLE = "differences"

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        return answers.differences(frame, casting.columns("response")[0], casting.columns("group")[0])

    def answer_details(self, result: answers.Answer) -> str:
        blocks = [
            report_html.section(_("The groups"), self.report_table(result.groups)),
            report_html.section(_("Details"), report_html.summary_table([
                (_("Test"), _(result.test)),
                (_("p-value"), report_html.format_number(result.pvalue)),
                (_(result.effect_name), report_html.format_number(result.effect)),
            ])),
        ]
        return "".join(blocks)

    def next_steps(self, result: answers.Answer) -> list[NextStep]:
        """On to what drives the measure; or again, with another measure or another grouping."""
        del result
        measure, group = self.measure, self.column("group")
        factors = [group, *other_columns(self._frame.columns, leave_out=(measure, group), limit=7)]
        return [
            self.next_step(_("What has the most impact on {measure} - {group} and the other columns, ranked")
                           .format(measure=measure, group=group), "What has the most impact?",
                           {"response": [measure], "factor": factors}),
            self.next_step(_("The same groups ({group}), another measure").format(group=group),
                           "Are there differences?", {"group": [group]}),
            self.next_step(_("{measure} again, grouped by another column").format(measure=measure),
                           "Are there differences?", {"response": [measure]}),
        ]

    def answer_table(self, result: answers.Answer) -> pd.DataFrame:
        return result.groups

    def answer_chart(self, result: answers.Answer, table: str) -> tuple:
        """Box plots of the measure by group, or a mosaic for categories - drawn from the source table."""
        del table
        measure, group = self.measure, self.column("group")
        # The chart leaves the hidden rows out, as the test did.
        shown = 'COALESCE("Hide", 0) = 0' if self._repo.has_column(self.table, "Hide") else ""
        title = _("{measure} by {group}").format(measure=measure, group=group)
        if result.numeric:
            sql, roles = self.series_select(self.table, {"value": measure, "group": group}, where=shown)
            return ("Box Plot", title, group, measure,
                    [(measure, sql + f" ORDER BY {self.quoted(group)}", roles, {})])
        sql, roles = self.series_select(self.table, {"x": group, "y": measure}, where=shown)
        return ("Mosaic Plot", title, group, measure, [(measure, sql, roles, {})])
