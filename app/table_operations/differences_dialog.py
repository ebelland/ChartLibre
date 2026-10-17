"""Quick answer: "Are there differences?" - a measure across the groups of another column.

For someone who does not know which test to run: choose what to compare and
what the groups are; app.analysis.answers.differences picks the test, and
the report opens with the answer in a sentence, then what it rests on.
OK writes the groups' summary as a table and draws the comparison - box
plots for numbers, a mosaic for categories - with the report in its notes.
"""
from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import answers
from app.plugin_api.table_operations.column_roles import NOMINAL, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


def answer_html(answer: answers.Answer) -> str:
    """The sentences, translated: the first as the answer, the rest beneath it."""
    def words(values: dict[str, Any]) -> dict[str, Any]:
        # The size and the test's name are words too, in the window's language.
        return {**values, **{key: _(values[key]) for key in ("size", "test", "unit") if key in values}}

    lines = [_(template).format(**words(values)) for template, values in answer.sentences]
    colour = "#1a7f37" if answer.verdict == answers.YES else "#57606a"
    head = f"<p style='font-size:125%; color:{colour}'><b>{html.escape(lines[0])}</b></p>"
    return head + "".join(f"<p>{html.escape(line)}</p>" for line in lines[1:])


class DifferencesDialog(TableOperationDialogBase):
    Name = "Are there differences?"
    Description = ("Is a measure different from one group to another? The right test is chosen for you, "
                   "and the answer is given in words.")
    Category = "Quick answers"
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

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        if not found and set(casting.columns("response")) & set(casting.columns("group")):
            found.append(_("The measure and the groups must be two different columns."))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        return answers.differences(frame, casting.columns("response")[0], casting.columns("group")[0])

    def report_subtitle(self, result: answers.Answer) -> str:
        casting = self.roles_widget.casting()
        return f"{self.table} · {casting.columns('response')[0]} / {casting.columns('group')[0]}"

    def report_body(self, result: answers.Answer) -> str:
        blocks = [
            answer_html(result),
            report_html.section(_("The groups"), self.report_table(result.groups)),
            report_html.section(_("Details"), report_html.summary_table([
                (_("Test"), _(result.test)),
                (_("p-value"), report_html.format_number(result.pvalue)),
                (_(result.effect_name), report_html.format_number(result.effect)),
            ])),
        ]
        return "".join(blocks)

    def apply_results(self, result: answers.Answer) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        casting = self.roles_widget.casting()
        measure, group = casting.columns("response")[0], casting.columns("group")[0]
        name = self.write_result_tables({"differences": result.groups})["differences"]
        # The chart leaves the hidden rows out, as the test did.
        shown = 'COALESCE("Hide", 0) = 0' if repo.has_column(table, "Hide") else ""
        if result.numeric:
            sql, roles = self.series_select(table, {"value": measure, "group": group}, where=shown)
            panel = ("Box Plot", _("{measure} by {group}").format(measure=measure, group=group), group, measure,
                     [(measure, sql + f" ORDER BY {self.quoted(group)}", roles, {})])
        else:
            sql, roles = self.series_select(table, {"x": group, "y": measure}, where=shown)
            panel = ("Mosaic Plot", _("{measure} by {group}").format(measure=measure, group=group), group, measure,
                     [(measure, sql, roles, {})])
        notes = self.report_document(result)
        figure_id = self.create_report_figure(f"{_(self.Name)} · {measure}", notes, [panel])
        self.created_figure_ids = [figure_id]
        return [{"table": name}, {"figure": figure_id, "name": measure}]
