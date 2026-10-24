"""The quick answers' base: a question answered in words - "Are there differences?", "Is it normal?"...

Each picks its test itself and opens its report with the answer in a
sentence (answer_html), then what the answer rests on. They share all but
the question: the measure (the "response" role) in the subtitle, one table
of what the answer rests on and one chart of it, drawn with the report in
its notes. A quick answer gives compute, RESULT_TABLE with answer_table and
answer_chart, its next steps, and answer_details for more under the answer.

Each keeps ``Category = "Quick answers"`` in its own body: the Analysis
panel finds the plugins by reading their source, not by importing them.
"""
from __future__ import annotations

import html
from typing import Any, ClassVar

import pandas as pd

from app.analysis import answers
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
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


class QuickAnswerDialogBase(TableOperationDialogBase):
    """A question on a measure, answered in words, with the table and the chart it rests on."""

    #: The key of the table OK writes (named after it and the source table).
    RESULT_TABLE: ClassVar[str] = "answer"

    @property
    def measure(self) -> str:
        """The column the question is about."""
        return self.column("response")

    def report_body(self, result: Any) -> str:
        return answer_html(result) + self.answer_details(result)

    def answer_details(self, result: Any) -> str:
        """More under the answer, as HTML sections; nothing by default."""
        del result
        return ""

    def answer_table(self, result: Any) -> pd.DataFrame:
        """The table OK writes: what the answer rests on."""
        raise NotImplementedError

    def answer_chart(self, result: Any, table: str) -> tuple:
        """The one panel of the answer's figure (see create_report_figure), drawn from *table* or the source."""
        raise NotImplementedError

    def apply_results(self, result: Any) -> list[dict[str, Any]]:
        name = self.write_result_tables({self.RESULT_TABLE: self.answer_table(result)})[self.RESULT_TABLE]
        return self.draw_report(result, self.measure, [self.answer_chart(result, name)], written=[{"table": name}])
