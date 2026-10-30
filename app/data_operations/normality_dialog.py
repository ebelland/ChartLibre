"""Is it normal? (Quick answer): does a series follow the normal distribution, in words.

Each checked series gets its own answer: the sentence that answers the question, then what it rests on
(the test, the shape that departs, the cautions about sample size), and the "What next?" links. OK draws
the series' normal Q-Q plot in a figure of its own. The engine is app.analysis.normality; nothing is
written to the project's tables.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.analysis import normality as nm
from app.data_operations.distribution_dialog import (
    qq_panel,
    series_values,
    value_series,
)
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase
from app.plugin_api.data_operations.results import OperationResult
from app.plugin_api.table_operations.next_steps import NextStep
from app.plugin_api.table_operations.quick_answer import answer_html
from app.utils import report_html
from app.utils.i18n import _

#: The FROM of a series' query: the table it reads, quoted or bare, last in the query.
_SOURCE = re.compile(r'\sFROM\s+("(?:[^"]|"")+"|\w+)(?=\s+(?:WHERE|ORDER|LIMIT|GROUP)\b|\s*$)',
                     re.IGNORECASE)


def source_column(query: str, alias: str) -> str:
    """The table column a series' query puts under *alias* (x or y), or "" when it does not say."""
    found = re.search(rf'("(?:[^"]|"")+"|\w+)\s+AS\s+{alias}\b', query, re.IGNORECASE)
    if found is not None:
        name = found.group(1)
        return name[1:-1].replace('""', '"') if name.startswith('"') else name
    # "SELECT x, y FROM t": the column is named as the alias, when the select list names it.
    select = re.split(r"\sFROM\s", query, maxsplit=1, flags=re.IGNORECASE)[0]
    return alias if re.search(rf'(?<![\w"]){alias}(?![\w"])', select) else ""


def source_table(query: str) -> str:
    """The table a series' query reads, or "" when it reads none that can be named.

    A series' query may carry the chart's own filter after the table (WHERE "Hide" = 0): the table is still the one.
    """
    found = _SOURCE.search(query)
    if found is None:
        return ""
    name = found.group(1)
    return name[1:-1].replace('""', '"') if name.startswith('"') else name


@dataclass(slots=True)
class NormalityResult(OperationResult):
    """One series' normality answer, and the figure that draws its Q-Q plot.

    ``query`` is the series' own SQL, ``column`` the column of its y values.
    """

    source_name: str
    query: str
    column: str
    answer: nm.NormalityAnswer

    def to_df(self) -> Any:
        return self.answer.qq

    def preview(self, dialog: NormalityDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def apply(self, dialog: NormalityDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def draw(self, dialog: NormalityDialog) -> None:
        """The normal Q-Q plot of the series, in a figure of its own."""
        series = value_series(self.source_name, self.query, self.column)
        dialog.create_report_figure(f"{_(dialog.Name)} · {self.source_name}",
                                    dialog.results_document(self.source_name, dialog.report_for(self)),
                                    [qq_panel(self.source_name, series, "norm")])


class NormalityDialog(DataOperationDialogBase):
    """Each series' answer to "is it normal?", with its normal Q-Q plot."""

    Name = "Is it normal?"
    Description = ("Does a series follow the normal (bell-shaped) distribution that t-tests, ANOVA and control "
                   "charts assume? The test is chosen for you, and the answer is given in words.")
    Category = "Quick answers"
    Order = 20
    Icon = """
    <path d="M2 19c3 0 4-12 10-12s7 12 10 12"/><path d="M2 21h20"/>
    """
    Keywords: str = (
        "è normale normalità normality normal gaussian gaussiana bell curve campana shapiro wilk "
        "anderson darling q-q qq plot skewness asimmetria kurtosis curtosi distribution distribuzione "
        "quick answer risposta rapida"
    )
    Goals: list[str] = ["describe"]

    DESTINATIONS = (DataOperationDialogBase.DEST_NONE, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_NEW_FIGURE
    SELECTS_ALL_SERIES = True

    def read_series(self, row: Any, name: str) -> Any:
        return series_values(self, row, name)

    def compute_series(self, name: str, data: Any, settings: Any) -> NormalityResult:
        del settings
        (_x, y), query, column = data
        frame = pd.DataFrame({name: np.asarray(y, dtype=float)})
        return NormalityResult(source_name=name, query=query, column=column, answer=nm.is_normal(frame, name))

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """The report draws its own figure: nothing is added to the series' axis."""
        del results
        return selected_axis_id

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_for(self, result: NormalityResult) -> str:
        """One series' answer: the sentences, the test it rests on, and the "What next?" links."""
        answer = result.answer
        box = ""
        if reading_guide.enabled():
            box = report_html.reading_guide(_("How to read this"), self.reading_for(result))
        body = report_html.section(f"{result.source_name} (N = {answer.rows})",
                                   report_html.with_reading_guide(answer_html(answer), box))
        return body + self.what_next(result)

    def reading_for(self, result: NormalityResult) -> list[str]:
        """One series' "How to read this" lines, in its own section (the base's box stays empty)."""
        answer = result.answer
        return [reading_guide.normality(answer.pvalue, test=answer.test, column=result.source_name)]

    def next_steps(self, result: NormalityResult) -> list[NextStep]:
        """On to its whole distribution, to differences between groups, or to whether it is in control."""
        measure = result.source_name
        steps = [NextStep(label=_("Its histogram, quantiles and fitted distributions: Distribution of {measure}")
                          .format(measure=measure), operation="Distribution", table="", roles={},
                          figure=self._figure_id)]
        table = source_table(result.query)
        column = source_column(result.query, "y")
        if table and column:
            steps.append(NextStep(label=_("Is {measure} different between groups?").format(measure=measure),
                                  operation="Are there differences?", table=table, roles={"response": [column]}))
            steps.append(NextStep(label=_("Is {measure} in control?").format(measure=measure),
                                  operation="Is it in control?", table=table, roles={"response": [column]}))
        return steps

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        subtitle = ", ".join(result.source_name for result in results)
        return self.results_document(subtitle, *(self.report_for(result) for result in results))
