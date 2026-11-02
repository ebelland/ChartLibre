"""Has it changed over time? (Quick answer): a measure along time, in words.

Each series - or each table's columns, read as its series - gets its answer: the sentence that answers the
question, what it rests on (the trend test, the change-point test), and the "What next?" links. OK draws the
measure over time with its trend line, or the medians before and after the change, in a figure of its own.
The lines are written into the chart's queries as values, so no table is written. The engine is
app.analysis.change.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from app.analysis import change as ch
from app.analysis import time_series as tsa
from app.data.data_source import row_value
from app.data_operations.normality_dialog import source_column, source_table
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase, xy_columns
from app.plugin_api.data_operations.next_steps import NextStep, other_columns
from app.plugin_api.data_operations.quick_answer import answer_html, measured_query, points_query
from app.plugin_api.data_operations.results import OperationResult
from app.utils import report_html
from app.utils.i18n import _

#: The time column the test reads: a column of the series' own, never a name a measure can take.
_TIME = "_time"


@dataclass(slots=True)
class ChangeSeriesResult(OperationResult):
    """One series' answer, and the figure that draws it.

    ``query`` is the series' own SQL, ``x_column`` and ``y_column`` its time and measure columns.
    """

    source_name: str
    query: str
    x_column: str
    y_column: str
    answer: ch.ChangeAnswer

    def to_df(self) -> pd.DataFrame:
        return self.answer.series

    def preview(self, dialog: ChangeDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def apply(self, dialog: ChangeDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def draw(self, dialog: ChangeDialog) -> None:
        """The measure over time, with its trend line or the two medians, in a figure of its own."""
        name = self.source_name
        measured = measured_query(self.query, self.x_column, self.y_column)
        series = [(name, measured, {"x": "x", "y": "y"}, {"linestyle": "-", "marker": "o"})]
        frame = self.answer.series
        if self.answer.shape == "drift":
            ends = frame.iloc[[0, -1]]
            series.append((_("Trend"), points_query(list(zip(ends["Order"], ends["Trend"], strict=True))),
                           {"x": "x", "y": "y"}, {"linestyle": "--", "marker": "", "color": "#d62728"}))
        elif self.answer.shape == "step":
            for segment, label in (("Before", _("Before")), ("After", _("After"))):
                part = frame[frame["Segment"] == segment]
                if part.empty:
                    continue
                points = [(part["Order"].iloc[0], part["Segment median"].iloc[0]),
                          (part["Order"].iloc[-1], part["Segment median"].iloc[-1])]
                series.append((label, points_query(points), {"x": "x", "y": "y"},
                               {"linestyle": "-", "marker": "", "color": "#d62728", "linewidth": 2}))
        options: dict[str, Any] = {"show_rolling": False}
        if self.answer.temporal:
            options.update(tsa.gap_options(frame["Order"]))
        panel = ("Time Series", _("{measure} over time").format(measure=name), self.x_column, name, series, options)
        dialog.create_report_figure(f"{_(dialog.Name)} · {name}",
                                    dialog.results_document(name, dialog.report_for(self)), [panel])


class ChangeDialog(DataOperationDialogBase):
    """Each series' answer to "has it changed over time?", with its trend or its medians drawn."""

    Name = "Has it changed over time?"
    Description = ("Has a measure moved - a steady trend or a sudden change? The tests are chosen for you, "
                   "and the answer is given in words.")
    Category = "Quick answers"
    Order = 30
    Icon = """
    <path d="M3 17l4-2 3 1 3-5 3-1 5-6"/><path d="M3 21h18"/><circle cx="13" cy="11" r="1.6"/>
    """
    Keywords: str = (
        "è cambiato nel tempo cambiamento change changed over time trend tendenza drift deriva step "
        "gradino change point punto di cambiamento before after prima dopo mann-kendall pettitt sen "
        "slope stabile stable quick answer risposta rapida"
    )
    Goals: ClassVar[list[str]] = ["signal", "process"]

    DESTINATIONS = (DataOperationDialogBase.DEST_NONE, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_NEW_FIGURE
    SELECTS_ALL_SERIES = True
    TABLE_CHARTS = ("Time Series", "Scatter Plot")

    def read_series(self, row: Any, name: str) -> Any:
        """The series' x and y, its query, and the columns its time and measure come from."""
        frame, roles = self._series_frame(row)
        x_column, y_column = xy_columns(frame, roles)
        query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        return self.series_xy(row, name), query, x_column, y_column

    def compute_series(self, name: str, data: Any, settings: Any) -> ChangeSeriesResult:
        del settings
        (x, y), query, x_column, y_column = data
        frame = pd.DataFrame({_TIME: np.asarray(x), name: np.asarray(y, dtype=float)})
        answer = ch.has_changed(frame, name, _TIME)
        return ChangeSeriesResult(source_name=name, query=query, x_column=x_column, y_column=y_column,
                                  answer=answer)

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """The report draws its own figure: nothing is added to the series' axis."""
        del results
        return selected_axis_id

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_for(self, result: ChangeSeriesResult) -> str:
        """One series' answer: the sentences, the tests it rests on, and the "What next?" links."""
        answer = result.answer
        box = ""
        if reading_guide.enabled():
            box = report_html.reading_guide(_("How to read this"), self.reading_for(result))
        body = report_html.section(f"{result.source_name} (N = {answer.rows})",
                                   report_html.with_reading_guide(answer_html(answer), box))
        return body + self.what_next(result)

    def reading_for(self, result: ChangeSeriesResult) -> list[str]:
        """One series' "How to read this" lines, in its own section (the base's box stays empty)."""
        answer, measure = result.answer, result.source_name
        return [
            reading_guide.p_value(answer.trend_p, null=_("{measure} had no trend").format(measure=measure),
                                  shown=_("{measure} has a trend").format(measure=measure)),
            reading_guide.p_value(answer.change_p,
                                  null=_("{measure} kept the same level throughout").format(measure=measure),
                                  shown=_("{measure} shifted at one point").format(measure=measure)),
        ]

    def next_steps(self, result: ChangeSeriesResult) -> list[NextStep]:
        """On to its trend, season and forecast, or to what drives it - when its columns come from a table."""
        table = source_table(result.query)
        if not table:
            return []
        columns = [str(column) for column in self._repo.table_frame(table).columns]
        measure, name = source_column(result.query, "y"), result.source_name
        # The time is the series' x when it is a column of the table; a chart of row numbers has none.
        time = [column] if (column := source_column(result.query, "x")) in columns else []
        steps = [NextStep(label=_("Its trend, season and forecast: Time Series of {measure}").format(measure=name),
                          operation="Time Series", table=table, roles={"response": [measure], "time": time})]
        others = other_columns(columns, leave_out=(measure, *time), limit=8)
        if others:
            steps.append(NextStep(label=_("What has the most impact on {measure}?").format(measure=name),
                                  operation="What has the most impact?", table=table,
                                  roles={"response": [measure], "factor": others}))
        return steps

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        subtitle = ", ".join(result.source_name for result in results)
        return self.results_document(subtitle, *(self.report_for(result) for result in results))
