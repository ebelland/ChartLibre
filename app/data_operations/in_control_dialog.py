"""Quick answer: "Is it in control?" - a measure on a control chart, in words.

Each series - or each table's columns, read as its series - gets its answer: its x orders the values (dates,
times, a run number), its y is the measure. app.analysis.in_control draws an Individuals chart with the
Nelson run rules and says whether the process is stable, and where it is not; then the "What next?" links.
OK draws the control chart - the values, the limits and the signals - in a figure of its own. The limits and
the signals are written into the chart's queries as values, so no table is written.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.analysis import in_control as ic
from app.analysis import time_series as tsa
from app.data.data_source import row_value
from app.data_operations.normality_dialog import source_column, source_table
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase, xy_columns
from app.plugin_api.data_operations.next_steps import NextStep
from app.plugin_api.data_operations.quick_answer import answer_html, measured_query, points_query
from app.plugin_api.data_operations.results import OperationResult
from app.utils import report_html
from app.utils.i18n import _

#: The order column the chart reads: a column of the series' own, never a name a measure can take.
_TIME = "_time"


@dataclass(slots=True)
class ControlSeriesResult(OperationResult):
    """One series' answer, and the control chart that draws it.

    ``query`` is the series' own SQL, ``x_column`` and ``y_column`` its order and measure columns.
    """

    source_name: str
    query: str
    x_column: str
    y_column: str
    answer: ic.ControlAnswer

    def to_df(self) -> pd.DataFrame:
        return self.answer.series

    def preview(self, dialog: InControlDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def apply(self, dialog: InControlDialog, axis_id: int) -> None:
        del axis_id
        self.draw(dialog)

    def draw(self, dialog: InControlDialog) -> None:
        """The individuals chart: its limits, the values and the signals, in a figure of its own."""
        name, answer = self.source_name, self.answer
        frame = answer.series
        ends = (frame["Order"].iloc[0], frame["Order"].iloc[-1])
        series = []
        for label, level, style in (
            ("UCL", answer.upper, {"linestyle": "--", "color": "#c62828"}),
            ("LCL", answer.lower, {"linestyle": "--", "color": "#c62828"}),
            ("CL", answer.center, {"linestyle": "-", "color": "#2e7d32"}),
        ):
            series.append((label, points_query([(ends[0], level), (ends[1], level)]), {"x": "x", "y": "y"},
                           {**style, "marker": "", "linewidth": 1.1, "show_in_legend": False}))
        series.append((name, measured_query(self.query, self.x_column, self.y_column), {"x": "x", "y": "y"},
                       {"linestyle": "-", "marker": "o", "markersize": 4.0}))
        signals = frame[frame["Signal"].notna()]
        if not signals.empty:
            series.append((_("Signals"), points_query(list(zip(signals["Order"], signals["Signal"], strict=True))),
                           {"x": "x", "y": "y"}, {"linestyle": "", "marker": "o", "markersize": 9.0,
                                                  "color": "#c62828", "show_in_legend": False}))
        options: dict[str, Any] = {"show_rolling": False}
        if answer.temporal:
            options.update(tsa.gap_options(frame["Order"]))
        panel = ("Time Series", _("{measure}: control chart").format(measure=name), self.x_column, name, series,
                 options)
        dialog.create_report_figure(f"{_(dialog.Name)} · {name}",
                                    dialog.results_document(name, dialog.report_for(self)), [panel])


class InControlDialog(DataOperationDialogBase):
    """Each series' answer to "is it in control?", with its control chart drawn."""

    Name = "Is it in control?"
    Description = ("Is a process stable, or did something special happen? A control chart with its run rules, "
                   "and the answer given in words.")
    Category = "Quick answers"
    Order = 40
    Icon = """
    <path d="M3 6h18" stroke-dasharray="2 2"/><path d="M3 18h18" stroke-dasharray="2 2"/>
    <path d="M3 12l3-2 3 3 3-4 3 2 3-1 3 2"/>
    """
    Keywords: str = (
        "è sotto controllo in control carta di controllo control chart spc shewhart stabile stable processo "
        "process nelson western electric run rules limiti limits ucl lcl special cause causa speciale "
        "quick answer risposta rapida"
    )
    Goals: list[str] = ["process", "anomalies"]

    DESTINATIONS = (DataOperationDialogBase.DEST_NONE, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_NEW_FIGURE
    SELECTS_ALL_SERIES = True
    TABLE_CHARTS = ("Time Series", "Scatter Plot")

    def read_series(self, row: Any, name: str) -> Any:
        """The series' order and values as its query returns them (dates kept as dates), its query, and the
        columns they come from."""
        frame, roles = self._series_frame(row)
        x_column, y_column = xy_columns(frame, roles)
        query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        return frame[x_column].to_numpy(), frame[y_column].to_numpy(), query, x_column, y_column

    def compute_series(self, name: str, data: Any, settings: Any) -> ControlSeriesResult:
        del settings
        x, y, query, x_column, y_column = data
        answer = ic.is_in_control(pd.DataFrame({_TIME: x, name: y}), name, _TIME)
        return ControlSeriesResult(source_name=name, query=query, x_column=x_column, y_column=y_column,
                                   answer=answer)

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """The report draws its own figure: nothing is added to the series' axis."""
        del results
        return selected_axis_id

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_for(self, result: ControlSeriesResult) -> str:
        """One series' answer: the sentences, the limits read, and the "What next?" links."""
        answer = result.answer
        box = ""
        if reading_guide.enabled():
            box = report_html.reading_guide(_("How to read this"), [
                reading_guide.control_limits(answer.center, answer.lower, answer.upper, signals=answer.signals)])
        body = report_html.section(f"{result.source_name} (N = {answer.rows})",
                                   report_html.with_reading_guide(answer_html(answer), box))
        return body + self.what_next(result)

    def next_steps(self, result: ControlSeriesResult) -> list[NextStep]:
        """On to whether it drifted, whether it is normal (the limits assume it), or another measure."""
        name = result.source_name
        table = source_table(result.query)
        measure = source_column(result.query, "y")
        if not table or not measure:
            # A series of no table: the next questions open on this chart.
            return [NextStep(label=_("Has {measure} changed over time?").format(measure=name),
                             operation="Has it changed over time?", table="", roles={}, figure=self._figure_id),
                    NextStep(label=_("Is {measure} normal?").format(measure=name), operation="Is it normal?",
                             table="", roles={}, figure=self._figure_id)]
        columns = [str(column) for column in self._repo.table_frame(table).columns]
        # The time is the series' x when it is a column of the table; a chart of row numbers has none.
        time = [column] if (column := source_column(result.query, "x")) in columns else []
        return [
            NextStep(label=_("Has {measure} changed over time?").format(measure=name),
                     operation="Has it changed over time?", table=table, roles={"response": [measure], "time": time}),
            NextStep(label=_("Is {measure} normal?").format(measure=name), operation="Is it normal?", table=table,
                     roles={"response": [measure]}),
        ]

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        subtitle = ", ".join(result.source_name for result in results)
        return self.results_document(subtitle, *(self.report_for(result) for result in results))
