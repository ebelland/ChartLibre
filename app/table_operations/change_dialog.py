"""Quick answer: "Has it changed over time?" - a measure along time, in words (Q-03).

Choose the measure and, if there is one, the column that orders it (dates,
times, a run number); without it the rows' own order is used.
app.analysis.change tests for a steady trend and for a sudden change and
says which one the values show. OK writes the values with the trend line and
the medians before and after, and draws them, with the report in the notes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import change as ch
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.plugin_api.table_operations.next_steps import NextStep, is_level, other_columns
from app.table_operations.differences_dialog import answer_html
from app.utils.i18n import _


class ChangeDialog(TableOperationDialogBase):
    Name = "Has it changed over time?"
    Description = ("Has a measure moved - a steady trend or a sudden change? The tests are chosen for you, "
                   "and the answer is given in words.")
    Category = "Quick answers"
    Icon = """
    <path d="M3 17l4-2 3 1 3-5 3-1 5-6"/><path d="M3 21h18"/><circle cx="13" cy="11" r="1.6"/>
    """
    Keywords: str = (
        "è cambiato nel tempo cambiamento change changed over time trend tendenza drift deriva step "
        "gradino change point punto di cambiamento before after prima dopo mann-kendall pettitt sen "
        "slope stabile stable quick answer risposta rapida"
    )
    Goals: list[str] = ["signal", "process"]

    ROLES = (
        ColumnRole("response", "Measure", "The numbers to follow over time.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1, maximum=1),
        ColumnRole("time", "Time or order", "Optional: dates, times or a run number. Without it, the rows' order.",
                   maximum=1),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        if not found and set(casting.columns("response")) & set(casting.columns("time")):
            found.append(_("The measure and the time must be two different columns."))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        time = casting.columns("time")
        return ch.has_changed(frame, casting.columns("response")[0], time[0] if time else ch.ROW_ORDER)

    def report_body(self, result: ch.ChangeAnswer) -> str:
        return answer_html(result)

    def report_subtitle(self, result: ch.ChangeAnswer) -> str:
        return f"{self.table} · {self.roles_widget.casting().columns('response')[0]}"

    def next_steps(self, result: ch.ChangeAnswer) -> list[NextStep]:
        """On to its trend, season and forecast, to what drives it, or to another measure over the same time."""
        del result
        casting = self.roles_widget.casting()
        measure, time = casting.columns("response")[0], casting.columns("time")
        steps = [self.next_step(_("Its trend, season and forecast: Time Series of {measure}").format(measure=measure),
                                "Time Series", {"response": [measure], "time": time})]
        others = other_columns(self._frame.columns, leave_out=(measure, *time), limit=8)
        if others:
            steps.append(self.next_step(_("What has the most impact on {measure}?").format(measure=measure),
                                        "What has the most impact?", {"response": [measure], "factor": others}))
        steps.append(self.next_step(_("Another measure over the same time"), "Has it changed over time?",
                                    {"time": time}))
        return steps

    def apply_results(self, result: ch.ChangeAnswer) -> list[dict[str, Any]]:
        casting = self.roles_widget.casting()
        measure = casting.columns("response")[0]
        time = casting.columns("time")
        name = self.write_result_tables({"change": result.series})["change"]

        series = []
        sql, roles = self.series_select(name, {"x": "Order", "y": "Value"})
        series.append((measure, sql, roles, {"linestyle": "-", "marker": "o"}))
        if result.shape == "drift":
            sql, roles = self.series_select(name, {"x": "Order", "y": "Trend"})
            series.append((_("Trend"), sql, roles, {"linestyle": "--", "marker": "", "color": "#d62728"}))
        elif result.shape == "step":
            for segment, label in (("Before", _("Before")), ("After", _("After"))):
                sql, roles = self.series_select(name, {"x": "Order", "y": "Segment median"},
                                                where=f"{self.quoted('Segment')} = {self.literal(segment)}")
                series.append((label, sql, roles, {"linestyle": "-", "marker": "", "color": "#d62728",
                                                   "linewidth": 2}))
        x_label = time[0] if time else _("Row")
        # No rolling average of its own: the trend or the two medians are the summary here.
        options: dict[str, Any] = {"show_rolling": False}
        if result.temporal:
            # The renderer breaks the line at gaps longer than 2 h by default,
            # which cut daily values into dots; a gap is five typical steps.
            moments = pd.to_datetime(result.series["Order"], errors="coerce", format="mixed").dropna()
            step = moments.diff().median() if len(moments) > 1 else pd.NaT
            if pd.notna(step) and step > pd.Timedelta(0):
                options["gap_threshold"] = f"{int((5 * step).total_seconds())}s"
        panel = ("Time Series", _("{measure} over time").format(measure=measure), x_label, measure, series, options)
        notes = self.report_document(result)
        figure_id = self.create_report_figure(f"{_(self.Name)} · {measure}", notes, [panel])
        self.created_figure_ids = [figure_id]
        return [{"table": name}, {"figure": figure_id, "name": measure}]
