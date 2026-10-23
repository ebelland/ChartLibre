"""Quick answer: "Is it in control?" - a measure on a control chart, in words.

Choose the measure and, if there is one, the column that orders it (dates,
times, a run number); without it the rows' own order is used.
app.analysis.in_control draws an Individuals chart with the Nelson run rules
and says whether the process is stable, and where it is not. OK writes the
values with the limits and draws the chart, with the report in the notes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import in_control as ic
from app.analysis.change import ROW_ORDER
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.plugin_api.table_operations.next_steps import NextStep
from app.table_operations.differences_dialog import answer_html
from app.utils.i18n import _


class InControlDialog(TableOperationDialogBase):
    Name = "Is it in control?"
    Description = ("Is a process stable, or did something special happen? A control chart with its run rules, "
                   "and the answer given in words.")
    Category = "Quick answers"
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

    ROLES = (
        ColumnRole("response", "Measure", "The process measurements to watch.",
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
        return ic.is_in_control(frame, casting.columns("response")[0], time[0] if time else ROW_ORDER)

    def report_body(self, result: ic.ControlAnswer) -> str:
        return answer_html(result)

    def report_subtitle(self, result: ic.ControlAnswer) -> str:
        return f"{self.table} · {self.roles_widget.casting().columns('response')[0]}"

    def next_steps(self, result: ic.ControlAnswer) -> list[NextStep]:
        """On to whether it drifted, whether it is normal (the limits assume it), or another measure."""
        del result
        casting = self.roles_widget.casting()
        measure, time = casting.columns("response")[0], casting.columns("time")
        return [
            self.next_step(_("Has {measure} changed over time?").format(measure=measure),
                           "Has it changed over time?", {"response": [measure], "time": time}),
            self.next_step(_("Is {measure} normal?").format(measure=measure), "Is it normal?",
                           {"response": [measure]}),
            self.next_step(_("Another measure of the same process"), "Is it in control?", {"time": time}),
        ]

    def apply_results(self, result: ic.ControlAnswer) -> list[dict[str, Any]]:
        casting = self.roles_widget.casting()
        measure = casting.columns("response")[0]
        time = casting.columns("time")
        name = self.write_result_tables({"control": result.series})["control"]

        series = []
        for column, label, style in (
            ("UCL", "UCL", {"linestyle": "--", "color": "#c62828"}),
            ("LCL", "LCL", {"linestyle": "--", "color": "#c62828"}),
            ("CL", "CL", {"linestyle": "-", "color": "#2e7d32"}),
        ):
            sql, roles = self.series_select(name, {"x": "Order", "y": column})
            series.append((label, sql, roles, {**style, "marker": "", "linewidth": 1.1, "show_in_legend": False}))
        sql, roles = self.series_select(name, {"x": "Order", "y": "Value"})
        series.append((measure, sql, roles, {"linestyle": "-", "marker": "o", "markersize": 4.0}))
        if result.signals:
            sql, roles = self.series_select(name, {"x": "Order", "y": "Signal"},
                                            where=f"{self.quoted('Signal')} IS NOT NULL")
            series.append((_("Signals"), sql, roles, {"linestyle": "", "marker": "o", "markersize": 9.0,
                                                      "color": "#c62828", "show_in_legend": False}))
        x_label = time[0] if time else _("Row")
        options: dict[str, Any] = {"show_rolling": False}
        panel = ("Time Series", _("{measure}: control chart").format(measure=measure), x_label, measure, series,
                 options)
        notes = self.report_document(result)
        figure_id = self.create_report_figure(f"{_(self.Name)} · {measure}", notes, [panel])
        self.created_figure_ids = [figure_id]
        return [{"table": name}, {"figure": figure_id, "name": measure}]
