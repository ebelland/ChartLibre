"""Quick answer: "Is it normal?" - does a measure follow the normal distribution, in words.

Choose the measure. app.analysis.normality tests it (Shapiro-Wilk, or
Anderson-Darling past 5000 values), names the shape that departs and adds
the cautions about sample size. OK writes the Q-Q values and draws the Q-Q
plot, with the report in the notes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import normality as nm
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.plugin_api.table_operations.next_steps import NextStep, other_columns
from app.table_operations.differences_dialog import answer_html
from app.utils.i18n import _


class NormalityDialog(TableOperationDialogBase):
    Name = "Is it normal?"
    Description = ("Does a measure follow the normal (bell-shaped) distribution that t-tests, ANOVA and control "
                   "charts assume? The test is chosen for you, and the answer is given in words.")
    Category = "Quick answers"
    Icon = """
    <path d="M2 19c3 0 4-12 10-12s7 12 10 12"/><path d="M2 21h20"/>
    """
    Keywords: str = (
        "è normale normalità normality normal gaussian gaussiana bell curve campana shapiro wilk "
        "anderson darling q-q qq plot skewness asimmetria kurtosis curtosi distribution distribuzione "
        "quick answer risposta rapida"
    )
    Goals: list[str] = ["describe"]

    ROLES = (
        ColumnRole("response", "Measure", "The numbers whose distribution to judge.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1, maximum=1),
    )


    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        return nm.is_normal(frame, casting.columns("response")[0])

    def report_body(self, result: nm.NormalityAnswer) -> str:
        return answer_html(result)

    def report_subtitle(self, result: nm.NormalityAnswer) -> str:
        return f"{self.table} · {self.columns('response')[0]}"

    def next_steps(self, result: nm.NormalityAnswer) -> list[NextStep]:
        """On to its whole distribution, to differences between groups, or to whether it is in control."""
        del result
        measure = self.columns("response")[0]
        steps = [self.next_step(_("Its histogram, quantiles and fitted distributions: Distribution of {measure}")
                                .format(measure=measure), "Distribution", {"response": [measure]})]
        others = other_columns(self._frame.columns, leave_out=(measure,), limit=1)
        if others:
            steps.append(self.next_step(_("Is {measure} different between groups?").format(measure=measure),
                                        "Are there differences?", {"response": [measure]}))
        steps.append(self.next_step(_("Is {measure} in control?").format(measure=measure), "Is it in control?",
                                    {"response": [measure]}))
        return steps

    def apply_results(self, result: nm.NormalityAnswer) -> list[dict[str, Any]]:
        measure = self.columns("response")[0]
        name = self.write_result_tables({"normality": result.qq})["normality"]
        sql, roles = self.series_select(name, {"x": "Theoretical", "y": "Line"})
        series = [(_("Normal"), sql, roles, {"linestyle": "--", "marker": "", "color": "#c62828",
                                             "show_in_legend": False})]
        sql, roles = self.series_select(name, {"x": "Theoretical", "y": "Sample"})
        series.append((measure, sql, roles, {"linestyle": "", "marker": "o", "markersize": 4.0}))
        panel = ("Scatter Plot", _("{measure}: normal Q-Q plot").format(measure=measure),
                 _("Normal quantile"), measure, series)
        return self.draw_report(result, measure, [panel], written=[{"table": name}])
