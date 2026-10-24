"""Quick answer: "What has the most impact?" - one response, many factors, ranked in words (Q-02).

Choose the result and the columns that might drive it; app.analysis.impact
ranks them (a random forest for how much, a linear model or a test per
factor for whether it is more than chance) and the report opens with the
answer in a sentence. OK writes the ranking as a table and draws it as a
Pareto chart, with the report in its notes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import impact as im
from app.plugin_api.table_operations.column_roles import ColumnCasting, ColumnRole
from app.plugin_api.table_operations.quick_answer import QuickAnswerDialogBase
from app.plugin_api.table_operations.next_steps import NextStep, is_level
from app.utils import report_html
from app.utils.i18n import _


class ImpactDialog(QuickAnswerDialogBase):
    Name = "What has the most impact?"
    Description = ("Which of many columns matters most for a result? The factors are ranked for you, "
                   "and the answer is given in words.")
    Category = "Quick answers"
    Icon = """
    <rect x="3" y="4" width="18" height="3.2" rx="1"/><rect x="3" y="10.4" width="11" height="3.2" rx="1"/>
    <rect x="3" y="16.8" width="5" height="3.2" rx="1"/>
    """
    Keywords: str = (
        "cosa impatta di più impatto impact most important factors fattori importanti drivers key "
        "driver importance importanza influence influenza cause cosa conta what matters ranking "
        "classifica quick answer risposta rapida"
    )
    Goals: list[str] = ["explain"]

    ROLES = (
        ColumnRole("response", "Result", "What you want to explain: numbers (a yield, a time) or categories "
                                         "(pass/fail).", minimum=1, maximum=1),
        ColumnRole("factor", "Factors", "The columns that might drive it.", minimum=1),
    )
    DISTINCT_ROLES = (("response", "factor", "The result cannot also be a factor."),)
    RESULT_TABLE = "impact"

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        return im.impact(frame, casting.columns("response")[0], casting.columns("factor"))

    def answer_details(self, result: im.ImpactAnswer) -> str:
        ranking = result.ranking.copy()
        ranking["Direction"] = [self._direction(value) for value in ranking["Direction"]]
        return report_html.section(
            _("The factors, most impact first"), self.report_table(ranking, p_columns=("p-value",)))

    @staticmethod
    def _direction(value: str) -> str:
        if value == "rises":
            return _("rises with it")
        if value == "falls":
            return _("falls as it rises")
        if value.startswith("highest at "):
            return _("highest at {level}").format(level=value.removeprefix("highest at "))
        return ""

    def next_steps(self, result: im.ImpactAnswer) -> list[NextStep]:
        """On to a model of the strongest factors, the differences across the first, or another result."""
        casting = self.roles_widget.casting()
        response, factors = casting.columns("response")[0], casting.columns("factor")
        top = [str(name) for name in result.ranking["Factor"].tolist()[:3]]
        steps = []
        if top:
            steps.append(self.next_step(_("A model of {response} on its strongest factors: {factors}").format(
                response=response, factors=", ".join(top)), "Fit Model", {"response": [response], "_x": top}))
            if is_level(casting.kinds, top[0]):
                steps.append(self.next_step(_("Are there differences in {response} across {factor}?").format(
                    response=response, factor=top[0]), "Are there differences?",
                    {"response": [response], "group": [top[0]]}))
        steps.append(self.next_step(_("Another result, the same factors"), "What has the most impact?",
                                    {"factor": factors}))
        return steps

    def answer_table(self, result: im.ImpactAnswer) -> pd.DataFrame:
        return result.ranking

    def answer_chart(self, result: im.ImpactAnswer, name: str) -> tuple:
        """The factors' impact, as a Pareto chart."""
        response = self.measure
        sql, roles = self.series_select(name, {"Y": "Impact %", "X": "Factor"})
        return ("Pareto Chart", _("What has the most impact on {response}").format(response=response), "",
                _("Impact %"), [(response, sql, roles, {})])
