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
from app.table_operations.column_roles import ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.table_operations.differences_dialog import answer_html
from app.utils import report_html
from app.utils.i18n import _


class ImpactDialog(TableOperationDialogBase):
    Name = "What has the most impact?"
    Description = ("Which of many columns matters most for a result? The factors are ranked for you, "
                   "and the answer is given in words.")
    Category = "Quick answers"
    Icon = """
    <rect x="3" y="4" width="18" height="3.2" rx="1"/><rect x="3" y="10.4" width="11" height="3.2" rx="1"/>
    <rect x="3" y="16.8" width="5" height="3.2" rx="1"/>
    """

    ROLES = (
        ColumnRole("response", "Result", "What you want to explain: numbers (a yield, a time) or categories "
                                         "(pass/fail).", minimum=1, maximum=1),
        ColumnRole("factor", "Factors", "The columns that might drive it.", minimum=1),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        if not found and set(casting.columns("response")) & set(casting.columns("factor")):
            found.append(_("The result cannot also be a factor."))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del params, model
        return im.impact(frame, casting.columns("response")[0], casting.columns("factor"))

    def format_results(self, result: im.ImpactAnswer) -> str:
        return report_html.document(_(self.Name), str(self.table), self._report(result))

    def _report(self, result: im.ImpactAnswer) -> str:
        ranking = result.ranking.copy()
        ranking["Direction"] = [self._direction(value) for value in ranking["Direction"]]
        return answer_html(result) + report_html.section(
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

    def apply_results(self, result: im.ImpactAnswer) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        response = self.roles_widget.casting().columns("response")[0]
        name = repo.free_table_name(f"{table}_impact")
        repo.snapshot_for_undo([name, *repo.DESCRIPTOR_TABLES], label=f"Impact on '{table}'")
        repo.import_dataframe(result.ranking, table_name=name, normalize_columns=False)
        sql, roles = self.series_select(name, {"Y": "Impact %", "X": "Factor"})
        panel = ("Pareto Chart", _("What has the most impact on {response}").format(response=response), "",
                 _("Impact %"), [(response, sql, roles, {})])
        notes = report_html.document(_(self.Name), f"{table} · {response}", self._report(result))
        figure_id = self.create_report_figure(f"{_(self.Name)} · {response}", notes, [panel])
        self.created_figure_ids = [figure_id]
        return [{"table": name}, {"figure": figure_id, "name": response}]
