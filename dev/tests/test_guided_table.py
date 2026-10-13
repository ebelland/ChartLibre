"""The guided analysis follows the selected table: what it holds, what it can answer, its questions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QPushButton

from app.analysis import catalog
from app.table_operations import table_profile
from app.widgets.guided_panel import GuidedAnalysisPanel

FRAME = pd.DataFrame({
    "peso": np.linspace(9.5, 12.0, 30),
    "linea": list("ABC") * 10,
    "turno": [1, 2] * 15,
    "giorno": pd.date_range("2026-01-01", periods=30).astype(str),
})
DECLARED = {"peso": "REAL", "linea": "TEXT", "turno": "INTEGER", "giorno": "TEXT"}


def test_the_profile_sorts_measures_groups_and_dates() -> None:
    profile = table_profile.profile("t", FRAME, DECLARED)
    assert (profile.measures, profile.groups, profile.times) == (["peso"], ["linea", "turno"], ["giorno"])
    assert table_profile.goal_missing(profile, "compare") is None
    assert table_profile.goal_missing(profile, "groups") == table_profile.NEEDS_TWO_MEASURES
    assert table_profile.suggestion_missing(profile, "chart:Mosaic Plot", catalog.CATEGORICAL) is None
    assert table_profile.suggestion_missing(profile, "chart:Scatter Plot", catalog.NUMERIC) is not None


def test_questions_cast_a_measure_and_a_group() -> None:
    found = table_profile.questions(table_profile.profile("t", FRAME, DECLARED))
    assert [question.roles for question in found] == [
        {"response": ["peso"], "factor": ["linea", "turno"]},
        {"response": ["peso"], "group": ["linea"]}, {"response": ["peso"], "group": ["turno"]}]
    assert found[0].template.format(**found[0].values) == "What has the most impact on “peso”?"
    assert found[1].template.format(**found[1].values) == "Does “peso” change between the groups of “linea”?"


def test_the_panel_follows_the_table(qapp) -> None:
    panel = GuidedAnalysisPanel()
    panel.set_loader(lambda table: (FRAME, DECLARED))
    asked = []
    panel.question_requested.connect(lambda entry, roles, kinds: asked.append((entry.key, roles)))
    panel.set_table("lines")
    assert panel.profile() is not None and "lines" in panel._table_label.text()
    panel._questions.findChildren(QPushButton)[1].click()
    assert asked == [("analysis:Are there differences?", {"response": ["peso"], "group": ["linea"]})]

    # Goals the table cannot answer go last; suggestions it cannot feed after the others, saying why.
    last = panel.goals.item(panel.goals.count() - 1)
    assert table_profile.goal_missing(panel.profile(), str(last.data(0x0100))) is not None
    panel.goals.setCurrentRow([str(panel.goals.item(row).data(0x0100)) for row in range(panel.goals.count())]
                              .index("relate"))
    cards = panel._cards()
    flags = [bool(card.property("missing")) for card in cards]
    assert flags == sorted(flags) and any(flags)

    panel.set_table(None)
    assert panel.profile() is None and not panel._questions.isVisibleTo(panel)


def test_a_question_opens_its_answer_with_the_columns_cast(qapp, tmp_path, monkeypatch) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.main_window.main_window import MainWindow
    from app.table_operations.dialog_base import TableOperationDialogBase

    repo = SqliteRepo(db_path=tmp_path / "g.dhub")
    repo.import_dataframe(FRAME, table_name="lines", normalize_columns=False)
    window = MainWindow(repo, tmp_path / "g.dhub")
    cast = []
    # On the base: the scanner loads the operation's own class afresh from its file.
    monkeypatch.setattr(TableOperationDialogBase, "exec", lambda self: cast.append(self.roles_widget.casting().roles) or 0)
    try:
        window._table_panel.reload()
        window._table_panel.select_table("lines")
        qapp.processEvents()
        panel = window._guided_panel
        assert panel.profile() is not None and panel.profile().table == "lines"
        panel._questions.findChildren(QPushButton)[1].click()
        assert cast and cast[0]["response"] == ["peso"] and cast[0]["group"] == ["linea"]
    finally:
        window.close()
        repo.close()
