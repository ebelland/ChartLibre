"""The guided analysis follows the selected table: what it holds, what it can answer, its questions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QPushButton, QWidget

from app.analysis import catalog
from app.plugin_api.table_operations import table_profile
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
        {"response": ["peso"], "time": ["giorno"]},
        {"response": ["peso"], "group": ["linea"]}]
    assert found[0].template.format(**found[0].values) == "What has the most impact on “peso”?"
    assert found[1].template.format(**found[1].values) == "Has “peso” changed over time?"
    assert found[2].template.format(**found[2].values) == "Does “peso” change between the groups of “linea”?"


def test_the_panel_follows_the_table(qapp) -> None:
    panel = GuidedAnalysisPanel()
    panel.set_loader(lambda table: (FRAME, DECLARED))
    asked = []
    panel.question_requested.connect(lambda entry, roles, kinds: asked.append((entry.key, roles)))
    panel.set_table("lines")
    assert panel.profile() is not None and "lines" in panel._table_label.text()
    panel._questions.findChildren(QPushButton)[2].click()
    assert asked == [("analysis:Are there differences?", {"response": ["peso"], "group": ["linea"]})]

    panel.set_table(None)
    assert panel.profile() is None and not panel._questions.isVisibleTo(panel)


def test_a_question_opens_its_answer_with_the_columns_cast(qapp, tmp_path, monkeypatch) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.main_window.main_window import MainWindow
    from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase

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
        panel._questions.findChildren(QPushButton)[2].click()
        assert cast and cast[0]["response"] == ["peso"] and cast[0]["group"] == ["linea"]
    finally:
        window.close()
        repo.close()


def _wizard():
    from app.dialogs.guided_wizard_dialog import GuidedWizardDialog

    return GuidedWizardDialog("lines", FRAME, table_profile.profile("lines", FRAME, DECLARED))


def test_the_wizard_offers_what_the_picked_columns_allow(qapp) -> None:
    """G-02: columns first; then charts sketched from the rows and questions in the columns' words."""
    from app.dialogs.guided_wizard_dialog import MAX_COLUMNS

    wizard = _wizard()
    try:
        assert not wizard._next.isEnabled()
        wizard.choose(["peso", "linea", "turno", "giorno"])
        assert wizard.chosen() == ["peso", "linea", "turno"][:MAX_COLUMNS]
        wizard.choose(["peso", "linea"])
        assert wizard.preview.model().columnCount() == 2 and wizard._next.isEnabled()
        wizard._go_next()
        assert not wizard.result.isVisibleTo(wizard)  # one measure: nothing to ask
        charts = [card.property("chart_type") for card in wizard.findChildren(QWidget, "wizardChart")
                  if card.isVisibleTo(wizard)]
        assert charts == ["Box Plot", "Violin Plot", "Bar Chart"]
        questions = [card.property("entry").key for card in wizard.findChildren(QWidget, "wizardQuestion")
                     if card.isVisibleTo(wizard)]
        assert questions[0] == "analysis:Are there differences?"
        asked = []
        wizard.question_requested.connect(lambda entry, roles, kinds: asked.append((entry.key, roles)))
        first = next(card for card in wizard.findChildren(QWidget, "wizardQuestion") if card.isVisibleTo(wizard))
        first.findChild(QPushButton).click()
        assert asked == [("analysis:Are there differences?", {"response": ["peso"], "group": ["linea"]})]
    finally:
        wizard.close()
    again = _wizard()  # the columns picked last, remembered for the table
    assert again.chosen() == ["peso", "linea"]
    again.close()


def test_not_sure_leads_to_the_goals_and_their_suggestions(qapp) -> None:
    wizard = _wizard()
    try:
        opened = []
        wizard.entry_requested.connect(opened.append)
        wizard._show_page(2)
        keys = [str(wizard.goals.item(i).data(0x0100)) for i in range(wizard.goals.count())]
        # Goals the table cannot answer go last.
        assert table_profile.goal_missing(wizard._profile, keys[-1]) is not None
        wizard.goals.setCurrentRow(keys.index("compare"))
        cards = [c for c in wizard.findChildren(QWidget, "guidedSuggestion") if c.isVisibleTo(wizard)]
        assert cards[0].property("entry").key == "analysis:Are there differences?"
        cards[0].findChild(QPushButton).click()
        assert [entry.key for entry in opened] == ["analysis:Are there differences?"]
    finally:
        wizard.close()


def test_create_chart_makes_the_figure_and_an_analysis_opens_cast(qapp, tmp_path, monkeypatch) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.main_window.main_window import MainWindow
    from app.plugin_api.table_operations import guided_plan
    from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase

    repo = SqliteRepo(db_path=tmp_path / "x.dhub")
    repo.import_dataframe(FRAME, table_name="lines", normalize_columns=False)
    window = MainWindow(repo, tmp_path / "x.dhub")
    opened = []
    monkeypatch.setattr(TableOperationDialogBase, "exec",
                        lambda self: opened.append(self.roles_widget.casting().roles) or 0)
    try:
        profile = table_profile.profile("lines", FRAME, DECLARED)
        idea = guided_plan.plan(profile, ["peso", "linea"]).charts[2]  # the mean of peso per linea
        figure_id = window._create_guided_chart("lines", idea)
        axis = repo.load_figure_descriptor(figure_id).axes[0]
        assert axis.name == "Bar Chart" and len(repo.query_df(axis.series[0].sql_query)) == 3
        question = guided_plan.plan(profile, ["peso", "linea", "turno"]).questions[0]
        window._open_catalog_entry_with_columns(
            next(e for e in catalog.build_catalog() if e.key == question.key), question.roles, question.kinds)
        assert opened[0]["response"] == ["peso"] and opened[0]["factor"] == ["linea", "turno"]
    finally:
        window.close()
        repo.close()


def test_the_panel_offers_the_common_charts(qapp) -> None:
    from app.widgets.guided_panel import COMMON_CHARTS, GuidedAnalysisPanel

    panel = GuidedAnalysisPanel()
    opened: list = []
    panel.entry_requested.connect(opened.append)
    buttons = {button.text(): button for button in panel.findChildren(type(panel.help_button))}
    for chart, label in COMMON_CHARTS:
        buttons[label].click()
    assert [entry.key for entry in opened] == [f"chart:{chart}" for chart, _label in COMMON_CHARTS]
