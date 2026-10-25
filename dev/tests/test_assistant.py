"""The assistant: its tools on a project, its permission levels, its conversation loop - with a fake provider,
never the API."""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ai import settings
from app.ai.assistant import Assistant
from app.ai.tools import AssistantTools
from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.ai.provider import ACCURATE, FAST, AIProvider, ChatSession, ToolCall, ToolSpec, Turn


@pytest.fixture
def repo(tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    rng = np.random.default_rng(3)
    repo.import_dataframe(pd.DataFrame({"species": np.repeat(["A", "B"], 30),
                                        "mass": np.r_[rng.normal(10, 1, 30), rng.normal(12, 1, 30)]}),
                          table_name="birds", normalize_columns=False)
    yield repo
    repo.close()


def _tools(repo, *, confirm=lambda _question: True, changed=None):
    made: list[list[int]] = []
    tools = AssistantTools(repo, confirm=confirm, changed=changed or made.append, current_figure=lambda: None)
    return tools, made


def test_reading_tools_and_the_permission_levels(qapp, repo, monkeypatch) -> None:
    monkeypatch.setattr(settings, "sample_rows", lambda: 3)
    tools, _made = _tools(repo)
    text, failed = tools.run("list_tables", {}, settings.READ_ONLY)
    assert not failed and "birds: 60 rows" in text
    text, failed = tools.run("describe_table", {"table": "birds"}, settings.READ_ONLY)
    assert not failed and "mass" in text and "continuous" in text and "First 3 rows" in text
    text, failed = tools.run("query", {"sql": "SELECT species, COUNT(*) AS n FROM birds GROUP BY species"},
                             settings.READ_ONLY)
    assert not failed and "A,30" in text
    # A query never writes, whatever the SQL guard says; and the analyses are not for Read only.
    assert tools.run("query", {"sql": "DELETE FROM birds"}, settings.READ_ONLY)[1]
    assert tools.run("run_analysis", {"analysis": "Distribution", "table": "birds",
                                      "roles": {"response": ["mass"]}}, settings.READ_ONLY)[1]
    assert "run_analysis" not in {spec.name for spec in tools.specs(settings.READ_ONLY)}
    assert "change_data" not in {spec.name for spec in tools.specs(settings.ANALYSES)}
    text, failed = tools.run("describe_table", {"tabel": "birds"}, settings.READ_ONLY)
    assert failed and text.startswith("Wrong arguments for describe_table")  # said to the model, not raised


def test_an_analysis_run_by_the_assistant_is_written_as_its_ok_writes_it(qapp, repo) -> None:
    tools, made = _tools(repo)
    text, failed = tools.run("run_analysis", {"analysis": "Are there differences?", "table": "birds",
                                              "roles": {"response": ["mass"], "group": ["species"]}},
                             settings.ANALYSES)
    assert not failed, text
    assert "Tables written: _birds_differences" in text and "Report:" in text
    assert made and made[0] and repo.load_figure_descriptor(made[0][0]) is not None
    assert [entry.label for entry in repo.undo_entries()] == ["Are there differences? on 'birds'"]


def test_a_change_is_confirmed_first_and_is_one_undo_step(qapp, repo) -> None:
    declined, _made = _tools(repo, confirm=lambda _question: False)
    text, _failed = declined.run("change_data", {"sql": "UPDATE birds SET mass = mass * 1000",
                                                 "description": "Grams"}, settings.EVERYTHING)
    assert text == "The user declined the change." and repo.query_df("SELECT MAX(mass) AS m FROM birds")["m"][0] < 100
    tools, _made = _tools(repo)
    assert tools.run("change_data", {"sql": "UPDATE birds SET mass = mass * 1000", "description": "Grams"},
                     settings.EVERYTHING) == ("Done: 60 rows changed.", False)
    assert [entry.label for entry in repo.undo_entries()] == ["Assistant: Grams"]
    assert tools.run("change_data", {"sql": "DROP TABLE birds", "description": "x"}, settings.EVERYTHING)[1]
    assert tools.run("change_data", {"sql": "UPDATE birds SET mass = 1", "description": "x"}, settings.ANALYSES)[1]


class _Session(ChatSession):
    def __init__(self) -> None:
        self.models: list[str] = []
        self.results: list = []

    def send(self, *, model: str, user_text: str | None = None, tool_results: Sequence = (), on_text,
             cancel) -> Turn:
        self.models.append(model)
        if user_text is not None:
            return Turn(stop="tools", tool_calls=[ToolCall("1", "list_tables", {})], model=model, input_tokens=5)
        self.results.extend(tool_results)
        on_text("Two species, ")
        on_text("sixty birds.")
        return Turn(text="Two species, sixty birds.", model=model, output_tokens=7)


class _Provider(AIProvider):
    Name = "Fake"

    def __init__(self) -> None:
        self.session = _Session()

    def models(self) -> Mapping[str, str]:
        return {ACCURATE: "big", FAST: "small"}

    def classify(self, text: str, *, api_key: str) -> str:
        return FAST

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        assert {spec.name for spec in tools} >= {"list_tables", "query"}
        return self.session


def test_the_conversation_runs_the_tools_the_model_asks_for_and_streams_the_answer(qapp, repo) -> None:
    from PySide6.QtWidgets import QApplication

    provider = _Provider()
    tools, _made = _tools(repo)
    assistant = Assistant(provider, tools, api_key="", level=settings.READ_ONLY, mode=settings.AUTO)
    streamed, used, answered = [], [], []
    assistant.text_arrived.connect(streamed.append)
    assistant.tool_used.connect(lambda name, summary: used.append(name))
    assistant.answered.connect(lambda model, tokens_in, tokens_out: answered.append((model, tokens_in, tokens_out)))
    assistant.ask("How many birds?")
    deadline = time.monotonic() + 10
    while not answered and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert used == ["list_tables"] and "birds: 60 rows" in provider.session.results[0].content
    assert "".join(streamed) == "Two species, sixty birds."
    assert answered == [("small", 5, 7)]  # Automatic: the request was simple, the fast model answered
    assert provider.session.models == ["small", "small"]


def test_the_claude_provider_is_found_and_needs_no_package_to_be_listed(qapp) -> None:
    from app.scanners.ai_provider_scanner import ai_providers, provider_class

    claude = provider_class(next(entry for entry in ai_providers if entry["value"] == "Claude"))()
    assert claude.models() == {ACCURATE: "claude-opus-5-5", FAST: "claude-haiku-5-5"}
    assert isinstance(claude.unavailable(), str)


def test_the_panel_says_what_is_missing_before_anything_is_sent(qapp, monkeypatch) -> None:
    from app.widgets.assistant_panel import AssistantPanel

    monkeypatch.setattr(settings, "consented", lambda: False)
    panel = AssistantPanel(None, repo=lambda: None, current_figure=lambda: None)
    assert not panel._send_button.isEnabled() and "Allow it in Settings" in panel._note.text()
    panel.deleteLater()


def _converse(qapp, assistant: Assistant, text: str) -> tuple[list[str], str, list]:
    """Ask *text*; the tools used, the answer streamed, what answered (or failed)."""
    used, streamed, ended = [], [], []
    assistant.tool_used.connect(lambda name, _summary: used.append(name))
    assistant.text_arrived.connect(streamed.append)
    assistant.answered.connect(lambda model, *_tokens: ended.append(model))
    assistant.failed.connect(lambda message: ended.append(f"failed: {message}"))
    assistant.ask(text)
    deadline = time.monotonic() + 60
    while not ended and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return used, "".join(streamed), ended


def test_the_demo_goes_through_every_tool_the_settings_allow_with_no_key(qapp, repo, monkeypatch) -> None:
    from app.ai_providers.demo_provider import DemoProvider

    monkeypatch.setattr("app.ai_providers.demo_provider._WORD_DELAY_S", 0)
    tools, _made = _tools(repo)
    used, text, ended = _converse(qapp, Assistant(DemoProvider(), tools, api_key="", level=settings.EVERYTHING,
                                                  mode=settings.AUTO), "Is the body mass different between the two species?")
    assert ended == ["demo"], ended  # not a short question: the "accurate" model
    assert used == ["list_tables", "describe_table", "run_analysis", "add_column"]
    assert "**birds**" in text and "Are there differences?" in text and "mass_z" in text
    assert "mass_z" in repo.get_columns("birds")
    assert repo.undo_entries()[0].label == "Add column 'mass_z' to 'birds'"  # the latest first


def test_the_demo_read_only_asks_the_database_and_changes_nothing(qapp, repo, monkeypatch) -> None:
    from app.ai_providers.demo_provider import DemoProvider

    monkeypatch.setattr("app.ai_providers.demo_provider._WORD_DELAY_S", 0)
    tools, made = _tools(repo)
    used, text, ended = _converse(qapp, Assistant(DemoProvider(), tools, api_key="", level=settings.READ_ONLY,
                                                  mode=settings.AUTO), "Tables?")
    assert ended == ["demo-fast"] and used == ["list_tables", "describe_table", "query"]
    assert "| A | 30 |" in text and not made and not repo.undo_entries()


def test_the_demo_needs_neither_consent_nor_key(qapp, monkeypatch) -> None:
    from app.widgets.assistant_panel import AssistantPanel

    monkeypatch.setattr(settings, "provider_name", lambda: "Demo")
    monkeypatch.setattr(settings, "consented", lambda: False)
    panel = AssistantPanel(None, repo=lambda: None, current_figure=lambda: None)
    assert panel._send_button.isEnabled() and "Demo" in panel._note.text()
    panel.deleteLater()


def test_the_assistant_can_be_left_out_of_the_navigation_bar(qapp, tmp_path: Path, monkeypatch) -> None:
    """Settings > Assistant > Navigation: unticked, the row is gone from the rail - its panel closed if open."""
    from app.data.sqlite_repo import SqliteRepo
    from app.main_window.main_window import MainWindow

    shown = {"value": False}
    monkeypatch.setattr(settings, "in_navigation", lambda: shown["value"])
    repo = SqliteRepo(db_path=tmp_path / "w.dhub")
    window = MainWindow(repo, tmp_path / "w.dhub")
    try:
        rail = window._left_panel
        assert rail.button("assistant") is None and rail.button("analysis") is not None
        rail.set_item_hidden("assistant", False)
        assert rail.button("assistant") is not None and rail.select("assistant")
        rail.set_item_hidden("assistant", True)  # while open: closed with it
        assert rail.button("assistant") is None and rail.selected_key is None
    finally:
        window.close()
        repo.close()
