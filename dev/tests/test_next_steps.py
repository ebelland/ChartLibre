"""G-02: each quick answer ends with "What next?" - links that open the next question, its columns cast."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.table_operations import next_steps
from app.table_operations.differences_dialog import DifferencesDialog
from app.table_operations.impact_dialog import ImpactDialog


def _plant() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    frame = pd.DataFrame({"line": rng.choice(["A", "B", "C"], n), "temp": rng.uniform(0, 100, n),
                          "noise": rng.normal(size=n)})
    frame["y"] = 0.05 * frame["temp"] + frame["line"].map({"A": 0.0, "B": 2.0, "C": 4.0}) + rng.normal(0, 0.3, n)
    return frame


def _wait(qapp, dialog) -> None:  # noqa: ANN001
    deadline = time.monotonic() + 60
    while dialog._task is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_a_link_carries_the_whole_question() -> None:
    step = next_steps.NextStep("label", "Fit Model", "my table", {"response": ["y"], "_x": ["a b", "c"]},
                               {"y": "nominal"})
    url = next_steps.step_url(step)
    assert next_steps.is_step_url(url) and not next_steps.is_step_url("https://example.com/next")
    back = next_steps.parse_step(url)
    assert (back.operation, back.table, back.roles, back.kinds) == (
        "Fit Model", "my table", {"response": ["y"], "_x": ["a b", "c"]}, {"y": "nominal"})
    assert next_steps.parse_step("https://chartlibre.invalid/next?op=x") is None  # no table


def test_differences_leads_to_impact_and_offers_another_column(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(_plant(), table_name="plant", normalize_columns=False)
    dialog = DifferencesDialog(repo=repo, table="plant")
    try:
        dialog.roles_widget.set_casting({"response": ["y"], "group": ["line"]})
        dialog.preview()
        _wait(qapp, dialog)
        steps = dialog.next_steps(dialog._result)
        assert [step.operation for step in steps] == ["What has the most impact?", "Are there differences?",
                                                      "Are there differences?"]
        assert steps[0].roles == {"response": ["y"], "factor": ["line", "temp", "noise"]}
        assert steps[1].roles == {"group": ["line"]} and steps[2].roles == {"response": ["y"]}
        assert "What next?" in dialog.report_document(dialog._result)
        assert next_steps.step_url(steps[0]).replace("&", "&amp;") in dialog.report_document(dialog._result)

        # Clicked in Results: this answer is kept (OK) and the next one asked for.
        dialog._follow_link(next_steps.step_url(steps[0]))
        _wait(qapp, dialog)
        assert dialog.next_request is not None and dialog.next_request.operation == "What has the most impact?"
        assert repo.table_frame("_plant_differences") is not None
    finally:
        dialog.close()


def test_impact_leads_to_a_model_of_the_strongest_factors(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(_plant(), table_name="plant", normalize_columns=False)
    dialog = ImpactDialog(repo=repo, table="plant")
    try:
        dialog.roles_widget.set_casting({"response": ["y"], "factor": ["line", "temp", "noise"]})
        dialog.preview()
        _wait(qapp, dialog)
        steps = dialog.next_steps(dialog._result)
        model = steps[0]
        assert model.operation == "Fit Model" and model.roles["response"] == ["y"]
        assert set(model.roles["_x"][:2]) == {"line", "temp"}
        assert steps[-1].operation == "What has the most impact?" and "response" not in steps[-1].roles
    finally:
        dialog.close()


def test_the_main_window_opens_the_next_question_on_its_table(qapp, tmp_path: Path, monkeypatch) -> None:
    from app.main_window import main_window as main_window_module

    repo = SqliteRepo(db_path=tmp_path / "n.dhub")
    repo.import_dataframe(_plant(), table_name="plant", normalize_columns=False)
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    opened: list = []
    monkeypatch.setattr(window, "_on_table_operation_requested",
                        lambda operation, *, roles=None, kinds=None: opened.append((operation["value"], roles)))
    try:
        step = next_steps.NextStep("", "Fit Model", "plant", {"response": ["y"], "_x": ["temp"]})
        window._open_next_step_url(next_steps.step_url(step))
        assert opened == [("Fit Model", {"response": ["y"], "_x": ["temp"]})]
        window._open_next_step(next_steps.NextStep("", "Fit Model", "gone", {}))  # a table since deleted
        assert len(opened) == 1
    finally:
        window.close()
        repo.close()


def test_a_report_link_reaches_its_pane_from_the_event_loop(qapp) -> None:
    from app.widgets.html_results import HtmlResultsView, _open_link

    pane = HtmlResultsView()
    received: list[str] = []
    pane.link_requested.connect(received.append)
    url = next_steps.step_url(next_steps.NextStep("", "Fit Model", "t", {}))
    _open_link(pane, url)
    assert received == []  # not inside the viewer's own callback
    qapp.processEvents()
    assert received == [url]
