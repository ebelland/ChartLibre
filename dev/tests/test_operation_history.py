"""Every Apply is recorded in the project's __operations__ (todo R-03)."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from app import APP_VERSION
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.operation_history_dialog import OperationHistoryDialog, history_html
from app.series_operations.dialog_base import SeriesOperationDialogBase
from app.series_operations.transform_dialog import SeriesTransformDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _fresh_dialog_state() -> Iterator[None]:
    clear_state("SeriesTransformDialog")
    yield
    clear_state("SeriesTransformDialog")


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    x = np.linspace(0.0, 10.0, 50)
    repo.import_dataframe(
        pd.DataFrame({"x": x, "a": np.exp(x / 5.0), "b": x**2}), table_name="data", normalize_columns=False
    )
    figure = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure, axis_index=0, chart_type="Scatter Plot",
        title="t", x_label="x", y_label="y", options={},
    ))
    for index, column in enumerate(("a", "b")):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=column,
            sql_query=f'SELECT x, {column} FROM "data"', roles={"x": "x", "y": column}, style={},
        )
    return figure


def _dialog(repo: SqliteRepo, figure_id: int) -> SeriesTransformDialog:
    dialog = SeriesTransformDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


def _wait_for(condition: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QApplication.processEvents()
        time.sleep(0.005)


def test_an_apply_is_recorded_with_its_inputs_and_outputs(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = _dialog(repo, figure_id)
    parameters = dialog.parameter_values()
    assert dialog.apply()
    dialog.close()

    [record] = repo.operations()
    assert record.operation == dialog.operation_label
    assert record.dialog == "transform_dialog:SeriesTransformDialog"
    assert record.app_version == APP_VERSION
    assert record.parameters == {key: value for key, value in parameters.items()}
    # The dialog's own choices, not its buttons.
    assert record.entries == {"model_combo": dialog.model_combo.currentText()}
    assert [source["name"] for source in record.sources] == ["a", "b"]
    assert all('FROM "data"' in source["sql_query"] for source in record.sources)
    tables = [result["table"] for result in record.results]
    assert len(tables) == 2 and all(repo.check_if_table_exists(table) for table in tables)
    assert record.report
    assert "T" in record.applied_at


def test_a_preview_is_not_recorded(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = _dialog(repo, figure_id)
    assert dialog.preview()
    dialog.close()
    assert repo.operations() == []


def test_undoing_an_apply_removes_its_record(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = _dialog(repo, figure_id)
    assert dialog.apply()
    assert dialog.apply()
    dialog.close()
    assert len(repo.operations()) == 2

    repo.undo_last()
    assert [record.id for record in repo.operations()] == [1]


def test_the_history_of_a_table(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = _dialog(repo, figure_id)
    assert dialog.apply()
    dialog.close()
    [record] = repo.operations()

    assert repo.operations(table="data") == [record]
    assert repo.operations(table=record.results[0]["table"]) == [record]
    assert repo.operations(table="unrelated") == []


def test_a_failure_to_record_does_not_lose_the_result(
    qapp, repo: SqliteRepo, figure_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_self: SqliteRepo, **_kwargs: object) -> int:
        raise RuntimeError("disk full")

    monkeypatch.setattr(SqliteRepo, "record_operation", broken)
    dialog = _dialog(repo, figure_id)
    assert dialog.apply()
    dialog.close()
    monkeypatch.undo()
    tables = [name for name in repo.list_table_names() if name.startswith("_Transform")]
    assert len(tables) == 2


@pytest.mark.background
def test_a_background_apply_records_the_series_it_started_with(
    qapp, repo: SqliteRepo, figure_id: int
) -> None:
    dialog = _dialog(repo, figure_id)
    assert SeriesOperationDialogBase.BACKGROUND_ENABLED and dialog.RUN_IN_BACKGROUND
    assert dialog.apply()
    # The window stays editable while the job runs; what is recorded is
    # what was computed.
    dialog.series_selector.reload(select_all_series=False)
    _wait_for(lambda: not dialog.evaluating)
    dialog.close()

    [record] = repo.operations()
    assert [source["name"] for source in record.sources] == ["a", "b"]


def test_the_history_window_lists_what_was_run(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = _dialog(repo, figure_id)
    assert dialog.apply()
    dialog.close()
    [record] = repo.operations()

    page = history_html("data", repo.operations(table="data"))
    assert record.operation in page
    assert record.results[0]["table"] in page
    for name in record.parameters:
        assert name in page

    window = OperationHistoryDialog(repo, "data")
    assert "data" in window.windowTitle()
    window.close()


def test_a_table_with_no_history_says_so(qapp, repo: SqliteRepo, figure_id: int) -> None:
    assert "No operation has been applied" in history_html("data", [])
