"""The table list's context menu: Query Builder edits the selected query, if any."""
from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import QItemSelectionModel

import app.widgets.table_list as table_list_module
from app.data.sqlite_repo import SqliteRepo
from app.widgets.table_list import TableListPanel


class _Recorder:
    """Stands in for QueryBuilderDialog: notes how it was opened, never shows."""

    opened: list[str | None] = []

    def __init__(self, repo, query_name: str | None = None, parent=None) -> None:
        _Recorder.opened.append(query_name)

    def exec(self) -> int:
        return 0


@pytest.fixture
def panel(qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch):
    repo.import_dataframe(pd.DataFrame({"a": [1, 2]}), table_name="numbers", normalize_columns=False)
    repo.save_query("evens", 'SELECT a FROM "numbers" WHERE a % 2 = 0')
    monkeypatch.setattr(table_list_module, "QueryBuilderDialog", _Recorder)
    _Recorder.opened = []
    widget = TableListPanel(repo, None)  # type: ignore[arg-type]
    widget.reload()
    yield widget
    widget.close()


def _select(panel: TableListPanel, name: str) -> None:
    for row in range(panel._model.rowCount()):
        item = panel._model.item(row, panel.COL_TABLE)
        if item is not None and item.data(panel.ROLE_TABLE_NAME) == name:
            index = panel._model.index(row, panel.COL_TABLE)
            panel._view.selectionModel().select(
                index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
            )
            panel._view.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
            return
    raise AssertionError(f"{name} is not in the list")


def test_with_a_saved_query_selected_the_builder_opens_on_that_query(panel: TableListPanel) -> None:
    _select(panel, "evens")
    assert panel._single_selected_query() == "evens"
    panel._open_query_builder()
    assert _Recorder.opened == ["evens"]


def test_with_a_table_selected_the_builder_opens_empty(panel: TableListPanel) -> None:
    _select(panel, "numbers")
    assert panel._single_selected_query() is None
    panel._open_query_builder()
    assert _Recorder.opened == [None]


def test_with_nothing_selected_the_builder_still_opens(panel: TableListPanel) -> None:
    panel._view.selectionModel().clearSelection()
    panel._open_query_builder()
    assert _Recorder.opened == [None]
