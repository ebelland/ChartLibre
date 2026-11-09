"""The table edits the data preview offers, from the table list and in the editor (todo N-01, N-06)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QInputDialog, QMessageBox

import app.widgets.table_list as table_list_module
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.table_editor_dialog import TableEditorDialog
from app.logs.logger import applogger
from app.widgets.table_list import TableListPanel

FRAME = pd.DataFrame({"name": ["alpha", "beta", "gamma", "delta"], "value": [1.0, 2.0, 3.0, None]})


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteRepo]:
    built = SqliteRepo(db_path=tmp_path / "edits.dhub")
    built.import_dataframe(FRAME, table_name="people", normalize_columns=False)
    built.save_query("big", 'SELECT * FROM "people" WHERE value > 1')
    yield built
    built.close()


@pytest.fixture
def told(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    said: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda _p, _t, text, *a, **k: said.append(text)))
    return said


# ----------------------------------------------------------------------
# N-01: the table list
# ----------------------------------------------------------------------
@pytest.fixture
def panel(qapp, repo: SqliteRepo) -> Iterator[TableListPanel]:
    widget = TableListPanel(repo, None)  # type: ignore[arg-type]
    widget.reload()
    yield widget
    widget.close()


def _select(panel: TableListPanel, name: str) -> None:
    for row in range(panel._model.rowCount()):
        item = panel._model.item(row, panel.COL_TABLE)
        if item is not None and item.data(panel.ROLE_TABLE_NAME) == name:
            index = panel._model.index(row, panel.COL_TABLE)
            selection = panel._view.selectionModel()
            selection.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
            selection.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
            return
    raise AssertionError(f"{name} is not in the list")


def test_duplicate_from_the_table_list_copies_the_rows(panel: TableListPanel, repo: SqliteRepo) -> None:
    _select(panel, "people")
    panel._duplicate_table()
    assert "people_copy" in repo.list_table_names()
    assert len(repo.query_df('SELECT * FROM "people_copy"')) == 4


def test_edit_from_the_table_list_opens_the_editor_on_that_table(
    panel: TableListPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    class _Editor:
        def __init__(self, _repo, table, _parent=None) -> None:
            opened.append(table)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(table_list_module, "TableEditorDialog", _Editor)
    _select(panel, "people")
    panel._edit_table()
    assert opened == ["people"]


def test_a_saved_query_is_neither_edited_nor_duplicated(panel: TableListPanel, repo: SqliteRepo) -> None:
    _select(panel, "big")
    assert panel._selected_table() is None
    panel._duplicate_table()
    assert "big_copy" not in repo.list_table_names()



def test_a_stored_function_is_listed_as_a_source_marked_f_and_deleted_as_one(
        panel: TableListPanel, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.data_operations import sympy_store

    sympy_store.save(repo, {"function_name": "Line", "source_type": "expression", "symbols": "a",
                            "code_string": "a*x", "values": {"a": 2.0}, "from": "0", "to": "2", "points": "3"})
    panel.reload()
    kinds = {panel._model.item(row, panel.COL_TABLE).text(): panel._model.item(row, panel.COL_HAS_LINK).data(
        panel.ROLE_KIND) for row in range(panel._model.rowCount())}
    assert kinds == {"people": "table", "big": "query", "Line": "function"}
    _select(panel, "Line")
    assert panel.selected_sources() == [("Line", "function")] and panel._selected_table() is None
    # Its values, read as the preview reads a query.
    source = repo.get_data_source("Line")
    assert source is not None and source.is_function
    assert repo.data_source_page(source, limit=10, offset=0)["y"].tolist() == [0.0, 2.0, 4.0]
    monkeypatch.setattr(table_list_module, "ask", lambda _parent, key, **_values: key == "function.confirm_delete")
    panel._delete_selected()
    assert repo.get_sympy_function("Line") is None and "people" in repo.list_table_names()

# ----------------------------------------------------------------------
# N-06: Hide rows in the editor
# ----------------------------------------------------------------------
@pytest.fixture
def editor(qapp, repo: SqliteRepo) -> Iterator[TableEditorDialog]:
    built = TableEditorDialog(repo, "people", None)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _select_column(editor: TableEditorDialog, column: int, row: int = 0) -> None:
    index = editor._model.index(row, column)
    editor.view.setCurrentIndex(index)
    editor.view.selectionModel().select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)


def _hidden(repo: SqliteRepo) -> list[str]:
    return list(repo.query_df('SELECT name FROM "people" WHERE "Hide" = 1 ORDER BY rowid')["name"])


def test_hide_rows_by_comparison(editor: TableEditorDialog, repo: SqliteRepo, told: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *_a, **_k: ("2", True)))
    _select_column(editor, 1)
    editor._flag_rows("Hide", ">=")
    assert _hidden(repo) == ["beta", "gamma"]
    assert told == ["2 row(s) hidden."]


def test_hide_empty_cells_and_the_selected_value(editor: TableEditorDialog, repo: SqliteRepo, told: list[str]) -> None:
    _select_column(editor, 1)
    editor._flag_rows("Hide", None)
    assert _hidden(repo) == ["delta"]
    _select_column(editor, 0, row=0)
    editor._flag_rows("Hide", "=", from_cell=True)
    assert _hidden(repo) == ["alpha", "delta"]


def test_cancel_puts_the_hidden_rows_back(
    editor: TableEditorDialog, repo: SqliteRepo, told: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_column(editor, 1)
    editor._flag_rows("Hide", None)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.StandardButton.Yes))
    editor.reject()
    assert _hidden(repo) == []


def test_a_stored_function_has_edit_which_opens_functions_on_it(
        qapp, panel: TableListPanel, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Right-click > Edit function...: the Functions operation, on the Function tab with that function."""
    from PySide6.QtWidgets import QWidget

    from app.data_operations import sympy_store
    from app.data_operations.function_dialog import SeriesFunctionDialog

    sympy_store.save(repo, {"function_name": "Line", "source_type": "expression", "symbols": "a",
                            "code_string": "a*x", "values": {"a": 2.0}})
    edited: list[str] = []

    class Window(QWidget):
        def edit_stored_function(self, name: str) -> None:
            edited.append(name)

    window = Window()
    panel.setParent(window)
    panel.reload()
    _select(panel, "Line")
    # Edit is the button beside the Tables title (no longer repeated in the right-click menu).
    panel.edit_selected()
    assert edited == ["Line"]

    figure_id = int(repo.create_figure_descriptor(name="F"))
    dialog = SeriesFunctionDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        assert dialog.edit_stored("Line") and not dialog.edit_stored("Missing")
        assert dialog._model_tabs.currentIndex() == 1 and dialog._stored_combo.currentData() == "Line"
    finally:
        dialog.close()
    panel.setParent(None)


def test_the_main_window_finds_the_functions_operation_it_opens() -> None:
    from app.scanners.data_operation_scanner import (
        data_operations,
        import_class_from_file,
    )

    entry = next(op for op in data_operations if op.get("name") == "SeriesFunctionDialog")
    assert hasattr(import_class_from_file(entry), "edit_stored")


def test_a_reload_keeps_the_selected_row_not_the_first(panel: TableListPanel) -> None:
    """After Column Info's OK (or any refresh) the list stayed selected on its first row, the preview with it."""
    _select(panel, "big")
    shown: list[str] = []
    panel.tableSelected.connect(shown.append)
    panel.reload()
    assert panel.selected_sources() == [("big", "query")] and shown == ["big"]
