"""Hand editing of a table: the cells, the rows, the columns.

Each of these drives the dialog the way the buttons do - set a selection,
call the slot - rather than calling the repository directly, because the
part that can be wrong is the step between them: which row is "the
selected row" once a rowid no longer matches a position.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.table_editor_dialog import EditableTableModel, TableEditorDialog
from app.logs.logger import applogger

FRAME = pd.DataFrame(
    {
        "name": ["alpha", "beta", "gamma", "delta"],
        "value": [1.0, 2.0, 3.0, 4.0],
    }
)


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    # The undo sidecar too, not just the database: it lives beside the
    # .dhub and outlives it, so a test that counts undo entries would be
    # counting the ones a previous run of itself left behind - and the
    # store caps the history, so the count stops moving once it is full.
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm", ".dhub.undo.db"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(FRAME, table_name="people", normalize_columns=False)
    yield built
    built.close()


@pytest.fixture
def dialog(qapp, repo: SqliteRepo):
    built = TableEditorDialog(repo, "people", None)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _rows(repo: SqliteRepo) -> list[tuple]:
    return [
        tuple(row)
        for row in repo._con.execute('SELECT name, value FROM "people" ORDER BY rowid')
    ]


def _columns(repo: SqliteRepo) -> list[str]:
    return [str(row[1]) for row in repo.table_info('"people"')]


def _select_cell(dialog: TableEditorDialog, row: int, column: int) -> None:
    index = dialog._model.index(row, column)
    dialog.view.setCurrentIndex(index)
    dialog.view.selectionModel().select(
        index, dialog.view.selectionModel().SelectionFlag.ClearAndSelect
    )


# ----------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------


def test_a_cell_edit_reaches_the_database(dialog: TableEditorDialog, repo: SqliteRepo) -> None:
    index = dialog._model.index(1, 1)
    assert dialog._model.setData(index, "22.5", Qt.ItemDataRole.EditRole)

    assert _rows(repo)[1] == ("beta", 22.5)


def test_an_emptied_cell_becomes_null_not_an_empty_string(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    """In a numeric column, "" would quietly make the column text."""
    dialog._model.setData(dialog._model.index(0, 1), "", Qt.ItemDataRole.EditRole)

    stored = repo._con.execute(
        'SELECT value FROM "people" ORDER BY rowid LIMIT 1'
    ).fetchone()[0]
    assert stored is None


def test_cells_are_editable_and_the_model_knows_its_rowids(
    dialog: TableEditorDialog,
) -> None:
    flags = dialog._model.flags(dialog._model.index(0, 0))
    assert flags & Qt.ItemFlag.ItemIsEditable

    assert dialog._model.rowid_at(0) == 1
    assert dialog._model.rowid_at(3) == 4
    assert dialog._model.rowid_at(99) is None


# ----------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------


def test_add_row_appends_an_empty_one(dialog: TableEditorDialog, repo: SqliteRepo) -> None:
    dialog._add_row()

    rows = _rows(repo)
    assert len(rows) == 5
    assert rows[-1] == (None, None)


def test_insert_row_puts_it_above_the_selected_one(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    _select_cell(dialog, 2, 0)  # gamma
    dialog._insert_row()

    names = [row[0] for row in _rows(repo)]
    assert names == ["alpha", "beta", None, "gamma", "delta"]


def test_delete_rows_removes_every_selected_one(
    dialog: TableEditorDialog, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    index_a = dialog._model.index(0, 0)
    index_c = dialog._model.index(2, 0)
    selection = dialog.view.selectionModel()
    # setCurrentIndex first: on a view it also *sets* the selection, so
    # calling it afterwards would throw the second cell away.
    dialog.view.setCurrentIndex(index_a)
    selection.select(index_a, selection.SelectionFlag.Select)
    selection.select(index_c, selection.SelectionFlag.Select)

    dialog._delete_rows()

    assert [row[0] for row in _rows(repo)] == ["beta", "delta"]


def test_a_rowid_is_not_a_position(dialog: TableEditorDialog, repo: SqliteRepo) -> None:
    """The reason the model tracks rowids at all.

    Delete the first row and the row that was at index 1 is now at index 0
    - but its rowid has not moved. An edit addressed by position after
    that would write to the wrong record.
    """
    repo.delete_table_rows("people", [1])
    dialog.reload()

    assert dialog._model.rowid_at(0) == 2
    dialog._model.setData(dialog._model.index(0, 0), "BETA", Qt.ItemDataRole.EditRole)

    assert [row[0] for row in _rows(repo)] == ["BETA", "gamma", "delta"]


# ----------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------


def test_add_column_appends_it(dialog: TableEditorDialog, repo: SqliteRepo) -> None:
    dialog._new_column_name.setText("score")
    dialog._new_column_type.setCurrentText("REAL")
    dialog._add_column()

    assert _columns(repo) == ["name", "value", "score"]
    assert dialog._new_column_name.text() == ""


def test_insert_column_puts_it_before_the_selected_one(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    _select_cell(dialog, 0, 1)  # the "value" column
    dialog._new_column_name.setText("unit")
    dialog._new_column_type.setCurrentText("TEXT")
    dialog._insert_column()

    assert _columns(repo) == ["name", "unit", "value"]


def test_inserting_a_column_keeps_every_row(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    """The positional insert rebuilds the table, so this is the thing to check."""
    _select_cell(dialog, 0, 0)
    dialog._new_column_name.setText("first")
    dialog._insert_column()

    assert _rows(repo) == [("alpha", 1.0), ("beta", 2.0), ("gamma", 3.0), ("delta", 4.0)]


def test_delete_column_removes_it(
    dialog: TableEditorDialog, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    _select_cell(dialog, 0, 1)
    dialog._delete_column()

    assert _columns(repo) == ["name"]


def test_rename_column_uses_the_name_box(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    _select_cell(dialog, 0, 1)
    dialog._new_column_name.setText("amount")
    dialog._rename_column()

    assert _columns(repo) == ["name", "amount"]


def test_a_duplicate_column_name_is_refused_not_applied(
    dialog: TableEditorDialog, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _p, _t, message, *a, **k: warned.append(message)
    )
    dialog._new_column_name.setText("value")
    dialog._add_column()

    assert _columns(repo) == ["name", "value"]
    assert warned and "already a column" in warned[0]


# ----------------------------------------------------------------------
# The managed columns, moved here from the preview's context menu
# ----------------------------------------------------------------------


def test_the_managed_columns_are_handled_here(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    dialog._managed(repo.ensure_hide_column, "hide")
    dialog._managed(repo.ensure_cluster_column, "cluster")

    columns = _columns(repo)
    assert "Hide" in columns
    assert "ClusterId" in columns


def test_invert_hide_swaps_shown_and_hidden(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    repo.ensure_hide_column("people")
    repo._con.execute('UPDATE "people" SET "Hide" = 1 WHERE rowid = 1')
    repo._commit()

    dialog._managed(repo.invert_hide, "invert")

    flags = [
        int(row[0])
        for row in repo._con.execute('SELECT "Hide" FROM "people" ORDER BY rowid')
    ]
    assert flags == [0, 1, 1, 1]


def test_the_preview_menu_no_longer_carries_them(qapp, repo: SqliteRepo) -> None:
    """They were moved, not copied: the menu should offer the editor instead."""
    from PySide6.QtCore import QPoint

    from app.widgets.table_preview import TablePreviewPanel

    panel = TablePreviewPanel(None, repo)
    try:
        panel.set_context(repo, "people")
        menu = panel._build_context_menu(QPoint(1, 1))
        assert menu is not None
        labels = [action.text() for action in menu.actions()]

        assert any("Edit table" in label for label in labels)
        for gone in ("Ensure Hide column", "Reset Hide to 0", "Invert Hide",
                     "Ensure ClusterId column", "Reset Clusters"):
            assert not any(gone in label for label in labels), gone
    finally:
        panel.deleteLater()
        applogger.set_status_bar(None)


# ----------------------------------------------------------------------
# Undo
# ----------------------------------------------------------------------


def test_a_whole_session_is_one_undo_entry(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    """Not one per keystroke.

    A snapshot copies the whole table, so an entry per cell edit would copy
    it again for every number corrected - unusable on anything large. The
    session shares one entry, which UndoStore fills once and then skips.
    """
    before = len(repo.undo_entries())

    dialog._model.setData(dialog._model.index(0, 1), "9.5", Qt.ItemDataRole.EditRole)
    dialog._model.setData(dialog._model.index(1, 1), "8.5", Qt.ItemDataRole.EditRole)
    dialog._model.setData(dialog._model.index(2, 1), "7.5", Qt.ItemDataRole.EditRole)
    dialog._add_row()

    assert len(repo.undo_entries()) == before + 1


def test_undoing_that_entry_restores_the_table(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    original = _rows(repo)

    dialog._model.setData(dialog._model.index(0, 0), "CHANGED", Qt.ItemDataRole.EditRole)
    dialog._add_row()
    assert _rows(repo) != original

    repo.undo_last()

    assert _rows(repo) == original


def test_an_edit_from_outside_the_dialog_gets_its_own_entry(repo: SqliteRepo) -> None:
    """The batching is the dialog's doing, not the repository's default."""
    before = len(repo.undo_entries())

    repo.append_table_row("people")
    repo.append_table_row("people")

    assert len(repo.undo_entries()) == before + 2


# ----------------------------------------------------------------------
# OK and Cancel
# ----------------------------------------------------------------------


def test_ok_keeps_the_changes(dialog: TableEditorDialog, repo: SqliteRepo) -> None:
    dialog._model.setData(dialog._model.index(0, 0), "KEPT", Qt.ItemDataRole.EditRole)
    dialog._add_row()

    dialog.accept()

    names = [row[0] for row in _rows(repo)]
    assert names[0] == "KEPT"
    assert len(names) == 5


def test_cancel_puts_the_table_back(
    dialog: TableEditorDialog, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    original = _rows(repo)
    original_columns = _columns(repo)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )

    dialog._model.setData(dialog._model.index(0, 0), "LOST", Qt.ItemDataRole.EditRole)
    dialog._add_row()
    dialog._new_column_name.setText("scratch")
    dialog._add_column()
    assert _rows(repo) != original

    dialog.reject()

    assert _rows(repo) == original
    assert _columns(repo) == original_columns


def test_cancel_asks_first_and_a_no_changes_nothing(
    dialog: TableEditorDialog, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    asked: list[str] = []

    def refuse(_parent, title, *a, **k):
        asked.append(title)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", refuse)
    dialog._model.setData(dialog._model.index(0, 0), "STAYS", Qt.ItemDataRole.EditRole)

    dialog.reject()

    assert asked, "Cancel discarded the edits without asking"
    assert [row[0] for row in _rows(repo)][0] == "STAYS"


def test_cancel_with_nothing_changed_does_not_ask(
    dialog: TableEditorDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to lose is nothing worth interrupting anybody about."""
    from PySide6.QtWidgets import QMessageBox

    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _p, title, *a, **k: (asked.append(title), QMessageBox.StandardButton.Yes)[1],
    )

    dialog.reject()

    assert asked == []


def test_ok_leaves_the_session_on_the_undo_stack(
    dialog: TableEditorDialog, repo: SqliteRepo
) -> None:
    """Keeping the changes does not mean giving up the ability to undo them."""
    before = len(repo.undo_entries())
    dialog._model.setData(dialog._model.index(0, 0), "KEPT", Qt.ItemDataRole.EditRole)
    dialog.accept()

    assert len(repo.undo_entries()) == before + 1

    repo.undo_last()
    assert [row[0] for row in _rows(repo)][0] == "alpha"
