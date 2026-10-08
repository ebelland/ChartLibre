"""Regression test for the TablePreviewPanel right-click menu.

``_show_context_menu`` used to build its menu from ``_model()``, a helper
that only ever returns a ``LazyTableModel`` (the real-table model). The
guard at the top of the method, ``if model is None: return``, therefore made
the *entire* menu disappear whenever the panel was previewing a saved query
instead - queries are shown through the read-only ``DataFrameTableModel``,
so ``_model()`` returned None for them even though a table was clearly
loaded on screen. Right-clicking a query preview silently did nothing.

The fix keeps the table-writing items (delete column, hide rows, ensure
hide/cluster columns, ...) table-only, since a query has nothing in the repo
for them to write to, but the menu itself - with at least Copy - must
still appear.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QItemSelectionModel, QPoint
from PySide6.QtWidgets import QWidget

from app.data.sqlite_repo import SqliteRepo
from app.widgets.table_preview import TablePreviewPanel


def _repo_with_table_and_query(db_path: Path) -> SqliteRepo:
    repo = SqliteRepo(db_path=db_path)
    repo.import_dataframe(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        table_name="t1",
        normalize_columns=False,
    )
    repo.save_query("q1", "SELECT * FROM t1")
    return repo


def test_context_menu_appears_for_a_real_table(qapp, tmp_db_path: Path) -> None:
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "t1")

    # _build_context_menu is the part of _show_context_menu that runs before
    # QMenu.exec() opens its own (blocking) local event loop, so it is the
    # part a headless test can call directly.
    menu = panel._build_context_menu(QPoint(0, 0))
    assert menu is not None
    assert menu is not None

    texts = {action.text() for action in menu.actions() if not action.isSeparator()}
    assert "Copy" in texts
    # The whole table - edit, export, reload - is the table list's, right
    # above this panel; the preview works on the cells selected in it.
    assert not texts & {"Edit table...", "Refresh data table", "Export rows..."}


def test_context_menu_still_appears_for_a_saved_query(qapp, tmp_db_path: Path) -> None:
    """The bug: this used to return None (no menu shown at all)."""
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "q1")

    menu = panel._build_context_menu(QPoint(0, 0))
    assert menu is not None

    assert menu is not None, "the context menu must not disappear for a saved query"
    texts = {action.text() for action in menu.actions() if not action.isSeparator()}
    assert "Copy" in texts
    # Table-writing actions do not apply to a query: it has no rowid to
    # address a cell by, and nothing in the repo backs a "hide" or
    # "cluster" column for it.
    assert "Edit table..." not in texts
    assert "Add column from SQL expression..." not in texts


def test_the_table_tools_are_in_the_menu_and_copy_writes_the_clipboard(qapp, tmp_db_path: Path) -> None:
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "t1")
    model = panel.view.model()
    panel.view.setCurrentIndex(model.index(0, 0))

    menu = panel._build_context_menu(QPoint(0, 0))
    assert menu is not None
    texts = {action.text() for action in menu.actions() if not action.isSeparator()}
    assert "Copy" in texts
    assert "Export selected rows..." not in texts  # one cell is not a block to export
    assert "Duplicate table" not in texts  # the table list's, not the preview's
    assert "Histogram of 'a'" in texts  # one column selected
    # Moved into Edit table..., with the rest of the editing.
    assert not texts & {
        "Group and aggregate...", "Add column from SQL expression...", "Delete column 'a'..."
    }

    selection = panel.view.selectionModel()
    for row in (0, 1):
        for col in (0, 1):
            selection.select(model.index(row, col), QItemSelectionModel.SelectionFlag.Select)
    panel._copy_selection()
    assert qapp.clipboard().text() == "1\t4\n2\t5"


def test_a_selected_block_can_be_exported(qapp, tmp_db_path: Path) -> None:
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "t1")
    model = panel.view.model()
    selection = panel.view.selectionModel()
    for row in (0, 1):
        selection.select(model.index(row, 0), QItemSelectionModel.SelectionFlag.Select)
    menu = panel._build_context_menu(QPoint(0, 0))
    assert menu is not None
    assert "Export selected rows..." in {action.text() for action in menu.actions()}
