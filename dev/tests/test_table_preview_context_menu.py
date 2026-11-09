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
for them to write to, but the menu itself - with its own entries - must
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
    assert "Copy" not in texts  # the Copy button beside the Data title
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
    assert "Copy" not in texts  # the Copy button beside the Data title
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
    assert "Copy" not in texts  # the Copy button beside the Data title
    assert "Export selected rows..." not in texts  # one cell is not a block to export
    assert "Duplicate table" not in texts  # the table list's, not the preview's
    assert "Histogram of 'a'" not in texts  # the chart button beside the Data title
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


def test_create_query_from_selection_selects_those_columns_and_rows(qapp, tmp_db_path: Path) -> None:
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "t1")
    model = panel.view.model()
    panel.view.setCurrentIndex(model.index(0, 0))
    texts = {action.text() for action in panel._build_context_menu(QPoint(0, 0)).actions()}
    assert "Create query from selection" in texts

    asked: list[str] = []
    panel.query_requested.connect(asked.append)
    selection = panel.view.selectionModel()
    selection.clearSelection()
    for row in (0, 2):  # rows 1 and 3 of column b
        selection.select(model.index(row, 1), QItemSelectionModel.SelectionFlag.Select)
    panel._request_query()
    assert repo.query_df(asked[-1])["b"].tolist() == [4, 6]
    assert list(repo.query_df(asked[-1]).columns) == ["b"]

    selection.clearSelection()
    for row in range(3):
        for col in range(model.columnCount()):
            selection.select(model.index(row, col), QItemSelectionModel.SelectionFlag.Select)
    panel._request_query()
    assert asked[-1].strip() == 'SELECT *\nFROM "t1"'


#: A right-click on no cell: the selection stays what the menu acts on.
OUTSIDE_CELLS = QPoint(-1, -1)


def _select_rows(panel: TablePreviewPanel, rows: list[int]) -> None:
    model = panel.view.model()
    selection = panel.view.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(model.index(row, 0), QItemSelectionModel.SelectionFlag.Select)


def _action(menu, text: str):
    return next(action for action in menu.actions() if action.text() == text)


def test_selected_rows_are_hidden_and_shown_again_from_the_menu(qapp, tmp_db_path: Path) -> None:
    from PySide6.QtCore import Qt

    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "t1")
    redrawn: list[bool] = []
    panel.refresh.connect(lambda: redrawn.append(True))

    _select_rows(panel, [0, 2])
    menu = panel._build_context_menu(OUTSIDE_CELLS)
    assert not _action(menu, "Show selected rows").isEnabled()  # nothing selected is hidden yet
    _action(menu, "Hide selected rows").trigger()
    assert sorted(repo.hidden_rowids("t1")) == [1, 3] and redrawn

    model = panel.view.model()
    hide = [model.column_name(c) for c in range(model.columnCount())].index("Hide")
    assert model.index(0, hide).data() == "1"
    assert model.index(0, 0).data(Qt.ItemDataRole.ForegroundRole) is not None  # greyed
    assert model.index(1, 0).data(Qt.ItemDataRole.ForegroundRole) is None

    _select_rows(panel, [2])
    menu = panel._build_context_menu(OUTSIDE_CELLS)
    assert _action(menu, "Show selected rows").isEnabled()
    _action(menu, "Show selected rows").trigger()
    assert repo.hidden_rowids("t1") == [1]


def test_set_rows_hidden_counts_only_the_rows_it_changed(tmp_db_path: Path) -> None:
    repo = _repo_with_table_and_query(tmp_db_path)
    assert repo.set_rows_hidden("t1", [1, 2], True) == 2
    assert repo.set_rows_hidden("t1", [1, 2, 3], True) == 1
    assert repo.set_rows_hidden("t1", [2], False) == 1
    assert sorted(repo.hidden_rowids("t1")) == [1, 3]


def test_statistics_of_a_column_opens_the_statistics_analysis(qapp, repo) -> None:
    import pandas as pd
    from PySide6.QtCore import QPoint

    from app.widgets.table_preview import TablePreviewPanel

    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0]}), table_name="t", normalize_columns=False)
    panel = TablePreviewPanel(None, repo=repo)
    panel.set_context(repo, "t")
    qapp.processEvents()
    model = panel.view.model()
    column = next(c for c in range(model.columnCount()) if model._columns[c] == "a")
    panel.view.setCurrentIndex(model.index(0, column))
    asked: list = []
    panel.statistics_requested.connect(lambda table, col: asked.append((table, col)))
    menu = panel._build_context_menu(QPoint(0, 0))
    entry = next(a for a in menu.actions() if a.text().startswith("Statistics of"))
    entry.trigger()
    assert asked == [("t", "a")]
    assert not any("unused generated" in a.text() for a in menu.actions())  # the Tables list's
