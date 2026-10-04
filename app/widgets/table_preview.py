"""Paged preview of a database table.

Rows are fetched in chunks through a lazy model rather than loaded up front, so
opening a table with millions of rows costs the same as opening a small one.
The panel also hosts the table context-menu operations (hide, filter, column
edit) that are implemented on the repository.
"""
from __future__ import annotations

from typing import Any, Optional

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QObject, QPersistentModelIndex, QPoint, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget, QFrame, QTableView, QHeaderView, QInputDialog, QVBoxLayout, QLineEdit, QMenu
from app.data.data_source import quote_identifier as _quote_table
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.config import get_constant
from app.styles.style import (
    CONTEXT_MENU_ICONS,
    MenuItem,
    apply_fusion_for_item_view_styling,
    create_menu,
    create_menu_item,
)
from app.utils.messages import show_message
from app.utils.i18n import _
from app.utils.coercion import to_numbers


# Rows loaded when previewing a saved query.  A preview is for judging shape
# and content, not for scrolling a whole result set.
QUERY_PREVIEW_ROW_LIMIT: int = get_constant("query_preview_row_limit", 1000)


def _type_code(decl_type: str | None) -> str:
    match (decl_type or "").strip().upper():
        case 'BLOB':
            return "BLB"
        case '':
            return "?"
        case 'INT'|'INTEGER'|'TINYINT'|'SMALLINT'|'MEDIUMINT'|'BIGINT'|'UNSIGNED BIG INT'|'INT2'|'INT8':
            return 'I'
        case 'REAL'|'DOUBLE'|'DOUBLE PRECISION'|'FLOAT'|'NUMERIC'|'DECIMAL(10,5)':
            return 'F'
        case 'BOOLEAN':
            return 'B'
        case 'DATE'|'DATETIME':
            return 'DT'
        case _ :
            return 'S'


        


class DataFrameTableModel(QAbstractTableModel):
    """Read-only view over an already-materialised DataFrame.

    Used for saved queries and for the query builder's Run preview.  A query
    result has no rowid, so the chunked model's rowid paging does not apply to
    it; a bounded page loaded up front is simpler and, for a preview, cheaper
    than inventing OFFSET paging that nobody scrolls through.
    """

    def __init__(self, frame: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._columns = [str(column) for column in frame.columns]

    @property
    def frame(self) -> Any:
        """Return the underlying DataFrame."""
        return self._frame

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(len(self._frame))

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._columns)

    def column_name(self, index: int) -> str | None:
        """Return the column name at a position, mirroring LazyTableModel."""
        return self._columns[index] if 0 <= index < len(self._columns) else None

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        try:
            value = self._frame.iat[index.row(), index.column()]
        except Exception:
            return None
        return "" if value is None else str(value)

    def headerData(self, section: int, orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section] if section < len(self._columns) else None
        return section + 1


class TablePreviewPanel(QWidget):
    """Reusable table preview panel with a context-sensitive right-click menu."""
    refresh = Signal()
    #: "New chart from selected columns": the source and its picked columns.
    chart_requested = Signal(str, list)
    #: "Histogram and statistics": the source and its one selected column.
    histogram_requested = Signal(str, str)

    def __init__(self, parent: QWidget, repo:SqliteRepo) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table: str|None = None

        self.view = QTableView(self)
        apply_fusion_for_item_view_styling(self.view)
        self.view.setShowGrid(False)
        # Finder's list view: alternating rows carry the eye across a wide
        # row, and the frame is dropped so the list meets the sidebar
        # hairline instead of drawing a second border beside it.
        self.view.setAlternatingRowColors(True)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setSortingEnabled(False)

        # QTableView is backed by a viewport. Right-click events may arrive on
        # the viewport, not on the view, so wire both and also filter events.
        for widget in (self.view, self.view.viewport()):
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.installEventFilter(self)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        self.view.viewport().customContextMenuRequested.connect(self._show_context_menu)

        header = self.view.horizontalHeader()
        header.setStretchLastSection(True)
        # Interactive rather than ResizeToContents: the latter locks every
        # column at its computed width and silently undoes a drag the moment
        # the user lets go of it. Columns are still sized to fit whenever a
        # new model is set - see resizeColumnsToContents() below - so the
        # panel opens looking the same as before; the difference is that a
        # resize the user makes now sticks.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setHighlightSections(False)

        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(32)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        self.setLayout(layout)

        copy = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self.view)
        copy.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy.activated.connect(self._copy_selection)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched in (self.view, self.view.viewport()):
            event_type = event.type()
            if event_type == QEvent.Type.ContextMenu:
                pos = event.pos()  # type: ignore[attr-defined]
                if watched is self.view:
                    pos = self.view.viewport().mapFrom(self.view, pos)
                self._show_context_menu(pos)
                event.accept()
                return True
            if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:  # type: ignore[attr-defined]
                pos = event.pos()  # type: ignore[attr-defined]
                if watched is self.view:
                    pos = self.view.viewport().mapFrom(self.view, pos)
                self._show_context_menu(pos)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def clear(self) -> None:
        self._repo = None
        self._table = None
        self.view.setModel(None)

    def set_context(self, repo: Optional[SqliteRepo], table: Optional[str]) -> None:
        if repo is None or not table:
            self.clear()
            return
        self._repo = repo
        self._table = table
        self._reload_model()

    def set_model(self, model) -> None:
        self._repo = None
        self._table = None
        self.view.setModel(model)
        self.view.resizeColumnsToContents()

    def _reload_model(self) -> None:
        """Show the current source, whether it is a table or a saved query.

        The source is resolved through the repository rather than assumed to be
        a table, which is what lets a saved query be previewed with no second
        code path here.  Queries get the DataFrame model because their result
        has no rowid for the chunked model to page on.
        """
        if self._repo is None or not self._table:
            self.view.setModel(None)
            return

        try:
            source = self._repo.get_data_source(self._table)
            if source is not None and source.is_query:
                frame = self._repo.data_source_page(
                    source, limit=QUERY_PREVIEW_ROW_LIMIT, offset=0
                )
                self.view.setModel(DataFrameTableModel(frame, parent=self.view))
                self.view.resizeColumnsToContents()
                return

            self.view.setModel(LazyTableModel(self._repo, self._table, parent=self.view))
            self.view.resizeColumnsToContents()
        except Exception as exc:
            applogger.exception("Preview model init failed for source=%s: %s", self._table, exc)
            self.view.setModel(None)

    def _model(self) -> "LazyTableModel | None":
        model = self.view.model()
        return model if isinstance(model, LazyTableModel) else None

    def _current_column_name(self) -> str | None:
        model = self._model()
        index = self.view.currentIndex()
        if model is None or not index.isValid():
            return None
        return model.column_name(index.column())

    def _show_context_menu(self, pos: QPoint) -> None:
        """Show the preview's right-click menu."""
        menu = self._build_context_menu(pos)
        if menu is None:
            return
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _build_context_menu(self, pos: QPoint) -> "QMenu | None":
        """Build the right-click menu for *pos*, or None to show nothing.

        Split out from _show_context_menu so a test can inspect the built
        menu's actions without going through QMenu.exec(), which opens a
        real (blocking) local event loop.

        The view is backed by one of two model kinds: LazyTableModel for a
        real table, or the read-only DataFrameTableModel for a saved query
        (see _reload_model). _model() only ever returns a LazyTableModel, so
        gating the whole menu on it being non-None made the menu disappear
        entirely while a query was on screen, instead of just hiding the
        items that write back to a table a query does not have. The guard
        below only requires some model to be loaded; lazy_model still gates
        the table-only items further down.
        """
        if self.view.model() is None:
            return None

        lazy_model = self._model()
        if lazy_model is not None:
            if self._repo is None:
                self._repo = lazy_model.repo
            if self._table is None:
                self._table = lazy_model.table
        if self._repo is None or not self._table:
            return None

        clicked = self.view.indexAt(pos)
        if clicked.isValid():
            self.view.setCurrentIndex(clicked)

        column = self._current_column_name() if lazy_model is not None else None
        items: list[MenuItem | None] = [
            # No shortcut on the item: Cmd/Ctrl+C is the view's own QShortcut
            # (see __init__), and a second binding would make both ambiguous.
            MenuItem(_("Copy"), callback=self._copy_selection, icon="copy"),
            None,
        ]
        if column:
            items.append(
                MenuItem(
                    _("Statistics of '{column}'").format(column=column),
                    callback=lambda _=False, col=column: self._show_column_stats(col),
                    icon="column_stats",
                    tooltip=_("Count, empty cells, distinct values, and min, max, mean and median"),
                )
            )
        _rows, positions = self._selected_block()
        selected = self._column_names(positions)
        if len(selected) == 1:
            items.append(
                MenuItem(
                    _("Histogram and statistics of '{column}'").format(column=selected[0]),
                    callback=lambda _=False, col=selected[0]: self.histogram_requested.emit(str(self._table), col),
                    icon="column_stats",
                    tooltip=_("A new figure with this column's histogram, and its statistics in the figure's notes"),
                )
            )
        items.append(
            MenuItem(
                _("New chart from selected columns"),
                callback=self._request_chart,
                icon="chart_from_columns",
                tooltip=_("Open New plot with the selected columns as x, y and z"),
            )
        )
        items.append(None)

        menu = create_menu(self, items, icons=CONTEXT_MENU_ICONS)

        if column:
            self._add_flag_menu(menu, column, "Hide")
            self._add_flag_menu(menu, column, "Selected")
            menu.addSeparator()

        # Table-writing actions need a real, LazyTableModel-backed table: a
        # saved query has no rowid and nothing in the repo to hide/cluster/add
        # a column to. A query preview keeps only the model-agnostic reload.
        # "Edit table..." replaces the Hide/ClusterId items, Delete column,
        # Add column from SQL expression and Group and aggregate that used
        # to sit here. They are edits, and they now live with the rest of
        # the editing - see TableEditorDialog.
        trailing_items: tuple[MenuItem | None, ...] = (
            MenuItem(
                _("Edit table..."),
                callback=self._edit_table,
                icon="document-edit",
                tooltip=_(
                    "Edit cells, add or delete rows and columns, and manage "
                    "the Hide and ClusterId columns"
                ),
            ),
            None,
            MenuItem(_("Export rows..."), callback=self._export_rows, icon="export_rows"),
            None,
            MenuItem(_("Refresh data table"), callback=self._reload_model, icon="reload"),
        ) if lazy_model is not None else (
            MenuItem(_("Export rows..."), callback=self._export_rows, icon="export_rows"),
            None,
            MenuItem(_("Refresh data table"), callback=self._reload_model, icon="reload"),
        )

        for item in trailing_items:
            if item is None:
                menu.addSeparator()
                continue
            create_menu_item(
                parent=self,
                icons=CONTEXT_MENU_ICONS,
                menu=menu,
                icon=item.icon,
                checkable=item.checkable,
                text=item.text,
                tooltip=item.tooltip,
                key=item.shortcut,
                action=item.callback,
            )

        return menu

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _selected_block(self) -> tuple[list[int], list[int]]:
        """Rows and columns that hold a selected cell, in view order."""
        indexes = self.view.selectedIndexes()
        if not indexes and self.view.currentIndex().isValid():
            indexes = [self.view.currentIndex()]
        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        return rows, columns

    def _column_names(self, positions: list[int]) -> list[str]:
        model = self.view.model()
        names: list[str] = []
        for position in positions:
            getter = getattr(model, "column_name", None)
            name = getter(position) if getter is not None else None
            if name:
                names.append(str(name))
        return names

    def _copy_selection(self) -> None:
        """Copy the selected cells as tab-separated text, rows on lines."""
        model = self.view.model()
        if model is None:
            return
        selected = {(i.row(), i.column()) for i in self.view.selectedIndexes()}
        if not selected and self.view.currentIndex().isValid():
            current = self.view.currentIndex()
            selected = {(current.row(), current.column())}
        if not selected:
            return
        rows = sorted({row for row, _col in selected})
        columns = sorted({col for _row, col in selected})
        lines = []
        for row in rows:
            cells = []
            for col in columns:
                value = model.data(model.index(row, col)) if (row, col) in selected else ""
                cells.append("" if value is None else str(value))
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Table tools
    # ------------------------------------------------------------------
    def _show_column_stats(self, column: str) -> None:
        if self._repo is None or not self._table:
            return
        try:
            stats = self._repo.column_stats(self._table, column)
        except Exception as exc:
            applogger.exception("Column statistics failed: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return
        lines = [
            _("Rows: {n}").format(n=stats["rows"]),
            _("Empty: {n}").format(n=stats["empty"]),
            _("Distinct values: {n}").format(n=stats["distinct"]),
        ]
        if "mean" in stats:
            lines += [
                "",
                _("Minimum: {v:.6g}").format(v=stats["min"]),
                _("Maximum: {v:.6g}").format(v=stats["max"]),
                _("Mean: {v:.6g}").format(v=stats["mean"]),
                _("Median: {v:.6g}").format(v=stats["median"]),
            ]
        QMessageBox.information(
            self, _("Statistics of '{column}'").format(column=column), "\n".join(lines)
        )

    def _request_chart(self) -> None:
        if not self._table:
            return
        _rows, positions = self._selected_block()
        self.chart_requested.emit(str(self._table), self._column_names(positions))

    def _duplicate_table(self) -> None:
        if self._repo is None or not self._table:
            return
        try:
            name = self._repo.duplicate_table(self._table)
        except Exception as exc:
            applogger.exception("Duplicate table failed: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return
        applogger.info("Table '%s' duplicated as '%s'.", self._table, name)
        self.refresh.emit()

    def _export_rows(self) -> None:
        """Save the selected block, or every visible row, as CSV or Excel."""
        model = self.view.model()
        if model is None or not self._table:
            return
        rows, positions = self._selected_block()
        several = len(rows) * len(positions) > 1
        try:
            if several:
                names = self._column_names(positions)
                frame = pd.DataFrame(
                    [[model.data(model.index(r, c)) for c in positions] for r in rows],
                    columns=names,
                )
                for name in names:
                    converted = to_numbers(frame[name])
                    if converted.notna().sum() == frame[name].replace("", None).notna().sum():
                        frame[name] = converted
            elif isinstance(model, DataFrameTableModel):
                frame = model.frame
            else:
                assert self._repo is not None
                frame = self._repo.visible_rows(self._table)
        except Exception as exc:
            applogger.exception("Export rows failed: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return

        path, chosen = QFileDialog.getSaveFileName(
            self,
            _("Export rows"),
            f"{self._table}.csv",
            _("CSV (*.csv);;Excel (*.xlsx)"),
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() not in (".csv", ".xlsx"):
            target = target.with_suffix(".xlsx" if "xlsx" in chosen else ".csv")
        try:
            if target.suffix.lower() == ".xlsx":
                frame.to_excel(target, index=False, engine="openpyxl")
            else:
                frame.to_csv(target, index=False)
        except Exception as exc:
            applogger.exception("Export rows failed: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return
        applogger.info("Exported %d rows to %s.", len(frame), target)

    def _edit_table(self) -> None:
        """Open the hand editor on this table, and show its result.

        Imported late: the editor builds on this module's own
        LazyTableModel, so importing it at module scope would be a cycle.
        """
        if self._repo is None or not self._table:
            return
        from app.dialogs.table_editor_dialog import TableEditorDialog

        dialog = TableEditorDialog(self._repo, self._table, self)
        dialog.exec()
        self._reload_model()
        # Not just this panel: the editor can add or drop a column, delete
        # rows, or flip Hide, and every one of those changes what the
        # charts built on this table draw. _ensure_cluster used to emit
        # this for the one case it covered; the editor covers more.
        self.refresh.emit()

    #: Per flag column: the submenu's title, then its four labels - the
    #: verb, the comparison tooltip, the empty-cells item and its tooltip,
    #: the same-as-cell item and its tooltip.
    _FLAG_MENU_TEXT: dict[str, tuple[str, str, str, str, str, str]] = {
        "Hide": (
            "Hide rows by selected column",
            "Hide rows where {column} {operator} value",
            "Hide NULL / empty values",
            "Hide rows where the selected column is empty",
            "Hide rows equal to selected cell",
            "Hide rows matching the selected cell value",
        ),
        "Selected": (
            "Select rows by selected column",
            "Mark Selected the rows where {column} {operator} value",
            "Select NULL / empty values",
            "Mark Selected the rows where the selected column is empty",
            "Select rows equal to selected cell",
            "Mark Selected the rows matching the selected cell value",
        ),
    }

    def _add_flag_menu(self, menu: QMenu, column: str, flag: str) -> None:
        """The submenu that sets *flag* (Hide or Selected) on rows by *column*."""
        title, compare_tip, empty_text, empty_tip, same_text, same_tip = self._FLAG_MENU_TEXT[flag]
        submenu = menu.addMenu(_(title))
        # The operator is the payload and stays as it is; only the label is
        # translated. Wrapping the operator too would compare against a
        # translated string instead of SQL.
        for label, operator in (
            (_("Equal to..."), "="),
            (_("Different from..."), "!="),
            (_("Lower than..."), "<"),
            (_("Lower or equal..."), "<="),
            (_("Higher than..."), ">"),
            (_("Higher or equal..."), ">="),
        ):
            create_menu_item(
                parent=self, icons=CONTEXT_MENU_ICONS, menu=submenu, icon=None, checkable=False,
                text=label, tooltip=_(compare_tip).format(column=column, operator=operator), key=None,
                action=lambda _=False, col=column, op=operator: self._flag_by_comparison(flag, col, op),
            )
        create_menu_item(
            parent=self, icons=CONTEXT_MENU_ICONS, menu=submenu, icon=None, checkable=False,
            text=_(empty_text), tooltip=_(empty_tip), key=None,
            action=lambda _=False, col=column: self._flag_special(flag, col, "null_or_empty"),
        )
        create_menu_item(
            parent=self, icons=CONTEXT_MENU_ICONS, menu=submenu, icon=None, checkable=False,
            text=_(same_text), tooltip=_(same_tip), key=None,
            action=lambda _=False: self._flag_selected_cell_value(flag),
        )

    def _flag_done(self, flag: str, count: int) -> None:
        self._reload_model()
        show_message(self, "preview.rows_hidden" if flag == "Hide" else "preview.rows_selected", count=count)

    def _flag_by_comparison(self, flag: str, column: str, operator: str) -> None:
        if self._repo is None or not self._table:
            return
        title = _("Hide rows") if flag == "Hide" else _("Select rows")
        value, ok = QInputDialog.getText(
            self, title, f"{title}: {column} {operator}", QLineEdit.EchoMode.Normal, "",
        )
        if not ok:
            return
        try:
            self._flag_done(flag, self._repo.flag_rows_by_value(self._table, flag, column, operator, value))
        except Exception as exc:
            applogger.exception("%s rows failed: %s", flag, exc)
            show_message(self, "preview.hide_rows_failed", error=exc)

    def _flag_special(self, flag: str, column: str, mode: str) -> None:
        if self._repo is None or not self._table:
            return
        try:
            self._flag_done(flag, self._repo.flag_rows_special(self._table, flag, column, mode))
        except Exception as exc:
            applogger.exception("%s rows failed: %s", flag, exc)
            show_message(self, "preview.hide_rows_failed", error=exc)

    def _flag_selected_cell_value(self, flag: str) -> None:
        index = self.view.currentIndex()
        column = self._current_column_name()
        if not index.isValid() or column is None or self._repo is None or not self._table:
            return
        value = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            self._flag_done(flag, self._repo.flag_rows_by_value(self._table, flag, column, "=", value))
        except Exception as exc:
            applogger.exception("%s selected value failed: %s", flag, exc)
            show_message(self, "preview.hide_rows_failed", error=exc)

    # The Hide entry points, kept for callers and tests that name them.
    def _hide_by_comparison(self, column: str, operator: str) -> None:
        self._flag_by_comparison("Hide", column, operator)

    def _hide_special(self, column: str, mode: str) -> None:
        self._flag_special("Hide", column, mode)

    def _hide_selected_cell_value(self) -> None:
        self._flag_selected_cell_value("Hide")


class LazyTableModel(QAbstractTableModel):
    def __init__(self, repo: SqliteRepo, table: str, parent=None) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table = table
        self._table_q = _quote_table(table)
        self._columns: list[str] = []
        self._codes: list[str] = []
        self._row_count: int = 0
        self._chunk_size = 5000
        self._cache: dict[int, list[list[Any]]] = {}
        self._chunk_rowid: dict[int, int] = {}
        self._load_schema_and_count()

    @property
    def repo(self) -> SqliteRepo:
        return self._repo

    @property
    def table(self) -> str:
        return self._table

    def _load_schema_and_count(self) -> None:
        if not self._repo.is_open:
            return
        rows = self._repo.table_info(self._table_q)
        self._columns = [str(row[1]) for row in rows]
        self._codes = [_type_code(str(row[2])) for row in rows]
        self._row_count = int(self._repo.row_count(self._table))

    def column_name(self, index: int) -> str | None:
        return self._columns[index] if 0 <= index < len(self._columns) else None

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._row_count

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section, orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return f"{self._columns[section]} ({self._codes[section]})"
            return None
        return str(section + 1)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        row_index = index.row()
        column_index = index.column()
        if row_index >= self._row_count or column_index >= len(self._columns):
            return None
        chunk = row_index // self._chunk_size
        offset = row_index % self._chunk_size
        rows = self._cache.get(chunk)
        if rows is None:
            rows = self._fetch_chunk(chunk)
            self._cache[chunk] = rows
        if offset >= len(rows):
            return None
        value = rows[offset][column_index]
        return "" if value is None else str(value)

    def _fetch_chunk(self, chunk_index: int) -> list[list[Any]]:
        last_rowid = self._chunk_rowid.get(chunk_index - 1, 0) if chunk_index > 0 else 0
        if not self._repo.is_open:
            return []
        rows = self._repo.read_rows_after_rowid(self._table, last_rowid, self._chunk_size)
        if not rows:
            return []
        self._chunk_rowid[chunk_index] = int(rows[-1][0])
        return [list(row[1:]) for row in rows]

    def clear_cache(self) -> None:
        self._cache.clear()
        self._chunk_rowid.clear()