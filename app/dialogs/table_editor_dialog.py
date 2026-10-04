"""Edit a data table by hand: its cells, its rows and its columns.

The preview panel is read-only on purpose - it is there to show what a
table holds while a chart is built from it - but a table that came in from
a CSV nearly always needs a value corrected, a stray row dropped or a
column added before it is worth plotting, and until this existed the only
ways to do that were to fix the CSV and re-import, or to write SQL.

Every change is written to the database as it is made - a SQLite table
has nowhere to be "saved" to - and yet *Cancel* still puts the table back.
Both are true because of the undo snapshot each change records (see the
repository's own hand-editing section): the whole session shares one
entry, taken before its first change, so *OK* simply leaves it in the
history like any other action, and *Cancel* restores it.

That is also what keeps this usable on a large table. A snapshot copies
the table, so an entry per edited cell would copy it per keystroke.

The *Managed columns* group is the second reason this dialog exists. Hide
and ClusterId are columns the application maintains and several of its
operations write to; their actions used to sit in the preview's
right-click menu, several levels down a menu that is mostly about looking
rather than changing. They are editing actions, so they live where the
editing is.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.data.repo.table_tools import CAST_TYPES
from app.styles.style import (
    CardFrame,
    action_presentation,
    apply_dialog_shell,
    apply_fusion_for_item_view_styling,
    create_action_button,
    mark_destructive_button,
    mark_icon_only,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.widgets.table_preview import LazyTableModel

#: What a new column can be declared as. SQLite's own storage classes, in
#: the order they are actually wanted: a measured quantity first.
COLUMN_TYPES: tuple[str, ...] = ("REAL", "INTEGER", "TEXT")


class EditableTableModel(LazyTableModel):
    """The preview's own lazy model, with the cells writable.

    Inherited rather than rewritten so the editor shows exactly what the
    preview shows - the same chunked reads, the same type codes in the
    header, the same handling of a table too big to hold in memory.

    What has to be added is the rowid. The preview reads ``rowid, *`` and
    then throws the rowid away, because it only ever displays; an edit has
    to address the row it is writing to, and a row's position in the view
    is not an address - it changes the moment anything above it is deleted.
    """

    def __init__(self, repo: SqliteRepo, table: str, parent=None) -> None:
        self._rowids: dict[int, list[int]] = {}
        #: Set by the dialog, so a cell edit joins the session's own undo
        #: entry rather than copying the table again for every keystroke.
        self.undo_entry_for: Any = lambda: None
        #: Set by the dialog: told (rowid, column) after each written cell.
        self.cell_written: Any = lambda _rowid, _column: None
        super().__init__(repo, table, parent)

    def _fetch_chunk(self, chunk_index: int) -> list[list[Any]]:
        last_rowid = self._chunk_rowid.get(chunk_index - 1, 0) if chunk_index > 0 else 0
        if not self._repo.is_open:
            return []
        rows = self._repo.read_rows_after_rowid(self._table, last_rowid, self._chunk_size)
        if not rows:
            self._rowids[chunk_index] = []
            return []
        self._chunk_rowid[chunk_index] = int(rows[-1][0])
        self._rowids[chunk_index] = [int(row[0]) for row in rows]
        return [list(row[1:]) for row in rows]

    def clear_cache(self) -> None:
        super().clear_cache()
        self._rowids.clear()

    def rowid_at(self, row_index: int) -> int | None:
        """Return the database rowid shown at *row_index*, loading if needed."""
        if not 0 <= row_index < self._row_count:
            return None
        chunk, offset = divmod(row_index, self._chunk_size)
        if chunk not in self._cache:
            self._cache[chunk] = self._fetch_chunk(chunk)
        rowids = self._rowids.get(chunk) or []
        return rowids[offset] if offset < len(rowids) else None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index: QModelIndex | QPersistentModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        column = self.column_name(index.column())
        rowid = self.rowid_at(index.row())
        if column is None or rowid is None:
            return False

        try:
            self._repo.update_table_cell(
                self._table,
                rowid,
                column,
                self._coerce(index, value),
                undo_entry=self.undo_entry_for(),
            )
        except Exception as exc:
            applogger.exception("Could not write the cell: %s", exc)
            return False

        self.cell_written(rowid, column)
        # Only this chunk is stale; re-reading the whole table to show one
        # changed cell would defeat the point of reading it in chunks.
        self._cache.pop(index.row() // self._chunk_size, None)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
        return True

    def _coerce(self, index: QModelIndex | QPersistentModelIndex, value: Any) -> Any:
        """Return *value* as the column's own type, or as text, or as NULL.

        An empty box means NULL rather than an empty string: in a numeric
        column the string would make the whole column text as far as
        anything reading it is concerned, which is a bigger change than the
        one that was asked for.
        """
        text = "" if value is None else str(value).strip()
        if text == "":
            return None
        code = self._codes[index.column()] if index.column() < len(self._codes) else ""
        if code in ("I", "F", "N"):
            try:
                if code == "I":
                    # "3.7" in an integer column is a number, not a label -
                    # int() alone rejects it and would store the string.
                    return int(text) if text.lstrip("+-").isdigit() else float(text)
                return float(text)
            except ValueError:
                # Deliberately not rejected: someone typing "n/a" into a
                # numeric column is telling us something, and SQLite's
                # dynamic typing can hold it.
                return text
        return text


class TableEditorDialog(QDialog):
    """Hand editing for one table: cells, rows, columns, and the data tools."""

    def __init__(
        self,
        repo: SqliteRepo,
        table: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table = str(table)
        self.setWindowTitle(_("Edit table '{table}'").format(table=self._table))

        self.view = QTableView(self)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setAlternatingRowColors(True)
        self.view.setSortingEnabled(False)
        apply_fusion_for_item_view_styling(self.view)
        # Rename a column where it is named: double-click its header.
        self.view.horizontalHeader().sectionDoubleClicked.connect(self._rename_from_header)

        #: One undo entry for everything done while this dialog is open.
        #: See the repository's hand-editing section: a snapshot copies the
        #: whole table, so an entry per cell would copy it per keystroke.
        #: It is also what Cancel and Restore roll back.
        self._undo_entry: int | None = None
        self._reset_counts()

        self._model = self._new_model()
        self.view.setModel(self._model)

        root = QVBoxLayout(self)
        # "medium" (900x640), not "large" (1020x700): 700 plus a title bar
        # and a dock does not fit the 768 a laptop screen still commonly
        # has, and a dialog that opens taller than the screen cannot be
        # resized back by dragging an edge that is off it.
        apply_dialog_shell(self, root, size="medium")
        root.addWidget(self._build_toolbar(), 0)
        root.addWidget(self.view, 1)

        self._status = QLabel(self)
        self._status.setProperty("muted", True)
        root.addWidget(self._status, 0)
        self._show_counts()

        closing = QHBoxLayout()
        stdSizeAndlayout(closing)
        self._restore_button = create_action_button(
            parent=self, action_id="table_restore", action=self._restore, layout=closing
        )
        mark_destructive_button(self._restore_button)
        closing.addStretch(1)
        create_action_button(
            parent=self,
            action_id="apply",
            action=self.accept,
            layout=closing,
            presentation=(
                action_presentation("apply")[0],
                _("OK"),
                _("Keep every change made here"),
            ),
        )
        create_action_button(
            parent=self,
            action_id="close",
            action=self.reject,
            layout=closing,
            presentation=(
                action_presentation("close")[0],
                _("Cancel"),
                _("Undo every change made here and close"),
            ),
        )
        root.addLayout(closing, 0)

    # ------------------------------------------------------------------
    # The toolbar: one row of icons, grouped; names are in the tooltips
    # ------------------------------------------------------------------

    def _tool(self, row: QHBoxLayout, action_id: str, action) -> QPushButton:
        """One square icon button from the action catalogue (SF Symbols on
        macOS, Segoe Fluent on Windows); its name is in the tooltip."""
        button = create_action_button(parent=self, action_id=action_id, action=action, layout=row)
        button.setAccessibleName(action_presentation(action_id)[1])
        mark_icon_only(button)
        return button

    @staticmethod
    def _gap(row: QHBoxLayout) -> None:
        row.addSpacing(18)

    def _build_toolbar(self) -> CardFrame:
        card = CardFrame(self, "tableEditorCard")
        row = QHBoxLayout()
        stdSizeAndlayout(row)
        card.layout().addLayout(row)

        self._tool(row, "table_add_row", self._add_row)
        self._tool(row, "table_insert_row", self._insert_row)
        self._tool(row, "table_delete_rows", self._delete_rows)
        self._gap(row)
        self._tool(row, "table_add_column", self._add_column)
        self._tool(row, "table_insert_column", self._insert_column)
        self._tool(row, "table_rename_column", self._rename_column)
        self._tool(row, "table_cast_column", self._cast_column)
        self._tool(row, "table_computed_column", self._computed_column)
        self._tool(row, "table_delete_column", self._delete_column)
        self._gap(row)
        self._tool(row, "table_sort_ascending", lambda: self._sort(descending=False))
        self._tool(row, "table_sort_descending", lambda: self._sort(descending=True))
        self._tool(row, "table_fill_missing", self._fill_missing)
        self._tool(row, "table_find_replace", self._find_replace)
        self._gap(row)
        self._tool(row, "table_group_aggregate", self._group_aggregate)
        row.addStretch(1)

        # Hide, Selected and ClusterId: the columns the application
        # maintains - used rarely, so behind one button.
        more = QToolButton(card)
        more.setObjectName("tableEditorMore")
        more_icon, more_text, more_tip = action_presentation("table_more")
        more.setIcon(more_icon)
        more.setAccessibleName(more_text)
        more.setToolTip(f"{more_text}: {more_tip}")
        more.setAutoRaise(True)
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(more)
        # Hide rows by the selected column: the same comparisons the data
        # preview offers, so the editor is not the one place they are
        # missing (todo N-06). Inside this session's undo entry.
        for flag, action_id, empty_text, same_text in (
            ("Hide", "table_hide_rows", _("Hide NULL / empty values"), _("Hide rows equal to selected cell")),
            ("Selected", "table_select_rows", _("Select NULL / empty values"), _("Select rows equal to selected cell")),
        ):
            flag_icon, flag_text, flag_tip = action_presentation(action_id)
            flag_menu = menu.addMenu(flag_icon, flag_text)
            flag_menu.setToolTipsVisible(True)
            flag_menu.setToolTip(flag_tip)
            # The operator is the payload; only the label is translated.
            for label, operator in (
                (_("Equal to..."), "="),
                (_("Different from..."), "!="),
                (_("Lower than..."), "<"),
                (_("Lower or equal..."), "<="),
                (_("Higher than..."), ">"),
                (_("Higher or equal..."), ">="),
            ):
                item = flag_menu.addAction(label)
                item.triggered.connect(lambda _checked=False, op=operator, f=flag: self._flag_rows(f, op))
            flag_menu.addSeparator()
            flag_menu.addAction(empty_text).triggered.connect(
                lambda _checked=False, f=flag: self._flag_rows(f, None)
            )
            flag_menu.addAction(same_text).triggered.connect(
                lambda _checked=False, f=flag: self._flag_rows(f, "=", from_cell=True)
            )
        menu.addSeparator()
        for action_id, action in (
            ("table_hide_ensure", self._repo.ensure_hide_column),
            ("table_hide_reset", self._repo.clear_hide_column),
            ("table_hide_invert", self._repo.invert_hide),
            (None, None),
            ("table_selected_ensure", self._repo.ensure_selected_column),
            ("table_selected_reset", self._repo.clear_selected_column),
            ("table_selected_invert", self._repo.invert_selected),
            (None, None),
            ("table_cluster_ensure", self._repo.ensure_cluster_column),
            ("table_cluster_reset", self._repo.clear_cluster_column),
        ):
            if action_id is None:
                menu.addSeparator()
                continue
            icon, text, tooltip = action_presentation(action_id)
            item = menu.addAction(icon, text)
            item.setToolTip(tooltip)
            item.triggered.connect(lambda _checked=False, run=action: self._managed(run))
        more.setMenu(menu)
        row.addWidget(more, 0)
        return card

    # ------------------------------------------------------------------
    # What changed: the status line
    # ------------------------------------------------------------------

    def _reset_counts(self) -> None:
        self._rows_added = 0
        self._rows_removed = 0
        self._rows_modified: set[int] = set()
        self._cells_bulk = 0
        self._columns_added = 0
        self._columns_removed = 0
        self._columns_modified: set[str] = set()

    def _show_counts(self) -> None:
        self._status.setText(
            _(
                "Rows: {ra} added, {rr} removed, {rm} modified  \u00b7  "
                "Columns: {ca} added, {cr} removed, {cm} modified"
            ).format(
                ra=self._rows_added,
                rr=self._rows_removed,
                rm=len(self._rows_modified) + self._cells_bulk,
                ca=self._columns_added,
                cr=self._columns_removed,
                cm=len(self._columns_modified),
            )
        )
        button = getattr(self, "_restore_button", None)
        if button is not None:
            button.setEnabled(self._undo_entry is not None)

    def _on_cell_written(self, rowid: int, column: str) -> None:
        self._rows_modified.add(int(rowid))
        self._show_counts()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _selected_column(self) -> str | None:
        index = self.view.currentIndex()
        if not index.isValid():
            return None
        return self._model.column_name(index.column())

    def _selected_rowids(self) -> list[int]:
        """Every distinct rowid with a selected cell, top to bottom."""
        rows = sorted({index.row() for index in self.view.selectedIndexes()})
        found = [self._model.rowid_at(row) for row in rows]
        return [rowid for rowid in found if rowid is not None]

    def _need_column(self) -> str | None:
        column = self._selected_column()
        if column is None:
            self._say(_("Select a column first."))
        return column

    # ------------------------------------------------------------------
    # Row actions
    # ------------------------------------------------------------------

    def _add_row(self) -> None:
        def run() -> None:
            self._repo.append_table_row(self._table, undo_entry=self._ensure_undo_entry())
            self._rows_added += 1

        self._guarded(run)

    def _insert_row(self) -> None:
        rowids = self._selected_rowids()
        if not rowids:
            self._say(_("Select the row to insert above first."))
            return

        def run() -> None:
            self._repo.insert_table_row_before(
                self._table, rowids[0], undo_entry=self._ensure_undo_entry()
            )
            self._rows_added += 1

        self._guarded(run)

    def _delete_rows(self) -> None:
        rowids = self._selected_rowids()
        if not rowids:
            self._say(_("Select the rows to delete first."))
            return
        if len(rowids) >= self._model.rowCount():
            self._say(_("A table needs at least one row. Add another row before deleting this one."))
            return
        confirmed = QMessageBox.question(
            self,
            _("Delete rows"),
            _("Delete {count} row(s)? This can be undone.").format(count=len(rowids)),
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        def run() -> None:
            self._repo.delete_table_rows(self._table, rowids, undo_entry=self._ensure_undo_entry())
            self._rows_removed += len(rowids)

        self._guarded(run)

    # ------------------------------------------------------------------
    # Column actions
    # ------------------------------------------------------------------

    def _add_column(
        self, before: str | None = None, *, name: str | None = None, kind: str | None = None
    ) -> None:
        if name is None:
            name, ok = QInputDialog.getText(self, _("Add column"), _("Name of the new column:"))
            if not ok:
                return
        name = (name or "").strip()
        if not name:
            return
        if kind is None:
            kind, ok = QInputDialog.getItem(
                self, _("Add column"), _("Type:"), list(COLUMN_TYPES), 0, False
            )
            if not ok:
                return
        column_type = str(kind)

        def run() -> None:
            self._repo.insert_table_column(
                self._table,
                name,
                column_type,
                before=before,
                undo_entry=self._ensure_undo_entry(),
            )
            self._columns_added += 1

        self._guarded(run)

    def _insert_column(self) -> None:
        before = self._selected_column()
        if before is None:
            self._say(_("Select the column to insert before first."))
            return
        self._add_column(before=before)

    def _rename_column(self) -> None:
        index = self.view.currentIndex()
        if not index.isValid():
            self._say(_("Select a column first."))
            return
        self._rename_from_header(index.column())

    def _rename_from_header(self, section: int) -> None:
        column = self._model.column_name(section)
        if column is None:
            return
        name, ok = QInputDialog.getText(
            self, _("Rename column"), _("New name for '{column}':").format(column=column),
            QLineEdit.EchoMode.Normal, column,
        )
        name = (name or "").strip()
        if ok and name and name != column:
            self._rename(column, name)

    def _rename(self, column: str, name: str) -> None:
        def run() -> None:
            self._ensure_undo_entry()
            self._repo.rename_table_column(self._table, column, name)
            self._columns_modified.discard(column)
            self._columns_modified.add(name)

        self._guarded(run)

    def _cast_column(self, kind: str | None = None) -> None:
        column = self._need_column()
        if column is None:
            return
        if kind is None:
            kind, ok = QInputDialog.getItem(
                self, _("Change type"), _("New type for '{column}':").format(column=column),
                list(CAST_TYPES), 0, False,
            )
            if not ok:
                return
        if kind not in CAST_TYPES:
            return

        def run() -> None:
            failed = self._repo.cast_column(
                self._table, column, kind, undo_entry=self._ensure_undo_entry()
            )
            self._columns_modified.add(column)
            if failed:
                self._say(
                    _("{count} value(s) could not be read as {kind} and were left empty.").format(
                        count=failed, kind=kind
                    )
                )

        self._guarded(run)

    def _computed_column(self) -> None:
        from app.dialogs.table_tools_dialogs import ComputedColumnDialog

        dialog = ComputedColumnDialog(self._repo, self._table, self)
        if not dialog.exec():
            return
        name, kind, expression = dialog.column_name(), dialog.column_type(), dialog.expression_text()

        def run() -> None:
            self._repo.add_column_from_expression(
                self._table, name, expression, kind, undo_entry=self._ensure_undo_entry()
            )
            self._columns_added += 1

        self._guarded(run)

    def _delete_column(self) -> None:
        column = self._need_column()
        if column is None:
            return
        # Hide, Selected and ClusterId are the application's; one column of data must stay.
        data_columns = [
            c for c in (self._model.column_name(i) for i in range(self._model.columnCount()))
            if c not in (None, "Hide", "Selected", "ClusterId")
        ]
        if column in data_columns and len(data_columns) <= 1:
            self._say(_("A table needs at least one column. Add another column before deleting this one."))
            return
        confirmed = QMessageBox.question(
            self,
            _("Delete column"),
            _("Delete '{column}' and everything in it? This can be undone.").format(
                column=column
            ),
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        def run() -> None:
            self._repo.delete_table_column(
                self._table, column, undo_entry=self._ensure_undo_entry()
            )
            self._columns_removed += 1
            self._columns_modified.discard(column)

        self._guarded(run)

    # ------------------------------------------------------------------
    # Data tools
    # ------------------------------------------------------------------

    def _hide_rows(self, operator: str | None, *, from_cell: bool = False) -> None:
        """Hide the rows where the selected column compares to a value."""
        self._flag_rows("Hide", operator, from_cell=from_cell)

    def _flag_rows(self, flag: str, operator: str | None, *, from_cell: bool = False) -> None:
        """Set *flag* (Hide or Selected) on the rows where the selected column
        compares to a value.

        *operator* None marks the empty cells; ``from_cell`` takes the value
        from the selected cell instead of asking for one.
        """
        column = self._need_column()
        if column is None:
            return
        value: Any = None
        if operator is not None:
            if from_cell:
                index = self.view.currentIndex()
                if not index.isValid():
                    self._say(_("Select a cell first."))
                    return
                value = index.data(Qt.ItemDataRole.DisplayRole)
            else:
                text, ok = QInputDialog.getText(
                    self,
                    _("Hide rows") if flag == "Hide" else _("Select rows"),
                    (_("Hide rows where {column} {operator}:") if flag == "Hide"
                     else _("Select rows where {column} {operator}:")).format(column=column, operator=operator),
                )
                if not ok:
                    return
                value = text
        hidden: list[int] = []

        def run() -> None:
            self._ensure_undo_entry()
            if operator is None:
                hidden.append(self._repo.flag_rows_special(self._table, flag, column, "null_or_empty"))
            else:
                hidden.append(self._repo.flag_rows_by_value(self._table, flag, column, operator, value))

        self._guarded(run)
        if hidden:
            message = _("{count} row(s) hidden.") if flag == "Hide" else _("{count} row(s) marked Selected.")
            self._say(message.format(count=hidden[0]))

    def _sort(self, *, descending: bool) -> None:
        column = self._need_column()
        if column is None:
            return
        self._guarded(lambda: self._repo.sort_table(
            self._table, column, descending=descending, undo_entry=self._ensure_undo_entry()
        ))

    def _fill_missing(self) -> None:
        column = self._need_column()
        if column is None:
            return
        from app.dialogs.table_tools_dialogs import FillMissingDialog

        dialog = FillMissingDialog(column, self)
        if not dialog.exec():
            return
        method = str(dialog.method.currentData())
        value = dialog.value.text()

        def run() -> None:
            count = self._repo.fill_missing(
                self._table, column, method, _number_or_text(value),
                undo_entry=self._ensure_undo_entry(),
            )
            self._cells_bulk += count
            if count:
                self._columns_modified.add(column)

        self._guarded(run)

    def _find_replace(self) -> None:
        from app.dialogs.table_tools_dialogs import FindReplaceDialog

        dialog = FindReplaceDialog(self._selected_column(), self)
        if not dialog.exec():
            return
        scope = dialog.scope.currentData()
        find, replace = dialog.find_edit.text(), dialog.replace_edit.text()
        whole = dialog.whole_cell.isChecked()

        def run() -> None:
            count = self._repo.find_replace(
                self._table, find, replace,
                columns=[scope] if scope else None,
                whole_cell=whole,
                undo_entry=self._ensure_undo_entry(),
            )
            self._cells_bulk += count
            if count and scope:
                self._columns_modified.add(str(scope))
            self._say(_("{count} cell(s) changed.").format(count=count))

        self._guarded(run)

    def _group_aggregate(self) -> None:
        """Summarise this table into a new one.

        This table is not changed, so the new one is not part of the
        session's undo entry: Cancel here keeps it, and Undo removes it.
        """
        from app.dialogs.table_tools_dialogs import GroupAggregateDialog

        dialog = GroupAggregateDialog(
            self._repo, self._table, self, initial_group=self._selected_column()
        )
        if not dialog.exec():
            return
        try:
            name = self._repo.group_aggregate(
                self._table,
                dialog.chosen_groups(),
                measures=dialog.measures(),
                new_name=dialog.name.text().strip() or None,
                include_hidden=dialog.include_hidden.isChecked(),
            )
        except Exception as exc:
            applogger.exception("Group and aggregate failed: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return
        applogger.info("Grouped '%s' into '%s'.", self._table, name)
        self._say(_("Table '{name}' created.").format(name=name))

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _managed(self, action) -> None:
        def run() -> None:
            self._ensure_undo_entry()
            action(self._table)

        self._guarded(run)

    def _new_model(self) -> EditableTableModel:
        model = EditableTableModel(self._repo, self._table, self)
        model.undo_entry_for = self._ensure_undo_entry
        model.cell_written = self._on_cell_written
        return model

    def _ensure_undo_entry(self) -> int | None:
        """Open this session's undo entry if it has not been opened yet."""
        if self._undo_entry is None:
            self._undo_entry = self._repo.snapshot_for_undo(
                [self._table], label=f"Edit table '{self._table}'"
            )
        return self._undo_entry

    def _guarded(self, action) -> None:
        """Run one edit, report what went wrong, and reload either way.

        Reloading even after a failure is deliberate: a half-applied change
        is exactly the case where what is on screen and what is in the
        database have parted company, and that is the worst moment to be
        showing the stale one.
        """
        try:
            action()
        except Exception as exc:
            applogger.exception("Table edit failed: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
        finally:
            self.reload()
            self._show_counts()

    def _say(self, message: str) -> None:
        QMessageBox.information(self, _("Edit table"), message)

    def _roll_back(self) -> bool:
        """Undo this session's entry. True when the table is back as it was."""
        if self._undo_entry is None:
            return True
        try:
            self._repo.undo_entry(self._undo_entry)
        except Exception as exc:
            applogger.exception("Could not undo the table edits: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return False
        self._undo_entry = None
        return True

    def _restore(self) -> None:
        """Put the table back as it was when this editor opened, and stay open."""
        if self._undo_entry is None:
            return
        confirmed = QMessageBox.question(
            self,
            _("Restore the table?"),
            _(
                "Every change made in this editor will be undone, and the "
                "table put back as it was.\n\nThis cannot be redone."
            ),
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        if self._roll_back():
            self._reset_counts()
        self.reload()
        self._show_counts()

    def accept(self) -> None:
        """Keep the changes. They are already written; just stop asking."""
        self._undo_entry = None
        super().accept()

    def reject(self) -> None:
        """Put the table back as it was, once the person confirms.

        Every edit was written as it was made, so Cancel means this
        session's undo entry, restored - by id, not whatever happens to be
        last. Restoring is itself not undoable, hence the confirmation.
        """
        if self._undo_entry is None:
            super().reject()
            return

        confirmed = QMessageBox.question(
            self,
            _("Discard the changes?"),
            _(
                "Every change made in this editor will be undone, and the "
                "table put back as it was.\n\nThis cannot be redone."
            ),
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        if self._roll_back():
            super().reject()

    def reload(self) -> None:
        """Rebuild the model, because a column change alters the schema."""
        previous = self._model
        self._model = self._new_model()
        self.view.setModel(self._model)
        previous.deleteLater()


def _number_or_text(text: str):
    """A typed fill value: a number when it reads as one, text otherwise."""
    stripped = (text or "").strip()
    if stripped == "":
        return None
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return int(number) if number.is_integer() and "." not in stripped else number
