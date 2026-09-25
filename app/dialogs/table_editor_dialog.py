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

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    action_presentation,
    apply_dialog_shell,
    apply_fusion_for_item_view_styling,
    create_action_button,
    icon_from_svg_source,
    mark_icon_only,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.widgets.table_preview import LazyTableModel

#: What a new column can be declared as. SQLite's own storage classes, in
#: the order they are actually wanted: a measured quantity first.
COLUMN_TYPES: tuple[str, ...] = ("REAL", "INTEGER", "TEXT")

# Stroke-only 24x24 glyphs, the same shape language as the rest of the
# application's inline artwork. Each one has to be distinguishable from its
# neighbours at 20px, which is what the pairing of a mark (plus, minus,
# arrow) with an orientation (a horizontal bar for a row, a vertical one
# for a column) is for.
_ROW_ADD = '<path d="M3 7h18"/><path d="M3 12h18"/><path d="M7 18h8"/><path d="M11 14v8"/>'
_ROW_INSERT = '<path d="M3 14h18"/><path d="M3 19h18"/><path d="M12 3v8"/><path d="M8.5 6.5L12 3l3.5 3.5"/>'
_ROW_DELETE = '<path d="M3 7h18"/><path d="M3 12h18"/><path d="M7 18h8"/>'
_COL_ADD = '<path d="M7 3v18"/><path d="M12 3v18"/><path d="M18 7h4"/><path d="M20 5v4"/>'
_COL_INSERT = '<path d="M14 3v18"/><path d="M19 3v18"/><path d="M3 12h8"/><path d="M6.5 8.5L3 12l3.5 3.5"/>'
_COL_DELETE = '<path d="M7 3v18"/><path d="M12 3v18"/><path d="M18 12h4"/>'
_PENCIL = '<path d="M4 20h4L20 8l-4-4L4 16z"/>'
_EYE_OFF = '<path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6z"/><circle cx="12" cy="12" r="2.5"/><path d="M4 20L20 4"/>'
_EYE = '<path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6z"/><circle cx="12" cy="12" r="2.5"/>'
_SWAP = '<path d="M4 9h13"/><path d="M13.5 5.5L17 9l-3.5 3.5"/><path d="M20 15H7"/><path d="M10.5 11.5L7 15l3.5 3.5"/>'
_CLUSTER = '<circle cx="7" cy="8" r="2"/><circle cx="16" cy="7" r="2"/><circle cx="12" cy="15" r="2"/><circle cx="18" cy="16" r="2"/>'
_CLUSTER_OFF = '<circle cx="7" cy="8" r="2"/><circle cx="16" cy="7" r="2"/><circle cx="12" cy="15" r="2"/><path d="M4 20L20 4"/>'


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
        super().__init__(repo, table, parent)

    def _fetch_chunk(self, chunk_index: int) -> list[list[Any]]:
        last_rowid = self._chunk_rowid.get(chunk_index - 1, 0) if chunk_index > 0 else 0
        if self._repo._con is None:
            return []
        cursor = self._repo._con.execute(self._select_sql, (last_rowid, self._chunk_size))
        rows = cursor.fetchall()
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

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index: QModelIndex, value: Any, role=Qt.ItemDataRole.EditRole) -> bool:
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

        # Only this chunk is stale; re-reading the whole table to show one
        # changed cell would defeat the point of reading it in chunks.
        self._cache.pop(index.row() // self._chunk_size, None)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
        return True

    def _coerce(self, index: QModelIndex, value: Any) -> Any:
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
    """Hand editing for one table: cells, rows, columns."""

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

        #: One undo entry for everything done while this dialog is open.
        #: See the repository's hand-editing section: a snapshot copies the
        #: whole table, so an entry per cell would copy it per keystroke.
        #: The cost is that Undo steps back over the whole session at once,
        #: which is also how it reads to the person who did it - "I was
        #: editing that table".
        self._undo_entry: int | None = None

        self._model = EditableTableModel(self._repo, self._table, self)
        self._model.undo_entry_for = self._ensure_undo_entry
        self.view.setModel(self._model)

        root = QVBoxLayout(self)
        # "medium" (900x640), not "large" (1020x700): 700 plus a title bar
        # and a dock does not fit the 768 a laptop screen still commonly
        # has, and a dialog that opens taller than the screen cannot be
        # resized back by dragging an edge that is off it.
        apply_dialog_shell(self, root, size="medium")
        root.addWidget(self._build_toolbar(), 0)
        root.addWidget(self.view, 1)

        closing = QHBoxLayout()
        stdSizeAndlayout(closing)
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
    # The three groups of actions
    # ------------------------------------------------------------------

    def _group_label(self, text: str, parent: QWidget) -> QLabel:
        """A small heading inside the toolbar rather than above it."""
        label = QLabel(text, parent)
        label.setProperty("muted", True)
        return label

    def _icon_button(
        self, row: QHBoxLayout, glyph: str, name: str, tooltip: str, action
    ) -> None:
        """One square icon button, with the words in its tooltip.

        Icon-only on purpose: three labelled rows of buttons put about 230
        pixels of chrome above the table, which on a laptop screen left
        barely enough of the dialog to see the data it is for. mark_icon_only
        is the application's own answer - both stylesheets already carry the
        [iconOnly] rule, and the macOS sheet hides icons on an ordinary
        QPushButton (icon-size: 0) precisely so that only the buttons meant
        to be icons are.
        """
        button = create_action_button(
            parent=self,
            action_id="",
            action=action,
            layout=row,
            presentation=(icon_from_svg_source(glyph, size=20), name, tooltip),
        )
        button.setAccessibleName(name)
        mark_icon_only(button)

    def _build_toolbar(self) -> CardFrame:
        """Every action, in two rows of icons above the table."""
        card = CardFrame(self, "tableEditorCard")
        layout = card.layout()

        structure = QHBoxLayout()
        stdSizeAndlayout(structure)
        structure.addWidget(self._group_label(_("Rows"), card), 0)
        self._icon_button(
            structure, _ROW_ADD, _("Add row"),
            _("Add an empty row at the end of the table"), self._add_row,
        )
        self._icon_button(
            structure, _ROW_INSERT, _("Insert row above"),
            _("Add an empty row above the selected one"), self._insert_row,
        )
        self._icon_button(
            structure, _ROW_DELETE, _("Delete rows"),
            _("Delete every row with a selected cell"), self._delete_rows,
        )

        structure.addSpacing(12)
        structure.addWidget(self._group_label(_("Columns"), card), 0)

        self._new_column_name = QLineEdit(self)
        self._new_column_name.setPlaceholderText(_("New column name"))
        stdSizeAndlayout(self._new_column_name)
        structure.addWidget(self._new_column_name, 1)

        self._new_column_type = QComboBox(self)
        self._new_column_type.addItems(COLUMN_TYPES)
        # The length matters: stdSizeAndlayout defaults a combo's minimum
        # contents length to 0, which collapsed the box to "I" and an arrow.
        stdSizeAndlayout(
            self._new_column_type,
            minimum_contents_length=max(len(name) for name in COLUMN_TYPES),
        )
        structure.addWidget(self._new_column_type, 0)

        self._icon_button(
            structure, _COL_ADD, _("Add column"),
            _("Add the named column at the end of the table"), self._add_column,
        )
        self._icon_button(
            structure, _COL_INSERT, _("Insert before selected"),
            _("Add the named column immediately before the selected one"),
            self._insert_column,
        )
        self._icon_button(
            structure, _PENCIL, _("Rename selected"),
            _("Rename the selected column to the name in the box"), self._rename_column,
        )
        self._icon_button(
            structure, _COL_DELETE, _("Delete selected"),
            _("Delete the selected column and everything in it"), self._delete_column,
        )
        layout.addLayout(structure)

        managed = QHBoxLayout()
        stdSizeAndlayout(managed)
        managed.addWidget(self._group_label(_("Managed columns"), card), 0)
        self._icon_button(
            managed, _EYE_OFF, _("Ensure Hide"),
            _("Add the Hide column if this table does not have one"),
            lambda: self._managed(self._repo.ensure_hide_column, _("Hide column ensured")),
        )
        self._icon_button(
            managed, _EYE, _("Reset Hide"),
            _("Set Hide back to 0 on every row, so nothing is hidden"),
            lambda: self._managed(self._repo.clear_hide_column, _("Hide reset")),
        )
        self._icon_button(
            managed, _SWAP, _("Invert Hide"),
            _("Swap hidden and shown rows"),
            lambda: self._managed(self._repo.invert_hide, _("Hide inverted")),
        )
        managed.addSpacing(12)
        self._icon_button(
            managed, _CLUSTER, _("Ensure ClusterId"),
            _("Add the ClusterId column if this table does not have one"),
            lambda: self._managed(self._repo.ensure_cluster_column, _("ClusterId column ensured")),
        )
        self._icon_button(
            managed, _CLUSTER_OFF, _("Reset clusters"),
            _("Clear every cluster label"),
            lambda: self._managed(self._repo.clear_cluster_column, _("Clusters reset")),
        )
        managed.addWidget(
            self._group_label(
                _("Hide marks rows the charts skip; ClusterId is written by the Clustering operation."),
                card,
            ),
            1,
        )
        layout.addLayout(managed)
        return card

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

    # ------------------------------------------------------------------
    # Row actions
    # ------------------------------------------------------------------

    def _add_row(self) -> None:
        self._guarded(lambda: self._repo.append_table_row(self._table, undo_entry=self._ensure_undo_entry()))

    def _insert_row(self) -> None:
        rowids = self._selected_rowids()
        if not rowids:
            self._say(_("Select the row to insert above first."))
            return
        self._guarded(lambda: self._repo.insert_table_row_before(
                self._table, rowids[0], undo_entry=self._ensure_undo_entry()
            ))

    def _delete_rows(self) -> None:
        rowids = self._selected_rowids()
        if not rowids:
            self._say(_("Select the rows to delete first."))
            return
        confirmed = QMessageBox.question(
            self,
            _("Delete rows"),
            _("Delete {count} row(s)? This can be undone.").format(count=len(rowids)),
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        self._guarded(lambda: self._repo.delete_table_rows(
                self._table, rowids, undo_entry=self._ensure_undo_entry()
            ))

    # ------------------------------------------------------------------
    # Column actions
    # ------------------------------------------------------------------

    def _add_column(self, before: str | None = None) -> None:
        name = self._new_column_name.text().strip()
        if not name:
            self._say(_("Type a name for the new column first."))
            return
        column_type = self._new_column_type.currentText()

        def run() -> None:
            self._repo.insert_table_column(
                self._table,
                name,
                column_type,
                before=before,
                undo_entry=self._ensure_undo_entry(),
            )
            self._new_column_name.clear()

        self._guarded(run)

    def _insert_column(self) -> None:
        before = self._selected_column()
        if before is None:
            self._say(_("Select the column to insert before first."))
            return
        self._add_column(before=before)

    def _rename_column(self) -> None:
        column = self._selected_column()
        if column is None:
            self._say(_("Select a column first."))
            return
        name = self._new_column_name.text().strip()
        if not name:
            self._say(_("Type the new name in the box first."))
            return

        def run() -> None:
            self._repo.rename_table_column(self._table, column, name)
            self._new_column_name.clear()

        self._guarded(run)

    def _delete_column(self) -> None:
        column = self._selected_column()
        if column is None:
            self._say(_("Select a column first."))
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
        self._guarded(lambda: self._repo.delete_table_column(self._table, column))

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _managed(self, action, done: str) -> None:
        def run() -> None:
            action(self._table)
            applogger.info("%s on '%s'.", done, self._table)

        self._guarded(run)

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

    def _say(self, message: str) -> None:
        QMessageBox.information(self, _("Edit table"), message)

    def accept(self) -> None:
        """Keep the changes. They are already written; just stop asking."""
        self._undo_entry = None
        super().accept()

    def reject(self) -> None:
        """Put the table back as it was, once the person confirms.

        Cancel has to mean something here even though every edit was
        written as it was made: what it means is this session's undo entry,
        restored. That is why the whole session shares one entry - it is
        both what keeps a large table editable and what Cancel rolls back.

        Restoring is itself not undoable, hence the confirmation. And
        because it restores the snapshot taken when this dialog was first
        edited, anything else that changed the same table meanwhile would
        go with it - which nothing can, the dialog being modal, but it is
        the reason this undoes *its own* entry by id rather than whatever
        happens to be last.
        """
        if self._undo_entry is None:
            # Nothing was changed, so there is nothing to lose and nothing
            # worth interrupting anybody about.
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

        try:
            self._repo.undo_entry(self._undo_entry)
        except Exception as exc:
            applogger.exception("Could not undo the table edits: %s", exc)
            QMessageBox.warning(self, _("Could not do that"), str(exc))
            return

        self._undo_entry = None
        super().reject()

    def reload(self) -> None:
        """Rebuild the model, because a column change alters the schema."""
        previous = self._model
        self._model = EditableTableModel(self._repo, self._table, self)
        self._model.undo_entry_for = self._ensure_undo_entry
        self.view.setModel(self._model)
        previous.deleteLater()
