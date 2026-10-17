"""Database Info: path, size, tables and import links, in one place.

A read-only overview that puts the whole database's shape (how many tables,
how big, which ones are fed by a link) on screen at once. Exporting,
renaming and refreshing a table's link are the Tables page's, in its
list's menu - one place for each action (todo H-14).

A plain embeddable QWidget, not a QDialog: it sits at the top of
MainWindow's Project page (see main_window._create_developer_page), with
its own Optimize DB action, rather than opening as its own modal window.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    TitledCard,
    create_action_button,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


def _human_size(num_bytes: int) -> str:
    """Return *num_bytes* as "12.3 MB" - the units a person actually reads."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


class DatabaseInfoPanel(QWidget):
    """Path, size, tables and import links for the connected database."""

    def __init__(
        self,
        repo: SqliteRepo,
        parent: QWidget | None = None,
        *,
        optimize_action: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo = repo

        root = QVBoxLayout(self)
        stdSizeAndlayout(root)

        self._titled_card = TitledCard(self, _("Database Info"), "databaseInfoCard")
        self._card = self._titled_card.card
        self._card_layout = self._card.layout()

        self._form_host = QWidget(self._card)
        self._form_layout = QFormLayout(self._form_host)
        stdSizeAndlayout(self._form_layout)
        self._card_layout.addWidget(self._form_host)
        self._build_form()

        if optimize_action is not None:
            # Lives here, not beside Query Builder (see main_window's own
            # Query Builder section): it is a maintenance action against
            # exactly the size/page stats the form above it shows - right
            # below "Last modified", its own last row - rather than against
            # the query workflow. Above the table list, not below it: it
            # acts on the database file as a whole, not on any one table.
            # Its own row, right-aligned, rather than beside the title -
            # the title moved outside the card's border (see TitledCard)
            # and an action button belongs inside it.
            optimize_row = QHBoxLayout()
            stdSizeAndlayout(optimize_row)
            optimize_row.addStretch(1)
            create_action_button(
                parent=self._card,
                action_id="optimize_db",
                action=optimize_action,
                layout=optimize_row,
            )
            self._card_layout.addLayout(optimize_row)

        self._table = QTableWidget(0, 3, self._card)
        self._table.setHorizontalHeaderLabels([_("Table"), _("Rows"), _("Linked")])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        mark_editor_panel(self._table)
        self._card_layout.addWidget(self._table, 1)

        root.addWidget(self._titled_card, 1)


        self._reload_tables()

    def set_repo(self, repo: SqliteRepo) -> None:
        """Point the panel at a newly-opened database and refresh everything."""
        self._repo = repo
        self._build_form()
        self._reload_tables()

    def _build_form(self) -> None:
        """(Re)build the path/size/pragma rows for the current repo."""
        while self._form_layout.rowCount():
            self._form_layout.removeRow(0)

        self._form_layout.addRow(_("Path:"), self._selectable_label(str(self._repo.db_path)))
        info = self._repo.project_info()
        if info.get("author"):
            self._form_layout.addRow(_("Author:"), self._selectable_label(info["author"]))
        if info.get("created"):
            from app.dialogs.project_info_dialog import readable_date

            self._form_layout.addRow(_("Created:"), self._selectable_label(readable_date(info["created"])))
        self._form_layout.addRow(_("Size on disk:"), self._selectable_label(self._db_size_text()))
        self._add_pragma_rows()

    def _selectable_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _db_size_text(self) -> str:
        try:
            return _human_size(self._repo.db_path.stat().st_size)
        except OSError:
            return _("unknown")

    def _add_pragma_rows(self) -> None:
        """Add everything SQLite and the filesystem themselves can say.

        A best effort, not a hard requirement of building this panel - a
        PRAGMA failing (a very old or unusual SQLite build) loses this
        section, not the whole panel.
        """
        try:
            info = self._repo.database_pragma_info()
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Could not read database PRAGMA info.")
            return

        form = self._form_layout
        form.addRow(
            _("Tables:"),
            self._selectable_label(
                _("{tables} table(s), {rows} row(s) total").format(
                    tables=f"{info.table_count:,}", rows=f"{info.total_rows:,}"
                )
            ),
        )
        if info.reclaimable_bytes:
            pages_text = _(
                "{count:,} x {size} B, {amount} reclaimable by Optimize DB"
            ).format(
                count=info.page_count,
                size=info.page_size,
                amount=_human_size(info.reclaimable_bytes),
            )
        else:
            pages_text = _("{count:,} x {size} B").format(
                count=info.page_count, size=info.page_size
            )
        form.addRow(_("Pages:"), self._selectable_label(pages_text))
        form.addRow(
            _("Encoding:"), self._selectable_label(f"{info.encoding} · {info.journal_mode}")
        )
        form.addRow(_("SQLite version:"), self._selectable_label(info.sqlite_version))
        if info.file_created is not None:
            form.addRow(
                _("File created:"),
                self._selectable_label(info.file_created.strftime("%Y-%m-%d %H:%M")),
            )
        if info.file_modified is not None:
            form.addRow(
                _("Last modified:"),
                self._selectable_label(info.file_modified.strftime("%Y-%m-%d %H:%M")),
            )

    def _reload_tables(self) -> None:
        """Repopulate the table list, keeping the current selection by name.

        include_internal=True: this list is the database's own inventory,
        so the app's "__..._descriptors__" tables belong in it same as any
        other - unlike list_user_tables()'s other callers (the chart
        data-source picker, the pragma row-count total), which only ever
        want tables a user could plot.
        """
        selected = self._selected_table()
        frame = self._repo.list_user_tables(include_internal=True)

        self._table.setRowCount(len(frame.index))
        for row, record in enumerate(frame.to_dict("records")):
            name = str(record.get("Table", ""))
            has_link = bool(record.get("has_link", False))
            rows = self._repo.row_count(name)

            self._table.setItem(row, 0, QTableWidgetItem(name))
            rows_item = QTableWidgetItem(f"{rows:,}")
            rows_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 1, rows_item)
            self._table.setItem(row, 2, QTableWidgetItem(_("Yes") if has_link else ""))

            if name == selected:
                self._table.selectRow(row)

    def _selected_table(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        return item.text() if item else None
