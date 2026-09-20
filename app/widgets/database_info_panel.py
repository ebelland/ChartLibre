"""Database Info: path, size, tables and import links, in one place.

Everything here already existed somewhere - TableListPanel's own context
menu already exports, renames and refreshes the link of a table - this is a
read-only overview that puts the whole database's shape (how many tables,
how big, which ones are fed by a link) on screen at once, with those same
actions one click away for whichever row is selected, rather than requiring
a right-click per table.

A plain embeddable QWidget, not a QDialog: it sits inside MainWindow's
"Database" nav page (see main_window._create_database_page), alongside the
Query Builder / Optimize DB actions the same page offers, rather than
opening as its own modal window.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
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
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.utils.import_runner import refresh_link


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
        self._table.itemSelectionChanged.connect(self._update_button_states)
        mark_editor_panel(self._table)
        self._card_layout.addWidget(self._table, 1)

        root.addWidget(self._titled_card, 1)

        # A separate card, not more rows tacked onto Database Info above:
        # these four act on whichever row is selected in the table list,
        # not on the database as a whole, and each gets the full row to
        # itself - one full-text button per line, rather than the
        # icon-only pair this replaces, reads best spelled out when there
        # is no longer a second button competing for the same row's width.
        self._selected_table_card = TitledCard(
            self, _("Table (selected table)"), "selectedTableCard"
        )
        selected_table_layout = self._selected_table_card.card.layout()

        update_row = QHBoxLayout()
        stdSizeAndlayout(update_row)
        self._update_link_button = create_action_button(
            parent=self,
            action_id="update_link",
            action=self._update_selected_link,
            layout=update_row,
            presentation=(
                load_icon("reload"),
                _("Update link"),
                _("Refresh the selected table from its import link"),
            ),
        )
        update_row.addStretch(1)
        selected_table_layout.addLayout(update_row)

        export_xlsx_row = QHBoxLayout()
        stdSizeAndlayout(export_xlsx_row)
        self._export_xlsx_button = create_action_button(
            parent=self,
            action_id="export_xlsx",
            action=self._export_selected_xlsx,
            layout=export_xlsx_row,
            presentation=(
                load_icon("export_xlsx"),
                _("Export as Excel"),
                _("Export as an Excel workbook"),
            ),
        )
        export_xlsx_row.addStretch(1)
        selected_table_layout.addLayout(export_xlsx_row)

        export_csv_row = QHBoxLayout()
        stdSizeAndlayout(export_csv_row)
        self._export_csv_button = create_action_button(
            parent=self,
            action_id="export_csv",
            action=self._export_selected_csv,
            layout=export_csv_row,
            presentation=(
                load_icon("export_csv"),
                _("Export as text file"),
                _("Export as a CSV file"),
            ),
        )
        export_csv_row.addStretch(1)
        selected_table_layout.addLayout(export_csv_row)

        rename_row = QHBoxLayout()
        stdSizeAndlayout(rename_row)
        self._rename_button = create_action_button(
            parent=self,
            action_id="rename",
            action=self._rename_selected_table,
            layout=rename_row,
        )
        rename_row.addStretch(1)
        selected_table_layout.addLayout(rename_row)

        root.addWidget(self._selected_table_card)

        self._reload_tables()
        self._update_button_states()

    def set_repo(self, repo: SqliteRepo) -> None:
        """Point the panel at a newly-opened database and refresh everything."""
        self._repo = repo
        self._build_form()
        self._reload_tables()
        self._update_button_states()

    def _build_form(self) -> None:
        """(Re)build the path/size/pragma rows for the current repo."""
        while self._form_layout.rowCount():
            self._form_layout.removeRow(0)

        self._form_layout.addRow(_("Path:"), self._selectable_label(str(self._repo.db_path)))
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
        except Exception:
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

    def _selected_table_has_link(self) -> bool:
        row = self._table.currentRow()
        if row < 0:
            return False
        item = self._table.item(row, 2)
        return bool(item and item.text())

    def _update_button_states(self) -> None:
        table = self._selected_table()
        self._export_csv_button.setEnabled(table is not None)
        self._export_xlsx_button.setEnabled(table is not None)
        self._rename_button.setEnabled(table is not None)
        self._update_link_button.setEnabled(
            table is not None and self._selected_table_has_link()
        )

    def _export_selected_csv(self) -> None:
        table = self._selected_table()
        if not table:
            return
        file_path, _unused = QFileDialog.getSaveFileName(
            self, _("Export CSV"), f"{table}.csv", "CSV files (*.csv)"
        )
        if not file_path:
            return
        try:
            self._repo.query_df(f'SELECT * FROM "{table}"').to_csv(
                file_path, index=False
            )
        except Exception as exc:  # noqa: BLE001
            applogger.exception("CSV export failed: %s", exc)

    def _export_selected_xlsx(self) -> None:
        table = self._selected_table()
        if not table:
            return
        file_path, _unused = QFileDialog.getSaveFileName(
            self, _("Export XLSX"), f"{table}.xlsx", "Excel files (*.xlsx)"
        )
        if not file_path:
            return
        try:
            self._repo.query_df(f'SELECT * FROM "{table}"').to_excel(
                file_path, index=False, engine="openpyxl"
            )
        except Exception as exc:  # noqa: BLE001
            applogger.exception("XLSX export failed: %s", exc)

    def _update_selected_link(self) -> None:
        table = self._selected_table()
        if not table:
            return
        link = self._repo.get_table_link(table)
        if not link:
            return

        source = (link.get("settings") or {}).get("source") or {}
        password: str | None = None
        if source.get("kind") in ("postgres", "mysql"):
            entered, ok = QInputDialog.getText(
                self,
                _("Update link"),
                _("Password for {username}@{host}:").format(
                    username=source.get("username", ""), host=source.get("host", "")
                ),
                QLineEdit.EchoMode.Password,
            )
            if not ok:
                return
            password = entered

        try:
            refresh_link(self._repo, link_id=int(link["id"]), password=password)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Link refresh failed: %s", exc)
        self._reload_tables()

    def _rename_selected_table(self) -> None:
        """Rename the selected table - the same action TableListPanel's own
        context menu offers (see its _rename_table), reached from here too
        since this panel already tracks which table is selected."""
        table = self._selected_table()
        if not table:
            return
        new, ok = QInputDialog.getText(
            self,
            _("Rename table"),
            _("New name for table '{table}':").format(table=table),
            QLineEdit.EchoMode.Normal,
            table,
        )
        if not ok:
            return
        new = (new or "").strip()
        if not new or new == table:
            return
        try:
            self._repo.rename_table(table, new)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Rename failed: %s", exc)
            return
        self._reload_tables()
