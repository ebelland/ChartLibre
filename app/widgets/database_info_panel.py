"""Database Info: path, size, tables and import links, in one place.

Everything here already existed somewhere - TableListPanel's own context
menu already exports a table and refreshes its link - this is a read-only
overview that puts the whole database's shape (how many tables, how big,
which ones are fed by a link) on screen at once, with those same actions
one click away for whichever row is selected, rather than requiring a
right-click per table.

A plain embeddable QWidget, not a QDialog: it sits inside MainWindow's
"Database" nav page (see main_window._create_database_page), alongside the
Query Builder / Optimize DB actions the same page offers, rather than
opening as its own modal window.
"""
from __future__ import annotations

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
    CardFrame,
    create_action_button,
    create_section_title,
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

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo

        root = QVBoxLayout(self)
        stdSizeAndlayout(root)

        self._card = CardFrame(self, "databaseInfoCard")
        self._card_layout = self._card.layout()
        self._card_layout.addWidget(create_section_title(_("Database Info"), self._card))

        self._form_host = QWidget(self._card)
        self._form_layout = QFormLayout(self._form_host)
        stdSizeAndlayout(self._form_layout)
        self._card_layout.addWidget(self._form_host)
        self._build_form()

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

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        self._export_csv_button = create_action_button(
            parent=self,
            action_id="export_csv",
            action=self._export_selected_csv,
            layout=action_row,
        )
        self._export_xlsx_button = create_action_button(
            parent=self,
            action_id="export_xlsx",
            action=self._export_selected_xlsx,
            layout=action_row,
        )
        self._update_link_button = create_action_button(
            parent=self,
            action_id="update_link",
            action=self._update_selected_link,
            layout=action_row,
            presentation=(
                load_icon("reload"),
                _("Update link"),
                _("Refresh the selected table from its import link"),
            ),
        )
        action_row.addStretch(1)
        self._card_layout.addLayout(action_row)

        root.addWidget(self._card, 1)

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
        """Repopulate the table list, keeping the current selection by name."""
        selected = self._selected_table()
        frame = self._repo.list_user_tables()

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
