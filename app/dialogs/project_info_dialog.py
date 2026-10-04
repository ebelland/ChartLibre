"""Project info: who made the project, when, what it is about, its sources.

Edits the project's own ``__project_info__`` entries (see
app/data/repo/project_info.py): author, notes and bibliographic references.
The creation date and the preview are shown, not edited - the date is a fact
about the file, and the preview is redrawn from the first figure each time
the project is saved. The author typed here is also remembered for this
computer, so a new project starts with it already filled in.
"""
from __future__ import annotations

from datetime import datetime

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.repo.project_info import dump_references, parse_references
from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils.config import get_value, set_value
from app.utils.dialog_state import restore_window_geometry, save_window_geometry
from app.utils.i18n import _

#: user.json key: the author last typed, offered to the next new project.
AUTHOR_SETTING = "project_author"

_STATE_KEY = "project_info_dialog"


def readable_date(iso: str) -> str:
    """An ISO 8601 timestamp as local date and time, or the text unchanged."""
    try:
        moment = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def default_author() -> str:
    """The author remembered on this computer, or an empty string."""
    return str(get_value(AUTHOR_SETTING, "") or "")


class ProjectInfoDialog(QDialog):
    """Author, creation date, notes, references and preview of the open project."""

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.setWindowTitle(_("Project info"))
        self.setWindowIcon(load_icon("project_info"))
        info = repo.project_info()

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")
        card = CardFrame(self, "projectInfoCard")
        card_layout = card.layout()
        form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(create_section_title(_("Project"), card))

        self.file_label = QLabel(str(repo.db_path or ""), card)
        self.file_label.setProperty("muted", True)
        self.file_label.setWordWrap(True)
        form.addRow(_("File"), self.file_label)

        self.author_edit = QLineEdit(info.get("author", ""), card)
        self.author_edit.setPlaceholderText(_("Your name, or your group's"))
        form.addRow(_("Author"), self.author_edit)

        self.created_label = QLabel(readable_date(info.get("created", "")) or _("unknown"), card)
        self.created_label.setToolTip(_("When the project was created. A project older than this record "
                                        "shows the date its file was created."))
        form.addRow(_("Created"), self.created_label)

        self.notes_edit = QPlainTextEdit(info.get("notes", ""), card)
        self.notes_edit.setPlaceholderText(_("What the project is about, where the data came from, what is left to do..."))
        self.notes_edit.setMinimumHeight(90)
        form.addRow(_("Notes"), self.notes_edit)

        # The first figure, as the project's preview shows it.
        top = QHBoxLayout()
        stdSizeAndlayout(top)
        top.addLayout(form, 1)
        self.preview_label = QLabel(card)
        self.preview_label.setFixedSize(self.PREVIEW_SIZE)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setToolTip(_("The project's first figure, drawn again each time the project is saved."))
        self._show_preview(info.get("preview_path"))
        top.addWidget(self.preview_label, 0, Qt.AlignmentFlag.AlignTop)
        card_layout.addLayout(top)

        card_layout.addWidget(create_section_title(_("References"), card))
        self.references_table = QTableWidget(0, 3, card)
        self.references_table.setHorizontalHeaderLabels([_("Citation"), _("DOI"), _("Link")])
        self.references_table.setToolTip(_("The sources of the data and methods: a citation, and if it has them a DOI "
                                           "(10.xxxx/...) and a web link. Reports list them with their links."))
        header = self.references_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(1, 150)
        header.resizeSection(2, 150)
        self.references_table.verticalHeader().setVisible(False)
        self.references_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.references_table.setMinimumHeight(110)
        for entry in parse_references(info.get("references")):
            self._append_reference(entry)
        card_layout.addWidget(self.references_table, 1)
        reference_buttons = QHBoxLayout()
        stdSizeAndlayout(reference_buttons)
        create_action_button(parent=card, action_id="add", action=lambda: self._append_reference({}, edit=True),
                             layout=reference_buttons)
        create_action_button(parent=card, action_id="delete", action=self._remove_references, layout=reference_buttons)
        reference_buttons.addStretch(1)
        card_layout.addLayout(reference_buttons)
        root.addWidget(card, 1)

        buttons = QHBoxLayout()
        stdSizeAndlayout(buttons)
        buttons.addStretch(1)
        create_action_button(parent=self, action_id="apply", action=self.accept, layout=buttons)
        create_action_button(parent=self, action_id="close", action=self.reject, layout=buttons)
        root.addLayout(buttons, 0)
        restore_window_geometry(self, _STATE_KEY)

    #: The preview, at the size the Load demo dialog shows it, halved.
    PREVIEW_SIZE = QSize(200, 125)

    def _show_preview(self, entry: str | None) -> None:
        from app.utils.project_preview import resolve_preview

        path = resolve_preview(Path(str(self._repo.db_path)), entry) if self._repo.db_path else None
        picture = QPixmap(str(path)) if path is not None else QPixmap()
        if picture.isNull():
            self.preview_label.setText(_("No preview yet: it is drawn when the project is saved."))
            self.preview_label.setWordWrap(True)
            self.preview_label.setProperty("muted", True)
            return
        ratio = self.devicePixelRatioF()
        scaled = picture.scaled(self.PREVIEW_SIZE * ratio, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(ratio)
        self.preview_label.setPixmap(scaled)

    def _append_reference(self, entry: dict[str, str], *, edit: bool = False) -> None:
        row = self.references_table.rowCount()
        self.references_table.insertRow(row)
        for column, key in enumerate(("citation", "doi", "url")):
            self.references_table.setItem(row, column, QTableWidgetItem(entry.get(key, "")))
        if edit:
            self.references_table.setCurrentCell(row, 0)
            self.references_table.editItem(self.references_table.item(row, 0))

    def _remove_references(self) -> None:
        rows = sorted({index.row() for index in self.references_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.references_table.removeRow(row)

    def references(self) -> list[dict[str, str]]:
        """The references as typed, rows with no citation left out."""
        result = []
        for row in range(self.references_table.rowCount()):
            values = {}
            for column, key in enumerate(("citation", "doi", "url")):
                item = self.references_table.item(row, column)
                text = item.text().strip() if item is not None else ""
                if text:
                    values[key] = text
            if values.get("citation"):
                result.append(values)
        return result

    def accept(self) -> None:
        """Save the entries; close only once they are saved."""
        author = self.author_edit.text().strip()
        try:
            self._repo.set_project_info({
                "author": author,
                "notes": self.notes_edit.toPlainText(),
                "references": dump_references(self.references()),
            })
        except DatabaseError as exc:
            applogger.exception("Could not save the project info: %s", exc)
            return
        if author:
            set_value(AUTHOR_SETTING, author)
        save_window_geometry(self, _STATE_KEY)
        super().accept()

    def reject(self) -> None:
        save_window_geometry(self, _STATE_KEY)
        super().reject()
