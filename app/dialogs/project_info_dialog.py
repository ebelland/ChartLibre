"""Project info: who made the project, when, and what it is about.

Edits the project's own ``__project_info__`` entries (see
app/data/repo/project_info.py). The creation date is shown, not edited - it
is a fact about the file. The author typed here is also remembered for this
computer, so a new project starts with it already filled in.
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

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
    """Author, creation date and notes of the open project."""

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.setWindowTitle(_("Project info"))
        self.setWindowIcon(load_icon("project_info"))
        info = repo.project_info()

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")
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
        self.notes_edit.setMinimumHeight(120)
        form.addRow(_("Notes"), self.notes_edit)
        card_layout.addLayout(form)
        root.addWidget(card, 1)

        buttons = QHBoxLayout()
        stdSizeAndlayout(buttons)
        buttons.addStretch(1)
        create_action_button(parent=self, action_id="apply", action=self.accept, layout=buttons)
        create_action_button(parent=self, action_id="close", action=self.reject, layout=buttons)
        root.addLayout(buttons, 0)
        restore_window_geometry(self, _STATE_KEY)

    def accept(self) -> None:
        """Save the entries; close only once they are saved."""
        author = self.author_edit.text().strip()
        try:
            self._repo.set_project_info({"author": author, "notes": self.notes_edit.toPlainText()})
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
