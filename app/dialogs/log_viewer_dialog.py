from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import (
    QCloseEvent,
    QFontDatabase,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import LOG_FILE, clear_log_file
from app.styles.style import (
    apply_dialog_shell,
    CardFrame,
    create_action_button,
    create_section_title,
    load_icon,
    mark_destructive_button,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.utils.messages import ask


class LogViewerDialog(QDialog):
    def __init__(self, parent: QWidget):
        self._parent = parent

        super().__init__(parent)

        self.setWindowTitle(_("Application Log"))
        self.setWindowIcon(load_icon("log_viewer"))

        try:
            log_path = Path(LOG_FILE)

            if log_path.exists():
                log_text = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                line_count = log_text.count("\n")
                if log_text and not log_text.endswith("\n"):
                    line_count += 1
            else:
                log_text = f"Log file not found:\n{LOG_FILE}"
                line_count = 0

        except OSError as exc:
            log_text = f"Unable to read log file:\n\n{exc}"
            line_count = 0

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        card = CardFrame(self, "logViewerCard")
        card_layout = card.layout()

        card_layout.addWidget(
            create_section_title(_("Application Log"), card)
        )

        viewer = QPlainTextEdit(card)
        viewer.setObjectName("logViewerText")
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        viewer.setFont(
            QFontDatabase.systemFont(
                QFontDatabase.SystemFont.FixedFont
            )
        )
        viewer.setPlainText(log_text)
        viewer.moveCursor(
            viewer.textCursor().MoveOperation.End
        )

        mark_editor_panel(viewer)
        card_layout.addWidget(viewer, 1)

        caption = QLabel(
            f"{line_count} lines loaded from {LOG_FILE}",
            card,
        )
        caption.setProperty("muted", True)
        caption.setWordWrap(True)
        card_layout.addWidget(caption, 0)

        root.addWidget(card, 1)

        self._viewer = viewer
        self._caption = caption

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        # Red and on its own at the left, away from Copy and Close: the one
        # button here whose effect cannot be taken back.
        clear_button = create_action_button(
            parent=self,
            action_id="clear",
            action=self._clear_log,
            layout=action_row,
            presentation=(load_icon("clear"), _("Clear log"), _("Empty the application log")),
        )
        mark_destructive_button(clear_button)
        action_row.addStretch(1)

        create_action_button(
            parent=self,
            action_id="copy",
            action=lambda: QApplication.clipboard().setText(viewer.toPlainText()),
            layout=action_row,
        )

        # "Exit", not the catalogue's "Cancel": reading a log changes nothing,
        # so there is nothing to cancel.
        create_action_button(
            parent=self,
            action_id="close",
            action=self.accept,
            layout=action_row,
            presentation=(load_icon("close"), _("Exit"), _("Close the log")),
        )

        root.addLayout(action_row, 0)

    def _clear_log(self) -> None:
        """Empty the log file, after asking, and the viewer with it."""
        if not ask(self, "log.confirm_clear"):
            return
        try:
            clear_log_file()
        except OSError as exc:
            self._caption.setText(_("Could not clear the log: {reason}").format(reason=exc))
            return
        self._viewer.clear()
        self._caption.setText(_("The log was cleared."))

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
