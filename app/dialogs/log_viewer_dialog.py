from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QCloseEvent, QFontDatabase
from PySide6.QtWidgets import QApplication, QLabel, QPlainTextEdit, QWidget

from app.logs.logger import LOG_FILE, clear_log_file
from app.dialogs.two_panels_dialog_base import PanelDialog
from app.styles.style import load_icon, mark_destructive_button, mark_editor_panel
from app.utils.i18n import _
from app.utils.messages import ask


class LogViewerDialog(PanelDialog):
    def __init__(self, parent: QWidget):
        self._parent = parent

        super().__init__(parent, title=_("Application Log"), icon=load_icon("log_viewer"), size="medium")

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

        card = self.add_card(_("Application Log"), "logViewerCard", stretch=1)
        viewer = QPlainTextEdit(card)
        viewer.setObjectName("logViewerText")
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        viewer.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        viewer.setPlainText(log_text)
        viewer.moveCursor(viewer.textCursor().MoveOperation.End)
        mark_editor_panel(viewer)
        card.layout().addWidget(viewer, 1)

        caption = QLabel(f"{line_count} lines loaded from {LOG_FILE}", card)
        caption.setProperty("muted", True)
        caption.setWordWrap(True)
        card.layout().addWidget(caption, 0)
        self._viewer = viewer
        self._caption = caption

        # Red and on its own at the left, away from Copy and Close: the one
        # button here whose effect cannot be taken back.
        clear_button = self.add_action(
            "clear", self._clear_log, left=True,
            presentation=(load_icon("clear"), _("Clear log"), _("Empty the application log")),
        )
        mark_destructive_button(clear_button)
        self.add_action("copy", lambda: QApplication.clipboard().setText(viewer.toPlainText()))
        # "Exit", not the catalogue's "Cancel": reading a log changes nothing,
        # so there is nothing to cancel.
        self.add_action("close", self.accept, presentation=(load_icon("close"), _("Exit"), _("Close the log")))

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
