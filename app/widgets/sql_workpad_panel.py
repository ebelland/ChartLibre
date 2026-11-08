"""The SQL workpad's panel: one statement run at a time, its rows under it, writes on a copy (app.data.sql_workpad)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import _
from app.widgets.nav_bar import NavPanel
from app.data.sql_workpad import RefusedStatement, SqlSession


class SqlWorkpadPanel(NavPanel):
    """SQL on the project *repo()*; changes are kept only with Apply."""

    #: The project changed (Apply): the window reloads its tables and charts.
    project_changed = Signal(list)

    def __init__(self, parent: QWidget | None, *, repo: Callable[[], Any]) -> None:
        super().__init__(parent)
        self._session = SqlSession(repo)
        self._history: list[str] = []
        self._position = 0

        box = QWidget(self)
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self._command = QPlainTextEdit(box)
        self._command.setPlaceholderText(_("e.g. SELECT species, AVG(body_mass_g) FROM Penguins GROUP BY species"))
        self._command.setFixedHeight(self._command.fontMetrics().lineSpacing() * 5 + 12)
        column.addWidget(self._command)
        QShortcut(QKeySequence("Ctrl+Return"), self._command).activated.connect(self.run)
        QShortcut(QKeySequence("Ctrl+Up"), self._command).activated.connect(lambda: self._recall(-1))
        QShortcut(QKeySequence("Ctrl+Down"), self._command).activated.connect(lambda: self._recall(1))
        row = QHBoxLayout()
        self._status = QLabel(box)
        self._status.setWordWrap(True)
        row.addWidget(self._status, 1)
        self._cancel = QPushButton(_("Cancel"), box)
        self._cancel.clicked.connect(self.cancel)
        row.addWidget(self._cancel)
        self._apply = QPushButton(_("Apply"), box)
        self._apply.clicked.connect(self.apply)
        row.addWidget(self._apply)
        run = QPushButton(_("Run"), box)
        run.setDefault(True)
        run.clicked.connect(self.run)
        row.addWidget(run)
        column.addLayout(row)
        self.add_frame(_("SQL"), box, object_name="sqlWorkpadCommandCard")

        self._result = QTableWidget(self)
        self._result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._result.setMinimumHeight(120)
        self._result.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.add_frame(_("Result"), self._result, object_name="sqlWorkpadResultCard", stretch=1)
        self._message = QLabel(self)
        self._message.setWordWrap(True)
        self.add_frame(_("Message"), self._message, object_name="sqlWorkpadMessageCard")
        self._update_state()

    def _update_state(self) -> None:
        on_copy = self._session.on_copy
        self._apply.setEnabled(on_copy)
        self._cancel.setEnabled(on_copy)
        self._status.setText(
            f"<span style='color:#b45309;'>{_('Working on a copy - Apply to keep, Cancel to throw away')}</span>"
            if on_copy else f"<span style='color:#6b7280;'>{_('One statement per Run · Ctrl+Return')}</span>")

    def _recall(self, step: int) -> None:
        if not self._history:
            return
        self._position = max(0, min(len(self._history), self._position + step))
        text = self._history[self._position] if self._position < len(self._history) else ""
        self._command.setPlainText(text)

    def run(self, source: str | None = None) -> None:
        """Run the statement typed (or *source*) and show its rows."""
        text = (source if isinstance(source, str) else self._command.toPlainText()).strip()
        if not text:
            return
        self._history.append(text)
        self._position = len(self._history)
        try:
            columns, rows, message = self._session.run(text)
        except RefusedStatement as error:
            self._show_message(str(error), "#b45309")
        except Exception as error:  # the user's SQL: its error is the answer
            self._show_message(f"{type(error).__name__}: {error}", "#dc2626")
        else:
            self._show_rows(columns, rows)
            self._show_message(message, "")
        self._update_state()

    def _show_message(self, text: str, colour: str) -> None:
        self._message.setStyleSheet(f"color:{colour};" if colour else "")
        self._message.setText(text)

    def _show_rows(self, columns: list[str], rows: list[tuple]) -> None:
        self._result.clear()
        self._result.setColumnCount(len(columns))
        self._result.setRowCount(len(rows))
        self._result.setHorizontalHeaderLabels(columns)
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                self._result.setItem(r, c, QTableWidgetItem("" if value is None else str(value)))

    def apply(self) -> None:
        """Keep the copy's changes in the project (one Undo entry)."""
        try:
            applied = self._session.apply()
        except Exception as error:
            self._show_message(f"{type(error).__name__}: {error}", "#dc2626")
            return
        if applied:
            self._show_message(_("Applied. Undo takes the whole of it back."), "")
            self.project_changed.emit([])
        self._update_state()

    def cancel(self) -> None:
        """Throw the copy away."""
        self._session.cancel()
        self._show_message(_("The copy was thrown away; the project is as it was."), "")
        self._update_state()

    def discard(self) -> None:
        """The project is closing: the copy goes, unasked."""
        self._session.cancel()
        self._update_state()
