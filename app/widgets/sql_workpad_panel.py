"""The SQL workpad's panel: one statement run at a time, its rows under it, writes on a copy (app.data.sql_workpad)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QCompleter,
    QInputDialog,
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
        self._repo = repo
        self._last_select = ""
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
        # Table and column names completed as you type (Ctrl+Space shows them all).
        self._names = QStringListModel(self)
        self._completer = QCompleter(self._names, self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setWidget(self._command)
        self._completer.activated.connect(self._complete)
        self._command.textChanged.connect(self._offer_names)
        QShortcut(QKeySequence("Ctrl+Space"), self._command).activated.connect(lambda: self._offer_names(True))
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
        result_box = QWidget(self)
        result_column = QVBoxLayout(result_box)
        result_column.setContentsMargins(0, 0, 0, 0)
        result_column.addWidget(self._result, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._copy_button = QPushButton(_("Copy"), result_box)
        self._copy_button.setToolTip(_("Copy the rows as tab-separated text, for a spreadsheet"))
        self._copy_button.clicked.connect(self.copy_rows)
        buttons.addWidget(self._copy_button)
        self._save_query_button = QPushButton(_("Save as query…"), result_box)
        self._save_query_button.setToolTip(_("Keep this SELECT as a saved query of the project"))
        self._save_query_button.clicked.connect(self.save_as_query)
        buttons.addWidget(self._save_query_button)
        result_column.addLayout(buttons)
        self.add_frame(_("Result"), result_box, object_name="sqlWorkpadResultCard", stretch=1)
        self._message = QLabel(self)
        self._message.setWordWrap(True)
        self.add_frame(_("Message"), self._message, object_name="sqlWorkpadMessageCard")
        self._update_state()

    def _update_state(self) -> None:
        on_copy = self._session.on_copy
        # A query is saved on the project, so only one that reads the project itself (not the copy).
        self._save_query_button.setEnabled(bool(self._last_select) and not on_copy)
        self._copy_button.setEnabled(self._result.rowCount() > 0)
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
            self._last_select = text if columns else ""
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

    # ------------------------------------------------------------------ comfort
    def _word_before_cursor(self) -> str:
        cursor = self._command.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfWord, QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText()

    def _offer_names(self, everything: bool = False) -> None:
        prefix = self._word_before_cursor()
        if not everything and len(prefix) < 2:
            self._completer.popup().hide()
            return
        repo = self._repo()
        names: set[str] = set()
        if repo is not None:
            try:
                for table in repo.list_table_names():
                    names.add(table)
                    names.update(repo.get_columns(table))
            except Exception:
                return
        self._names.setStringList(sorted(names, key=str.lower))
        self._completer.setCompletionPrefix(prefix)
        if self._completer.completionCount() == 0 or (self._completer.completionCount() == 1
                                                      and self._completer.currentCompletion() == prefix):
            self._completer.popup().hide()
            return
        rect = self._command.cursorRect()
        rect.setWidth(220)
        self._completer.complete(rect)

    def _complete(self, name: str) -> None:
        cursor = self._command.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfWord, QTextCursor.MoveMode.KeepAnchor)
        quoted = name if name.isidentifier() else '"' + name.replace('"', '""') + '"'
        cursor.insertText(quoted)
        self._command.setTextCursor(cursor)

    def copy_rows(self) -> None:
        """The rows shown, with their header, as tab-separated text on the clipboard."""
        table = self._result
        lines = ["\t".join(table.horizontalHeaderItem(c).text() for c in range(table.columnCount()))]
        for r in range(table.rowCount()):
            lines.append("\t".join((table.item(r, c).text() if table.item(r, c) else "")
                                   for c in range(table.columnCount())))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def save_as_query(self) -> None:
        """The last SELECT run, kept as a saved query (Q in the Tables list)."""
        repo = self._repo()
        if repo is None or not self._last_select:
            return
        name, ok = QInputDialog.getText(self, _("Save as query"), _("Name of the query:"))
        if not ok or not name.strip():
            return
        try:
            repo.save_query(name.strip(), self._last_select)
        except Exception as error:
            self._show_message(f"{type(error).__name__}: {error}", "#dc2626")
            return
        self._show_message(_("Saved as query \"{name}\".").format(name=name.strip()), "")
        self.project_changed.emit([])

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
