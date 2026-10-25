"""The Assistant panel: a conversation with an AI about the project, in the navigation bar.

Two frames, as every panel: the conversation - the answers rendered from the
Markdown the model writes, the tools it used shown as they ran - and the
question, with Send, Stop, New conversation and the model (Automatic, Fast,
Accurate). What the assistant may do, what it may send and the key are
Settings > Assistant's; the panel says what is missing before anything is sent.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.ai import settings
from app.ai.assistant import Assistant
from app.ai.tools import AssistantTools
from app.styles.style import create_action_button, stdSizeAndlayout
from app.utils.i18n import _
from app.widgets.nav_bar import NavPanel


def _load_provider(name: str) -> Any:
    """The provider called *name* (Settings), an instance; None when there is none."""
    from app.scanners.ai_provider_scanner import ai_providers, provider_class

    entry = next((entry for entry in ai_providers if entry["value"] == name), ai_providers[0] if ai_providers else None)
    return provider_class(entry)() if entry is not None else None


class AssistantPanel(NavPanel):
    """The conversation and the question; the project through *repo()* and the window's callbacks."""

    #: The project changed (an analysis written, a chart, a column): the figures made, to show.
    project_changed = Signal(list)
    #: Settings > Assistant asked for (the key, the consent, the permissions).
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None, *, repo: Callable[[], Any],
                 current_figure: Callable[[], int | None]) -> None:
        super().__init__(parent)
        self._repo = repo
        self._current_figure = current_figure
        self._assistant: Assistant | None = None
        self._assistant_settings: tuple[Any, ...] | None = None
        #: The conversation as Markdown: what the panel shows, rendered again as text arrives.
        self._entries: list[str] = []
        self._answer = ""

        self._transcript = QTextBrowser(self)
        self._transcript.setOpenExternalLinks(True)
        self._transcript.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._transcript.setMinimumHeight(120)
        self.add_frame(_("Conversation"), self._transcript, object_name="assistantConversationCard", stretch=1)

        ask = QWidget(self)
        column = QVBoxLayout(ask)
        column.setContentsMargins(0, 0, 0, 0)
        self._note = QLabel(ask)
        self._note.setWordWrap(True)
        self._note.setProperty("muted", True)
        self._settings_button = QPushButton(_("Settings..."), ask)
        self._settings_button.clicked.connect(self.settings_requested.emit)
        note_row = QHBoxLayout()
        note_row.addWidget(self._note, 1)
        note_row.addWidget(self._settings_button)
        column.addLayout(note_row)
        self._question = QPlainTextEdit(ask)
        self._question.setPlaceholderText(_("Ask about the data - e.g. Is body mass different between species?"))
        self._question.setFixedHeight(self._question.fontMetrics().lineSpacing() * 4 + 12)
        column.addWidget(self._question)
        # Ctrl+Return - Cmd+Return on the Mac - sends, as in a mail.
        QShortcut(QKeySequence("Ctrl+Return"), self._question).activated.connect(self._send)
        row = QHBoxLayout()
        stdSizeAndlayout(row)
        self._mode_combo = QComboBox(ask)
        for key, label in settings.MODES:
            self._mode_combo.addItem(_(label), key)
        self._mode_combo.setToolTip(_("Automatic: the fast model for simple requests, the accurate one for the "
                                      "rest. Or always one of them."))
        self._mode_combo.setCurrentIndex(max(0, self._mode_combo.findData(settings.mode())))
        self._mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._mode_combo.setMinimumWidth(self._mode_combo.sizeHint().width())
        self._mode_combo.currentIndexChanged.connect(self._mode_changed)
        row.addWidget(self._mode_combo)
        row.addStretch(1)
        self._stop_button = create_action_button(parent=ask, action_id="operation_stop", action=self._stop,
                                                 layout=row)
        self._send_button = create_action_button(parent=ask, action_id="assistant_send", action=self._send,
                                                 layout=row)
        column.addLayout(row)
        # Below: start again, and what the last answer cost.
        under = QHBoxLayout()
        stdSizeAndlayout(under)
        self._new_button = create_action_button(parent=ask, action_id="assistant_new", action=self._new_conversation,
                                                layout=under)
        self._status = QLabel(ask)
        self._status.setProperty("muted", True)
        self._status.setWordWrap(True)
        under.addWidget(self._status, 1)
        column.addLayout(under)
        self.add_frame(_("Ask"), ask, object_name="assistantAskCard")
        self._set_busy(False)
        self.refresh()

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Say what is missing before the assistant can be used: the provider's package, the consent, the key."""
        provider = _load_provider(settings.provider_name())
        problem = ""
        if provider is None:
            problem = _("No assistant provider is installed.")
        elif provider.unavailable():
            problem = _(provider.unavailable())
        elif not settings.consented():
            problem = _("The assistant sends questions and data from the project to {provider}. Allow it in "
                        "Settings > Assistant.").format(provider=provider.Name)
        elif not settings.api_key(provider.KEY_NAME):
            problem = _("No API key: set it in Settings > Assistant (or in the environment).")
        level = dict(settings.PERMISSIONS)[settings.permission()]
        self._note.setText(problem or _("{provider} · {level}").format(
            provider=provider.Name if provider else "", level=_(level)))
        self._settings_button.setVisible(bool(problem))
        self._ready = not problem
        self._send_button.setEnabled(self._ready and not self._busy)

    def _conversation(self) -> Assistant:
        """The current conversation - a new one when the settings it was started with changed."""
        provider = _load_provider(settings.provider_name())
        wanted = (settings.provider_name(), settings.permission(), self._mode_combo.currentData(),
                  settings.api_key(provider.KEY_NAME), id(self._repo()))
        if self._assistant is None or wanted != self._assistant_settings:
            if self._assistant is not None:
                self._add_note(_("New conversation: the settings or the project changed."))
            tools = AssistantTools(self._repo(), confirm=self._confirm, changed=self.project_changed.emit,
                                   current_figure=self._current_figure)
            self._assistant = Assistant(provider, tools, api_key=wanted[3], level=wanted[1],
                                        mode=str(wanted[2]), parent=self)
            self._assistant.text_arrived.connect(self._text_arrived)
            self._assistant.tool_used.connect(self._tool_used)
            self._assistant.answered.connect(self._answered)
            self._assistant.failed.connect(self._failed)
            self._assistant.busy_changed.connect(self._set_busy)
            self._assistant_settings = wanted
        return self._assistant

    def _send(self) -> None:
        text = self._question.toPlainText().strip()
        if not text or self._busy:
            return
        self.refresh()
        if not self._ready:
            return
        self._question.clear()
        self._entries.append(f"**{_('You')}:** {text}")
        self._answer = ""
        self._render()
        self._status.setText(_("Thinking..."))
        self._conversation().ask(text)

    def _stop(self) -> None:
        if self._assistant is not None:
            self._assistant.stop()

    def _new_conversation(self) -> None:
        self._stop()
        self._assistant, self._assistant_settings = None, None
        self._entries, self._answer = [], ""
        self._render()
        self._status.clear()

    def _mode_changed(self) -> None:
        settings.save(provider=settings.provider_name(), permission_level=settings.permission(),
                      consent=settings.consented(), rows=settings.sample_rows(),
                      assistant_mode=str(self._mode_combo.currentData()))

    # ------------------------------------------------------------------

    def _text_arrived(self, text: str) -> None:
        self._answer += text
        self._render()

    def _tool_used(self, name: str, summary: str) -> None:
        self._close_answer()
        self._entries.append(f"> *{name}* - {summary}")
        self._render()

    def _answered(self, model: str, tokens_in: int, tokens_out: int) -> None:
        self._close_answer()
        self._render()
        self._status.setText(_("{model} · {tokens_in} tokens in, {tokens_out} out").format(
            model=model, tokens_in=tokens_in, tokens_out=tokens_out))

    def _failed(self, message: str) -> None:
        self._close_answer()
        self._add_note(_(message))
        self._status.clear()

    def _close_answer(self) -> None:
        if self._answer.strip():
            self._entries.append(self._answer)
        self._answer = ""

    def _add_note(self, text: str) -> None:
        self._entries.append(f"*{text}*")
        self._render()

    def _render(self) -> None:
        self._transcript.setMarkdown("\n\n---\n\n".join([*self._entries, *([self._answer] if self._answer else [])]))
        bar = self._transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._stop_button.setVisible(busy)
        self._send_button.setEnabled(getattr(self, "_ready", False) and not busy)
        self._question.setReadOnly(busy)

    def _confirm(self, question: str) -> bool:
        """A change the assistant wants to make: the user says yes before it runs."""
        answer = QMessageBox.question(self, _("Assistant"), _("The assistant wants to change the project:")
                                      + "\n\n" + question, QMessageBox.StandardButton.Yes
                                      | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes
