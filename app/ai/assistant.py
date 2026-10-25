"""One conversation with the assistant: the model's turns off the GUI thread, the tools on it.

A question goes to the provider on a worker thread - first, in Automatic
mode, to the fast model to decide which model answers it. The text streams
in as it is written (kept in a buffer the GUI thread empties every 80 ms: a
worker touches no Qt object). When the model asks for tools, they are run
here, on the GUI thread - they read and change the project, and may ask the
user - and their results go back for the next turn, until the model is done.
"""
from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from app.ai import settings
from app.ai.tools import AssistantTools
from app.plugin_api.ai.provider import ACCURATE, FAST, AIProvider, ToolResult, Turn
from app.utils.background import BackgroundTask, run_in_background

#: Turns of tool calls one question may take before the assistant is stopped (a loop that does not end).
MAX_TOOL_TURNS = 25

SYSTEM = """You are the assistant inside ChartLibre, a desktop application for charts and statistical analysis of \
data kept in a SQLite project, in the manner of JMP.

- Answer in the language the user writes in.
- Use the tools to look at the data before you answer: never invent a table, a column or a number.
- Write your answers in Markdown: short paragraphs, tables where they help. Numbers with sensible precision.
- When you run an analysis, say what it found in plain words, then the key numbers.
- {permission}
- When a tool fails, say so and what you would try; do not pretend it worked."""

_PERMISSIONS = {
    settings.READ_ONLY: "You may only read the project: answer with what you find. If the user asks for an "
                        "analysis or a change, explain how to do it in the application instead.",
    settings.ANALYSES: "You may read the project, run analyses and draw charts; you may not change the data. If "
                       "the user asks for a change to the data, explain how to do it in the application.",
    settings.EVERYTHING: "You may read the project, run analyses, draw charts and change the data. Every change "
                         "is confirmed by the user before it runs and can be undone; describe each one exactly.",
}


class Assistant(QObject):
    """The conversation; its signals are the panel's."""

    #: Text of the answer being written, as it arrives.
    text_arrived = Signal(str)
    #: A tool was run: its name and what it gave back (shortened).
    tool_used = Signal(str, str)
    #: The answer is complete: the model that wrote it, the tokens of the question so far (in, out).
    answered = Signal(str, int, int)
    #: Why the question could not be answered.
    failed = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, provider: AIProvider, tools: AssistantTools, *, api_key: str, level: str, mode: str,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._provider = provider
        self._tools = tools
        self._api_key = api_key
        self._level = level
        self._mode = mode
        self._models = dict(provider.models())
        self._session = provider.start(api_key=api_key, system=SYSTEM.format(permission=_PERMISSIONS[level]),
                                       tools=tools.specs(level))
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(80)
        self._flush_timer.timeout.connect(self._flush)
        self._task: BackgroundTask | None = None
        self._tier = ACCURATE
        self._tool_turns = 0
        self._tokens = [0, 0]

    @property
    def busy(self) -> bool:
        return self._task is not None and self._task.running

    def ask(self, text: str) -> None:
        """A question: routed (Automatic), then answered."""
        if self.busy:
            return
        self._tool_turns, self._tokens = 0, [0, 0]
        self.busy_changed.emit(True)
        if self._mode == settings.AUTO:
            self._start(lambda _cancel: self._provider.classify(text, api_key=self._api_key),
                        lambda tier: self._send(tier=str(tier), user_text=text))
        else:
            self._send(tier=FAST if self._mode == FAST else ACCURATE, user_text=text)

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    # ------------------------------------------------------------------

    def _start(self, work: Any, then: Any) -> None:
        self._task = run_in_background(work, then, self._failed)

    def _send(self, *, tier: str, user_text: str | None = None, tool_results: tuple[ToolResult, ...] = ()) -> None:
        self._tier = tier if tier in self._models else ACCURATE
        model = self._models[self._tier]
        self._flush_timer.start()
        self._start(lambda cancel: self._session.send(model=model, user_text=user_text, tool_results=tool_results,
                                                       on_text=self._buffered, cancel=cancel),
                    self._on_turn)

    def _buffered(self, text: str) -> None:  # worker thread: no Qt here
        with self._lock:
            self._buffer.append(text)

    def _flush(self) -> None:
        with self._lock:
            text, self._buffer = "".join(self._buffer), []
        if text:
            self.text_arrived.emit(text)

    def _on_turn(self, turn: Turn) -> None:
        self._flush()
        self._tokens[0] += turn.input_tokens
        self._tokens[1] += turn.output_tokens
        if turn.stop == "tools":
            self._tool_turns += 1
            if self._tool_turns > MAX_TOOL_TURNS:
                self._end(failure="The assistant was stopped: too many steps for one question.")
                return
            results = []
            for call in turn.tool_calls:
                content, is_error = self._tools.run(call.name, call.input, self._level)
                results.append(ToolResult(call.id, content, is_error))
                self.tool_used.emit(call.name, content.splitlines()[0][:200] if content else "")
            self._send(tier=self._tier, tool_results=tuple(results))
            return
        messages = {"stopped": "Stopped.", "refused": "The model declined to answer this request.",
                    "too_long": "The answer was too long and was cut before a step could run."}
        if turn.stop in messages:
            self._end(failure=messages[turn.stop])
            return
        self._end(model=turn.model)

    def _failed(self, error: BaseException) -> None:
        self._flush()
        self._end(failure=str(error) or type(error).__name__)

    def _end(self, *, model: str = "", failure: str = "") -> None:
        self._flush_timer.stop()
        self._task = None
        if failure:
            self.failed.emit(failure)
        else:
            self.answered.emit(model, *self._tokens)
        self.busy_changed.emit(False)
