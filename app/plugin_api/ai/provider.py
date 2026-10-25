"""The assistant's providers: a plugin family, as the charts, the functions and the operations are.

A provider answers the Assistant panel's conversation through some AI service.
It knows that service's API and nothing of ChartLibre: the tools it is given
are described here (ToolSpec), the calls it asks for come back as ToolCalls,
and the application runs them - on its own thread, with its confirmations and
its Undo - then hands the results back. Claude is the one shipped
(app/ai_providers); another provider is a file beside it, or in the user's
own ai_providers folder, deriving from AIProvider.

A provider is optional: one whose package is not installed says so through
``unavailable`` and the panel explains how to enable it - nothing imports the
service's package before the panel is used.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The two kinds of work, and so the two models a provider offers: the conversation and the analysis work
#: ("accurate"), and the simple jobs - routing a request, a title, a summary ("fast").
ACCURATE, FAST = "accurate", "fast"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the assistant may call: its name, what it does, and its input as a JSON schema."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call the model asked for: the provider's id for it, the tool and its input."""

    id: str
    name: str
    input: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool gave back, as text, for the call *id*; *is_error* when it failed."""

    id: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class Turn:
    """One answer of the model: its text, the tool calls it asks for, why it stopped, the tokens it used."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: "done", "tools" (run tool_calls and continue), "stopped", "refused", "too_long".
    stop: str = "done"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class ChatSession:
    """One conversation with a provider: it keeps the history in the service's own form.

    ``send`` runs on a worker thread: it must not touch a Qt object or the project. It streams the text as it
    comes to *on_text* (from that thread) and checks *cancel* between chunks.
    """

    def send(
        self,
        *,
        model: str,
        user_text: str | None = None,
        tool_results: Sequence[ToolResult] = (),
        on_text: Callable[[str], None],
        cancel: threading.Event,
    ) -> Turn:
        """Add the user's message, or the results of the last turn's tool calls, and get the next turn from
        *model* - which may change from one message to the next (the routing), the history staying one."""
        raise NotImplementedError


class AIProvider:
    """An AI service the Assistant can talk to. Subclasses set Name and implement the methods below."""

    #: How the provider is listed in Settings.
    Name: str = ""
    Description: str = ""
    #: The account's key, kept by the system's keychain: the name it is filed under; "" when none is needed.
    KEY_NAME: str = ""
    #: Whether the questions and the data leave the computer: then nothing is sent before the user consents.
    SENDS_DATA: bool = True

    def unavailable(self) -> str:
        """Why the provider cannot be used here (a package missing...), or "" when it can."""
        return ""

    def check(self, api_key: str) -> str:
        """Try *api_key* with the smallest request the service answers - on a worker thread: "" when it works,
        else why not, in words the user can act on. The default: nothing to try."""
        del api_key
        return ""

    def models(self) -> Mapping[str, str]:
        """The model for each kind of work: {ACCURATE: ..., FAST: ...}."""
        raise NotImplementedError

    def classify(self, text: str, *, api_key: str) -> str:
        """FAST when *text* is a simple request (a direct question, one lookup), else ACCURATE - with the fast
        model, on a worker thread. The default: ACCURATE, always."""
        del text, api_key
        return ACCURATE

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        """A new conversation, with *system* as its instructions and *tools* to call."""
        raise NotImplementedError
