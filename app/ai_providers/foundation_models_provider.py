"""Apple's on-device model (Foundation Models): the assistant answered by the Mac itself, with no key.

Python has no usable binding on this application's Python, so a small Swift helper
(foundation_models_helper.swift) talks to the framework: built with swiftc on first use into the user's cache
folder, one process a conversation, JSON lines on its stdin and stdout. Nothing leaves the computer.
"""
from __future__ import annotations

import hashlib
import json
import platform
import queue
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.plugin_api.ai.provider import ACCURATE, FAST, AIProvider, ChatSession, ToolCall, ToolResult, ToolSpec, Turn

MODEL = "apple-on-device"
_SOURCE = Path(__file__).with_name("foundation_models_helper.swift")
_BUILD_LOCK = threading.Lock()

_REASONS = {
    "device_not_eligible": "This Mac cannot run Apple Intelligence.",
    "apple_intelligence_off": "Turn on Apple Intelligence in System Settings to use this model.",
    "model_not_ready": "Apple's model is still downloading: try again later.",
}


def helper_path() -> Path:
    """The helper built from this source, named after it: a changed source is built again."""
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    return Path.home() / "Library" / "Caches" / "ChartLibre" / f"foundation_models_helper_{digest}"


def build_helper() -> Path:
    """The helper, compiled the first time; RuntimeError, in words to act on, when it cannot be."""
    target = helper_path()
    with _BUILD_LOCK:
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        building = target.with_suffix(".building")
        result = subprocess.run(
            ["swiftc", "-O", "-swift-version", "5", "-parse-as-library", str(_SOURCE), "-o", str(building)],
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("Could not build the helper for Apple's model: "
                               f"{result.stderr.strip()[-400:] or 'swiftc failed'}")
        building.replace(target)
    return target


def helper_command() -> list[str]:
    """How the helper is started (the tests give a script of their own)."""
    return [str(build_helper())]


class FoundationModelsSession(ChatSession):
    """One conversation: the framework keeps the history, in the helper's process."""

    def __init__(self, system: str, tools: Sequence[ToolSpec]) -> None:
        self._start = {"type": "start", "instructions": system,
                       "tools": [{"name": spec.name, "description": spec.description,
                                  "schema": dict(spec.input_schema)} for spec in tools]}
        self._process: subprocess.Popen[str] | None = None
        self._events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._turn = 0
        self._calls = 0

    def _write(self, message: Mapping[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            try:
                self._events.put(json.loads(line))
            except ValueError:
                continue
        self._events.put(None)

    def _ensure_started(self) -> None:
        if self._process is not None:
            return
        self._process = subprocess.Popen(helper_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        threading.Thread(target=self._read, daemon=True).start()
        self._write(self._start)

    def send(self, *, model: str, user_text: str | None = None, tool_results: Sequence[ToolResult] = (),
             on_text: Callable[[str], None], cancel: threading.Event) -> Turn:
        self._ensure_started()
        if user_text is not None:
            self._turn += 1
            self._write({"type": "user", "turn": self._turn, "text": user_text})
        for result in tool_results:
            self._write({"type": "tool_result", "content": result.content, "is_error": result.is_error})
        turn = Turn(model=model)
        while True:
            if cancel.is_set():
                self._write({"type": "cancel"})
                turn.stop = "stopped"
                return turn
            try:
                event = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is None:
                self._process = None
                raise RuntimeError("Apple's model stopped unexpectedly: ask again.")
            kind = event.get("type")
            if event.get("turn") not in (self._turn, 0):
                continue  # from a turn that was stopped
            if kind == "text":
                turn.text += str(event.get("text") or "")
                on_text(str(event.get("text") or ""))
            elif kind == "tool_call":
                self._calls += 1
                turn.stop = "tools"
                turn.tool_calls = [ToolCall(f"call_{self._calls}", str(event.get("name") or ""),
                                            event.get("input") if isinstance(event.get("input"), dict) else {})]
                return turn
            elif kind == "done":
                return turn
            elif kind == "error":
                if event.get("kind") == "refused":
                    turn.stop = "refused"
                    return turn
                if event.get("kind") == "context" or "maximum allowed" in str(event.get("message")):
                    raise RuntimeError("The conversation no longer fits Apple's model: start a new one.")
                raise RuntimeError(str(event.get("message") or "Apple's model failed."))


class FoundationModelsProvider(AIProvider):
    Name = "Apple Foundation Models"
    Description = "The model built into macOS (Apple Intelligence): no key, and nothing leaves the computer."
    SENDS_DATA = False
    # Measured: 4096 tokens in all, and the analyses' catalogue alone overflows them; a tool's free-form
    # roles reach the model as an empty object. Listing, describing and querying tables work.
    READ_ONLY = True
    MAX_RESULT = 1500

    def unavailable(self) -> str:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return "Apple's model needs a Mac with an Apple chip."
        try:
            major = int(platform.mac_ver()[0].split(".")[0])
        except ValueError:
            major = 0
        if major < 26:
            return "Apple's model needs macOS 26 or newer."
        if not helper_path().exists() and shutil.which("swiftc") is None:
            return "Install Apple's command line tools to use this model: xcode-select --install"
        return ""

    def check(self, api_key: str) -> str:
        del api_key  # none: on this computer
        try:
            result = subprocess.run([*helper_command(), "--check"], capture_output=True, text=True, timeout=30,
                                    check=False)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            return str(error)
        reason = result.stdout.strip()
        return _REASONS.get(reason, f"Apple's model is not available: {reason}") if reason else ""

    def models(self) -> Mapping[str, str]:
        return {ACCURATE: MODEL, FAST: MODEL}

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        del api_key
        return FoundationModelsSession(system, tools)
