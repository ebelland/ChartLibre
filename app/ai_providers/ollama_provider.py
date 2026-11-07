"""Ollama: the assistant answered by a model running on this computer (ollama.com), through Ollama's HTTP API.

Nothing to install but Ollama itself - the requests are the standard library's. The questions and the data
stay on the computer while the address is this computer's (no consent asked); an Ollama on another machine
of the network is data leaving it, and asks the consent as Claude does.

Two models, chosen in Settings > Assistant: the accurate one holds the conversation and runs the analyses,
the fast one (the same by default) routes each request in Automatic mode and answers the simple ones. The
model must call tools - llama3.1, qwen2.5, mistral-nemo... A small local model is far less reliable than
Claude at several steps: choosing an analysis, casting its columns, reading its report.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.plugin_api.ai.http_chat import (
    ProviderHTTPError,
    get_json,
    json_lines,
    post_json,
    stream_lines,
)
from app.plugin_api.ai.provider import (
    ACCURATE,
    FAST,
    AIProvider,
    ChatSession,
    ProviderOption,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
)

DEFAULT_ADDRESS = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
#: Tokens a conversation may hold, unless Settings says otherwise. Ollama's own default (4096) drops the oldest
#: messages without a word. More takes more memory: llama3.1 at 16384 needs 7 GB and swamps an 8 GB Mac.
DEFAULT_CONTEXT = "8192"

_ROUTER = (
    "You route requests for a data-analysis application. Reply with one word. SIMPLE: a direct question "
    "answerable with at most one look at a table, a short explanation, a title or a summary. COMPLEX: anything "
    "that needs several steps, an analysis to run, a choice among methods, or a change to the project."
)


def _arguments(raw: Any) -> dict[str, Any]:
    """A tool call's arguments: Ollama gives an object; some models write it as a JSON string."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def text_tool_calls(text: str, names: set[str]) -> list[tuple[str, dict[str, Any]]]:
    """Tool calls a small model wrote as text instead of making them: (name, arguments) for each one found.

    llama3.2 often answers with the call itself - {"name": "create_chart", "parameters": {...}}, sometimes in a
    code fence, sometimes several - and the request then ends with nothing done. Only a known tool's name counts.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    decoder = json.JSONDecoder()
    index = 0
    while (start := text.find("{", index)) != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        index = end
        if isinstance(value, dict) and value.get("function") and isinstance(value["function"], dict):
            value = value["function"]
        name = value.get("name") if isinstance(value, dict) else None
        if name in names:
            arguments = value.get("parameters", value.get("arguments", {}))
            found.append((str(name), _arguments(arguments)))
    return found


class OllamaSession(ChatSession):
    """One conversation: the history in /api/chat's form, sent whole on every turn."""

    def __init__(self, address: str, system: str, tools: Sequence[ToolSpec], context: int = 8192) -> None:
        self._context = context
        self._url = f"{address.rstrip('/')}/api/chat"
        self._tools = [{"type": "function", "function": {"name": spec.name, "description": spec.description,
                                                         "parameters": dict(spec.input_schema)}} for spec in tools]
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        #: The tool each call id names: a result goes back with its tool's name.
        self._calls: dict[str, str] = {}
        self._counter = 0

    def send(self, *, model: str, user_text: str | None = None, tool_results: Sequence[ToolResult] = (),
             on_text: Callable[[str], None], cancel: threading.Event) -> Turn:
        if user_text is not None:
            self._messages.append({"role": "user", "content": user_text})
        for result in tool_results:
            content = f"Error: {result.content}" if result.is_error else result.content
            self._messages.append({"role": "tool", "content": content, "tool_name": self._calls.get(result.id, "")})
        turn = Turn(model=model)
        calls: list[ToolCall] = []
        payload = {"model": model, "messages": self._messages, "tools": self._tools, "stream": True,
                   "options": {"num_ctx": self._context}}
        try:
            for item in json_lines(stream_lines(self._url, payload, cancel=cancel)):
                message = item.get("message") or {}
                text = str(message.get("content") or "")
                if text:
                    turn.text += text
                    on_text(text)
                for call in message.get("tool_calls") or ():
                    function = call.get("function") or {}
                    self._counter += 1
                    call_id = str(call.get("id") or f"call_{self._counter}")
                    calls.append(ToolCall(call_id, str(function.get("name") or ""),
                                          _arguments(function.get("arguments"))))
                if item.get("done"):
                    turn.input_tokens = int(item.get("prompt_eval_count") or 0)
                    turn.output_tokens = int(item.get("eval_count") or 0)
                    if item.get("done_reason") == "length" and calls:
                        turn.stop = "too_long"
        except ProviderHTTPError as error:
            raise RuntimeError(_readable(str(error), model)) from error
        if cancel.is_set():
            turn.stop = "stopped"
            return turn
        if turn.stop == "too_long":
            return turn  # a tool call cut short is not kept: its result would be missing from the next request
        if not calls:
            # A call written as text is made: the request goes on instead of ending with nothing done.
            known = {tool["function"]["name"] for tool in self._tools}
            for name, arguments in text_tool_calls(turn.text, known):
                self._counter += 1
                calls.append(ToolCall(f"call_{self._counter}", name, arguments))
        assistant: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if calls:
            assistant["tool_calls"] = [{"function": {"name": call.name, "arguments": dict(call.input)}}
                                       for call in calls]
            self._calls.update({call.id: call.name for call in calls})
            turn.stop, turn.tool_calls = "tools", calls
        self._messages.append(assistant)
        return turn


    # -- the form (app.ai.form): a small model fills one form instead of choosing tools -----------------------

    def fill(self, *, model: str, instructions: str, schema: Mapping[str, Any], text: str,
             cancel: threading.Event) -> dict[str, Any]:
        """The form *schema* filled for *text*: Ollama's structured output (format) keeps it to the schema."""
        del cancel  # one short, unstreamed answer
        payload = {"model": model, "stream": False, "format": dict(schema), "options": {"num_ctx": self._context},
                   "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": text}]}
        try:
            answer = post_json(self._url, payload, timeout=300)
        except ProviderHTTPError as error:
            raise RuntimeError(_readable(str(error), model)) from error
        return _arguments(str((answer.get("message") or {}).get("content") or "{}"))

    def say(self, *, model: str, text: str, on_text: Callable[[str], None], cancel: threading.Event) -> Turn:
        """An answer to *text* in words, no tools offered; kept in the conversation's history."""
        self._messages.append({"role": "user", "content": text})
        turn = Turn(model=model)
        payload = {"model": model, "messages": self._messages, "stream": True, "options": {"num_ctx": self._context}}
        try:
            for item in json_lines(stream_lines(self._url, payload, cancel=cancel)):
                piece = str((item.get("message") or {}).get("content") or "")
                if piece:
                    turn.text += piece
                    on_text(piece)
                if item.get("done"):
                    turn.input_tokens = int(item.get("prompt_eval_count") or 0)
                    turn.output_tokens = int(item.get("eval_count") or 0)
        except ProviderHTTPError as error:
            raise RuntimeError(_readable(str(error), model)) from error
        if cancel.is_set():
            turn.stop = "stopped"
        self._messages.append({"role": "assistant", "content": turn.text})
        return turn


def _readable(problem: str, model: str) -> str:
    if "not found" in problem and "model" in problem:
        return f"The model '{model}' is not installed in Ollama: ollama pull {model}"
    if "does not support tools" in problem:
        return f"The model '{model}' cannot call tools: choose one that can (llama3.1, qwen2.5, mistral-nemo)."
    return problem


class OllamaProvider(AIProvider):
    Name = "Ollama"
    SMALL_MODEL = True
    Description = "A model running on this computer through Ollama (ollama.com): nothing leaves the computer."
    OPTIONS = (
        ProviderOption("address", "Address", DEFAULT_ADDRESS,
                       "Where Ollama answers: this computer's by default; another machine's sends it the data."),
        ProviderOption("accurate_model", "Model", DEFAULT_MODEL,
                       "The conversation and the analyses: a model that calls tools (llama3.1, qwen2.5...)."),
        ProviderOption("context", "Context", DEFAULT_CONTEXT,
                       "Tokens a conversation may hold: more remembers more, and takes more memory."),
        ProviderOption("fast_model", "Fast model", "",
                       "Routes and answers the simple requests in Automatic mode; empty: the model above."),
    )

    @property
    def address(self) -> str:
        """Where Ollama answers; an address typed without its http:// gets it, so it can be read and reached."""
        address = self.option("address").strip() or DEFAULT_ADDRESS
        return address if "://" in address else f"http://{address}"

    @property
    def SENDS_DATA(self) -> bool:
        return (urllib.parse.urlparse(self.address).hostname or "") not in _LOCAL_HOSTS

    def unavailable(self) -> str:
        # Asked when the panel is shown: quickly, so it says so before a question is typed.
        try:
            get_json(f"{self.address.rstrip('/')}/api/version", timeout=2)
        except ProviderHTTPError:
            return (f"Ollama is not running at {self.address}: start it (ollama serve), or install it from "
                    "ollama.com.")
        return ""

    def models(self) -> Mapping[str, str]:
        accurate = self.option("accurate_model").strip() or DEFAULT_MODEL
        return {ACCURATE: accurate, FAST: self.option("fast_model").strip() or accurate}

    def installed(self) -> list[str]:
        """The models Ollama has (/api/tags)."""
        answer = get_json(f"{self.address.rstrip('/')}/api/tags")
        return sorted(str(model.get("name") or "") for model in answer.get("models") or ())

    def check(self, api_key: str) -> str:
        del api_key  # none: a local service
        try:
            names = self.installed()
        except ProviderHTTPError as error:
            return str(error)
        missing = [model for model in dict.fromkeys(self.models().values())
                   if model not in names and f"{model}:latest" not in names]
        if missing:
            return (f"Not installed in Ollama: {', '.join(missing)} - ollama pull {missing[0]}. Installed: "
                    f"{', '.join(names) or 'none'}.")
        return self._too_big()

    def _too_big(self) -> str:
        """A warning when the model will not fit beside the application in this computer's memory."""
        if self.SENDS_DATA:
            return ""  # another machine's memory
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            answer = get_json(f"{self.address.rstrip('/')}/api/tags")
        except (ValueError, OSError, AttributeError, ProviderHTTPError):
            return ""
        sizes = {str(model.get("name") or ""): int(model.get("size") or 0) for model in answer.get("models") or ()}
        accurate = self.models()[ACCURATE]
        size = sizes.get(accurate) or sizes.get(f"{accurate}:latest") or 0
        if size > memory * 0.4:
            return (f"'{accurate}' takes {size / 2**30:.1f} GB of this computer's {memory / 2**30:.0f} GB: everything "
                    "will slow down. Choose a smaller model (llama3.2).")
        return ""

    def classify(self, text: str, *, api_key: str) -> str:
        del api_key
        models = self.models()
        if models[FAST] == models[ACCURATE]:
            return ACCURATE  # one model: nothing to route
        try:
            answer = post_json(f"{self.address.rstrip('/')}/api/chat", {
                "model": models[FAST], "stream": False,
                "messages": [{"role": "system", "content": _ROUTER}, {"role": "user", "content": text}]},
                timeout=60)
        except ProviderHTTPError:
            return ACCURATE  # in doubt, the careful model
        word = str((answer.get("message") or {}).get("content") or "").upper()
        return FAST if "SIMPLE" in word and "COMPLEX" not in word else ACCURATE

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        del api_key
        try:
            context = max(2048, int(self.option("context")))
        except ValueError:
            context = int(DEFAULT_CONTEXT)
        return OllamaSession(self.address, system, tools, context)
