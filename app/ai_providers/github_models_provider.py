"""GitHub Copilot's models, through GitHub Models: the assistant answered with a GitHub account's token.

GitHub Models serves the models Copilot uses (OpenAI's, and others) behind an OpenAI-compatible chat API,
to a personal access token with the "models" permission (github.com/settings/tokens). The address and the
models are settings: the same provider talks to any OpenAI-compatible chat endpoint - Azure, a local server
that speaks it - with that service's key.

The questions and the data go to GitHub (consent asked). Requests are the standard library's: nothing to
install. The answer is streamed (server-sent events) and its tool calls are put together from their pieces.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.plugin_api.ai.http_chat import (
    ProviderHTTPError,
    post_json,
    server_events,
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

DEFAULT_ENDPOINT = "https://models.github.ai/inference/chat/completions"
DEFAULT_ACCURATE = "openai/gpt-4.1"
DEFAULT_FAST = "openai/gpt-4.1-mini"
#: The finish reasons, OpenAI's names, as a Turn's stop.
_STOPS = {"stop": "done", "tool_calls": "tools", "length": "too_long", "content_filter": "refused"}

_ROUTER = (
    "You route requests for a data-analysis application. Reply with one word. SIMPLE: a direct question "
    "answerable with at most one look at a table, a short explanation, a title or a summary. COMPLEX: anything "
    "that needs several steps, an analysis to run, a choice among methods, or a change to the project."
)


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def _readable(problem: str) -> str:
    if " 401" in problem or "Unauthorized" in problem or "Bad credentials" in problem:
        return "The token was refused: check it in Settings > Assistant (a GitHub token with the models permission)."
    if " 403" in problem:
        return "This token may not use the model: give it the models permission, or choose another model."
    if " 429" in problem:
        return "Too many requests for now (GitHub Models' free limits): wait a moment and ask again."
    return problem


class GitHubModelsSession(ChatSession):
    """One conversation: the history in the chat completions' form, sent whole on every turn."""

    def __init__(self, endpoint: str, api_key: str, system: str, tools: Sequence[ToolSpec]) -> None:
        self._endpoint, self._headers = endpoint, _headers(api_key)
        self._tools = [{"type": "function", "function": {"name": spec.name, "description": spec.description,
                                                         "parameters": dict(spec.input_schema)}} for spec in tools]
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def send(self, *, model: str, user_text: str | None = None, tool_results: Sequence[ToolResult] = (),
             on_text: Callable[[str], None], cancel: threading.Event) -> Turn:
        if user_text is not None:
            self._messages.append({"role": "user", "content": user_text})
        for result in tool_results:
            content = f"Error: {result.content}" if result.is_error else result.content
            self._messages.append({"role": "tool", "tool_call_id": result.id, "content": content})
        turn = Turn(model=model)
        #: The tool calls, by their index in the answer: id, name and the arguments' text as it streams in.
        pieces: dict[int, dict[str, str]] = {}
        finish = "stop"
        payload = {"model": model, "messages": self._messages, "tools": self._tools, "stream": True,
                   "stream_options": {"include_usage": True}}
        try:
            for item in server_events(stream_lines(self._endpoint, payload, cancel=cancel, headers=self._headers)):
                usage = item.get("usage") or {}
                if usage:
                    turn.input_tokens = int(usage.get("prompt_tokens") or 0)
                    turn.output_tokens = int(usage.get("completion_tokens") or 0)
                turn.model = str(item.get("model") or turn.model)
                for choice in item.get("choices") or ():
                    delta = choice.get("delta") or {}
                    text = str(delta.get("content") or "")
                    if text:
                        turn.text += text
                        on_text(text)
                    for call in delta.get("tool_calls") or ():
                        piece = pieces.setdefault(int(call.get("index") or 0), {"id": "", "name": "", "arguments": ""})
                        piece["id"] = str(call.get("id") or piece["id"])
                        function = call.get("function") or {}
                        piece["name"] += str(function.get("name") or "")
                        piece["arguments"] += str(function.get("arguments") or "")
                    finish = str(choice.get("finish_reason") or finish)
        except ProviderHTTPError as error:
            raise RuntimeError(_readable(str(error))) from error
        if cancel.is_set():
            turn.stop = "stopped"
            return turn
        turn.stop = _STOPS.get(finish, "done")
        if pieces and turn.stop == "done":
            turn.stop = "tools"  # some models say "stop" with their calls
        if turn.stop in ("too_long", "refused") and pieces:
            return turn  # not kept: a tool call without its result would break the next request
        assistant: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
        if pieces and turn.stop == "tools":
            calls = []
            for index in sorted(pieces):
                piece = pieces[index]
                call_id = piece["id"] or f"call_{index}"
                try:
                    arguments = json.loads(piece["arguments"] or "{}")
                except ValueError:
                    arguments = {}  # written badly: the tool says what is missing, and the model tries again
                calls.append(ToolCall(call_id, piece["name"], arguments if isinstance(arguments, dict) else {}))
            assistant["tool_calls"] = [{"id": call.id, "type": "function",
                                        "function": {"name": call.name, "arguments": pieces[index]["arguments"] or "{}"}}
                                       for index, call in zip(sorted(pieces), calls, strict=True)]
            turn.tool_calls = calls
        self._messages.append(assistant)
        return turn


class GitHubModelsProvider(AIProvider):
    Name = "GitHub Copilot"
    Description = ("GitHub Copilot's models through GitHub Models, with a GitHub token - or any OpenAI-compatible "
                   "chat endpoint.")
    KEY_NAME = "github_models_token"
    OPTIONS = (
        ProviderOption("endpoint", "Endpoint", DEFAULT_ENDPOINT,
                       "GitHub Models' chat completions; any OpenAI-compatible chat endpoint works the same."),
        ProviderOption("accurate_model", "Model", DEFAULT_ACCURATE,
                       "The conversation and the analyses, as GitHub Models names it (openai/gpt-4.1...)."),
        ProviderOption("fast_model", "Fast model", DEFAULT_FAST,
                       "Routes and answers the simple requests in Automatic mode."),
    )

    @property
    def endpoint(self) -> str:
        return self.option("endpoint").strip() or DEFAULT_ENDPOINT

    def models(self) -> Mapping[str, str]:
        accurate = self.option("accurate_model").strip() or DEFAULT_ACCURATE
        return {ACCURATE: accurate, FAST: self.option("fast_model").strip() or accurate}

    def check(self, api_key: str) -> str:
        # A few tokens of the fast model: the token, its permission and the connection, all three.
        try:
            post_json(self.endpoint, {"model": self.models()[FAST], "max_tokens": 4,
                                      "messages": [{"role": "user", "content": "Reply with: OK"}]},
                      headers=_headers(api_key), timeout=30)
        except ProviderHTTPError as error:
            return _readable(str(error))
        return ""

    def classify(self, text: str, *, api_key: str) -> str:
        models = self.models()
        if models[FAST] == models[ACCURATE]:
            return ACCURATE
        try:
            answer = post_json(self.endpoint, {"model": models[FAST], "max_tokens": 4, "messages": [
                {"role": "system", "content": _ROUTER}, {"role": "user", "content": text}]},
                headers=_headers(api_key), timeout=60)
        except ProviderHTTPError:
            return ACCURATE  # in doubt, the careful model
        choices = answer.get("choices") or [{}]
        word = str((choices[0].get("message") or {}).get("content") or "").upper()
        return FAST if "SIMPLE" in word and "COMPLEX" not in word else ACCURATE

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        return GitHubModelsSession(self.endpoint, api_key, system, tools)
