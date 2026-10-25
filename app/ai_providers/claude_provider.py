"""Claude, through Anthropic's Python SDK (``pip install anthropic``) - the assistant's provider shipped.

Claude Opus 5.5 holds the conversation and does the analysis work; Claude
Haiku 5.5 routes each request (is it a simple one?) and answers the simple
ones. Every turn is streamed, the system prompt and the tools - the same on
every message - are cached, and a request Opus declines on a safety ground is
answered by the fallback model Anthropic chooses (server-side fallbacks).

The SDK is imported only when a conversation starts: without it the
provider says how to install it, and the rest of the application is the same.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.plugin_api.ai.provider import ACCURATE, FAST, AIProvider, ChatSession, ToolCall, ToolResult, ToolSpec, Turn

OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"
#: A request Opus declines is answered by the model Anthropic picks for the refusal's category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Room for the longest answer; streamed, so no request timeout.
MAX_TOKENS = 64000

_ROUTER = (
    "You route requests for a data-analysis application. Reply with one word. SIMPLE: a direct question "
    "answerable with at most one look at a table, a short explanation, a title or a summary. COMPLEX: anything "
    "that needs several steps, an analysis to run, a choice among methods, or a change to the project."
)


def _sdk() -> Any:
    import anthropic

    return anthropic


def _readable(error: BaseException) -> str:
    """An SDK error in words the user can act on."""
    anthropic = _sdk()
    if isinstance(error, anthropic.AuthenticationError):
        return "The API key was refused: check it in Settings > Assistant."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "This API key may not use the model."
    if isinstance(error, anthropic.RateLimitError):
        return "Too many requests for now: wait a moment and ask again."
    if isinstance(error, anthropic.APIConnectionError):
        return "No connection to Anthropic: check the internet connection."
    if isinstance(error, anthropic.BadRequestError) and "credit balance" in str(error.message).lower():
        return "The account has no credit left: add some in Anthropic's Console (Billing)."
    if isinstance(error, anthropic.APIStatusError):
        return f"Anthropic answered with an error ({error.status_code}): {error.message}"
    return str(error)


class ClaudeSession(ChatSession):
    """One conversation: the history in the Messages API's form, sent whole on every turn."""

    def __init__(self, client: Any, system: str, tools: Sequence[ToolSpec]) -> None:
        self._client = client
        # Stable first: tools, then system, then the conversation - the prefix the cache reuses.
        self._system = system
        self._tools = [{"name": spec.name, "description": spec.description, "input_schema": dict(spec.input_schema),
                        "eager_input_streaming": True} for spec in tools]
        self._messages: list[dict[str, Any]] = []

    def send(self, *, model: str, user_text: str | None = None, tool_results: Sequence[ToolResult] = (),
             on_text: Callable[[str], None], cancel: threading.Event) -> Turn:
        anthropic = _sdk()
        if user_text is not None:
            self._messages.append({"role": "user", "content": user_text})
        if tool_results:
            self._messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": result.id, "content": result.content,
                 "is_error": result.is_error} for result in tool_results]})
        options: dict[str, Any] = {}
        if model == OPUS:
            options = {"betas": [FALLBACK_BETA], "fallbacks": "default", "output_config": {"effort": "medium"}}
        else:
            options = {"output_config": {"effort": "low"}}
        turn = Turn(model=model)
        unparsed = 0
        while True:
            try:
                with self._client.beta.messages.stream(
                    model=model, max_tokens=MAX_TOKENS, system=self._system, tools=self._tools,
                    messages=self._messages, cache_control={"type": "ephemeral"}, **options,
                ) as stream:
                    for event in stream:
                        if cancel.is_set():
                            turn.stop = "stopped"
                            return turn
                        if event.type == "text":
                            turn.text += event.text
                            on_text(event.text)
                    response = stream.get_final_message()
            except ValueError:
                # A tool's input the SDK could not parse at all: the turn is asked again (twice at most).
                unparsed += 1
                if unparsed > 2:
                    raise
                continue
            except anthropic.APIError as error:
                raise RuntimeError(_readable(error)) from error
            turn.input_tokens += int(response.usage.input_tokens or 0)
            turn.output_tokens += int(response.usage.output_tokens or 0)
            turn.model = str(response.model or model)
            if response.stop_reason == "pause_turn":
                self._messages.append({"role": "assistant", "content": response.content})
                continue
            calls = [block for block in response.content if block.type == "tool_use"]
            if response.stop_reason == "refusal":
                turn.stop = "refused"
            elif response.stop_reason == "max_tokens" and calls:
                turn.stop = "too_long"  # a tool input cut short is not run
            if turn.stop in ("refused", "too_long") and calls:
                return turn  # not kept: a tool call without its result would break the next request
            self._messages.append({"role": "assistant", "content": response.content})
            if calls and turn.stop == "done":
                turn.stop = "tools"
                turn.tool_calls = [ToolCall(block.id, block.name, block.input if isinstance(block.input, dict)
                                            else {}) for block in calls]
            return turn


class ClaudeProvider(AIProvider):
    Name = "Claude"
    Description = "Anthropic's Claude: Opus 5.5 for the conversation and the analyses, Haiku 5.5 for simple requests."
    KEY_NAME = "anthropic_api_key"

    def unavailable(self) -> str:
        try:
            _sdk()
        except ImportError:
            return "Install Anthropic's package to use Claude: pip install anthropic"
        return ""

    def models(self) -> Mapping[str, str]:
        return {ACCURATE: OPUS, FAST: HAIKU}

    def check(self, api_key: str) -> str:
        # A few tokens of Haiku: the key, the account's credit and the connection, all three.
        anthropic = _sdk()
        try:
            anthropic.Anthropic(api_key=api_key or None, max_retries=0, timeout=30).messages.create(
                model=HAIKU, max_tokens=8, messages=[{"role": "user", "content": "Reply with: OK"}])
        except anthropic.APIError as error:
            return _readable(error)
        except anthropic.AnthropicError as error:  # no key at all: the client says so before any request
            return str(error) or type(error).__name__
        return ""

    def classify(self, text: str, *, api_key: str) -> str:
        anthropic = _sdk()
        try:
            response = anthropic.Anthropic(api_key=api_key or None).messages.create(
                model=HAIKU, max_tokens=1024, output_config={"effort": "low"}, system=_ROUTER,
                messages=[{"role": "user", "content": text}],
            )
        except anthropic.APIError:
            return ACCURATE  # in doubt, the careful model
        answer = " ".join(block.text for block in response.content if block.type == "text").upper()
        return FAST if "SIMPLE" in answer and "COMPLEX" not in answer else ACCURATE

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        # No key given: the SDK's own resolution (ANTHROPIC_API_KEY, an `ant auth login` profile).
        return ClaudeSession(_sdk().Anthropic(api_key=api_key or None), system, tools)
