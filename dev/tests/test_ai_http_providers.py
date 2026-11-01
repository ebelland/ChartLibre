"""Ollama and GitHub Copilot (GitHub Models), against a fake server on this computer - never the real ones.

The conversation streamed, a tool call and its result sent back, Stop, a server that is not running, a model
that is not installed, the routing, and whether the data leave the computer.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.ai_providers.github_models_provider import GitHubModelsProvider
from app.ai_providers.ollama_provider import OllamaProvider
from app.plugin_api.ai.provider import ACCURATE, FAST, ToolResult, ToolSpec

TOOLS = [ToolSpec("list_tables", "The project's tables.", {"type": "object", "properties": {}})]


class _Fake:
    """A server answering each POST with the next of *answers* (lines), recording what it was sent."""

    def __init__(self) -> None:
        self.answers: list[list[str]] = []
        self.received: list[dict] = []
        self.gets: dict[str, dict] = {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # quiet
                pass

            def do_GET(self) -> None:
                body = fake.gets.get(self.path)
                self.send_response(200 if body is not None else 404)
                self.end_headers()
                self.wfile.write(json.dumps(body or {"error": "not found"}).encode())

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                fake.received.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      **json.loads(self.rfile.read(length) or b"{}")})
                lines = fake.answers.pop(0) if fake.answers else []
                status = 404 if lines and lines[0].startswith("HTTP404") else 200
                self.send_response(status)
                self.end_headers()
                for line in lines[1:] if status == 404 else lines:
                    self.wfile.write((line + "\n").encode())
                    self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.address = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake() -> Iterator[_Fake]:
    server = _Fake()
    yield server
    server.close()


def _ollama(fake: _Fake, **options: str) -> OllamaProvider:
    provider = OllamaProvider()
    provider.overrides = {"address": fake.address, "accurate_model": "llama3.1", **options}
    return provider


def _chunk(content: str = "", calls: list | None = None, done: bool = False, **more) -> str:
    message = {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}
    return json.dumps({"message": message, "done": done, **more})


def test_ollama_streams_calls_a_tool_and_takes_its_result_back(fake: _Fake) -> None:
    fake.answers = [
        [_chunk("Let me "), _chunk("look.", [{"function": {"name": "list_tables", "arguments": {}}}]),
         _chunk(done=True, prompt_eval_count=30, eval_count=5)],
        [_chunk("Two tables."), _chunk(done=True)],
    ]
    session = _ollama(fake).start(api_key="", system="You help.", tools=TOOLS)
    seen: list[str] = []
    turn = session.send(model="llama3.1", user_text="What tables?", on_text=seen.append, cancel=threading.Event())
    assert seen == ["Let me ", "look."] and turn.stop == "tools"
    assert [(call.name, dict(call.input)) for call in turn.tool_calls] == [("list_tables", {})]
    assert (turn.input_tokens, turn.output_tokens) == (30, 5)
    first = fake.received[0]
    assert first["path"] == "/api/chat" and first["stream"] is True
    assert first["tools"][0]["function"]["name"] == "list_tables"
    assert first["messages"][0] == {"role": "system", "content": "You help."}

    turn = session.send(model="llama3.1", tool_results=[ToolResult(turn.tool_calls[0].id, "a, b")],
                        on_text=lambda _t: None, cancel=threading.Event())
    assert turn.stop == "done" and turn.text == "Two tables."
    assert fake.received[1]["messages"][-1] == {"role": "tool", "content": "a, b", "tool_name": "list_tables"}


def test_ollama_stop_ends_the_turn(fake: _Fake) -> None:
    fake.answers = [[_chunk("a"), _chunk("b"), _chunk(done=True)]]
    cancel = threading.Event()
    turn = _ollama(fake).start(api_key="", system="", tools=TOOLS).send(
        model="llama3.1", user_text="Hi", on_text=lambda _t: cancel.set(), cancel=cancel)
    assert turn.stop == "stopped" and turn.text == "a"


def test_ollama_not_running_and_a_model_not_installed_are_said(fake: _Fake) -> None:
    gone = OllamaProvider()
    gone.overrides = {"address": "http://127.0.0.1:9"}  # nothing listens there
    assert "Ollama is not running at http://127.0.0.1:9" in gone.unavailable()

    fake.gets = {"/api/version": {"version": "0.9"}, "/api/tags": {"models": [{"name": "qwen2.5:latest"}]}}
    provider = _ollama(fake)
    assert provider.unavailable() == ""
    assert provider.check("").startswith("Not installed in Ollama: llama3.1 - ollama pull llama3.1")
    assert _ollama(fake, accurate_model="qwen2.5").check("") == ""

    fake.answers = [["HTTP404", json.dumps({"error": "model 'llama3.1' not found"})]]
    with pytest.raises(RuntimeError, match="not installed in Ollama: ollama pull llama3.1"):
        provider.start(api_key="", system="", tools=TOOLS).send(
            model="llama3.1", user_text="Hi", on_text=lambda _t: None, cancel=threading.Event())


def test_ollama_routes_only_with_two_models_and_sends_data_only_off_this_computer(fake: _Fake) -> None:
    one = _ollama(fake)
    assert one.models() == {ACCURATE: "llama3.1", FAST: "llama3.1"}
    assert one.classify("hi", api_key="") == ACCURATE and not fake.received  # nothing asked
    fake.answers = [[json.dumps({"message": {"content": "SIMPLE"}, "done": True})]]
    assert _ollama(fake, fast_model="qwen2.5:0.5b").classify("How many rows?", api_key="") == FAST
    assert not one.SENDS_DATA
    remote = OllamaProvider()
    remote.overrides = {"address": "http://192.168.1.20:11434"}
    assert remote.SENDS_DATA  # another machine: the consent is asked


def _event(delta: dict | None = None, finish: str | None = None, **more) -> str:
    choice = {"index": 0, "delta": delta or {}, "finish_reason": finish}
    return "data: " + json.dumps({"choices": [choice], **more})


def _github(fake: _Fake) -> GitHubModelsProvider:
    provider = GitHubModelsProvider()
    provider.overrides = {"endpoint": f"{fake.address}/inference/chat/completions"}
    return provider


def test_github_models_puts_a_tool_call_together_from_its_pieces(fake: _Fake) -> None:
    fake.answers = [
        [_event({"content": "Checking"}),
         _event({"tool_calls": [{"index": 0, "id": "call_7", "function": {"name": "list_", "arguments": "{\"li"}}]}),
         _event({"tool_calls": [{"index": 0, "function": {"name": "tables", "arguments": "mit\": 3}"}}]}),
         _event(finish="tool_calls"),
         "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9}}),
         "data: [DONE]"],
        [_event({"content": "Done."}), _event(finish="stop"), "data: [DONE]"],
    ]
    session = _github(fake).start(api_key="ghp_test", system="You help.", tools=TOOLS)
    turn = session.send(model="openai/gpt-4.1", user_text="Tables?", on_text=lambda _t: None,
                        cancel=threading.Event())
    assert turn.stop == "tools" and turn.text == "Checking"
    assert [(call.id, call.name, dict(call.input)) for call in turn.tool_calls] == [("call_7", "list_tables", {"limit": 3})]
    assert (turn.input_tokens, turn.output_tokens) == (40, 9)
    sent = fake.received[0]
    assert sent["auth"] == "Bearer ghp_test" and sent["stream"] is True and sent["model"] == "openai/gpt-4.1"

    turn = session.send(model="openai/gpt-4.1", tool_results=[ToolResult("call_7", "a, b, c")],
                        on_text=lambda _t: None, cancel=threading.Event())
    assert turn.stop == "done" and turn.text == "Done."
    history = fake.received[1]["messages"]
    assert history[-2]["tool_calls"][0]["function"] == {"name": "list_tables", "arguments": "{\"limit\": 3}"}
    assert history[-1] == {"role": "tool", "tool_call_id": "call_7", "content": "a, b, c"}


def test_github_models_check_says_a_refused_token_in_words(fake: _Fake) -> None:
    fake.answers = [["HTTP404"]]
    assert _github(fake).check("bad").startswith(f"{fake.address}/inference/chat/completions answered 404")
    provider = GitHubModelsProvider()
    assert provider.KEY_NAME == "github_models_token" and provider.SENDS_DATA
    assert provider.models() == {ACCURATE: "openai/gpt-4.1", FAST: "openai/gpt-4.1-mini"}


def test_the_providers_options_are_kept_per_provider() -> None:
    assert OllamaProvider.option_key("address") == "assistant_ollama_address"
    assert GitHubModelsProvider.option_key("endpoint") == "assistant_github_copilot_endpoint"


def test_settings_shows_the_chosen_providers_own_rows_and_tries_them(qapp) -> None:
    """Ollama chosen: its address and models are shown (GitHub Copilot's hidden), and Try needs no key."""
    from app.dialogs.settings_dialog import SettingsDialog

    dialog = SettingsDialog()
    try:
        combo = dialog._assistant_provider
        combo.setCurrentIndex(combo.findData("Ollama"))
        form = dialog._assistant_form
        ollama, github = dialog._assistant_options["Ollama"], dialog._assistant_options["GitHub Copilot"]
        assert all(form.isRowVisible(edit) for edit, _value in ollama.values())
        assert not any(form.isRowVisible(edit) for edit, _value in github.values())
        assert dialog._assistant_check.isEnabled() and not dialog._assistant_key.isEnabled()
        assert ollama["address"][0].placeholderText() == "http://localhost:11434"
    finally:
        dialog.close()
