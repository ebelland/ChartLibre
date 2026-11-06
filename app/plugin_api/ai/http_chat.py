"""A provider's HTTP, with the standard library only: a JSON request, and a streamed answer read line by line.

Ollama answers a chat as newline-delimited JSON; the OpenAI-style APIs (GitHub Models, a local server that
speaks it) as server-sent events, ``data: {...}`` lines ending with ``data: [DONE]``. Both are read here a
line at a time, so the Stop button is looked at between lines and the text reaches the panel as it comes.
No package to install: urllib, as the data sources already use.
"""
from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

#: Long enough for a local model's first token on a slow machine; streamed after that, so not a limit on the
#: answer's length.
TIMEOUT = 300


class ProviderHTTPError(RuntimeError):
    """The service answered with an error, or could not be reached - said in words the user can act on."""


def _request(url: str, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None) -> urllib.request.Request:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    return request


def _error(url: str, error: BaseException) -> ProviderHTTPError:
    if isinstance(error, urllib.error.HTTPError):
        try:
            detail = error.read().decode("utf-8", "replace").strip()
        except OSError:
            detail = ""
        try:
            parsed = json.loads(detail)
            detail = str(parsed.get("error", {}).get("message") if isinstance(parsed.get("error"), dict)
                         else parsed.get("error") or parsed.get("message") or detail)
        except (ValueError, AttributeError):
            pass
        return ProviderHTTPError(f"{url} answered {error.code}: {detail or error.reason}")
    reason = getattr(error, "reason", error)
    return ProviderHTTPError(f"No answer from {url}: {reason}")


def get_json(url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 10) -> Any:
    """GET *url* and read its JSON."""
    try:
        with urllib.request.urlopen(_request(url, None, headers), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _error(url, error) from error


def post_json(url: str, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None,
              timeout: float = TIMEOUT) -> Any:
    """POST *payload* to *url* and read its JSON answer (not streamed)."""
    try:
        with urllib.request.urlopen(_request(url, payload, headers), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _error(url, error) from error


def stream_lines(url: str, payload: Mapping[str, Any], *, cancel: threading.Event,
                 headers: Mapping[str, str] | None = None, timeout: float = TIMEOUT) -> Iterator[str]:
    """POST *payload* and yield the answer's lines as they arrive; stops early once *cancel* is set."""
    try:
        response = urllib.request.urlopen(_request(url, payload, headers), timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _error(url, error) from error
    with response:
        try:
            for raw in response:
                if cancel.is_set():
                    return
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    yield line
        except (OSError, http.client.HTTPException) as error:  # the service went away in mid-answer
            raise _error(url, error) from error


def json_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Ollama's newline-delimited JSON, an object a line."""
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            if item.get("error"):
                raise ProviderHTTPError(str(item["error"]))
            yield item


def server_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Server-sent events' ``data:`` objects, until ``data: [DONE]``."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            item = json.loads(data)
        except ValueError:
            continue
        if isinstance(item, dict):
            if item.get("error"):
                error = item["error"]
                raise ProviderHTTPError(str(error.get("message") if isinstance(error, dict) else error))
            yield item
