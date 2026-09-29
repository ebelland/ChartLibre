"""Machine translation of catalogue strings, through ``deep_translator``.

Edit Localization's "Translate missing (auto)" fills empty translations
with a first draft to review. ``deep_translator`` is a soft dependency:
the editor works fully without it, just without that one button.

Two services, both free and without an API key: Google Translate, the
better translation, and MyMemory, which answers when Google does not -
Google turns away addresses that ask too often, and says so with an HTTP
429 on the very first request.

What makes a catalogue string different from prose is its placeholders.
``{count}``, ``%s`` and ``<b>`` are code, not words: a service translates
``{count}`` into ``{conteggio}`` and the running application then fails
to format the string. So each placeholder is swapped for a numbered token
before the text leaves (``{0}``, ``{1}`` - digits survive every service
tried), put back afterwards, and a translation that lost or invented one is
thrown away rather than saved: an empty translation falls back to the
English text, a broken one crashes a dialog.

No Qt here: the dialog runs :func:`translate_all` on a worker thread.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

#: The services offered, as (key, name shown in the dialog).
PROVIDERS: tuple[tuple[str, str], ...] = (
    ("google", "Google Translate"),
    ("mymemory", "MyMemory"),
)

#: Google allows five requests a second; this stays under it.
_MIN_INTERVAL_S: dict[str, float] = {"google": 0.25, "mymemory": 0.0}

#: Most characters one request carries. MyMemory's free tier takes 500 a
#: request; Google takes 5000.
_BLOCK_CHARS: dict[str, int] = {"google": 4000, "mymemory": 450}

#: Consecutive failures after which the service is taken to be refusing
#: (rate limit, no network) and the run stops instead of asking again.
MAX_CONSECUTIVE_FAILURES: int = 3

#: A placeholder: a str.format field, a %-conversion, or an HTML tag.
_PLACEHOLDER_RE = re.compile(
    r"\{[^{}]*\}"                                   # {count}, {0}, {value:.2f}
    r"|%(?:\([^)]*\))?[-+ #0]*\d*(?:\.\d+)?[sdifrxXeEgGc%]"  # %s, %(name)s, %.2f
    r"|</?[A-Za-z][^<>]*>"                          # <b>, </a>, <br/>
)
_TOKEN_RE = re.compile(r"\{\s*(\d+)\s*\}")


def available() -> bool:
    """True when ``deep_translator`` is installed."""
    try:
        import deep_translator  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def protect(text: str) -> tuple[str, list[str]]:
    """Return *text* with its placeholders swapped for ``{0}``, ``{1}``…"""
    placeholders: list[str] = []

    def swap(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return "{" + str(len(placeholders) - 1) + "}"

    return _PLACEHOLDER_RE.sub(swap, text), placeholders


def restore(translated: str, placeholders: list[str]) -> str | None:
    """Put the placeholders back, or None if the service lost or added one."""
    found = [int(index) for index in _TOKEN_RE.findall(translated)]
    if sorted(found) != list(range(len(placeholders))):
        return None
    return _TOKEN_RE.sub(lambda match: placeholders[int(match.group(1))], translated)


def provider_code(codes: Iterable[str], language: str) -> str | None:
    """Return the service's code for a catalogue's two-letter *language*.

    Each service spells languages its own way - "it" for Google, "it-IT"
    for MyMemory, "zh-CN" for Chinese on both - so take the exact code if
    the service has it, else the first regional one it starts.
    """
    wanted = language.lower()
    listed = list(codes)
    exact = next((code for code in listed if code.lower() == wanted), None)
    if exact is not None:
        return exact
    return next((code for code in listed if code.lower().startswith(wanted + "-")), None)


def _make_translator(provider: str, language: str) -> Any:
    """Return a deep_translator translator from English to *language*."""
    from deep_translator import (  # pyright: ignore[reportMissingImports]
        GoogleTranslator,
        MyMemoryTranslator,
    )
    from deep_translator.constants import (  # pyright: ignore[reportMissingImports]
        GOOGLE_LANGUAGES_TO_CODES,
        MY_MEMORY_LANGUAGES_TO_CODES,
    )

    if provider == "mymemory":
        kind, codes, source = MyMemoryTranslator, MY_MEMORY_LANGUAGES_TO_CODES, "en-GB"
    else:
        kind, codes, source = GoogleTranslator, GOOGLE_LANGUAGES_TO_CODES, "en"
    target = provider_code(codes.values(), language)
    if target is None:
        raise ValueError(f"{dict(PROVIDERS).get(provider, provider)} does not translate into {language!r}")
    return kind(source=source, target=target)


@dataclass(slots=True)
class TranslationRun:
    """What a run produced, and why it stopped short if it did."""

    translations: dict[str, str] = field(default_factory=dict)
    #: Strings the service translated but whose placeholders did not survive.
    rejected: int = 0
    #: Strings the service did not translate at all, or never got to.
    failed: int = 0
    #: The last error the service gave, for the message at the end.
    error: str = ""
    stopped: bool = False


def make_blocks(texts: list[str], limit: int) -> list[list[str]]:
    """Group *texts* into blocks of at most *limit* characters, one per line.

    A block travels as one request, its strings joined by newlines, which
    is what makes a long catalogue take minutes instead of the better part
    of an hour. A string with a newline of its own travels alone: the
    newlines are how the answer is split back into strings.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if "\n" in text or len(text) >= limit:
            if current:
                blocks.append(current)
                current, size = [], 0
            blocks.append([text])
            continue
        if current and size + len(text) + 1 > limit:
            blocks.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text) + 1
    if current:
        blocks.append(current)
    return blocks


def translate_all(
    texts: Iterable[str],
    *,
    language: str,
    provider: str = "google",
    cancel_event: threading.Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[str, str], None] | None = None,
    translator: Any = None,
) -> TranslationRun:
    """Translate every text in *texts* from English into *language*.

    In blocks (see :func:`make_blocks`), paced for the service. A block
    whose answer does not split back into as many lines as it sent is asked
    again one string at a time rather than guessed at. *on_result* hears
    each translation as it arrives, so a caller can show them - and keep
    them - before the run ends. Never raises for a service problem: stops
    after :data:`MAX_CONSECUTIVE_FAILURES` failed requests in a row and
    reports what it has. *translator* is for tests - anything with a
    ``translate(text) -> str`` method.
    """
    pending = list(dict.fromkeys(text for text in texts if text.strip()))
    run = TranslationRun()
    if translator is None:
        translator = _make_translator(provider, language)
    interval = _MIN_INTERVAL_S.get(provider, 0.0)
    state = {"failures_in_a_row": 0, "last_request": 0.0}

    def ask(masked: str) -> str | None:
        wait = interval - (time.monotonic() - state["last_request"])
        if wait > 0:
            time.sleep(wait)
        state["last_request"] = time.monotonic()
        try:
            answer = str(translator.translate(masked) or "")
        except Exception as exc:  # noqa: BLE001 - the service's problem, reported
            run.error = str(exc).split(".")[0][:160] or type(exc).__name__
            state["failures_in_a_row"] += 1
            return None
        state["failures_in_a_row"] = 0
        return answer

    def keep(text: str, translated: str, placeholders: list[str]) -> None:
        restored = restore(translated.strip(), placeholders)
        if restored is None or not restored.strip():
            run.rejected += 1
            return
        # Services trim; the catalogue's spacing is part of the string
        # (" cm" is a suffix after a number).
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        restored = lead + restored.strip() + trail
        run.translations[text] = restored
        if on_result is not None:
            on_result(text, restored)

    def one_by_one(block: list[str]) -> None:
        for text in block:
            if giving_up():
                return
            masked, placeholders = protect(text)
            answer = ask(masked)
            if answer is None:
                run.failed += 1
            else:
                keep(text, answer, placeholders)

    def giving_up() -> bool:
        return (
            state["failures_in_a_row"] >= MAX_CONSECUTIVE_FAILURES
            or (cancel_event is not None and cancel_event.is_set())
        )

    done = 0
    for block in make_blocks(pending, _BLOCK_CHARS.get(provider, 450)):
        if giving_up():
            break
        if progress is not None:
            progress(done, len(pending))
        protected = [protect(text) for text in block]
        before = len(run.translations) + run.rejected + run.failed
        if len(block) == 1:
            one_by_one(block)
        else:
            answer = ask("\n".join(masked for masked, _placeholders in protected))
            lines = answer.split("\n") if answer is not None else []
            if answer is None:
                run.failed += len(block)
            elif len(lines) == len(block):
                for text, (_masked, placeholders), line in zip(block, protected, lines):
                    keep(text, line, placeholders)
            else:
                one_by_one(block)
        done += len(run.translations) + run.rejected + run.failed - before

    run.stopped = cancel_event is not None and cancel_event.is_set()
    run.failed += len(pending) - done
    if progress is not None:
        progress(len(pending), len(pending))
    return run
