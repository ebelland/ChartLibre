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
from collections.abc import Callable, Iterable, Sequence
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

#: How long to wait before asking again after "too many requests".
_RATE_LIMIT_PAUSE_S: float = 2.0

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
    #: Strings no service translated, or that were never reached.
    failed: int = 0
    #: The last error a service gave, for the message at the end.
    error: str = ""
    stopped: bool = False
    #: (service that refused, service that took over), in order.
    switched: list[tuple[str, str]] = field(default_factory=list)
    #: Services that refused for their request limit - Google's "too many
    #: requests", MyMemory's free daily quota - by name, in order.
    limited: list[str] = field(default_factory=list)


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
    fallbacks: Sequence[str] = (),
    cancel_event: threading.Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[str, str], None] | None = None,
    translator: Any = None,
    make_translator: Callable[[str, str], Any] | None = None,
) -> TranslationRun:
    """Translate every text in *texts* from English into *language*.

    In blocks (see :func:`make_blocks`), paced for the service. A block
    whose answer does not split back into as many lines as it sent is asked
    again one string at a time rather than guessed at. *on_result* hears
    each translation as it arrives, so a caller can show them - and keep
    them - before the run ends.

    Never raises for a service problem. A service that fails
    :data:`MAX_CONSECUTIVE_FAILURES` requests in a row - Google answering
    "too many requests" - is left, and what it did not translate goes to
    the next of *fallbacks*; with none left, the run reports what it has.
    *translator* and *make_translator* are for tests: anything with a
    ``translate(text) -> str`` method, and a (provider, language) factory.
    """
    pending = list(dict.fromkeys(text for text in texts if text.strip()))
    run = TranslationRun()
    factory = make_translator or _make_translator
    chain = [provider, *(name for name in fallbacks if name != provider)]
    names = dict(PROVIDERS)
    state = {"done": 0}

    def advance(count: int) -> None:
        state["done"] += count
        if progress is not None:
            progress(state["done"], len(pending))

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    todo = pending
    for position, name in enumerate(chain):
        if not todo or cancelled():
            break
        if position > 0:
            run.switched.append((names.get(chain[position - 1], chain[position - 1]), names.get(name, name)))
        try:
            service = translator if (position == 0 and translator is not None) else factory(name, language)
        except ValueError as exc:  # this service does not have the language
            run.error = str(exc)
            continue
        todo = _translate_with(service, name, todo, run, cancelled, advance, on_result)

    run.failed += len(todo)
    run.stopped = cancelled()
    if progress is not None:
        progress(len(pending), len(pending))
    return run


def _is_rate_limit(exc: BaseException) -> bool:
    return "too many requests" in str(exc).lower() or type(exc).__name__ == "TooManyRequests"


def _translate_with(
    translator: Any,
    provider: str,
    texts: list[str],
    run: TranslationRun,
    cancelled: Callable[[], bool],
    advance: Callable[[int], None],
    on_result: Callable[[str, str], None] | None,
) -> list[str]:
    """Translate *texts* with one service; return those it did not translate."""
    interval = _MIN_INTERVAL_S.get(provider, 0.0)
    name = dict(PROVIDERS).get(provider, provider)
    state = {"failures_in_a_row": 0, "last_request": 0.0}
    unanswered: list[str] = []

    def ask(masked: str) -> str | None:
        for attempt in (1, 2):
            wait = interval - (time.monotonic() - state["last_request"])
            if wait > 0:
                time.sleep(wait)
            state["last_request"] = time.monotonic()
            try:
                answer = str(translator.translate(masked) or "")
                # MyMemory can answer its quota warning as if it were the
                # translation; it must never reach the catalogue.
                if answer.lstrip().upper().startswith("MYMEMORY WARNING"):
                    raise RuntimeError("Too many requests: " + answer.strip()[:120])
            except Exception as exc:  # noqa: BLE001 - the service's problem, reported
                run.error = str(exc).split(".")[0][:160] or type(exc).__name__
                if _is_rate_limit(exc) and name not in run.limited:
                    # deep_translator words every 429 as Google's limit,
                    # MyMemory's daily quota included: say whose it was.
                    run.limited.append(name)
                # One pause and one more try, the first time a service says
                # it is being asked too often: a burst can trip the limit
                # without the service refusing for good.
                if attempt == 1 and state["failures_in_a_row"] == 0 and _is_rate_limit(exc):
                    time.sleep(_RATE_LIMIT_PAUSE_S)
                    continue
                state["failures_in_a_row"] += 1
                return None
            state["failures_in_a_row"] = 0
            return answer
        return None

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

    def giving_up() -> bool:
        return state["failures_in_a_row"] >= MAX_CONSECUTIVE_FAILURES or cancelled()

    def one_by_one(block: list[str]) -> None:
        for index, text in enumerate(block):
            if giving_up():
                unanswered.extend(block[index:])
                return
            masked, placeholders = protect(text)
            answer = ask(masked)
            if answer is None:
                unanswered.append(text)
            else:
                keep(text, answer, placeholders)
                advance(1)

    blocks = make_blocks(texts, _BLOCK_CHARS.get(provider, 450))
    for number, block in enumerate(blocks):
        if giving_up():
            unanswered.extend(text for rest in blocks[number:] for text in rest)
            break
        if len(block) == 1:
            one_by_one(block)
            continue
        protected = [protect(text) for text in block]
        answer = ask("\n".join(masked for masked, _placeholders in protected))
        lines = answer.split("\n") if answer is not None else []
        if answer is None:
            unanswered.extend(block)
        elif len(lines) == len(block):
            for text, (_masked, placeholders), line in zip(block, protected, lines):
                keep(text, line, placeholders)
            advance(len(block))
        else:
            one_by_one(block)
    return unanswered
