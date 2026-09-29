"""Machine translation for Edit Localization: what leaves, what comes back.

No network: every test hands translate_all a translator of its own.
"""
from __future__ import annotations

import threading

import pytest

from app.utils import machine_translation as mt


class Echo:
    """Translates by upper-casing, placeholders untouched - a good service."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def translate(self, text: str) -> str:
        self.asked.append(text)
        return text.upper()


def test_placeholders_do_not_reach_the_service_and_come_back() -> None:
    echo = Echo()
    run = mt.translate_all(
        ["Delete {count} rows of %s in <b>{table}</b>"], language="it", translator=echo
    )
    assert echo.asked == ["Delete {0} rows of {1} in {2}{3}{4}"]
    assert run.translations == {
        "Delete {count} rows of %s in <b>{table}</b>": "DELETE {count} ROWS OF %s IN <b>{table}</b>"
    }


def test_a_translation_that_loses_a_placeholder_is_discarded() -> None:
    class Loses:
        def translate(self, text: str) -> str:
            return "Elimina righe"

    run = mt.translate_all(["Delete {count} rows"], language="it", translator=Loses())
    assert run.translations == {}
    assert run.rejected == 1


def test_a_service_that_keeps_refusing_is_not_asked_again() -> None:
    class Refuses:
        calls = 0

        def translate(self, text: str) -> str:
            Refuses.calls += 1
            raise RuntimeError("Connection refused. Check the network")

    # One line each with a newline of its own, so each is a request of its own.
    run = mt.translate_all([f"text\n{i}" for i in range(10)], language="it", translator=Refuses())
    assert Refuses.calls == mt.MAX_CONSECUTIVE_FAILURES
    assert run.failed == 10
    assert run.error == "Connection refused"


def test_a_service_that_refuses_hands_over_to_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google answering "too many requests": one pause, then MyMemory."""
    monkeypatch.setattr(mt, "_RATE_LIMIT_PAUSE_S", 0.0)

    class TooMany:
        calls = 0

        def translate(self, text: str) -> str:
            TooMany.calls += 1
            raise RuntimeError("Server Error: You made too many requests to the server. Wait")

    services = {"google": TooMany(), "mymemory": Echo()}
    run = mt.translate_all(
        ["Open", "Close"], language="it", provider="google", fallbacks=["mymemory"],
        make_translator=lambda provider, language: services[provider],
    )
    # One block, one request - asked once more after the pause.
    assert TooMany.calls == 2
    assert run.translations == {"Open": "OPEN", "Close": "CLOSE"}
    assert run.switched == [("Google Translate", "MyMemory")]
    assert run.limited == ["Google Translate"]
    assert run.failed == 0


def test_a_quota_warning_is_never_kept_as_a_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mt, "_RATE_LIMIT_PAUSE_S", 0.0)

    class Quota:
        def translate(self, text: str) -> str:
            return "MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY"

    run = mt.translate_all(["Open", "Close"], language="it", provider="mymemory", translator=Quota())
    assert run.translations == {}
    assert run.failed == 2
    assert run.limited == ["MyMemory"]


def test_stop_ends_the_run_with_what_it_has() -> None:
    stop = threading.Event()

    class StopsAfterOne(Echo):
        def translate(self, text: str) -> str:
            stop.set()
            return super().translate(text)

    texts = ["one\n1", "two\n2", "three\n3"]
    run = mt.translate_all(texts, language="it", translator=StopsAfterOne(), cancel_event=stop)
    assert run.translations == {"one\n1": "ONE\n1"}
    assert run.stopped


def test_short_strings_travel_in_blocks_and_keep_their_spacing() -> None:
    echo = Echo()
    heard: list[tuple[str, str]] = []
    texts = [" cm", "Close", "Delete {count} rows"]
    run = mt.translate_all(texts, language="it", translator=echo, on_result=lambda s, t: heard.append((s, t)))
    assert echo.asked == [" cm\nClose\nDelete {0} rows"]
    assert run.translations == {" cm": " CM", "Close": "CLOSE", "Delete {count} rows": "DELETE {count} ROWS"}
    assert heard == list(run.translations.items())


def test_a_block_that_does_not_split_back_is_asked_line_by_line() -> None:
    class Merges(Echo):
        def translate(self, text: str) -> str:
            return super().translate(text).replace("\n", " ")

    service = Merges()
    run = mt.translate_all(["Open", "Close"], language="it", translator=service)
    assert service.asked == ["Open\nClose", "Open", "Close"]
    assert run.translations == {"Open": "OPEN", "Close": "CLOSE"}


def test_blocks_respect_the_size_limit() -> None:
    blocks = mt.make_blocks(["a" * 40, "b" * 40, "c" * 40, "multi\nline"], 90)
    assert [len(block) for block in blocks] == [2, 1, 1]


@pytest.mark.parametrize(
    ("codes", "language", "expected"),
    [
        (["it", "fr", "zh-CN", "zh-TW"], "it", "it"),
        (["it", "fr", "zh-CN", "zh-TW"], "zh", "zh-CN"),
        (["it-IT", "en-GB", "pt-PT"], "it", "it-IT"),
        (["it-IT", "en-GB"], "xx", None),
    ],
)
def test_each_service_gets_its_own_language_code(codes: list[str], language: str, expected: str | None) -> None:
    assert mt.provider_code(codes, language) == expected


def test_the_dialog_fills_only_empty_cells(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtCore import Qt

    import app.dialogs.edit_localization_dialog as dialog_module

    monkeypatch.setattr(mt, "available", lambda: True)
    monkeypatch.setattr(mt, "_make_translator", lambda provider, language: Echo())
    monkeypatch.setattr(dialog_module, "ask_before_auto_translate", lambda parent: True)
    shown: list[tuple[str, dict]] = []
    monkeypatch.setattr(dialog_module, "show_message", lambda parent, key, **kw: shown.append((key, kw)))

    dialog = dialog_module.EditLocalizationDialog()
    table = dialog._table
    rows = [
        row for row in range(table.rowCount())
        if not table.item(row, 1).text().strip()  # pyright: ignore[reportOptionalMemberAccess]
    ][:3]
    # Keep the run short: every other empty cell gets a hand translation.
    for row in range(table.rowCount()):
        cell = table.item(row, 1)
        if row not in rows and cell is not None and not cell.text().strip():
            cell.setText("typed")
    dialog._on_auto_translate_missing()
    task = dialog._translation_task
    assert task is not None
    for _ in range(200):
        qapp.processEvents()
        if dialog._translation_task is None:
            break
        threading.Event().wait(0.01)

    for row in rows:
        source = str(table.item(row, 0).data(Qt.ItemDataRole.UserRole))  # pyright: ignore[reportOptionalMemberAccess]
        if source.strip():
            assert table.item(row, 1).text() == mt.restore(*[  # pyright: ignore[reportOptionalMemberAccess]
                mt.protect(source)[0].upper(), mt.protect(source)[1]
            ])
    assert shown and shown[-1][0] == "dev.localization_auto_translated"
    dialog.close()
