"""Machine translation for Edit Localization: what leaves, what comes back.

No network: every test hands translate_all a translator of its own.
"""
from __future__ import annotations

import io
import json
import threading
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from app.utils import machine_translation as mt
from dev.tests._cases import check_all


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


_CASES_EACH_SERVICE_GETS_ITS_OWN_LANGUAGE_CODE = [
        (["it", "fr", "zh-CN", "zh-TW"], "it", "it"),
        (["it", "fr", "zh-CN", "zh-TW"], "zh", "zh-CN"),
        (["it-IT", "en-GB", "pt-PT"], "it", "it-IT"),
        (["it-IT", "en-GB"], "xx", None),
    ]


def _each_service_gets_its_own_language_code(codes: list[str], language: str, expected: str | None) -> None:
    assert mt.provider_code(codes, language) == expected


def test_each_service_gets_its_own_language_code() -> None:
    check_all(_each_service_gets_its_own_language_code, _CASES_EACH_SERVICE_GETS_ITS_OWN_LANGUAGE_CODE)


def test_the_dialog_fills_only_empty_cells(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtCore import Qt

    import app.developer_helpers.localization_helper as dialog_module

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


# ----------------------------------------------------------------------
# DeepL (todo A-08)
# ----------------------------------------------------------------------
class _Answer:
    """What urlopen returns, as a context manager."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Answer":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _deepl_opener(sent: list[urllib.request.Request]):
    def opener(request: urllib.request.Request, timeout: float = 0) -> _Answer:
        sent.append(request)
        fields = urllib.parse.parse_qs(bytes(request.data or b"").decode("utf-8"))  # pyright: ignore[reportArgumentType]
        return _Answer({"translations": [{"text": text.upper()} for text in fields["text"]]})

    return opener


def test_deepl_sends_the_key_in_the_header_and_each_line_as_a_text() -> None:
    sent: list[urllib.request.Request] = []
    deepl = mt.DeepLTranslator("secret:fx", "IT", opener=_deepl_opener(sent))
    assert deepl.translate("one\ntwo {0}") == "ONE\nTWO {0}"
    (request,) = sent
    assert request.full_url == mt.DEEPL_FREE_URL
    assert request.get_header("Authorization") == "DeepL-Auth-Key secret:fx"
    fields = urllib.parse.parse_qs(bytes(request.data or b"").decode("utf-8"))  # pyright: ignore[reportArgumentType]
    assert fields["text"] == ["one", "two {0}"]
    assert fields["target_lang"] == ["IT"]
    assert "secret" not in request.full_url


def test_a_paid_deepl_key_goes_to_the_paid_host_and_long_blocks_are_split() -> None:
    sent: list[urllib.request.Request] = []
    deepl = mt.DeepLTranslator("paid-key", "FR", opener=_deepl_opener(sent))
    lines = [f"line {index}" for index in range(120)]
    assert deepl.translate("\n".join(lines)).split("\n") == [line.upper() for line in lines]
    assert [request.full_url for request in sent] == [mt.DEEPL_PRO_URL] * 3


_CASES_DEEPL_REFUSALS_ARE_NAMED = [(403, "API key", False), (456, "quota", True), (429, "Too many requests", True)]


def _deepl_refusals_are_named(status: int, words: str, limited: bool) -> None:
    def refuses(request: urllib.request.Request, timeout: float = 0) -> _Answer:
        raise urllib.error.HTTPError(request.full_url, status, "refused", Message(), None)

    run = mt.translate_all(
        ["Hello"], language="it", translator=mt.DeepLTranslator("k:fx", "IT", opener=refuses)
    )
    assert run.failed == 1 and words in run.error
    # A test translator runs under the default provider's name.
    assert bool(run.limited) == limited


def test_deepl_refusals_are_named() -> None:
    check_all(_deepl_refusals_are_named, _CASES_DEEPL_REFUSALS_ARE_NAMED)


def test_deepl_is_offered_only_with_a_key_and_needs_no_deep_translator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mt, "available", lambda: False)
    monkeypatch.setattr(mt, "argos_available", lambda: False)
    assert mt.available_providers("") == []
    assert mt.available_providers("k:fx") == [("deepl", "DeepL")]
    monkeypatch.setattr(mt, "available", lambda: True)
    assert [key for key, _name in mt.available_providers("")] == ["google", "mymemory"]


_CASES_DEEPL_LANGUAGE_CODES = [("it", "IT"), ("pt", "PT-BR"), ("zh", "ZH-HANS"), ("en", "EN-GB")]


def _deepl_language_codes(language: str, target: str) -> None:
    deepl = mt._make_translator("deepl", language, deepl_key="k:fx")
    assert isinstance(deepl, mt.DeepLTranslator) and deepl.target == target


def test_deepl_language_codes() -> None:
    check_all(_deepl_language_codes, _CASES_DEEPL_LANGUAGE_CODES)


def test_deepl_without_a_key_hands_over_to_the_next_service() -> None:
    made: list[str] = []

    def factory(name: str, language: str):
        made.append(name)
        if name == "deepl":
            return mt._make_translator(name, language, deepl_key="")
        return Echo()

    run = mt.translate_all(["Hello"], language="it", provider="deepl", fallbacks=["google"], make_translator=factory)
    assert made == ["deepl", "google"]
    assert run.translations == {"Hello": "HELLO"}


def test_settings_keep_the_deepl_key_in_user_json(qapp) -> None:
    from app.dialogs.settings_dialog import SettingsDialog
    from app.utils import config

    dialog = SettingsDialog()
    dialog._deepl_key_edit.setText("  my-key:fx ")
    dialog._save()
    assert config.get_value(mt.DEEPL_KEY_SETTING) == "my-key:fx"
    assert mt.DEEPL_KEY_SETTING not in json.loads(config.CONFIG_PATH.read_text(encoding="utf-8"))
    config.set_value(mt.DEEPL_KEY_SETTING, "")


def _argos_archive(top: str) -> bytes:
    """An .argosmodel as Argos ships it: one top folder, stanza/ included."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(f"{top}/model/model.bin", b"weights")
        bundle.writestr(f"{top}/sentencepiece.model", b"pieces")
        bundle.writestr(f"{top}/metadata.json", "{}")
        bundle.writestr(f"{top}/stanza/en/tokenize/ewt.pt", b"splitter")
    return buffer.getvalue()


class ArgosSite:
    """Argos's index and one model archive, served without a network."""

    def __init__(self, archive: bytes) -> None:
        self.archive = archive
        self.asked: list[str] = []

    def __call__(self, url: str, timeout: float = 0) -> io.BytesIO:
        self.asked.append(url)
        if url == mt.ARGOS_INDEX_URL:
            return io.BytesIO(json.dumps([
                {"from_code": "en", "to_code": "it", "links": ["https://example.org/en_it.argosmodel", "ipfs://x"]},
            ]).encode())
        return io.BytesIO(self.archive)


def test_an_argos_model_is_downloaded_once_without_its_sentence_splitter(tmp_path: Path) -> None:
    site = ArgosSite(_argos_archive("translate-en_it-1_0"))
    folder = mt.argos_model("it", tmp_path, opener=site)
    assert folder == tmp_path / "en_it"
    assert (folder / "model" / "model.bin").read_bytes() == b"weights"
    assert (folder / "sentencepiece.model").is_file()
    assert not (folder / "stanza").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["en_it"]  # no archive left behind
    assert site.asked == [mt.ARGOS_INDEX_URL, "https://example.org/en_it.argosmodel"]

    assert mt.argos_model("it", tmp_path, opener=site) == folder
    assert len(site.asked) == 2  # already there: nothing asked


def test_argos_without_the_language_or_a_good_archive_says_so(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no model for 'xx'"):
        mt.argos_model("xx", tmp_path, opener=ArgosSite(b""))
    with pytest.raises(RuntimeError, match="damaged"):
        mt.argos_model("it", tmp_path, opener=ArgosSite(b"not a zip"))
    assert list(tmp_path.iterdir()) == []


def test_argos_that_cannot_download_hands_over_to_the_next_service() -> None:
    def factory(name: str, language: str):
        if name == "argos":
            raise RuntimeError("Argos Translate's model could not be downloaded (no network)")
        return Echo()

    run = mt.translate_all(["Hello"], language="it", provider="argos", fallbacks=["google"], make_translator=factory)
    assert run.translations == {"Hello": "HELLO"}
    assert run.switched == [("Argos Translate (offline)", "Google Translate")]


def test_argos_is_offered_when_its_engine_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mt, "available", lambda: False)
    monkeypatch.setattr(mt, "argos_available", lambda: True)
    assert mt.available_providers("") == [("argos", "Argos Translate (offline)")]


@pytest.mark.skipif(
    not (mt.argos_available() and (mt.argos_models_dir() / "en_it" / "model" / "model.bin").is_file()),
    reason="needs ctranslate2, sentencepiece and the English-Italian model already downloaded",
)
def test_argos_translates_a_block_and_keeps_its_placeholders() -> None:
    run = mt.translate_all(
        ["Name cannot be empty.", "A file named {file} already exists.", "  cm"], language="it", provider="argos"
    )
    assert run.translations["Name cannot be empty."] == "Il nome non può essere vuoto."
    assert "{file}" in run.translations["A file named {file} already exists."]
    assert run.failed == run.rejected == 0
