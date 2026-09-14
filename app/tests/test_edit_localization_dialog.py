"""Edit Localization: a table editor for a .po catalogue (todo.txt P3-5).

app.utils.i18n.LOCALES_DIR is monkeypatched to a tmp_path in every test
here - this tool writes real .po/.mo files, and app/locales/ is not a
place a test run should ever leave litter in, success or failure. Every
test that could open a real QMessageBox (show_message, the auto-translate
confirmation) mocks it - a modal popped during an automated run blocks
the whole suite on a click nobody is there to give.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.dialogs import edit_localization_dialog as module
from app.dialogs.edit_localization_dialog import EditLocalizationDialog
from app.utils import i18n


@pytest.fixture(autouse=True)
def locales_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(i18n, "LOCALES_DIR", tmp_path)
    monkeypatch.setattr(module, "_po_path", lambda language: tmp_path / language / "LC_MESSAGES" / "datahub.po")
    return tmp_path


@pytest.fixture
def seeded_it_catalog(locales_dir: Path) -> Path:
    path = locales_dir / "it" / "LC_MESSAGES" / "datahub.po"
    path.parent.mkdir(parents=True)
    path.write_text(
        '# Italian translation for ChartLibre.\n'
        'msgid ""\n'
        'msgstr ""\n'
        '"Project-Id-Version: ChartLibre\\n"\n'
        '"Language: it\\n"\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        '\n'
        'msgid "Hello"\n'
        'msgstr "Ciao"\n'
        '\n'
        'msgid "Untranslated one"\n'
        'msgstr ""\n',
        encoding="utf-8",
    )
    return path


def _mock_show_message(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        module, "show_message",
        lambda _parent, message_id, **fields: calls.append((message_id, fields)),
    )
    return calls


# ----------------------------------------------------------------------
# i18n._write_po / _po_escape - the round-trip this whole tool depends on
# ----------------------------------------------------------------------
def test_write_po_round_trips_through_parse_po(tmp_path: Path) -> None:
    entries = {
        "": "Content-Type: text/plain; charset=UTF-8\n",
        "Plain": "Semplice",
        "With \"quotes\" and \\backslash\\": "Con \"virgolette\" e \\backslash\\",
        "Multi\nline": "Multi\nlinea",
    }
    path = tmp_path / "roundtrip.po"

    i18n._write_po(entries, path)
    result = i18n._parse_po(path)

    assert result == entries


def test_write_po_preserves_entry_order(tmp_path: Path) -> None:
    entries = {"": "x", "Zebra": "z", "Apple": "a", "Mango": "m"}
    path = tmp_path / "order.po"

    i18n._write_po(entries, path)

    ids_in_order = [
        line.split(" ", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("msgid ") and line != 'msgid ""'
    ]
    assert ids_in_order == ['"Zebra"', '"Apple"', '"Mango"']


def test_source_translator_calls_finds_a_known_string() -> None:
    found = i18n.source_translator_calls()
    assert "Cancel" in found or len(found) > 100  # sanity: a real, sizeable sweep


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
@pytest.fixture
def dialog(qapp, seeded_it_catalog: Path) -> EditLocalizationDialog:
    built = EditLocalizationDialog()
    yield built
    built.close()


def test_english_is_not_offered_as_a_translation_target(dialog) -> None:
    codes = [dialog._language_combo.itemData(i) for i in range(dialog._language_combo.count())]
    assert "en" not in codes
    assert "it" in codes


def test_existing_translations_are_loaded(dialog) -> None:
    assert dialog._entries.get("Hello") == "Ciao"


def test_an_untranslated_entry_shows_empty(dialog) -> None:
    assert dialog._entries.get("Untranslated one") == ""


def test_missing_only_hides_translated_rows(dialog) -> None:
    dialog._missing_only_check.setChecked(True)

    for row in range(dialog._table.rowCount()):
        source = dialog._table.item(row, 0).text()
        hidden = dialog._table.isRowHidden(row)
        if source == "Hello":
            assert hidden is True
        elif source == "Untranslated one":
            assert hidden is False


def test_editing_a_cell_and_saving_persists_it(
    dialog, seeded_it_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_show_message(monkeypatch)
    row = next(
        r for r in range(dialog._table.rowCount())
        if dialog._table.item(r, 0).text() == "Untranslated one"
    )
    dialog._table.item(row, 1).setText("Uno non tradotto")

    dialog._on_save()

    saved = i18n._parse_po(seeded_it_catalog)
    assert saved["Untranslated one"] == "Uno non tradotto"
    assert saved["Hello"] == "Ciao"  # untouched entries survive the save


def test_saving_recompiles_the_mo(
    dialog, seeded_it_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_show_message(monkeypatch)

    dialog._on_save()

    mo_path = seeded_it_catalog.with_suffix(".mo")
    assert mo_path.exists()


def test_saving_reports_success(
    dialog, seeded_it_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_show_message(monkeypatch)

    dialog._on_save()

    assert calls and calls[0][0] == "dev.localization_saved"


def test_a_new_language_is_created_with_every_source_string(
    dialog, locales_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module.QInputDialog, "getText", staticmethod(lambda *a, **k: ("fr", True)))

    dialog._on_new_language()

    created = locales_dir / "fr" / "LC_MESSAGES" / "datahub.po"
    assert created.exists()
    entries = i18n._parse_po(created)
    assert all(value == "" for key, value in entries.items() if key)
    assert len(entries) > 100  # seeded from the real source sweep
    assert dialog._language_combo.currentData() == "fr"


def test_an_invalid_language_code_is_rejected(
    dialog, locales_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_show_message(monkeypatch)
    monkeypatch.setattr(module.QInputDialog, "getText", staticmethod(lambda *a, **k: ("123", True)))

    dialog._on_new_language()

    assert not (locales_dir / "123").exists()
    assert calls and calls[0][0] == "dev.validation_error"


def test_a_language_that_already_exists_is_rejected(
    dialog, seeded_it_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_show_message(monkeypatch)
    monkeypatch.setattr(module.QInputDialog, "getText", staticmethod(lambda *a, **k: ("it", True)))

    dialog._on_new_language()

    assert calls and calls[0][0] == "dev.validation_error"


def test_cancelling_the_new_language_prompt_does_nothing(
    dialog, locales_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module.QInputDialog, "getText", staticmethod(lambda *a, **k: ("fr", False)))

    dialog._on_new_language()

    assert not (locales_dir / "fr").exists()


# ----------------------------------------------------------------------
# Auto-translate: never a real network call in a test
# ----------------------------------------------------------------------
def test_translate_available_is_false_without_the_package() -> None:
    # The "translate" package is not a hard dependency (see the module's
    # own docstring) and is not installed in this environment - confirms
    # the soft-dependency guard degrades cleanly rather than raising.
    assert module._translate_available() is False


def test_auto_translate_returns_none_without_the_package() -> None:
    assert module._auto_translate("Hello", to_lang="it") is None


def test_auto_translate_missing_fills_only_empty_cells(
    dialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_show_message(monkeypatch)
    monkeypatch.setattr(module, "ask_before_auto_translate", lambda _parent: True)
    monkeypatch.setattr(module, "_auto_translate", lambda text, *, to_lang: f"[{to_lang}] {text}")

    dialog._on_auto_translate_missing()

    row = next(
        r for r in range(dialog._table.rowCount())
        if dialog._table.item(r, 0).text() == "Untranslated one"
    )
    assert dialog._table.item(row, 1).text() == "[it] Untranslated one"
    hello_row = next(
        r for r in range(dialog._table.rowCount())
        if dialog._table.item(r, 0).text() == "Hello"
    )
    assert dialog._table.item(hello_row, 1).text() == "Ciao"  # already translated, left alone
    assert calls and calls[0][0] == "dev.localization_auto_translated"


def test_declining_the_auto_translate_confirmation_changes_nothing(
    dialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "ask_before_auto_translate", lambda _parent: False)
    called = []
    monkeypatch.setattr(module, "_auto_translate", lambda *a, **k: called.append(1))

    dialog._on_auto_translate_missing()

    assert called == []
