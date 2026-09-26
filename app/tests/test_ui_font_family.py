"""The Fluent sheet must never name a font that is not installed.

The sheet used to open with

    font-family: "Segoe UI Variable", "Segoe UI", "Inter", "Arial", sans-serif;

and "Segoe UI Variable" is not a family any Windows installs: Windows 11
registers the variable UI face under three optical sizes ("Segoe UI Variable
Small" / "Text" / "Display") and nothing under the bare name. Asking Qt for it
anyway sends the Windows font database off to walk every alias it knows before
falling through to the second name, which it reported in every run's log:

    Qt warning: Populating font family aliases took 200 ms. Replace uses of
    missing font family "Segoe UI Variable" with one that exists to avoid
    this cost.

So the family is resolved against QFontDatabase before it reaches the sheet -
the same rule _best_fluent_font_family already applied to the icon font.
"""
from __future__ import annotations

import pytest

from PySide6.QtGui import QFontDatabase

from app.styles.style import (
    UI_FONT_FAMILY_TOKEN,
    apply_platform_style,
    substitute_ui_font,
    ui_font_family,
)


def test_the_resolved_family_is_one_this_machine_actually_has(qapp) -> None:
    """The whole point: whatever comes back, Qt can find it without a
    search. Asserted against the font database rather than against a
    hard-coded name, because the right answer differs per machine - Segoe
    UI Variable Text on Windows 11, Segoe UI on Windows 10, the system UI
    font on a Linux CI box with none of them."""
    assert ui_font_family().strip('"') in set(QFontDatabase.families())


def test_the_family_is_quoted_for_the_sheet(qapp) -> None:
    """Unquoted, a family whose name has spaces is not one QSS value."""
    resolved = ui_font_family()
    assert resolved.startswith('"') and resolved.endswith('"')


def test_a_sheet_without_the_token_is_untouched(qapp) -> None:
    """macos_native.qss names no font at all, and a sheet of the user's own
    need not know this token exists."""
    sheet = "QWidget { color: red; }"
    assert substitute_ui_font(sheet) == sheet


def test_the_token_is_replaced(qapp) -> None:
    assert UI_FONT_FAMILY_TOKEN not in substitute_ui_font(
        f"* {{ font-family: {UI_FONT_FAMILY_TOKEN}; }}"
    )


def test_the_applied_fluent_sheet_carries_no_unresolved_token(qapp) -> None:
    """End to end: a token left in the sheet is not a font-family Qt can
    read, so the whole rule would be dropped and every widget would fall
    back to the default font."""
    apply_platform_style(qapp, "fluent_win11")
    try:
        sheet = qapp.styleSheet()
        assert sheet, "the Fluent sheet did not load"
        assert UI_FONT_FAMILY_TOKEN not in sheet
    finally:
        qapp.setStyleSheet("")


@pytest.mark.parametrize("sheet_key", ["fluent_win11", "macos_native"])
def test_no_shipped_sheet_names_the_family_that_does_not_exist(
    qapp, sheet_key: str
) -> None:
    apply_platform_style(qapp, sheet_key)
    try:
        assert '"Segoe UI Variable"' not in qapp.styleSheet()
    finally:
        qapp.setStyleSheet("")
