"""A .po string is read back as it was written: a typed backslash-n is not a line break."""
from __future__ import annotations

from app.utils import i18n


def test_escapes_round_trip() -> None:
    for text in ("type \\n where a line ends", "two\nlines", 'a "quote"', "a tab\there", "back\\slash"):
        assert i18n._po_string(i18n._po_escape(text)) == text
