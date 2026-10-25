"""The translation function itself.

The catalogues are kept by hand, with Edit Localization: no test checks what
they cover - only how ``_()`` behaves.
"""
from __future__ import annotations

from app.utils import i18n


def test_an_empty_string_stays_empty() -> None:
    """gettext translates "" as the catalogue's header: a report's blank column title read "Content-Type: ..."."""
    previous = i18n.language()
    try:
        i18n.set_language("it")
        assert i18n.tr("") == ""
    finally:
        i18n.set_language(previous)
