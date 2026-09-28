"""Tests for the tables a series operation writes.

They used to be named like anything else, so a project with a few fits and a
spectral analysis showed more generated tables in the source list than imported
ones - and the imported ones are what the user came to find.  Every generated
table now starts with an underscore, which is enough for the list to hide the
whole class of them without keeping a registry.
"""
from __future__ import annotations

from pathlib import Path

from app.series_operations.dialog_base import (
    GENERATED_TABLE_PREFIX,
    generated_table_name,
)

APP_DIR = Path(__file__).resolve().parents[2] / "app"


# ----------------------------------------------------------------------
# The name
# ----------------------------------------------------------------------
def test_a_generated_name_is_prefixed() -> None:
    assert generated_table_name("Fit_series_1").startswith(GENERATED_TABLE_PREFIX)


def test_unsafe_characters_are_replaced() -> None:
    """The name goes into SQL, and a series can be called anything."""
    assert generated_table_name("Fit: my series (2)") == "_Fit_my_series_2"


# ----------------------------------------------------------------------
# Every operation uses it
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The toggle
# ----------------------------------------------------------------------


