"""A colour cycle of RGB tuples survives the style editor's table (issue #7: "invalid RGBA argument")."""
from __future__ import annotations

import cycler
import numpy as np
from matplotlib.colors import to_rgba

from app.dialogs import edit_mpl_styles_dialog as editor


def test_a_cycler_of_rgb_tuples_is_written_as_hex_and_read_back() -> None:
    original = cycler.cycler("color", [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05, 0.5)])
    text = editor._to_table_str(original)
    assert "(" not in text.split("[", 1)[1]  # no tuple left inside the list
    restored = editor._parse_cycler_text(text)
    assert restored is not None
    colours = [entry["color"] for entry in restored]
    got = np.array([to_rgba(c) for c in colours])
    want = np.array([to_rgba((0.12, 0.47, 0.71)), to_rgba((1.0, 0.5, 0.05, 0.5))])
    np.testing.assert_allclose(got, want, atol=0.01)


def test_a_table_string_that_already_holds_tuples_is_read_too() -> None:
    """Styles saved before the fix hold "(r, g, b)" strings."""
    restored = editor._parse_cycler_text("cycler('color', ['(0.12, 0.47, 0.71)', '1f77b4', 'red'])")
    assert restored is not None
    for entry in restored:
        to_rgba(entry["color"])  # raises "Invalid RGBA argument" if unreadable


def test_ordinary_hex_cyclers_are_unchanged() -> None:
    text = editor._to_table_str(cycler.cycler("color", ["#1f77b4", "#ff7f0e"]))
    assert text == "cycler('color', ['1f77b4', 'ff7f0e'])"
    restored = editor._parse_cycler_text(text)
    assert [e["color"] for e in restored] == ["#1f77b4", "#ff7f0e"]


def test_text_that_is_not_a_cycler_gives_none() -> None:
    assert editor._parse_cycler_text("cycler(") is None
