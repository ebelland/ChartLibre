"""Markers beyond the combo's list: kept as the text typed, drawn as what matplotlib takes.

    "o", "s", "^"...           matplotlib's own codes, as they are
    "★", "A", "♣"              one character, drawn as mathtext "$★$" in DejaVu Sans
    "$\\clubsuit$"             matplotlib's mathtext, as it is
    "(5, 1, 0)"                matplotlib's (sides, style, angle): 0 polygon, 1 star, 2 asterisk

A character is accepted only when DejaVu Sans - the font shipped inside matplotlib, the same on macOS, Windows
and Linux - has it: a system font would draw it differently on each, or not at all (emoji), and the chart would
not look the same where the project is opened.
"""
from __future__ import annotations

import ast
from functools import cache
from typing import Any

from matplotlib.markers import MarkerStyle

_SPECIAL = {"$": r"\$", "\\": r"\backslash", "_": r"\_", "^": r"\^{}", "{": r"\{", "}": r"\}",
            "#": r"\#", "%": r"\%", "&": r"\&", "~": r"\sim", " ": r"\ "}


@cache
def _dejavu_has(char: str) -> bool:
    import os

    import matplotlib
    from matplotlib.ft2font import FT2Font

    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")  # inside matplotlib itself
    return FT2Font(path).get_char_index(ord(char)) != 0


@cache
def _valid(marker: Any) -> bool:
    try:
        MarkerStyle(marker)
    except (ValueError, TypeError):
        return False
    if isinstance(marker, str) and marker.startswith("$"):
        from matplotlib.mathtext import MathTextParser

        try:
            MathTextParser("path").parse(marker)
        except ValueError:
            return False
    return True


#: matplotlib's tick and caret markers are the integers 0-11; the combo keeps them as text. "1"-"4" and "8" are
#: string codes of their own (tri, octagon), so they stay strings.
_INTEGER_CODES = {"0", "5", "6", "7", "9", "10", "11"}


def to_matplotlib(text: Any) -> Any:
    """What matplotlib draws for *text*; None when it is none of the forms above."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped in _INTEGER_CODES:
        return int(stripped)
    if stripped.startswith("("):
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return None
        if (isinstance(value, tuple) and len(value) in (2, 3) and all(isinstance(v, (int, float)) for v in value)
                and isinstance(value[0], int) and 2 <= value[0] <= 24 and value[1] in (0, 1, 2)):
            return value
        return None
    if len(stripped) == 1 and stripped.isprintable() and _dejavu_has(stripped):
        if stripped in _SPECIAL:
            return f"${_SPECIAL[stripped]}$"
        if MarkerStyle.markers.get(stripped) is None and stripped not in MarkerStyle.markers:
            return f"${stripped}$"
        return stripped
    if len(stripped) == 1:
        return None
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 2:
        return stripped if _valid(stripped) else None
    return stripped if stripped in MarkerStyle.markers else None


def is_custom(text: Any) -> bool:
    """Whether *text* is a marker of its own (not a code of matplotlib's) that can be drawn."""
    if isinstance(text, str) and text.startswith("image:") and len(text) > 6:
        return True  # a picture of the project (charts.picture_markers)
    return (isinstance(text, str) and text.strip() not in MarkerStyle.markers
            and to_matplotlib(text) is not None)
