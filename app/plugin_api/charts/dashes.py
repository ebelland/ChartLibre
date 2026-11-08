"""Line styles beyond matplotlib's four names: dash patterns written as text, kept as text, drawn as tuples.

A series' style keeps its line style as a string - it is saved in the project as one - so a custom pattern is a
string too, turned into what matplotlib draws only when the chart is drawn (kwarg_spec.resolve):

    "-", "--", "-.", ":", "none", "solid", "dashed"...   matplotlib's own names, as they are
    "..---", "-- . --", "- "                             "-" a dash, "." a dot, " " a longer gap
    "(0, (5, 2, 1, 2))"                                  matplotlib's (offset, (on, off, ...)) in line widths

Lengths are in line widths, as matplotlib scales its own dashes: a pattern looks the same on a thin line and a
thick one.
"""
from __future__ import annotations

import ast
from typing import Any

#: matplotlib's named styles, kept as they are.
NAMED = {"-", "--", "-.", ":", "none", "None", "", " ", "solid", "dashed", "dashdot", "dotted"}

DASH, DOT, GAP, SPACE = 3.7, 1.0, 1.6, 2.4  # on, on, off between marks, extra off for a " "


def _from_marks(text: str) -> tuple[float, tuple[float, ...]] | None:
    """"..---" -> (0, (1, 1.6, 1, 1.6, 3.7, 1.6, ...)); None when the text is not only "-", "." and spaces."""
    if not text or set(text) - {"-", ".", " "} or not set(text) & {"-", "."}:
        return None
    sequence: list[float] = []
    for mark in text:
        if mark == " ":
            if sequence:
                sequence[-1] += SPACE
            continue
        sequence += [DASH if mark == "-" else DOT, GAP]
    return 0.0, tuple(sequence)


def _from_tuple(text: str) -> tuple[float, tuple[float, ...]] | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if (isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, float))
            and isinstance(value[1], (tuple, list)) and len(value[1]) >= 2 and len(value[1]) % 2 == 0
            and all(isinstance(item, (int, float)) and item >= 0 for item in value[1]) and any(value[1])):
        return float(value[0]), tuple(float(item) for item in value[1])
    return None


def to_matplotlib(text: Any) -> Any:
    """What matplotlib draws for *text*: a name as it is, a pattern as an (offset, sequence) tuple; None when the
    text is neither (the option is then left out, and the style sheet decides)."""
    if not isinstance(text, str):
        return text
    if text in NAMED:
        return text
    stripped = text.strip()
    if stripped.startswith("("):
        return _from_tuple(stripped)
    return _from_marks(text)


def is_custom(text: str) -> bool:
    """Whether *text* is a pattern of its own (not one of matplotlib's names) that can be drawn."""
    return isinstance(text, str) and text not in NAMED and to_matplotlib(text) is not None
