"""Panel letters on a multi-axis figure: (a), (b), (c) - what a paper's text refers to (todo G-18).

A figure option (``panel_labels``) names the style, ``panel_label_position`` says
where the letter goes. The letters follow the axes' order in the grid - left to
right, top to bottom - and a twin axis, which shares its target's panel, gets
none. No Qt: the properties widget lists STYLES and the renderer writes the text.
"""
from __future__ import annotations

from typing import Any

#: (stored value, label) of each style; "" is no letters.
STYLES: tuple[tuple[str, str], ...] = (
    ("", "None"),
    ("a", "a, b, c"),
    ("A", "A, B, C"),
    ("(a)", "(a), (b), (c)"),
    ("a)", "a), b), c)"),
    ("(A)", "(A), (B), (C)"),
    ("A.", "A., B., C."),
)

#: (stored value, label) of where the letter sits.
POSITIONS: tuple[tuple[str, str], ...] = (
    ("outside", "Above the corner"),
    ("inside", "Inside the corner"),
)


def label_text(style: str, index: int) -> str:
    """The text of panel *index* (0 = first) in *style*: 'a' -> 'a', '(A)' -> '(C)' for index 2; '' for none.

    Past the 26 letters it goes on as aa, ab, ... (AA, AB ... for capitals).
    """
    style = str(style or "")
    letter_base = next((char for char in style if char.isalpha()), "")
    if not letter_base:
        return ""
    base = ord("A" if letter_base.isupper() else "a")
    count, offset = divmod(int(index), 26)
    letters = (chr(base + count - 1) if count else "") + chr(base + offset)
    return style.replace(letter_base, letters, 1)


def apply_panel_labels(ordered_axes: list[Any], options: dict[str, Any]) -> None:
    """Write each of *ordered_axes*' letter, in the order given; none when the style is empty."""
    style = str(options.get("panel_labels") or "")
    if not style:
        return
    inside = str(options.get("panel_label_position") or "outside") == "inside"
    for index, ax in enumerate(ordered_axes):
        text = label_text(style, index)
        if not text:
            continue
        artist = ax.annotate(
            text,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(5, -5) if inside else (0, 4),
            textcoords="offset points",
            ha="left",
            va="top" if inside else "bottom",
            fontsize="large",
            fontweight="bold",
            annotation_clip=False,
        )
        artist._dhub_panel_label = True  # noqa: SLF001 - our own attribute
