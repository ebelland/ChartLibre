"""Text on a chart over several lines: "\\n" typed in a one-line field, and the lines' alignment.

A title, an axis label or a legend entry is typed into a one-line field, so a
line break cannot be typed into it; Matplotlib breaks a line at "\\n". So the
two characters backslash and n, typed, mean a new line - as in Matplotlib's
own Python, where "Time\\n(s)" is two lines. Not inside $...$: there
"\\nu" is the Greek letter. A doubled backslash ("\\\\n") keeps them as typed.

How the lines of such a text line up with each other is Matplotlib's
``multialignment`` - left, center or right - apart from where the whole
block sits (its ``ha``).
"""
from __future__ import annotations

import re

MULTIALIGNMENTS: tuple[str, ...] = ("center", "left", "right")
_BREAK = re.compile(r"(?<!\\)\\n")


def display_text(text: str) -> str:
    """*text* with every typed "\\n" outside $...$ made a line break ("\\\\n" stays "\\n")."""
    if "\\" not in text:
        return text
    parts = re.split(r"(?<!\\)(\$)", text)  # "$" delimiters kept, escaped "\$" not split on
    inside = False
    out: list[str] = []
    for part in parts:
        if part == "$":
            inside = not inside
            out.append(part)
        elif inside:
            out.append(part)
        else:
            out.append(_BREAK.sub("\n", part).replace("\\\\n", "\\n"))
    return "".join(out)


def multialignment(options: dict, key: str = "label_multialignment") -> str | None:
    """The lines' alignment an options dict asks for; None leaves Matplotlib's own."""
    value = str(options.get(key) or "")
    return value if value in MULTIALIGNMENTS else None
