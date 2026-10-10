"""Write the manual's appendix of equivalents: the same task in ChartLibre, JMP, SPSS, Minitab, Origin and Excel.

The table is app.analysis.catalog.EQUIVALENTS - the rows the search also
finds things by - so the manual and the application cannot disagree. Run
``python -m dev.manual.make_equivalents`` after editing it; the file written
is docs/manual/sections/equivalents.typ, included by user_manual.typ.
"""
from __future__ import annotations

from pathlib import Path

from app.analysis.catalog import EQUIVALENTS, PROGRAMS

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "manual" / "sections" / "equivalents.typ"

_INTRO = """\
= Equivalents in other programs <equivalents>

Coming from JMP, SPSS, Minitab, Origin or Excel? Find the task you know in the left column: the second says where
it is in ChartLibre, the others where it is in each program. A dash means the program has no direct equivalent.
The search (the *Search* frame of the Analysis panel, or #emph[Ctrl+K] - #emph[Cmd+K] on macOS) knows these names
too: typing "Fit Y by X", "Crosstabs" or "Control Chart Builder" finds the ChartLibre function.

#note[
  Menu paths are those of JMP 18, SPSS 29, Minitab 22, Origin 2024 and Excel 365 with the Analysis ToolPak. They are
  indicative: versions move things, and an equivalent does not always compute exactly the same statistics or offer
  the same options.
]

"""


def _cell(text: str) -> str:
    """*text* as a Typst content block: its markup characters escaped."""
    escaped = text
    for char in ("\\", "#", "[", "]", "*", "_", "@", "<", "$", "`"):
        escaped = escaped.replace(char, "\\" + char)
    return f"[{escaped}]"


def render() -> str:
    """The appendix, as Typst."""
    headers = ["Task", "ChartLibre", *PROGRAMS]
    rows = []
    for task, where, _key, programs in EQUIVALENTS:
        cells = [f"[*{_cell(task)[1:-1]}*]", _cell(where), *(_cell(programs.get(p, "—")) for p in PROGRAMS)]
        rows.append("  " + ", ".join(cells) + ",")
    table = "\n".join([
        "#set text(size: 7.6pt)",
        "#set par(justify: false)",
        "#table(",
        "  columns: (1.1fr, 1.25fr, 1.4fr, 1.4fr, 1.4fr, 1.4fr),",
        "  stroke: 0.4pt + rgb(\"#d0d0d6\"),",
        "  inset: 4pt,",
        "  fill: (_, row) => if row == 0 { rgb(\"#eef3fb\") },",
        "  table.header(" + ", ".join(f"[*{h}*]" for h in headers) + "),",
        *rows,
        ")",
    ])
    # Landscape, the introduction on the table's own first page: six columns
    # of menu paths need the width.
    return '#import "helpers.typ": *\n\n#page(flipped: true)[\n' + _INTRO + table + "\n]\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(EQUIVALENTS)} rows)")


if __name__ == "__main__":
    main()
