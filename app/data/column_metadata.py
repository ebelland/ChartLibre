"""What a table's columns are, beyond their values: label, unit, notes, modelling type, format (R-08).

As JMP's Column Info. Kept with the table's information (``__table_descriptors__``,
key "columns": one entry per column name), so it travels with the project and
undoes with it. What it changes:

* kind - the modelling type forced (continuous, ordinal, nominal): the
  analyses cast the column so, whatever its values suggest;
* a cell with no value is missing: SQL NULL, and in a text column an empty
  or blank string, as in JMP. Codes standing for one (-999, "NA") are made
  empty cells in the data itself, by the table editor's Fill missing values;
* format - how its values are shown (Best, Fixed, Percent, Scientific,
  thousands separator, a date/time pattern): the table preview's text only,
  never the values themselves;
* label, unit, notes - said where the column is described.

No Qt: the Column Info dialog and the table analyses both read it here.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

INFO_KEY = "columns"
KINDS: tuple[str, ...] = ("continuous", "ordinal", "nominal")
#: How a column's values may be shown, as JMP's Format: (key, label). "" is Best: the value as stored.
FORMATS: tuple[tuple[str, str], ...] = (
    ("", "Best"), ("fixed", "Fixed decimals"), ("percent", "Percent"), ("scientific", "Scientific"),
    ("thousands", "Thousands separator"), ("datetime", "Date / time"),
)
DEFAULT_DATE_PATTERN = "%Y-%m-%d"


@dataclass
class ColumnInfo:
    label: str = ""
    unit: str = ""
    notes: str = ""
    #: The modelling type forced; "" lets the values decide.
    kind: str = ""
    #: How the values are shown (FORMATS); "" shows them as stored.
    format: str = ""
    decimals: int = 2
    #: The strftime pattern of the "datetime" format.
    pattern: str = DEFAULT_DATE_PATTERN

    def is_empty(self) -> bool:
        return not (self.label or self.unit or self.notes or self.kind
                    or self.format)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ColumnInfo:
        kind = str(values.get("kind") or "")
        shown = str(values.get("format") or "")
        try:
            decimals = max(0, min(15, int(values.get("decimals", 2))))
        except (TypeError, ValueError):
            decimals = 2
        return cls(label=str(values.get("label") or ""), unit=str(values.get("unit") or ""),
                   notes=str(values.get("notes") or ""), kind=kind if kind in KINDS else "",
                   format=shown if shown in dict(FORMATS) else "", decimals=decimals,
                   pattern=str(values.get("pattern") or DEFAULT_DATE_PATTERN))


def column_info(repo: Any, table: str) -> dict[str, ColumnInfo]:
    """Every described column of *table*: {name: ColumnInfo}."""
    stored = repo.get_table_info(table).get(INFO_KEY) or {}
    return {str(name): ColumnInfo.from_dict(values) for name, values in dict(stored).items()
            if isinstance(values, Mapping)}


def set_column_info(repo: Any, table: str, column: str, info: ColumnInfo | None) -> None:
    """Keep *info* for *column* (None, or nothing in it, forgets it)."""
    stored = dict(repo.get_table_info(table).get(INFO_KEY) or {})
    if info is None or info.is_empty():
        stored.pop(column, None)
    else:
        stored[column] = asdict(info)
    repo.set_table_info(table, INFO_KEY, stored or None)


def rename_column_info(repo: Any, table: str, old: str, new: str) -> None:
    """The column's description follows it to its new name."""
    stored = dict(repo.get_table_info(table).get(INFO_KEY) or {})
    if old in stored:
        stored[new] = stored.pop(old)
        repo.set_table_info(table, INFO_KEY, stored)


def blank_as_missing(column: pd.Series) -> pd.Series:
    """*column* with its empty and blank strings missing: a text cell with no value entered.

    NULL is missing already (NaN, read from SQLite); "" or "   " in a text column is no value either, as
    JMP reads an empty character cell - else an analysis takes it for a level of its own. A number column
    is returned as it is.
    """
    if not (pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column)):
        return column
    blank = column.str.strip().eq("").fillna(False).astype(bool)  # non-text values give NaN here: not blank
    return column.mask(blank, np.nan) if blank.any() else column


def apply_to_frame(frame: pd.DataFrame, infos: Mapping[str, ColumnInfo]) -> pd.DataFrame:
    """*frame* as the analyses should read it: cells with no value empty.

    Value labels (1 -> "male") were applied here; taken out for now - a query joining a table of labels does
    it, and the analyses read the query."""
    result = frame.copy()
    for name in result.columns:
        result[name] = blank_as_missing(result[name])
    return result


def forced_kinds(infos: Mapping[str, ColumnInfo]) -> dict[str, str]:
    """{column: modelling type} for the columns whose type is forced."""
    return {name: info.kind for name, info in infos.items() if info.kind}


def format_value(value: Any, info: ColumnInfo) -> str:
    """*value* as the column's format shows it; as stored when it does not apply (text in a number format)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    kind, decimals = info.format, info.decimals
    if kind == "datetime":
        moment = pd.to_datetime(value, errors="coerce") if not isinstance(value, (int, float)) else pd.NaT
        if pd.isna(moment):
            return str(value)
        try:
            return moment.strftime(info.pattern or DEFAULT_DATE_PATTERN)
        except ValueError:
            return str(value)
    if kind in ("fixed", "percent", "scientific", "thousands"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if kind == "fixed":
            return f"{number:.{decimals}f}"
        if kind == "percent":
            return f"{number * 100:.{decimals}f}%"
        if kind == "scientific":
            return f"{number:.{decimals}e}"
        return f"{number:,.{decimals}f}"
    return str(value)


def best_text(value: Any, fits: Callable[[str], bool]) -> str:
    """The Best format: *value* as stored when it *fits* the cell, else with as many significant digits as fit
    (1/3 in a narrow column 0.333, in a wide one 0.333333333333) - then scientific, then "#" (a spreadsheet's
    General format). Text that is not a number is shown as it is."""
    text = "" if value is None else str(value)
    if fits(text):
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    for digits in range(15, 0, -1):
        shown = f"{number:.{digits}g}"
        # "g" turns to scientific notation past 1e{digits}: kept only once nothing plainer fits.
        if "e" not in shown and fits(shown):
            return shown
    for digits in range(6, -1, -1):
        shown = f"{number:.{digits}e}"
        if fits(shown):
            return shown
    return "#"


def source_table_of(sql: str) -> str | None:
    """The one table a plain "SELECT ... FROM table" reads; None for anything more involved."""
    import re

    text = str(sql or "")
    if re.search(r"\b(JOIN|UNION|WITH)\b|\(\s*SELECT", text, re.IGNORECASE):
        return None
    match = re.search(r'\bFROM\s+("(?:[^"]|"")+"|`[^`]+`|\[[^\]]+\]|\w+)', text, re.IGNORECASE)
    if match is None:
        return None
    name = match.group(1)
    if name[0] in '"`[':
        name = name[1:-1].replace('""', '"')
    return name or None


def axis_label_of(column: str, info: ColumnInfo | None) -> str:
    """What an axis showing *column* is called: its label and unit, "Voltage (V)"; "" when it has neither."""
    if info is None or not (info.label or info.unit):
        return ""
    name = info.label or column
    return f"{name} ({info.unit})" if info.unit else name
