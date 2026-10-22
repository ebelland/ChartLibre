"""What a table's columns are, beyond their values: label, unit, notes, modelling type, missing codes, value labels (R-08).

As JMP's Column Info. Kept with the table's information (``__table_descriptors__``,
key "columns": one entry per column name), so it travels with the project and
undoes with it. What it changes:

* kind - the modelling type forced (continuous, ordinal, nominal): the
  analyses cast the column so, whatever its values suggest;
* missing - codes that mean "no value" (-999, "NA", "n/d"): the analyses
  read them as empty;
* value_labels - {stored value: label} (1 -> "male"): the analyses see the
  labels, as levels;
* format - how its values are shown (Best, Fixed, Percent, Scientific,
  thousands separator, a date/time pattern): the table preview's text only,
  never the values themselves;
* label, unit, notes - said where the column is described.

No Qt: the Column Info dialog and the table analyses both read it here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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
    missing: list[str] = field(default_factory=list)
    value_labels: dict[str, str] = field(default_factory=dict)
    #: How the values are shown (FORMATS); "" shows them as stored.
    format: str = ""
    decimals: int = 2
    #: The strftime pattern of the "datetime" format.
    pattern: str = DEFAULT_DATE_PATTERN

    def is_empty(self) -> bool:
        return not (self.label or self.unit or self.notes or self.kind or self.missing or self.value_labels
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
                   missing=[str(code) for code in values.get("missing") or []],
                   value_labels={str(k): str(v) for k, v in dict(values.get("value_labels") or {}).items()},
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


def _matches(values: pd.Series, codes: list[str]) -> pd.Series:
    """Where *values* equal one of *codes*: as text, or as numbers (-999 matches -999.0)."""
    text = values.astype(str).str.strip()
    hits = text.isin([code.strip() for code in codes])
    numbers = pd.to_numeric(pd.Series(codes), errors="coerce").dropna()
    if not numbers.empty:
        hits |= pd.to_numeric(values, errors="coerce").isin(numbers.to_numpy())
    return hits & values.notna()


def apply_to_frame(frame: pd.DataFrame, infos: Mapping[str, ColumnInfo]) -> pd.DataFrame:
    """*frame* as the analyses should read it: missing codes empty, value labels in place of the codes."""
    result = frame.copy()
    for name, info in infos.items():
        if name not in result.columns:
            continue
        column = result[name]
        if info.missing:
            column = column.mask(_matches(column, info.missing), np.nan)
        if info.value_labels:
            keys = {code.strip(): label for code, label in info.value_labels.items()}

            def label_of(value: Any) -> Any:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return value
                text = str(value).strip()
                if text in keys:
                    return keys[text]
                if isinstance(value, float) and value.is_integer():  # 1.0 stored for 1
                    return keys.get(str(int(value)), value)
                return value

            column = column.map(label_of)
        result[name] = column
    return result


def forced_kinds(infos: Mapping[str, ColumnInfo]) -> dict[str, str]:
    """{column: modelling type} for the columns whose type is forced."""
    return {name: info.kind for name, info in infos.items() if info.kind}


def value_label_of(value: Any, labels: Mapping[str, str]) -> str | None:
    """The label of a stored *value* ({"1": "male"}); None when it has none.

    A stored 1.0 is the key "1": a number column keeps its integers as floats.
    """
    keys = [str(value)]
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        keys.append(str(int(value)))
    for key in keys:
        if key in labels:
            return labels[key]
    return None


def format_value(value: Any, info: ColumnInfo) -> str:
    """*value* as the column's format shows it; as stored when it does not apply (text in a number format)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if info.value_labels:
        label = value_label_of(value, info.value_labels)
        if label is not None:
            return label
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
