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


@dataclass
class ColumnInfo:
    label: str = ""
    unit: str = ""
    notes: str = ""
    #: The modelling type forced; "" lets the values decide.
    kind: str = ""
    missing: list[str] = field(default_factory=list)
    value_labels: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.label or self.unit or self.notes or self.kind or self.missing or self.value_labels)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ColumnInfo:
        kind = str(values.get("kind") or "")
        return cls(label=str(values.get("label") or ""), unit=str(values.get("unit") or ""),
                   notes=str(values.get("notes") or ""), kind=kind if kind in KINDS else "",
                   missing=[str(code) for code in values.get("missing") or []],
                   value_labels={str(k): str(v) for k, v in dict(values.get("value_labels") or {}).items()})


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
