"""A chart's axis read as a table, for the analyses of a table ("Data from: Chart").

The axis is the unit: every series it draws, together, each as the points the
chart shows - its own SQL, its filters and expressions. A series' measure (its
y, or its value) is a column named after the series. A role every series has
with the same values - their x, most often - is one column, named after the
axis' label for it; otherwise each series keeps its own ("<series> x"). Series
of different lengths are not lined up by x, which would drop points without
saying so: they stand side by side, a row per position.
"""
from __future__ import annotations

import re

import pandas as pd

from app.data.data_source import parse_roles
from app.data.descriptors import AxisDescriptor

#: The role that is what a series measures - its column is named after the series.
MEASURE_ROLES = ("y", "value", "values", "z")


def axis_frame(repo, axis: AxisDescriptor) -> pd.DataFrame:  # noqa: ANN001 - a SqliteRepo
    """*axis*'s series as one frame: see the module's description."""
    per_series: list[tuple[str, dict[str, pd.Series]]] = []
    for series in axis.series:
        sql = str(series.sql_query or "").strip()
        if not sql:
            continue
        data = repo.query_df(sql)
        roles = {str(role): str(column) for role, column in (parse_roles(series.roles) or {}).items() if column}
        if not roles:
            roles = {role: role for role in ("x", "y") if role in data.columns}
        columns = {role: data[column].reset_index(drop=True) for role, column in roles.items() if column in data.columns}
        if columns:
            per_series.append((str(series.name or f"Series {len(per_series) + 1}"), columns))
    if not per_series:
        return pd.DataFrame()

    labels = {"x": axis.x_label, "y": axis.y_label, "z": axis.z_label}
    measures = {name: next((role for role in MEASURE_ROLES if role in columns), list(columns)[-1])
                for name, columns in per_series}
    every_role = list(dict.fromkeys(role for _name, columns in per_series for role in columns))
    # A role every series has with the same values, and is no series' measure: one column for all.
    shared = [role for role in every_role
              if role not in measures.values()
              and all(role in columns and columns[role].equals(per_series[0][1].get(role, pd.Series(dtype=float)))
                      for _name, columns in per_series)]
    out: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series) -> None:
        unique, count = name, 2
        while unique in out:
            unique, count = f"{name} {count}", count + 1
        out[unique] = values

    for role in shared:
        add(str(labels.get(role) or role), per_series[0][1][role])
    for name, columns in per_series:
        for role, values in columns.items():
            if role not in shared:
                add(name if role == measures[name] else f"{name} {role}", values)
    return pd.concat(out, axis=1)


def axis_table_name(figure_name: str, axis: AxisDescriptor) -> str:
    """A table name for *axis*' data: its figure and its title, as an identifier."""
    text = f"{figure_name} {axis.title or axis.name or ''}"
    return re.sub(r"\W+", "_", text).strip("_") or "chart_data"
