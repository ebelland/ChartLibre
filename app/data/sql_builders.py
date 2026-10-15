"""SQL for the Tables operations of JMP and the like (AN-11): join, union, stack, split, transpose, recode, bin.

The Query Builder writes these statements into its editor, where they can
be read, changed, run and saved as a query (read live) or as a table - so
each operation is plain SQLite, not a hidden calculation. Every function
takes names and values and returns the statement; nothing here reads the
project. Split, transpose and the bins need values from the data (the
labels, the edges): the dialog reads those and passes them in, so the
statement holds them literally, as one written by hand would.

* join - rows of two tables matched on one or more pairs of columns:
  inner, left, right or full outer; a column both tables have gets the
  second table's name as a suffix; optionally a Match column (Both, Left
  only, Right only).
* union - tables one under the other, columns matched by name (NULL where
  a table lacks one), optionally with a Source column; UNION ALL keeps
  repeated rows, UNION drops them.
* stack - wide to long: each chosen column becomes a row of (Label, Data),
  the other columns repeated, in the source's row order.
* split - long to wide: a column's values become columns filled from
  another, one row per combination of the grouping columns.
* transpose - columns become rows, the rows columns (named by a label
  column, or Row 1, Row 2...).
* recode - a column's values mapped to new ones (a new column, or in place).
* bin - a numeric column cut at the given edges into labelled intervals.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

JOIN_KINDS: tuple[str, ...] = ("INNER", "LEFT", "RIGHT", "FULL OUTER")
SPLIT_AGGREGATES: tuple[str, ...] = ("MAX", "MIN", "SUM", "AVG", "COUNT")
#: The most columns a split or a transpose writes: a wide table past this is a mistake, not a result.
MAX_NEW_COLUMNS = 500
#: Between the items of a long SELECT list: one per line, aligned under the first.
_LIST_SEPARATOR = ",\n       "


def q(name: str) -> str:
    """A quoted identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def literal(value: object) -> str:
    """A SQL literal: a number as it is, anything else as quoted text, None as NULL."""
    if value is None:
        return "NULL"
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return repr(float(value)) if isinstance(value, (float, np.floating)) else str(int(value))
    return "'" + str(value).replace("'", "''") + "'"


def _typed(text: str) -> str:
    """A value typed by a person: a number literal when it reads as one, else text."""
    try:
        float(text)
    except ValueError:
        return literal(text)
    return text.strip()


def join(left: str, left_columns: Sequence[str], right: str, right_columns: Sequence[str],
         pairs: Sequence[tuple[str, str]], *, kind: str = "INNER", suffix: str = "_2",
         match_column: bool = False) -> str:
    kind = kind.upper()
    if kind not in JOIN_KINDS:
        raise ValueError(f"Unknown join '{kind}'.")
    if not pairs:
        raise ValueError("Choose at least one pair of columns to match on.")
    # A right key with the same name as its left key is the same value: listed once,
    # and from whichever side has it (a full or right join leaves the left one empty).
    shared_keys = {b for a, b in pairs if a == b}
    taken = {c.lower() for c in left_columns}
    select = []
    for column in left_columns:
        if column in shared_keys and kind in ("RIGHT", "FULL OUTER"):
            select.append(f"COALESCE(a.{q(column)}, b.{q(column)}) AS {q(column)}")
        else:
            select.append(f"a.{q(column)}")
    for column in right_columns:
        if column in shared_keys:
            continue
        name = column if column.lower() not in taken else f"{column}{suffix}"
        taken.add(name.lower())
        select.append(f"b.{q(column)}" + (f" AS {q(name)}" if name != column else ""))
    if match_column:
        a_key, b_key = pairs[0]
        select.append(f"CASE WHEN a.{q(a_key)} IS NULL THEN 'Right only' WHEN b.{q(b_key)} IS NULL "
                      f"THEN 'Left only' ELSE 'Both' END AS {q('Match')}")
    on = " AND ".join(f"a.{q(a)} = b.{q(b)}" for a, b in pairs)
    return (f"SELECT {', '.join(select)}\nFROM {q(left)} AS a\n{kind} JOIN {q(right)} AS b\n  ON {on}")


def union(tables: Mapping[str, Sequence[str]], *, source_column: str | None = "Source",
          distinct: bool = False) -> str:
    """*tables*: name -> its columns. Every column of any table, matched by name."""
    if len(tables) < 2:
        raise ValueError("Choose at least two tables.")
    columns: list[str] = []
    for table_columns in tables.values():
        columns += [c for c in table_columns if c not in columns]
    parts = []
    for name, table_columns in tables.items():
        present = set(table_columns)
        select = [f"{literal(name)} AS {q(source_column)}"] if source_column else []
        select += [q(c) if c in present else f"NULL AS {q(c)}" for c in columns]
        parts.append(f"SELECT {', '.join(select)} FROM {q(name)}")
    return f"\n{'UNION' if distinct else 'UNION ALL'}\n".join(parts)


def stack(table: str, columns: Sequence[str], keep: Sequence[str], *, label: str = "Label",
          value: str = "Data", drop_missing: bool = False) -> str:
    if not columns:
        raise ValueError("Choose the columns to stack.")
    if label == value or {label, value} & set(keep):
        raise ValueError("The Label and Data names must be new and different.")
    kept = "".join(f"{q(c)}, " for c in keep)
    parts = [f"SELECT rowid AS __row, {index} AS __col, {kept}{literal(column)} AS {q(label)}, "
             f"{q(column)} AS {q(value)} FROM {q(table)}" for index, column in enumerate(columns)]
    where = f"\nWHERE {q(value)} IS NOT NULL" if drop_missing else ""
    return (f"SELECT {kept}{q(label)}, {q(value)} FROM (\n  " + "\n  UNION ALL\n  ".join(parts)
            + f"\n){where}\nORDER BY __row, __col")


def split(table: str, label: str, value: str, labels: Sequence[object], group_by: Sequence[str], *,
          aggregate: str = "MAX") -> str:
    aggregate = aggregate.upper()
    if aggregate not in SPLIT_AGGREGATES:
        raise ValueError(f"Unknown aggregate '{aggregate}'.")
    if label == value:
        raise ValueError("The split column and the values must be two different columns.")
    if not labels:
        raise ValueError(f"'{label}' has no values to make columns of.")
    if len(labels) > MAX_NEW_COLUMNS:
        raise ValueError(f"'{label}' has {len(labels)} values: too many columns.")
    groups = [g for g in group_by if g not in (label, value)]
    source = q(table)
    if not groups:
        # As JMP's Split with no group: the n-th value of every label in row n.
        source = (f"(SELECT rowid AS __rowid, *, ROW_NUMBER() OVER (PARTITION BY {q(label)} ORDER BY rowid) "
                  f"AS __n FROM {q(table)})")
    select = [q(g) for g in groups]
    taken = {g.lower() for g in groups}
    for level in labels:
        name = str(level)
        while name.lower() in taken:
            name += "_2"
        taken.add(name.lower())
        select.append(f"{aggregate}(CASE WHEN {q(label)} = {literal(level)} THEN {q(value)} END) AS {q(name)}")
    sql = f"SELECT {', '.join(select)}\nFROM {source}"
    if groups:
        group_sql = ", ".join(q(g) for g in groups)
        return sql + f"\nGROUP BY {group_sql}\nORDER BY MIN(rowid)"
    return sql + "\nGROUP BY __n\nORDER BY __n"


def transpose(table: str, columns: Sequence[str], *, label_column: str | None = None,
              labels: Sequence[object] = (), rows: int = 0) -> str:
    """With *label_column*, its *labels* name the new columns; without, *rows* numbered columns."""
    if not columns:
        raise ValueError("Choose the columns to transpose.")
    if label_column:
        names = [str(level) for level in labels]
        if len(set(names)) != len(names):
            raise ValueError(f"'{label_column}' repeats values: every new column needs its own name.")
        tests = [f"{q(label_column)} = {literal(level)}" for level in labels]
    else:
        names = [f"Row {i + 1}" for i in range(rows)]
        tests = [f"rowid = (SELECT rowid FROM {q(table)} ORDER BY rowid LIMIT 1 OFFSET {i})" for i in range(rows)]
    if not names:
        raise ValueError("The table has no rows to make columns of.")
    if len(names) > MAX_NEW_COLUMNS:
        raise ValueError(f"{len(names)} rows: too many columns for a transpose.")
    parts = []
    for column in columns:
        cells = ", ".join(f"MAX(CASE WHEN {test} THEN {q(column)} END) AS {q(name)}"
                          for test, name in zip(tests, names))
        parts.append(f"SELECT {literal(column)} AS {q('Label')}, {cells} FROM {q(table)}")
    return "\nUNION ALL\n".join(parts)


def recode(table: str, columns: Sequence[str], column: str, mapping: Mapping[str, str], *,
           new_column: str | None = None) -> str:
    """*mapping*: value (as text) -> new value. In place when *new_column* is None."""
    if not mapping:
        raise ValueError("Give at least one new value.")
    cases = "\n".join(f"    WHEN {literal(old)} THEN {_typed(new)}" for old, new in mapping.items())
    expression = f"CASE CAST({q(column)} AS TEXT)\n{cases}\n    ELSE {q(column)} END"
    if new_column:
        if new_column in columns:
            raise ValueError(f"'{new_column}' is already a column.")
        select = [q(c) for c in columns] + [f"{expression} AS {q(new_column)}"]
    else:
        select = [f"{expression} AS {q(c)}" if c == column else q(c) for c in columns]
    return "SELECT " + _LIST_SEPARATOR.join(select) + f"\nFROM {q(table)}"


def bin_edges(values: np.ndarray, method: str, count: int = 5, edges: Sequence[float] = ()) -> np.ndarray:
    """The cut points: *count* equal widths, *count* quantiles (equal counts), or the *edges* given."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if method == "edges":
        cuts = np.unique(np.asarray(list(edges), dtype=float))
    else:
        if finite.size == 0:
            raise ValueError("The column has no numbers to bin.")
        if count < 1:
            raise ValueError("At least one bin.")
        if method == "equal width":
            low, high = float(finite.min()), float(finite.max())
            cuts = np.linspace(low, high if high > low else low + 1.0, count + 1)
        elif method == "quantiles":
            cuts = np.unique(np.quantile(finite, np.linspace(0, 1, count + 1)))
        else:
            raise ValueError(f"Unknown binning '{method}'.")
    if cuts.size < 2:
        raise ValueError("At least two edges are needed.")
    return cuts


def bin_labels(cuts: Sequence[float], style: str = "intervals") -> list[str]:
    pairs = list(zip(cuts[:-1], cuts[1:]))
    if style == "numbers":
        return [str(i + 1) for i in range(len(pairs))]
    if style == "midpoints":
        return [f"{(a + b) / 2:g}" for a, b in pairs]
    return [f"[{a:g}, {b:g}" + ("]" if i == len(pairs) - 1 else ")") for i, (a, b) in enumerate(pairs)]


def bin_column(table: str, columns: Sequence[str], column: str, cuts: Sequence[float], *, new_column: str,
               style: str = "intervals") -> str:
    """A new column with each value's interval: [a, b), the last one closed - so the maximum is in it."""
    if new_column in columns:
        raise ValueError(f"'{new_column}' is already a column.")
    labels = bin_labels(cuts, style)
    value = f"CAST({q(column)} AS REAL)"
    cases = []
    for i, (a, b, name) in enumerate(zip(cuts[:-1], cuts[1:], labels)):
        upper = "<=" if i == len(labels) - 1 else "<"
        label_sql = name if style in ("numbers", "midpoints") else literal(name)
        cases.append(f"    WHEN {value} >= {literal(float(a))} AND {value} {upper} {literal(float(b))} THEN {label_sql}")
    expression = "CASE\n" + "\n".join(cases) + "\n  END"
    select = [q(c) for c in columns] + [f"{expression} AS {q(new_column)}"]
    return "SELECT " + _LIST_SEPARATOR.join(select) + f"\nFROM {q(table)}"
