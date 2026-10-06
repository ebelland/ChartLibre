"""Whole-table and whole-column tools: the things a spreadsheet's Data menu does.

Duplicate, sort, change a column's type, fill its gaps, find and replace,
summarise, group and aggregate. Like the hand edits in ``editing.py``, each
change records an undo snapshot first (joining ``undo_entry`` when one is
open), because none of them can be recomputed from a source. A tool that
creates a table records that table as created, so Undo drops it again.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from app.data.repo._common import RepoHost, _is_ident, _quote_ident, ensure_connection_wrapper
from app.utils.coercion import to_numbers

#: The declared types a column can be changed to, in SQLite's own names.
CAST_TYPES: tuple[str, ...] = ("REAL", "INTEGER", "TEXT")

#: How fill_missing can fill a gap.
FILL_METHODS: tuple[str, ...] = ("constant", "mean", "median", "previous", "linear")

#: Aggregates group_aggregate offers, as SQL function names.
AGGREGATES: tuple[str, ...] = ("COUNT", "SUM", "AVG", "MIN", "MAX", "COUNT_DISTINCT")


class TableToolsMixin(RepoHost):
    """Table-level tools with undo."""

    __slots__ = ()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _schema(self, table_name: str) -> list[tuple[str, str]]:
        """(column, declared type) pairs, in table order."""
        return [(str(row[1]), str(row[2] or "")) for row in self.table_info(table_name)]

    @ensure_connection_wrapper
    def free_table_name(self, base: str) -> str:
        """*base*, or *base* with the first free numeric suffix."""
        assert self._con is not None
        taken = {
            str(row[0]).lower()
            for row in self._con.execute("SELECT name FROM sqlite_master").fetchall()
        }
        if base.lower() not in taken:
            return base
        index = 2
        while f"{base}_{index}".lower() in taken:
            index += 1
        return f"{base}_{index}"

    def _require_column(self, table_name: str, column_name: str | None) -> str:
        """Return *column_name* if the table has it, or raise a readable error."""
        if not column_name or column_name not in {n for n, _k in self._schema(table_name)}:
            raise ValueError(f"'{column_name}' is not a column of '{table_name}'")
        return column_name

    def _rebuild(
        self,
        table_name: str,
        schema: Sequence[tuple[str, str]],
        select_sql: str,
    ) -> None:
        """Replace *table_name* with *schema*, filled by *select_sql* (no rowids kept)."""
        assert self._con is not None
        temporary = _quote_ident(f"__rebuild_{table_name}")
        columns_sql = ", ".join(f"{_quote_ident(name)} {kind}".strip() for name, kind in schema)
        names = ", ".join(_quote_ident(name) for name, _kind in schema)
        self._con.execute(f"DROP TABLE IF EXISTS {temporary}")
        self._con.execute(f"CREATE TABLE {temporary} ({columns_sql})")
        self._con.execute(f"INSERT INTO {temporary} ({names}) {select_sql}")
        self._con.execute(f"DROP TABLE {_quote_ident(table_name)}")
        self._con.execute(f"ALTER TABLE {temporary} RENAME TO {_quote_ident(table_name)}")
        self._commit()

    # ------------------------------------------------------------------
    # Table tools
    # ------------------------------------------------------------------
    @ensure_connection_wrapper
    def duplicate_table(self, table_name: str, new_name: str | None = None) -> str:
        """Copy a table - columns, declared types and rows - and return the copy's name."""
        assert self._con is not None
        name = self.free_table_name(new_name or f"{table_name}_copy")
        if not _is_ident(name):
            raise ValueError(f"'{name}' is not a usable table name")
        schema = self._schema(table_name)
        if not schema:
            raise ValueError(f"'{table_name}' has no columns to copy")
        self.snapshot_for_undo([name], label=f"Duplicate '{table_name}' as '{name}'")
        columns_sql = ", ".join(f"{_quote_ident(c)} {kind}".strip() for c, kind in schema)
        names = ", ".join(_quote_ident(c) for c, _kind in schema)
        self._con.execute(f"CREATE TABLE {_quote_ident(name)} ({columns_sql})")
        self._con.execute(
            f"INSERT INTO {_quote_ident(name)} (rowid, {names}) "
            f"SELECT rowid, {names} FROM {_quote_ident(table_name)}"
        )
        # The copy keeps the original's notes and information (a DOE design).
        self._con.execute(
            "INSERT OR REPLACE INTO __table_descriptors__ (name, notes, info_json) "
            "SELECT ?, notes, info_json FROM __table_descriptors__ WHERE name = ?",
            (name, table_name),
        )
        self._commit()
        return name

    def sort_table(
        self,
        table_name: str,
        column_name: str,
        *,
        descending: bool = False,
        undo_entry: int | None = None,
    ) -> None:
        """Reorder the rows by one column, empty cells last.

        The row order *is* the rowid order, so this rewrites the table the
        way a spreadsheet's Sort does. Refused on a table whose rowid is one
        of its own columns (INTEGER PRIMARY KEY): renumbering would change
        the user's key values.
        """
        if self.has_integer_primary_key(table_name):
            raise ValueError("this table's row ids are its own key column and cannot be renumbered")
        schema = self._schema(table_name)
        if column_name not in {name for name, _kind in schema}:
            raise ValueError(f"'{column_name}' is not a column of this table")
        self.snapshot_for_undo(
            [table_name], label=f"Sort '{table_name}' by '{column_name}'", entry_id=undo_entry
        )
        column = _quote_ident(column_name)
        direction = "DESC" if descending else "ASC"
        names = ", ".join(_quote_ident(name) for name, _kind in schema)
        self._rebuild(
            table_name,
            schema,
            f"SELECT {names} FROM {_quote_ident(table_name)} "
            f"ORDER BY {column} IS NULL, {column} {direction}, rowid",
        )

    def cast_column(
        self,
        table_name: str,
        column_name: str,
        new_type: str,
        *,
        undo_entry: int | None = None,
    ) -> int:
        """Change a column's declared type and convert its values.

        Returns how many non-empty values could not be converted; those are
        left empty (NULL) rather than stored as text in a numeric column.
        Rowids are kept.
        """
        kind = str(new_type or "").strip().upper()
        if kind not in CAST_TYPES:
            raise ValueError(f"'{new_type}' is not one of {', '.join(CAST_TYPES)}")
        schema = self._schema(table_name)
        if column_name not in {name for name, _kind in schema}:
            raise ValueError(f"'{column_name}' is not a column of this table")
        self.snapshot_for_undo(
            [table_name],
            label=f"Change '{column_name}' to {kind} in '{table_name}'",
            entry_id=undo_entry,
        )
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        values = self._con.execute(
            f"SELECT rowid, {_quote_ident(column_name)} FROM {table_sql}"
        ).fetchall()

        converted: list[tuple[Any, int]] = []
        failed = 0
        for rowid, value in values:
            new_value = _convert(value, kind)
            if new_value is None and value is not None and str(value).strip() != "":
                failed += 1
            converted.append((new_value, int(rowid)))

        new_schema = [(name, kind if name == column_name else declared) for name, declared in schema]
        names = ", ".join(_quote_ident(name) for name, _kind in schema)
        temporary = _quote_ident(f"__rebuild_{table_name}")
        columns_sql = ", ".join(f"{_quote_ident(n)} {k}".strip() for n, k in new_schema)
        self._con.execute(f"DROP TABLE IF EXISTS {temporary}")
        self._con.execute(f"CREATE TABLE {temporary} ({columns_sql})")
        self._con.execute(
            f"INSERT INTO {temporary} (rowid, {names}) SELECT rowid, {names} FROM {table_sql}"
        )
        self._con.executemany(
            f"UPDATE {temporary} SET {_quote_ident(column_name)} = ? WHERE rowid = ?", converted
        )
        self._con.execute(f"DROP TABLE {table_sql}")
        self._con.execute(f"ALTER TABLE {temporary} RENAME TO {table_sql}")
        self._commit()
        return failed

    def column_stats(self, table_name: str, column_name: str) -> dict[str, Any]:
        """Count, empties, distinct values and - when numeric - min/max/mean/median."""
        self._require_column(table_name, column_name)
        assert self._con is not None
        column = _quote_ident(column_name)
        values = pd.Series(
            [row[0] for row in self._con.execute(
                f"SELECT {column} FROM {_quote_ident(table_name)}"
            ).fetchall()],
            dtype=object,
        )
        empty = values.isna() | (values.astype(str).str.strip() == "")
        present = values[~empty]
        stats: dict[str, Any] = {
            "rows": int(len(values)),
            "empty": int(empty.sum()),
            "distinct": int(present.astype(str).nunique()),
        }
        numbers = to_numbers(present).dropna()
        if len(numbers):
            stats.update(
                min=float(numbers.min()),
                max=float(numbers.max()),
                mean=float(numbers.mean()),
                median=float(numbers.median()),
                numeric=int(len(numbers)),
            )
        return stats

    def fill_missing(
        self,
        table_name: str,
        column_name: str,
        method: str,
        value: Any = None,
        *,
        undo_entry: int | None = None,
    ) -> int:
        """Fill the empty cells of one column, in row order. Returns how many were filled.

        ``constant`` writes *value*; ``mean`` and ``median`` use the column's
        own numbers; ``previous`` carries the last value down; ``linear``
        interpolates between the numbers either side (gaps at the ends stay
        empty - there is nothing to interpolate from).
        """
        if method not in FILL_METHODS:
            raise ValueError(f"unknown fill method '{method}'")
        self._require_column(table_name, column_name)
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        column = _quote_ident(column_name)
        rows = self._con.execute(f"SELECT rowid, {column} FROM {table_sql} ORDER BY rowid").fetchall()
        if not rows:
            return 0
        rowids = [int(row[0]) for row in rows]
        raw = pd.Series([row[1] for row in rows], dtype=object)
        empty = raw.isna() | (raw.astype(str).str.strip() == "")

        if method == "constant":
            filled = raw.where(~empty, value)
        elif method in ("mean", "median", "linear"):
            numbers = to_numbers(raw.where(~empty))
            if method == "mean":
                filled = numbers.fillna(numbers.mean())
            elif method == "median":
                filled = numbers.fillna(numbers.median())
            else:
                filled = numbers.interpolate(method="linear", limit_area="inside")
            filled = filled.astype(object).where(empty, raw)
        else:
            filled = raw.where(~empty).ffill()

        updates = [
            (_plain(filled.iat[i]), rowids[i])
            for i in range(len(rowids))
            if empty.iat[i] and _plain(filled.iat[i]) is not None
        ]
        if not updates:
            return 0
        self.snapshot_for_undo(
            [table_name], label=f"Fill empty '{column_name}' in '{table_name}'", entry_id=undo_entry
        )
        self._con.executemany(f"UPDATE {table_sql} SET {column} = ? WHERE rowid = ?", updates)
        self._commit()
        return len(updates)

    def find_replace(
        self,
        table_name: str,
        find: str,
        replace: str,
        *,
        columns: Sequence[str] | None = None,
        whole_cell: bool = False,
        undo_entry: int | None = None,
    ) -> int:
        """Replace text in some or all columns. Returns how many cells changed.

        *whole_cell* replaces cells whose entire value equals *find* (and
        stores a number as a number); otherwise every occurrence inside a
        text cell is replaced. Numeric cells are only ever matched whole.
        """
        if find == "":
            raise ValueError("there is nothing to find")
        schema = self._schema(table_name)
        known = {name: kind.upper() for name, kind in schema}
        targets = [c for c in (columns or [n for n, _k in schema]) if c in known and c.lower() != "rowid"]
        self.snapshot_for_undo(
            [table_name], label=f"Replace '{find}' in '{table_name}'", entry_id=undo_entry
        )
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        replacement: Any = _number_or_text(replace)
        changed = 0
        for name in targets:
            column = _quote_ident(name)
            if whole_cell or known[name] in ("REAL", "INTEGER", "NUMERIC", "FLOAT", "DOUBLE", "INT"):
                # A number is matched as a number too: 5 finds 5.0.
                number = _number_or_text(find)
                cursor = self._con.execute(
                    f"UPDATE {table_sql} SET {column} = ? "
                    f"WHERE CAST({column} AS TEXT) = ? OR (typeof({column}) IN ('integer', 'real') AND {column} = ?)",
                    (replacement, find, number if not isinstance(number, str) else None),
                )
            else:
                cursor = self._con.execute(
                    f"UPDATE {table_sql} SET {column} = REPLACE({column}, ?, ?) "
                    f"WHERE typeof({column}) = 'text' AND instr({column}, ?) > 0",
                    (find, replace, find),
                )
            changed += int(cursor.rowcount or 0)
        self._commit()
        return changed

    def _group_select(
        self,
        table_name: str,
        group_by: Sequence[str],
        measures: Sequence[tuple[str, str | None]],
        include_hidden: bool = False,
    ) -> str:
        """The SELECT behind group_aggregate, checked against the schema.

        *measures* are (aggregate, column) pairs; COUNT with no column counts
        rows. No grouping column is allowed too: one row of totals.
        """
        schema_names = [name for name, _kind in self._schema(table_name)]
        groups = [g for g in group_by if g in schema_names]
        if not measures:
            raise ValueError("add at least one measure")
        selected: list[str] = [_quote_ident(g) for g in groups]
        labels: set[str] = {g.lower() for g in groups}
        for aggregate, column in measures:
            func = str(aggregate or "").upper()
            if func not in AGGREGATES:
                raise ValueError(f"'{aggregate}' is not one of {', '.join(AGGREGATES)}")
            if column is None and func == "COUNT":
                measure, label = "COUNT(*)", "count"
            elif column not in schema_names:
                raise ValueError(f"choose the column to aggregate with {func}")
            elif func == "COUNT_DISTINCT":
                measure, label = f"COUNT(DISTINCT {_quote_ident(str(column))})", f"distinct_{column}"
            else:
                measure, label = f"{func}({_quote_ident(str(column))})", f"{func.lower()}_{column}"
            unique, index = label, 2
            while unique.lower() in labels:
                unique, index = f"{label}_{index}", index + 1
            labels.add(unique.lower())
            selected.append(f"{measure} AS {_quote_ident(unique)}")
        where = (
            'WHERE COALESCE("Hide", 0) = 0'
            if "Hide" in schema_names and not include_hidden
            else ""
        )
        sql = f"SELECT {', '.join(selected)} FROM {_quote_ident(table_name)} {where}"
        if groups:
            group_sql = ", ".join(_quote_ident(g) for g in groups)
            sql += f" GROUP BY {group_sql} ORDER BY {group_sql}"
        return sql

    def group_table_name(self, table_name: str, group_by: Sequence[str]) -> str:
        """The name group_aggregate gives its table when none is asked for."""
        groups = [str(g) for g in group_by]
        base = f"{table_name}_by_{'_'.join(groups)}" if groups else f"{table_name}_summary"
        return self.free_table_name(base)

    @ensure_connection_wrapper
    def group_aggregate_preview(
        self,
        table_name: str,
        group_by: Sequence[str],
        measures: Sequence[tuple[str, str | None]],
        *,
        include_hidden: bool = False,
        limit: int = 50,
    ) -> tuple[pd.DataFrame, int]:
        """The first *limit* rows group_aggregate would write, and how many in all."""
        assert self._con is not None
        sql = self._group_select(table_name, group_by, measures, include_hidden)
        total = int(self._con.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0])
        frame = pd.read_sql_query(f"SELECT * FROM ({sql}) LIMIT ?", self._con, params=(int(limit),))
        return frame, total

    def group_aggregate(
        self,
        table_name: str,
        group_by: Sequence[str],
        aggregate: str | None = None,
        value_column: str | None = None,
        new_name: str | None = None,
        *,
        measures: Sequence[tuple[str, str | None]] | None = None,
        include_hidden: bool = False,
    ) -> str:
        """Write GROUP BY *group_by* to a new table; return its name.

        One measure as *aggregate* of *value_column*, or several as
        *measures*. Rows marked Hide are left out, as they are from the
        charts, unless *include_hidden*.
        """
        if measures is None:
            measures = [(str(aggregate or ""), value_column)]
        sql = self._group_select(table_name, group_by, measures, include_hidden)
        name = (new_name or "").strip() or self.group_table_name(
            table_name, [g for g in group_by if g in {n for n, _k in self._schema(table_name)}]
        )
        if not _is_ident(name):
            raise ValueError(f"'{name}' is not a usable table name")
        if name != self.free_table_name(name):
            raise ValueError(f"a table or query named '{name}' already exists")
        self.snapshot_for_undo([name], label=f"Group '{table_name}' into '{name}'")
        assert self._con is not None
        self._con.execute(f"CREATE TABLE {_quote_ident(name)} AS {sql}")
        self._commit()
        return name


def _plain(value: Any) -> Any:
    """A value sqlite3 can bind: numpy scalars unboxed, NaN as NULL."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _number_or_text(text: str) -> Any:
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return int(number) if number.is_integer() and "." not in str(text) else number


def _convert(value: Any, kind: str) -> Any:
    if value is None or str(value).strip() == "":
        return None
    if kind == "TEXT":
        if isinstance(value, float) and value.is_integer():
            return str(int(value)) if abs(value) < 1e15 else str(value)
        return str(value)
    try:
        number = float(str(value).strip().replace(",", ".")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    if kind == "INTEGER":
        return int(round(number))
    return number
