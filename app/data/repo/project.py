"""The project around the tables: table-level tools with undo, saved queries and indexes, integrity
checks and compaction, the operations record and the project info."""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from app.data.repo._common import (
    ROW_ID,
    DatabasePragmaInfo,
    DatabaseReport,
    SavedQuery,
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    ensure_connection_wrapper,
    ensure_read_only_select,
)
from app.logs.logger import applogger
from app.utils.coercion import to_numbers

#: The declared types a column can be changed to, in SQLite's own names.
CAST_TYPES: tuple[str, ...] = ("REAL", "INTEGER", "TEXT")


#: How fill_missing can fill a gap.
FILL_METHODS: tuple[str, ...] = ("constant", "mean", "median", "previous", "linear")


#: Aggregates group_aggregate offers, as SQL function names.
AGGREGATES: tuple[str, ...] = ("COUNT", "SUM", "AVG", "MIN", "MAX", "COUNT_DISTINCT")


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


OPERATIONS_TABLE = "__operations__"


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """One applied operation, as recorded."""

    id: int
    #: UTC, ISO 8601 to the second.
    applied_at: str
    #: What the user saw: "Smoothing", "Control chart"...
    operation: str
    #: ``module:Class`` of the dialog, what re-running it needs.
    dialog: str
    #: The declared parameters' values, by name.
    parameters: dict[str, Any] = field(default_factory=dict)
    #: Every entry of the dialog, as it remembers them between uses.
    entries: dict[str, Any] = field(default_factory=dict)
    #: The series read: id, name, SQL and roles of each.
    sources: list[dict[str, Any]] = field(default_factory=list)
    #: The tables written, and the result names.
    results: list[dict[str, Any]] = field(default_factory=list)
    app_version: str = ""
    #: The report shown in the results pane, as text or markup.
    report: str = ""


def _json(value: Any) -> str:
    # default=str: a parameter holding a date or a NumPy number is still
    # worth recording as text rather than failing the Apply it belongs to.
    return json.dumps(value, ensure_ascii=False, default=str)


def _parsed(text: str | None, empty: Any) -> Any:
    try:
        value = json.loads(text) if text else empty
    except (TypeError, ValueError):
        return empty
    return value if isinstance(value, type(empty)) else empty


def _touches(record: OperationRecord, table: str) -> bool:
    if any(str(result.get("table", "")) == table for result in record.results):
        return True
    for source in record.sources:
        sql = str(source.get("sql_query", ""))
        if f'"{table}"' in sql or f"[{table}]" in sql:
            return True
        if any(token == table for token in sql.replace(",", " ").replace("(", " ").replace(")", " ").split()):
            return True
    return False


PROJECT_INFO_TABLE = "__project_info__"


#: The project's preview picture, embedded: one row holding the PNG itself. A picture
#: kept as a file beside the project is lost whenever the project alone is copied,
#: sent or packaged (issue #8).
PROJECT_PREVIEW_TABLE = "__project_preview__"


def file_created_at(path: Path) -> datetime | None:
    """When *path* was created, as the file system knows it, in UTC.

    macOS and Windows record a creation time; Linux, as Python sees it,
    only the last modification, which is the best there is.
    """
    try:
        info = os.stat(path)
    except OSError:
        return None
    stamp = getattr(info, "st_birthtime", None)
    if stamp is None:
        stamp = info.st_ctime if os.name == "nt" else info.st_mtime
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


if TYPE_CHECKING:

    class _Host:
        """What this part uses of SqliteRepo: copies of the signatures, for the type checker only."""

        db_path: Path
        _con: sqlite3.Connection | None
        @staticmethod
        def ensure_dhub_extension(path: Path) -> Path:
            ...
        def _commit(self) -> None:
            ...
        DESCRIPTOR_TABLES: ClassVar[tuple[str, ...]]
        def snapshot_for_undo(self, tables: Sequence[str], *, label: str, entry_id: int | None=None) -> int | None:
            ...
        def list_user_tables(self, *, include_internal: bool=False) -> pd.DataFrame:
            ...
        @ensure_connection_wrapper
        def table_info(self, table: str) -> list[sqlite3.Row]:
            ...
        @ensure_connection_wrapper
        def ensure_row_id(self, table: str) -> None:
            ...
        @ensure_connection_wrapper
        def row_count(self, table: str) -> int:
            ...
        def query_source_table(self, sql_query: str) -> str:
            ...
        @staticmethod
        def is_table_backed_sql(sql_query: str) -> bool:
            ...
        def sql_with_row_id(self, sql_query: str) -> str:
            ...
        @ensure_connection_wrapper
        def list_table_names(self) -> list[str]:
            ...
        @ensure_connection_wrapper
        def check_if_table_exists(self, table_name: str) -> bool:
            ...
        def has_integer_primary_key(self, table_name: str) -> bool:
            ...
else:
    _Host = object


class ProjectMixin(_Host):
    """The project around the tables: table-level tools with undo, saved queries and indexes, integrity A part of SqliteRepo (sqlite_repo.py)."""

    __slots__ = ()

    # ==================================================================
    # TableTools: Table-level tools with undo.
    # ==================================================================
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _schema(self, table_name: str) -> list[tuple[str, str]]:
        """(column, declared type) pairs, in table order - the row id's as "INTEGER PRIMARY KEY", for a copy to keep it."""
        return [
            (str(row[1]), "INTEGER PRIMARY KEY" if str(row[1]) == ROW_ID and int(row[5] or 0) else str(row[2] or ""))
            for row in self.table_info(table_name)
        ]

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

    @ensure_connection_wrapper
    def free_hidden_table_name(self, base: str) -> str:
        """A free name for a table the user should not see: "_base_".

        The Tables list, the chart data sources and the project's counts leave
        out the names that begin and end with an underscore (the application's
        own "__operations__" is one), so a table that only feeds a chart is
        marked as such by its name. The numeric suffix of a taken name goes
        inside the underscores, or the name would stop being hidden.
        """
        assert self._con is not None
        taken = {
            str(row[0]).lower()
            for row in self._con.execute("SELECT name FROM sqlite_master").fetchall()
        }
        name, index = f"_{base}_", 2
        while name.lower() in taken:
            name = f"_{base}_{index}_"
            index += 1
        return name

    def _require_column(self, table_name: str, column_name: str | None) -> str:
        """Return *column_name* if the table has it, or raise a readable error."""
        if not column_name or column_name == ROW_ID or column_name not in {n for n, _k in self._schema(table_name)}:
            raise ValueError(f"'{column_name}' is not a column of '{table_name}'")
        return column_name

    def _rebuild(
        self,
        table_name: str,
        schema: Sequence[tuple[str, str]],
        select_sql: str,
    ) -> None:
        """Replace *table_name* with *schema*, filled by *select_sql* (no rowids kept).

        *select_sql* returns the schema's columns but the row id (ROW_ID),
        which numbers the rows anew in the order it returns them.
        """
        assert self._con is not None
        temporary = _quote_ident(f"__rebuild_{table_name}")
        columns_sql = ", ".join(f"{_quote_ident(name)} {kind}".strip() for name, kind in schema)
        names = ", ".join(_quote_ident(name) for name, _kind in schema if name != ROW_ID)
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
        if column_name == ROW_ID or column_name not in {name for name, _kind in schema}:
            raise ValueError(f"'{column_name}' is not a column of this table")
        self.snapshot_for_undo(
            [table_name], label=f"Sort '{table_name}' by '{column_name}'", entry_id=undo_entry
        )
        column = _quote_ident(column_name)
        direction = "DESC" if descending else "ASC"
        names = ", ".join(_quote_ident(name) for name, _kind in schema if name != ROW_ID)
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
        if column_name == ROW_ID or column_name not in {name for name, _kind in schema}:
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
        targets = [c for c in (columns or [n for n, _k in schema]) if c in known and c.lower() != "rowid" and c != ROW_ID]
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
        schema_names = [name for name, _kind in self._schema(table_name) if name != ROW_ID]
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
        self.ensure_row_id(name)
        self._commit()
        return name

    @ensure_connection_wrapper
    def create_table_from_query(self, name: str, sql: str) -> str:
        """Write what *sql* returns as a new table *name* (undoable); its name.

        The rows are fixed, unlike a saved query, which is read again each
        time: the Query Builder's Save as table.
        """
        name = name.strip()
        if not _is_ident(name):
            raise ValueError(f"'{name}' is not a usable table name")
        if name != self.free_table_name(name):
            raise ValueError(f"a table or query named '{name}' already exists")
        self.snapshot_for_undo([name], label=f"Save a query as '{name}'")
        assert self._con is not None
        self._con.execute(f"CREATE TABLE {_quote_ident(name)} AS SELECT * FROM ({sql.strip().rstrip(';')})")
        self.ensure_row_id(name)
        self._commit()
        return name

    # ==================================================================
    # Queries: Saved queries and explicit indexing.
    # ==================================================================
    @ensure_connection_wrapper
    def create_queries_table(self) -> None:
        """Create the saved-query table; called once, as the project opens."""
        assert self._con is not None
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __queries__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                sql           TEXT NOT NULL,
                settings_json TEXT
            );
            """
        )

    @ensure_connection_wrapper
    def save_query(
        self,
        name: str,
        sql: str,
        settings: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert or update a named SQL query and return its id."""
        assert self._con is not None

        clean_name = name.strip()
        clean_sql = sql.strip()
        if not clean_name:
            applogger.error("Query name is required")
        if not clean_sql:
            applogger.error("Query SQL is required")
        ensure_read_only_select(clean_sql)

        self._con.execute(
            """
            INSERT INTO __queries__ (name, sql, settings_json)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                sql = excluded.sql,
                settings_json = excluded.settings_json
            """,
            (clean_name, clean_sql, _dumps_json(settings)),
        )
        row = self._con.execute(
            "SELECT id FROM __queries__ WHERE name = ?",
            (clean_name,),
        ).fetchone()
        if row is None:
            applogger.error(f"Failed to save query: {clean_name}")
            return 0
        return int(row["id"])

    @ensure_connection_wrapper
    def get_query(self, name: str) -> SavedQuery | None:
        """Return a saved query by name."""
        assert self._con is not None

        row = self._con.execute(
            """
            SELECT id, name, sql, settings_json
            FROM __queries__
            WHERE name = ?
            """,
            (name,),
        ).fetchone()
        if row is None:
            return None
        return SavedQuery(
            id=int(row["id"]),
            name=str(row["name"]),
            sql=str(row["sql"]),
            settings=_loads_json(row["settings_json"]),
        )

    @ensure_connection_wrapper
    def list_queries(self) -> list[SavedQuery]:
        """Return all saved queries ordered by name."""
        assert self._con is not None

        rows = self._con.execute(
            """
            SELECT id, name, sql, settings_json
            FROM __queries__
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()
        return [
            SavedQuery(
                id=int(row["id"]),
                name=str(row["name"]),
                sql=str(row["sql"]),
                settings=_loads_json(row["settings_json"]),
            )
            for row in rows
        ]

    @ensure_connection_wrapper
    def new_query(
        self,
        name: str = "",
        *,
        table: str | None = None,
        sql: str = "",
    ) -> SavedQuery:
        """Return a new saved-query descriptor - named, seeded, not yet stored.

        Nothing is written.  A query is only worth a row once it runs, and a
        brand-new one does not: ``save_query`` refuses empty SQL and the
        builder refuses to save a statement that fails validation, so writing
        here would either need a second set of rules or leave rows behind that
        no chart can read.  The dialog holds the draft and stores it on Save,
        exactly like a new document in an editor.

        What the repository does own is the naming - an unnamed draft gets the
        first free ``Query <n>`` from :meth:`next_query_name` - and the seed
        statement, which is a plain SELECT over *table* when one is named and
        empty otherwise.  Nothing beyond the table is guessed: a WHERE or a
        JOIN invented here would produce a query that runs and returns the
        wrong rows, which is worse than one that does not run.

        The name is checked but not enforced: a name that is already a table's
        can never be selected (the table always wins) and a name already taken
        by a saved query will overwrite it on Save.  Both are the caller's
        decision - the builder asks about the second one - so this reports them
        to the log and hands the draft back either way.
        """
        clean_name = str(name or "").strip() or self.next_query_name()

        if self.check_if_table_exists(clean_name):
            applogger.warning(
                "New query %r has the name of a table; it could never be "
                "selected, because a table of the same name always wins.",
                clean_name,
            )
        elif self.get_query(clean_name) is not None:
            applogger.warning(
                "New query %r has the name of a saved query; saving it will "
                "replace that one.",
                clean_name,
            )

        clean_sql = str(sql or "").strip()
        if not clean_sql and table:
            clean_sql = f"SELECT * FROM {_quote_ident(table)}"

        # id None, not 0: None is what "no row yet" means everywhere else in
        # this module, and a 0 would compare equal to a real id in a falsy
        # test while looking like one in a log line.
        return SavedQuery(id=None, name=clean_name, sql=clean_sql, settings={})

    @ensure_connection_wrapper
    def next_query_name(self, base: str = "Query") -> str:
        """Return a free name for a new saved query, as ``"<base> <n>"``.

        Free of *both* saved queries and tables.  A query named after a table
        can never be selected - ``list_data_sources`` resolves the name against
        the schema first, so the table always wins - which is why the builder
        refuses such a name on Save; suggesting one here would be offering a
        name that is about to be rejected.

        Compared case-insensitively: SQLite resolves a table name whatever its
        case, so "Query 1" would still be shadowed by a table called "query 1",
        and two saved queries differing only in case are a trap for whoever has
        to tell them apart in the list.
        """
        stem = str(base or "Query").strip() or "Query"

        taken = {saved.name.strip().lower() for saved in self.list_queries()}
        taken.update(name.strip().lower() for name in self.list_table_names())

        index = 1
        while f"{stem} {index}".lower() in taken:
            index += 1
        return f"{stem} {index}"

    @ensure_connection_wrapper
    def delete_query(self, name: str) -> bool:
        """Delete a saved query by name."""
        assert self._con is not None

        cur = self._con.execute(
            "DELETE FROM __queries__ WHERE name = ?",
            (name,),
        )
        return cur.rowcount > 0

    # =====================================================================
    # Stored SymPy/Python functions
    # =====================================================================
    @ensure_connection_wrapper
    def create_sympy_table(self) -> None:
        """Create the internal function store used by Function and Fit."""
        assert self._con is not None
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __sympy__ (
                function_name TEXT PRIMARY KEY,
                source_type   TEXT NOT NULL,
                symbols       TEXT NOT NULL DEFAULT '',
                code_string   TEXT NOT NULL,
                "From"        REAL NOT NULL DEFAULT 0.0,
                "To"          REAL NOT NULL DEFAULT 10.0,
                Points        INTEGER NOT NULL DEFAULT 200,
                Description   TEXT NOT NULL DEFAULT '',
                "YFrom"       REAL NOT NULL DEFAULT 0.0,
                "YTo"         REAL NOT NULL DEFAULT 10.0,
                YPoints       INTEGER NOT NULL DEFAULT 200,
                "Values"      TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {str(row[1]) for row in self._con.execute("PRAGMA table_info(__sympy__)").fetchall()}
        migrations = (
            ("From", 'ALTER TABLE __sympy__ ADD COLUMN "From" REAL NOT NULL DEFAULT 0.0'),
            ("To", 'ALTER TABLE __sympy__ ADD COLUMN "To" REAL NOT NULL DEFAULT 10.0'),
            ("Points", 'ALTER TABLE __sympy__ ADD COLUMN Points INTEGER NOT NULL DEFAULT 200'),
            ("Description", "ALTER TABLE __sympy__ ADD COLUMN Description TEXT NOT NULL DEFAULT ''"),
            # The y range of a function of x and y (z = f(x, y)), as From / To / Points are x's.
            ("YFrom", 'ALTER TABLE __sympy__ ADD COLUMN "YFrom" REAL NOT NULL DEFAULT 0.0'),
            ("YTo", 'ALTER TABLE __sympy__ ADD COLUMN "YTo" REAL NOT NULL DEFAULT 10.0'),
            ("YPoints", "ALTER TABLE __sympy__ ADD COLUMN YPoints INTEGER NOT NULL DEFAULT 200"),
            # The parameters' values, as JSON ({"a": 2.0}): a chart's defaults when the function is its source.
            ("Values", """ALTER TABLE __sympy__ ADD COLUMN "Values" TEXT NOT NULL DEFAULT ''"""),
        )
        for column, statement in migrations:
            if column not in columns:
                self._con.execute(statement)

    @ensure_connection_wrapper
    def list_sympy_functions(self) -> list[dict[str, str]]:
        self.create_sympy_table()
        assert self._con is not None
        rows = self._con.execute(
            "SELECT function_name, source_type, symbols, code_string, "
            '"From", "To", Points, Description, "YFrom", "YTo", YPoints, "Values" '
            "FROM __sympy__ ORDER BY function_name COLLATE NOCASE"
        ).fetchall()
        return [
            {
                "function_name": str(row["function_name"]),
                "source_type": str(row["source_type"]),
                "symbols": str(row["symbols"] or ""),
                "code_string": str(row["code_string"] or ""),
                "from": str(row["From"]), "to": str(row["To"]),
                "points": str(row["Points"]), "description": str(row["Description"] or ""),
                "y_from": str(row["YFrom"]), "y_to": str(row["YTo"]), "y_points": str(row["YPoints"]),
                "values": str(row["Values"] or ""),
            }
            for row in rows
        ]

    @ensure_connection_wrapper
    def get_sympy_function(self, name: str) -> dict[str, str] | None:
        self.create_sympy_table()
        assert self._con is not None
        row = self._con.execute(
            "SELECT function_name, source_type, symbols, code_string, "
            '"From", "To", Points, Description, "YFrom", "YTo", YPoints, "Values" '
            "FROM __sympy__ WHERE function_name = ?",
            (str(name).strip(),),
        ).fetchone()
        if row is None:
            return None
        return {
            "function_name": str(row["function_name"]),
            "source_type": str(row["source_type"]),
            "symbols": str(row["symbols"] or ""),
            "code_string": str(row["code_string"] or ""),
            "from": str(row["From"]), "to": str(row["To"]),
            "points": str(row["Points"]), "description": str(row["Description"] or ""),
            "y_from": str(row["YFrom"]), "y_to": str(row["YTo"]), "y_points": str(row["YPoints"]),
            "values": str(row["Values"] or ""),
        }

    @ensure_connection_wrapper
    def save_sympy_function(
        self, function_name: str, source_type: str, symbols: str, code_string: str,
        from_value: float = 0.0, to_value: float = 10.0, points: int = 200,
        description: str = "", y_from_value: float = 0.0, y_to_value: float = 10.0, y_points: int = 200,
        values: str = "",
    ) -> None:
        self.create_sympy_table()
        assert self._con is not None
        self._con.execute(
            """
            INSERT INTO __sympy__
                (function_name, source_type, symbols, code_string, "From", "To", Points, Description,
                 "YFrom", "YTo", YPoints, "Values")
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(function_name) DO UPDATE SET
                source_type = excluded.source_type,
                symbols = excluded.symbols,
                code_string = excluded.code_string,
                "From" = excluded."From",
                "To" = excluded."To",
                Points = excluded.Points,
                Description = excluded.Description,
                "YFrom" = excluded."YFrom",
                "YTo" = excluded."YTo",
                YPoints = excluded.YPoints,
                "Values" = excluded."Values"
            """,
            (function_name.strip(), source_type.strip(), symbols.strip(), code_string,
             float(from_value), float(to_value), int(points), description,
             float(y_from_value), float(y_to_value), int(y_points), str(values or "")),
        )
        self._commit()

    @ensure_connection_wrapper
    def delete_sympy_function(self, name: str) -> bool:
        self.create_sympy_table()
        assert self._con is not None
        cursor = self._con.execute(
            "DELETE FROM __sympy__ WHERE function_name = ?", (str(name).strip(),)
        )
        self._commit()
        return cursor.rowcount > 0

    # ==================================================================
    # Maintenance: Integrity checks, compaction, and saving the project elsewhere.
    # ==================================================================
    def repair_misplaced_hide_filters(self) -> list[str]:
        """Take off a Hide filter appended where it broke a series' query; returns the series fixed.

        Before the filter looked at the outer query only, a series written
        as "WITH source AS (... WHERE ...) SELECT ... FROM source" - what
        Geometry creates - was given " AND \"Hide\" = 0" at the end, which
        is not SQL: the series failed and was not drawn. Run when a project
        opens. Only a query that fails and works without the trailing
        filter is changed.
        """
        assert self._con is not None
        trailing = re.compile(r'\s+(?:AND|WHERE)\s+"Hide"\s*=\s*0\s*$', re.IGNORECASE)
        fixed: list[str] = []
        for row in self._con.execute(
            "SELECT id, name, sql_query FROM __series_descriptors__"
        ).fetchall():
            sql = str(row["sql_query"] or "")
            if not trailing.search(sql) or self._compiles(sql):
                continue
            repaired = trailing.sub("", sql)
            if self._compiles(repaired):
                self._con.execute(
                    "UPDATE __series_descriptors__ SET sql_query = ? WHERE id = ?",
                    (repaired, int(row["id"])),
                )
                fixed.append(str(row["name"]))
        if fixed:
            self._commit()
            applogger.warning(
                "Repaired the query of %d series broken by a misplaced Hide filter: %s",
                len(fixed), ", ".join(fixed),
            )
        return fixed

    @ensure_connection_wrapper
    def add_row_ids_to_series(self) -> int:
        """Every series' query selecting its table's row id too (sql_with_row_id); how many changed.

        Run when a project opens, after its tables got their row id column:
        a series made before R-07 then names its table rows like a new one.
        A query that would no longer run is left as it was.
        """
        assert self._con is not None
        changed = 0
        for row in self._con.execute("SELECT id, sql_query FROM __series_descriptors__").fetchall():
            sql = str(row["sql_query"] or "")
            with_id = self.sql_with_row_id(sql)
            if with_id != sql.strip().rstrip(";") and self._compiles(with_id):
                self._con.execute(
                    "UPDATE __series_descriptors__ SET sql_query = ? WHERE id = ?", (with_id, int(row["id"]))
                )
                changed += 1
        if changed:
            self._commit()
        return changed

    def _compiles(self, sql: str) -> bool:
        assert self._con is not None
        try:
            self._con.execute(f"EXPLAIN {sql}")
        except sqlite3.Error:
            return False
        return True

    @ensure_connection_wrapper
    def optimize_db(self) -> DatabaseReport:
        """Check the database, report what is wrong, then compact it.

        Checks first, compaction second: VACUUM rewrites the file, so anything
        it silently drops would no longer be reportable afterwards.

        Inside an open transaction the compaction is skipped rather than
        attempted.  SQLite refuses to VACUUM there, and the resulting
        OperationalError used to travel up through whatever multi-step
        operation was running and roll the whole thing back - which is how a
        spectral analysis of five series ended up applying one.  A missed
        compaction costs some disk space; an aborted operation costs the
        user's work.

        Returns the report as well as logging it, so callers can show it.
        """
        assert self._con is not None

        report = self.check_database()
        report.log()

        if self._con.in_transaction:
            applogger.warning(
                "Skipping VACUUM: a transaction is open. Call optimize_db "
                "after committing.",
            )
            return report

        self._con.execute("VACUUM")
        self._con.execute("ANALYZE")
        self._con.execute("PRAGMA optimize")
        applogger.info("Database optimized: VACUUM, ANALYZE, PRAGMA optimize.")
        return report

    @ensure_connection_wrapper
    def checkpoint(self) -> None:
        """Fold the WAL into the .dhub file itself, on disk, right now.

        Every change is already committed through SQLite's WAL as it
        happens - there is no "unsaved" state for ``Save`` to flush the way
        there would be for a document editor. What TRUNCATE actually buys is
        the .dhub file being the *whole* database on its own again (WAL mode
        otherwise leaves recent writes in the -wal side file until SQLite
        checkpoints on its own schedule) and that side file shrunk back to
        empty - useful before copying the file, backing it up, or handing it
        to a tool that only reads the .dhub. Safe inside an open transaction,
        unlike VACUUM: a checkpoint does not rewrite the file, only merges
        into it.
        """
        assert self._con is not None
        self._con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        applogger.info("Checkpointed: WAL folded into the database file.")

    @ensure_connection_wrapper
    def database_pragma_info(self) -> DatabasePragmaInfo:
        """Return everything SQLite and the filesystem can say about this
        database file, for the Database Info dialog's own "Info" section.

        Not a health check (see check_database for that) - just what the
        engine already tracks (page accounting, encoding, journal mode)
        and what the OS already knows (the file's own timestamps) without
        computing anything new, beyond a row count per table which SQLite
        itself has no PRAGMA for.
        """
        con = self._con
        assert con is not None

        def pragma(name: str) -> Any:
            row = con.execute(f"PRAGMA {name}").fetchone()
            return row[0] if row is not None else None

        tables = self.list_user_tables()
        table_names = [str(name) for name in tables["Table"]] if not tables.empty else []
        total_rows = sum(self.row_count(name) for name in table_names)

        file_created: datetime | None = None
        file_modified: datetime | None = None
        try:
            stat = Path(self.db_path).stat()
            file_modified = datetime.fromtimestamp(stat.st_mtime)
            # macOS/BSD only - st_birthtime does not exist on Linux, where
            # there is no true file-creation time to read at all (st_ctime
            # is "last metadata change", not creation, and reporting it as
            # one would be reporting the wrong thing with confidence).
            birthtime = getattr(stat, "st_birthtime", None)
            if birthtime is not None:
                file_created = datetime.fromtimestamp(birthtime)
        except OSError:
            pass

        return DatabasePragmaInfo(
            sqlite_version=sqlite3.sqlite_version,
            page_size=int(pragma("page_size") or 0),
            page_count=int(pragma("page_count") or 0),
            freelist_count=int(pragma("freelist_count") or 0),
            encoding=str(pragma("encoding") or ""),
            journal_mode=str(pragma("journal_mode") or ""),
            application_id=int(pragma("application_id") or 0),
            user_version=int(pragma("user_version") or 0),
            table_count=len(table_names),
            total_rows=total_rows,
            file_created=file_created,
            file_modified=file_modified,
        )

    @ensure_connection_wrapper
    def save_as(self, target_path: Path) -> Path:
        """Write a compacted copy of this database to *target_path*.

        Uses ``VACUUM INTO`` rather than copying the file on disk: with WAL
        mode active (see __init__), the .dhub file is not the whole database
        on its own until its -wal side file is checkpointed into it, so a
        plain filesystem copy of just the .dhub file can silently miss
        recent writes. ``VACUUM INTO`` instead builds one self-contained,
        up-to-date file directly from the live connection.

        Refuses inside an open transaction, on the same principle as
        :meth:`optimize_db`: SQLite refuses to VACUUM there, and letting the
        error travel up would abort whatever multi-step operation triggered
        it. An existing file at *target_path* is removed first - the target
        came out of a save dialog that has already confirmed the overwrite -
        because VACUUM INTO refuses to write over one itself.
        """
        assert self._con is not None
        target = self.ensure_dhub_extension(target_path)

        if self._con.in_transaction:
            raise RuntimeError(
                "Cannot save as while a transaction is open. Commit first."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()

        self._con.execute("VACUUM INTO ?", (str(target),))
        applogger.info("Saved a copy of the database to %s", target)
        return target

    @ensure_connection_wrapper
    def check_database(self) -> DatabaseReport:
        """Run integrity checks and look for orphaned descriptor rows/tables."""
        assert self._con is not None

        report = DatabaseReport()
        self._check_integrity(report)
        self._check_foreign_keys(report)
        self._check_zombies(report)
        return report

    def _check_integrity(self, report: DatabaseReport) -> None:
        """Run PRAGMA integrity_check and record anything but 'ok'."""
        assert self._con is not None
        try:
            rows = self._con.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.Error as exc:
            report.integrity_errors.append(f"integrity_check failed: {exc}")
            return

        for row in rows:
            message = str(row[0])
            if message.strip().lower() != "ok":
                report.integrity_errors.append(message)

    def _check_foreign_keys(self, report: DatabaseReport) -> None:
        """Record rows whose foreign key points at nothing."""
        assert self._con is not None
        try:
            rows = self._con.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.Error as exc:
            report.integrity_errors.append(f"foreign_key_check failed: {exc}")
            return

        for row in rows:
            # (table, rowid, parent table, fkid)
            report.foreign_key_errors.append(
                f"{row[0]} rowid={row[1]} references missing row in {row[2]}"
            )

    def _check_zombies(self, report: DatabaseReport) -> None:
        """Find descriptors pointing at nothing, and data nothing points at.

        Two directions, because they are different problems:

        * a **dangling reference** is a series or import link naming a table
          that no longer exists - the chart is broken and the user should know;
        * an **unreferenced table** is data no chart or link uses. That is not
          an error - it is often simply a table waiting to be plotted - so it
          is reported separately and only as information.
        """
        assert self._con is not None

        existing = set(self.list_table_names())
        # Hidden ones too ("_name_", a chart's own): a series over one is not broken.
        every_table = self._every_table()

        # Series whose SQL selects from a table that is gone.
        for row in self._con.execute(
            "SELECT id, name, sql_query FROM __series_descriptors__"
        ).fetchall():
            sql = str(row["sql_query"] or "")
            # A series over a saved query reads from a subquery; its inner
            # tables are checked through that query, not through this name.
            if not self.is_table_backed_sql(sql):
                continue
            table = self.query_source_table(sql)
            if table and table not in every_table:
                report.dangling_series.append(
                    f"series id={row['id']} '{row['name']}' reads from missing table '{table}'"
                )

        # Axes whose figure is gone, and series whose axis is gone.  The schema
        # declares these foreign keys, but only enforces them when the pragma
        # was on for every writer that ever touched the file.
        for label, child, parent, child_key in (
            ("axis", "__axis_descriptors__", "__figure_descriptors__", "figure_id"),
            ("series", "__series_descriptors__", "__axis_descriptors__", "axis_id"),
        ):
            for row in self._con.execute(
                f"SELECT c.id AS id, c.{child_key} AS parent_id FROM {child} AS c "
                f"LEFT JOIN {parent} AS p ON p.id = c.{child_key} "
                "WHERE p.id IS NULL"
            ).fetchall():
                report.orphan_descriptors.append(
                    f"{label} id={row['id']} belongs to missing parent id={row['parent_id']}"
                )

        # Import links whose destination table is gone.
        for row in self._con.execute(
            "SELECT id, table_name FROM __import_links__"
        ).fetchall():
            if str(row["table_name"]) not in every_table:
                report.dangling_links.append(
                    f"import link id={row['id']} targets missing table '{row['table_name']}'"
                )

        # Tables no series and no link refers to.
        referenced: set[str] = set()
        for row in self._con.execute(
            "SELECT sql_query FROM __series_descriptors__"
        ).fetchall():
            sql = str(row["sql_query"] or "")
            if not self.is_table_backed_sql(sql):
                # Credit the tables the subquery mentions, so a table used only
                # through a saved query is not reported as unreferenced.
                referenced.update(re.findall(r'\bfrom\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', sql, flags=re.IGNORECASE))
                continue
            table = self.query_source_table(sql)
            if table:
                referenced.add(table)
        for row in self._con.execute("SELECT table_name FROM __import_links__").fetchall():
            referenced.add(str(row["table_name"]))

        report.unreferenced_tables.extend(sorted(existing - referenced))
        report.leftover_tables.extend(
            f"table '{name}' is no longer used" for name in self._leftover_tables()
        )

    def _every_table(self) -> set[str]:
        """Every table in the file, the hidden and the application's own included."""
        assert self._con is not None
        return {str(row[0]) for row in self._con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    def _leftover_tables(self) -> list[str]:
        """The tables the application made that nothing uses: "__rebuild_" ones, hidden "_name_" ones no series
        or saved query reads, and an analysis' "_name" results that no series reads and no recorded operation
        wrote - what a Preview left behind when its savepoint was lost (a commit ended it)."""
        assert self._con is not None
        names = [str(row[0]) for row in self._con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()]
        hidden = [n for n in names if n.startswith("__rebuild_")
                  or (len(n) > 1 and n.startswith("_") and not n.startswith("__"))]
        if not hidden:
            return []
        sql_texts = [str(row[0] or "") for row in self._con.execute(
            "SELECT sql_query FROM __series_descriptors__").fetchall()]
        sql_texts += [str(saved.sql or "") for saved in self.list_queries()]
        used = " ".join(sql_texts)
        written = " ".join(str(row[0] or "") for row in self._con.execute(
            "SELECT results_json FROM __operations__").fetchall()) if "__operations__" in names else ""

        def mentioned(name: str, text: str) -> bool:
            return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))

        return [n for n in hidden
                if n.startswith("__rebuild_")
                or (not mentioned(n, used) and (n.endswith("_") or not mentioned(n, written)))]

    @ensure_connection_wrapper
    def repair_database(self) -> list[str]:
        """Fix what check_database finds that can be fixed (undoable); what was done, one line each.

        Axes of a deleted figure and series of a deleted axis, series reading
        a table that is gone (their chart was broken), import links to a
        table that is gone, and the tables the application made that nothing
        uses any more. A table someone made or imported is never touched,
        used or not: it is data.
        """
        assert self._con is not None
        leftovers = self._leftover_tables()
        self.snapshot_for_undo([*self.DESCRIPTOR_TABLES, "__import_links__", *leftovers],
                               label="Fix the database")
        done: list[str] = []
        existing = self._every_table()
        dangling = [
            int(row["id"]) for row in self._con.execute(
                "SELECT id, sql_query FROM __series_descriptors__").fetchall()
            if self.is_table_backed_sql(str(row["sql_query"] or ""))
            and (table := self.query_source_table(str(row["sql_query"] or ""))) and table not in existing
        ]
        for statement, what in (
            ("DELETE FROM __axis_descriptors__ WHERE figure_id NOT IN (SELECT id FROM __figure_descriptors__)",
             "axis/axes of a deleted figure"),
            ("DELETE FROM __series_descriptors__ WHERE axis_id NOT IN (SELECT id FROM __axis_descriptors__)",
             "series of a deleted axis"),
        ):
            count = self._con.execute(statement).rowcount
            if count:
                done.append(f"{count} {what} deleted")
        if dangling:
            self._con.executemany("DELETE FROM __series_descriptors__ WHERE id = ?", [(i,) for i in dangling])
            done.append(f"{len(dangling)} series reading a missing table deleted")
        links = [str(row[0]) for row in self._con.execute("SELECT table_name FROM __import_links__").fetchall()
                 if str(row[0]) not in existing]
        if links:
            self._con.executemany("DELETE FROM __import_links__ WHERE table_name = ?", [(t,) for t in links])
            done.append(f"{len(links)} import link(s) to a missing table deleted")
        for name in leftovers:
            self._con.execute(f"DROP TABLE IF EXISTS {_quote_ident(name)}")
            done.append(f"table '{name}' dropped")
        self._commit()
        for line in done:
            applogger.info("Database repair: %s", line)
        return done

    # ==================================================================
    # Operations: The ``__operations__`` record.
    # ==================================================================
    @ensure_connection_wrapper
    def create_operations_table(self) -> None:
        assert self._con is not None
        self._con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OPERATIONS_TABLE} (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                applied_at      TEXT NOT NULL,
                operation       TEXT NOT NULL,
                dialog          TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                entries_json    TEXT NOT NULL,
                sources_json    TEXT NOT NULL,
                results_json    TEXT NOT NULL,
                app_version     TEXT NOT NULL,
                report          TEXT NOT NULL
            )
            """
        )

    @ensure_connection_wrapper
    def record_operation(
        self,
        *,
        operation: str,
        dialog: str,
        parameters: Mapping[str, Any],
        entries: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        app_version: str,
        report: str = "",
    ) -> int:
        """Add one applied operation; returns its id.

        Committed with whatever transaction is open - the Apply's own, so
        the record and the results it describes land or roll back together.
        """
        assert self._con is not None
        self.create_operations_table()
        cursor = self._con.execute(
            f"""
            INSERT INTO {OPERATIONS_TABLE} (
                applied_at, operation, dialog, parameters_json, entries_json,
                sources_json, results_json, app_version, report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                str(operation),
                str(dialog),
                _json(dict(parameters)),
                _json(dict(entries)),
                _json([dict(source) for source in sources]),
                _json([dict(result) for result in results]),
                str(app_version),
                str(report or ""),
            ),
        )
        self._commit()
        return int(cursor.lastrowid or 0)

    @ensure_connection_wrapper
    def operations(self, *, table: str | None = None) -> list[OperationRecord]:
        """Every recorded operation, oldest first; with *table*, those that read or wrote it.

        "Read" means a source series whose SQL names the table - a plain
        text match on the quoted or bare name, which is what the series
        store.
        """
        assert self._con is not None
        self.create_operations_table()
        rows = self._con.execute(
            f"""
            SELECT id, applied_at, operation, dialog, parameters_json, entries_json,
                   sources_json, results_json, app_version, report
            FROM {OPERATIONS_TABLE}
            ORDER BY id
            """
        ).fetchall()
        records = [
            OperationRecord(
                id=int(row[0]),
                applied_at=str(row[1]),
                operation=str(row[2]),
                dialog=str(row[3]),
                parameters=_parsed(row[4], {}),
                entries=_parsed(row[5], {}),
                sources=_parsed(row[6], []),
                results=_parsed(row[7], []),
                app_version=str(row[8]),
                report=str(row[9]),
            )
            for row in rows
        ]
        if table is None:
            return records
        return [record for record in records if _touches(record, table)]

    # ==================================================================
    # ProjectInfo: The ``__project_info__`` entries.
    # ==================================================================
    @ensure_connection_wrapper
    def create_project_info_table(self) -> None:
        """Make the table if it is missing, and date the project when it is new to it."""
        assert self._con is not None
        existed = self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (PROJECT_INFO_TABLE,)
        ).fetchone()
        self._con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PROJECT_INFO_TABLE} (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        if existed:
            return
        path = Path(str(self.db_path)) if getattr(self, "db_path", None) else None
        # A file with nothing in it yet is a project being created now.
        has_content = self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE '__%__' ESCAPE '_' "
            "AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone() or self._con.execute("SELECT 1 FROM __figure_descriptors__ LIMIT 1").fetchone()
        born = (file_created_at(path) if (path is not None and has_content) else None) or datetime.now(timezone.utc)
        self._con.execute(
            f"INSERT OR IGNORE INTO {PROJECT_INFO_TABLE} (key, value) VALUES ('created', ?)", (_iso(born),)
        )
        self._commit()

    @ensure_connection_wrapper
    def project_info(self) -> dict[str, str]:
        """Every entry, by key; the ones never set are absent."""
        assert self._con is not None
        self.create_project_info_table()
        rows = self._con.execute(f"SELECT key, value FROM {PROJECT_INFO_TABLE}").fetchall()
        return {str(key): str(value) for key, value in rows}

    @ensure_connection_wrapper
    def set_project_info(self, entries: Mapping[str, str | None]) -> None:
        """Write *entries*; an empty or None value removes that entry."""
        assert self._con is not None
        self.create_project_info_table()
        for key, value in entries.items():
            text = "" if value is None else str(value).strip()
            if text:
                self._con.execute(
                    f"INSERT INTO {PROJECT_INFO_TABLE} (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(key), text),
                )
            else:
                self._con.execute(f"DELETE FROM {PROJECT_INFO_TABLE} WHERE key = ?", (str(key),))
        self._commit()

    @ensure_connection_wrapper
    def set_project_preview(self, png: bytes | None) -> None:
        """Embed *png* as the project's preview; None removes it."""
        assert self._con is not None
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {PROJECT_PREVIEW_TABLE} "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), png BLOB NOT NULL)"
        )
        if png:
            self._con.execute(
                f"INSERT OR REPLACE INTO {PROJECT_PREVIEW_TABLE} (id, png) VALUES (1, ?)", (sqlite3.Binary(png),)
            )
        else:
            self._con.execute(f"DELETE FROM {PROJECT_PREVIEW_TABLE}")
        self._commit()

    @ensure_connection_wrapper
    def project_preview(self) -> bytes | None:
        """The embedded preview PNG, or None."""
        assert self._con is not None
        try:
            row = self._con.execute(f"SELECT png FROM {PROJECT_PREVIEW_TABLE} WHERE id = 1").fetchone()
        except sqlite3.Error:
            return None
        return bytes(row[0]) if row and row[0] else None
