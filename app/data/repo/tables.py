"""The user tables: listing, reading, importing, renaming and deleting them, their Hide, Selected and
ClusterId flags, and edits made by hand to cells, rows and columns."""
from __future__ import annotations

import json
import re
import sqlite3
import struct
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

from app import APP_NAME
from app.data.data_source import DataSource, quote_identifier, resolve_role_column
from app.data.repo._common import (
    _RETURNS_ROWS_RE,
    READ_FAILURES,
    ROW_ID,
    SavedQuery,
    StoredFunction,
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    check_sql_expression,
    descriptor_write_wrapper,
    ensure_connection_wrapper,
    is_read_only_select,
    read_only,
)
from app.data.select_sql import (
    has_projection_alias,
    split_top_level_select_items,
    sql_insert_select_expression,
    top_level_from,
    top_level_match,
)
from app.logs.logger import applogger
from app.utils.coercion import column_of, to_numbers

if TYPE_CHECKING:

    class _Host:
        """What this part uses of SqliteRepo: copies of the signatures, for the type checker only."""

        _con: sqlite3.Connection | None
        _is_connected: bool
        _preview_savepoint_name: str | None
        def _connect(self):
            ...
        def invalidate_series_cache(self) -> None:
            ...
        def refuse_if_blocked(self, sql: str | None) -> None:
            ...
        def _commit(self) -> None:
            ...
        @contextmanager
        @ensure_connection_wrapper
        def transaction(self, *, immediate: bool = False) -> Iterator[None]:
            ...
        def snapshot_for_undo(self, tables: Sequence[str], *, label: str, entry_id: int | None=None) -> int | None:
            ...
        @ensure_connection_wrapper
        @descriptor_write_wrapper
        def update_series_sql_query(self, series_id: int, sql_query: str) -> None:
            ...
        @ensure_connection_wrapper
        def list_series_dict(self) -> list[dict[str, Any]]:
            ...
        @ensure_connection_wrapper
        def get_series_sql_query(self, series_id: int) -> str | None:
            ...
        def _schema(self, table_name: str) -> list[tuple[str, str]]:
            ...
        @ensure_connection_wrapper
        def get_query(self, name: str) -> SavedQuery | None:
            ...
        @ensure_connection_wrapper
        def list_queries(self) -> list[SavedQuery]:
            ...
        @ensure_connection_wrapper
        def list_sympy_functions(self) -> list[StoredFunction]:
            ...
        @ensure_connection_wrapper
        def get_sympy_function(self, name: str) -> StoredFunction | None:
            ...
else:
    _Host = object


class TablesMixin(_Host):
    """The user tables: listing, reading, importing, renaming and deleting them, their Hide, Selected and A part of SqliteRepo (sqlite_repo.py)."""

    __slots__ = ()

    # ==================================================================
    # Tables: User tables: listing, reading, importing, renaming, deleting.
    # ==================================================================
    def list_user_tables(self, *, include_internal: bool = False) -> pd.DataFrame:
        """List all user tables with link status and notes.

        Returns DataFrame with columns:
          - Table: table name
          - has_link: bool indicating if import link exists
          - Notes: user notes (or None)
          - source_path: import source path (or None)

        ``include_internal`` also lists this app's own "__..._descriptors__"
        and similar tables - off by default, since every other caller (the
        chart data-source picker, the pragma table/row count) wants only
        tables a user could plot, but the Database nav page's own table
        list wants the full picture of what the file actually holds.
        ``sqlite_%`` (SQLite's own bookkeeping, e.g. sqlite_sequence) is
        never included either way - that is not this application's state.
        """
        internal_filter = (
            "" if include_internal else "AND sm.name NOT LIKE '__%__' ESCAPE '_'"
        )
        sql = f"""
        SELECT
            sm.name AS "Table",
            (LENGTH(COALESCE(il.source_path, '')) > 0) AS has_link,
            td.notes AS "Notes",
            il.source_path AS "source_path"
        FROM sqlite_master sm
        LEFT JOIN __import_links__ il ON il.table_name = sm.name
        LEFT JOIN __table_descriptors__ td ON td.name = sm.name
        WHERE sm.type = 'table'
          AND sm.name NOT LIKE 'sqlite_%'
          {internal_filter}
        ORDER BY sm.name
        """
        df = self.query_df(sql)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Table", "has_link", "Notes", "source_path"])
        df["has_link"] = df["has_link"].astype(bool)
        return df

    def list_data_sources(self, *, functions: bool = False) -> pd.DataFrame:
        """List tables and saved queries - and, with *functions*, the stored functions - as one addressable list.

        Same columns as ``list_user_tables`` plus ``kind``, so the table list
        can render both without a second code path.  Saved queries have no
        import link and no source file; their SQL goes in ``source_path`` so it
        can be shown as a tooltip. A stored function (kind "function") has its
        formula there, and its description as its Notes.
        """
        tables = self.list_user_tables()
        tables = tables.assign(kind="table")

        queries = self.list_queries()
        stored = self.list_sympy_functions() if functions else []
        if not queries and not stored:
            return tables

        query_frame = pd.DataFrame(
            {
                "Table": [query.name for query in queries] + [function.name for function in stored],
                "has_link": [False] * (len(queries) + len(stored)),
                "Notes": [None] * len(queries) + [function.description or None for function in stored],
                "source_path": [query.sql for query in queries] + [function.code for function in stored],
                "kind": ["query"] * len(queries) + ["function"] * len(stored),
            }
        )

        combined = pd.concat([tables, query_frame], ignore_index=True)
        return combined.sort_values("Table", key=lambda s: s.str.lower(), ignore_index=True)

    @ensure_connection_wrapper
    def get_data_source(self, name: str) -> DataSource | None:
        """Resolve a name to a table, a saved query or a stored function.

        Tables win over queries when a name is used twice: the physical object
        is the one the rest of SQLite would resolve, so shadowing it here would
        make the preview and a hand-written query disagree. Queries win over
        functions, which came later.
        """
        assert self._con is not None
        clean = str(name or "").strip()
        if not clean:
            return None

        if self.check_if_table_exists(clean):
            return DataSource.table(clean)

        saved = self.get_query(clean)
        if saved is not None:
            return DataSource.query(saved.name, saved.sql)

        stored = self.get_sympy_function(clean)
        if stored is not None:
            from app.data_operations.sympy_store import source_sql  # SymPy: loaded when a function is used

            try:
                return DataSource.function(stored.name, source_sql(stored.to_dict()))
            except (ValueError, SyntaxError):
                applogger.exception("The stored function '%s' cannot be a data source", clean)
        return None

    @ensure_connection_wrapper
    def data_source_columns(self, source: DataSource) -> list[str]:
        """Return the column names a source yields, without reading any rows."""
        assert self._con is not None
        if not source.is_query:
            return self.get_columns(source.name)

        try:
            self.refuse_if_blocked(source.sql)
            cursor = self._con.execute(source.columns_sql())
            if cursor.description is None:
                return []
            return [str(item[0]) for item in cursor.description if item and item[0] and item[0] != ROW_ID]
        except READ_FAILURES:
            applogger.exception("Failed to read columns of query '%s'", source.name)
            return []

    @ensure_connection_wrapper
    def data_source_row_count(self, source: DataSource) -> int:
        """Return how many rows a source yields, or 0 when it cannot run."""
        assert self._con is not None
        try:
            if source.is_query:
                self.refuse_if_blocked(source.sql)
            row = self._con.execute(source.count_sql()).fetchone()
            return int(row[0]) if row else 0
        except READ_FAILURES:
            applogger.exception("Failed to count rows of source '%s'", source.name)
            return 0

    @ensure_connection_wrapper
    def data_source_page(self, source: DataSource, *, limit: int, offset: int) -> pd.DataFrame:
        """Return one page of a source's rows."""
        assert self._con is not None
        try:
            if source.is_query:
                self.refuse_if_blocked(source.sql)
            return pd.read_sql_query(source.page_sql(limit=limit, offset=offset), self._con)
        except READ_FAILURES:
            applogger.exception("Failed to read source '%s'", source.name)
            return pd.DataFrame()

    def visible_rows(self, table_name: str) -> pd.DataFrame:
        """Every row of a table that is not marked Hide, as the charts see it."""
        where = ' WHERE COALESCE("Hide", 0) = 0' if "Hide" in self.get_columns(table_name) else ""
        return self.query_df(f"SELECT * FROM {_quote_ident(table_name)}{where}")

    def table_frame(self, table_name: str) -> pd.DataFrame:
        """Every row of a table, hidden ones included, as a DataFrame."""
        return self.query_df(f"SELECT * FROM {_quote_ident(table_name)}")

    def validate_query(self, sql: str) -> tuple[bool, str]:
        """Return (ok, message) for a candidate saved query.

        The statement is prepared and run with ``LIMIT 0``: that catches syntax
        errors, unknown tables and unknown columns without materialising a
        single row, so validating a query over a huge table is instant.
        """
        text = str(sql or "").strip().rstrip(";").strip()

        ok, reason = is_read_only_select(text)
        if not ok:
            return False, reason

        if self._con is None:
            self._connect()
        assert self._con is not None

        try:
            with read_only(self._con):
                cursor = self._con.execute(f"SELECT * FROM ({text}) AS _probe LIMIT 0")
        except READ_FAILURES as exc:
            return False, str(exc)

        columns = [str(item[0]) for item in (cursor.description or []) if item]
        if not columns:
            return False, "The query returns no columns."
        return True, f"{len(columns)} column(s): {', '.join(columns)}"

    @ensure_connection_wrapper
    def table_info(self, table: str) -> list[sqlite3.Row]:
        """``PRAGMA table_info`` of a table: one row per column, in order.

        The one place the schema is read - column names, declared types,
        the primary key - so every caller quotes the name the same way (a
        name with a space in it used to fail here, and nowhere else). Empty
        when there is no such table.
        """
        assert self._con is not None
        try:
            return list(self._con.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall())
        except READ_FAILURES:
            applogger.exception("Failed to read table schema for %s", table)
            return []

    def get_columns(self, table: str, *, with_id: bool = False) -> list[str]:
        """A table's column names, in order; empty when there is no such table.

        Without its row id (ROW_ID) unless *with_id*: what a person sees
        and chooses from.
        """
        return [str(row[1]) for row in self.table_info(table) if with_id or str(row[1]) != ROW_ID]

    def has_column(self, table_name: str, col_name: str) -> bool:
        """True when the table has a column with this name."""
        return col_name in self.get_columns(table_name, with_id=True)

    def row_id_column(self, table: str) -> str | None:
        """The column that is *table*'s rowid - its INTEGER PRIMARY KEY - or None."""
        keys = [row for row in self.table_info(table) if int(row[5] or 0) > 0]
        if len(keys) == 1 and str(keys[0][2] or "").strip().upper() == "INTEGER":
            return str(keys[0][1])
        return None

    @ensure_connection_wrapper
    def ensure_row_id(self, table: str) -> None:
        """Give *table* its row id column (ROW_ID), numbered in row order - unless it has one.

        A rebuild of the table: SQLite cannot add a primary key to a table.
        An "_id" column already there that holds distinct integers (the
        table was saved from a query of another) keeps its values; any other
        is the person's own and is renamed out of the way, "_id_2".
        """
        assert self._con is not None
        info = self.table_info(table)
        if not info or self.row_id_column(table) is not None:
            return
        table_sql = _quote_ident(table)
        id_sql = _quote_ident(ROW_ID)
        source = "rowid"
        if any(str(row[1]) == ROW_ID for row in info):
            total, distinct, integers = self._con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT {id_sql}), "
                f"COALESCE(SUM(typeof({id_sql}) = 'integer'), 0) FROM {table_sql}"
            ).fetchone()
            if total == distinct == integers:
                source = id_sql
            else:
                taken = {str(row[1]).lower() for row in info}
                index = 2
                while f"{ROW_ID}_{index}".lower() in taken:
                    index += 1
                self._con.execute(
                    f"ALTER TABLE {table_sql} RENAME COLUMN {id_sql} TO {_quote_ident(f'{ROW_ID}_{index}')}"
                )
                info = self.table_info(table)
        columns = [(str(row[1]), str(row[2] or "")) for row in info if str(row[1]) != ROW_ID]
        temporary = _quote_ident(f"__rebuild_{table}")
        definition = ", ".join([f"{id_sql} INTEGER PRIMARY KEY"] + [f"{_quote_ident(n)} {k}".strip() for n, k in columns])
        names = "".join(f", {_quote_ident(n)}" for n, _k in columns)
        self._con.execute(f"DROP TABLE IF EXISTS {temporary}")
        self._con.execute(f"CREATE TABLE {temporary} ({definition})")
        self._con.execute(
            f"INSERT INTO {temporary} ({id_sql}{names}) SELECT {source}{names} FROM {table_sql} ORDER BY rowid"
        )
        self._con.execute(f"DROP TABLE {table_sql}")
        self._con.execute(f"ALTER TABLE {temporary} RENAME TO {table_sql}")
        self._commit()

    def ensure_row_ids(self) -> None:
        """Every user table with its row id: a project made before R-07, the first time it opens."""
        for table in self.list_user_tables()["Table"]:
            try:
                self.ensure_row_id(str(table))
            except sqlite3.Error:
                applogger.exception("Could not give table '%s' its row id column.", table)

    @ensure_connection_wrapper
    def set_table_notes(self, table: str, notes: str) -> None:
        """Set notes for a table in __table_descriptors__."""
        assert self._con is not None
        self._con.execute(
            """
            INSERT INTO __table_descriptors__ (name, notes)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET notes = excluded.notes
            """,
            (table, notes),
        )
        self._commit()

    @ensure_connection_wrapper
    def get_table_info(self, table: str) -> dict[str, Any]:
        """What the project records about *table* beyond its rows - its DOE design, say.

        A JSON object in ``__table_descriptors__.info_json``, one key per
        kind of information; empty when there is none or it cannot be read.
        """
        assert self._con is not None
        row = self._con.execute(
            "SELECT info_json FROM __table_descriptors__ WHERE name = ?", (table,)
        ).fetchone()
        if not row or not row[0]:
            return {}
        try:
            info = json.loads(row[0])
        except ValueError:
            applogger.warning("The information recorded for table %r is not valid JSON.", table)
            return {}
        return info if isinstance(info, dict) else {}

    @ensure_connection_wrapper
    def set_table_info(self, table: str, key: str, value: Any) -> None:
        """Record *value* under *key* for *table*; None removes the key."""
        assert self._con is not None
        info = self.get_table_info(table)
        if value is None:
            info.pop(key, None)
        else:
            info[key] = value
        self._con.execute(
            """
            INSERT INTO __table_descriptors__ (name, info_json)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET info_json = excluded.info_json
            """,
            (table, json.dumps(info, ensure_ascii=False) if info else None),
        )
        self._commit()

    @property
    def is_open(self) -> bool:
        """True while the connection is open.

        Every other method reopens a closed connection on demand; a widget
        that outlives a database switch asks this first, so it can show
        nothing instead of quietly reading the file that replaced its own.
        """
        return self._is_connected and self._con is not None

    @ensure_connection_wrapper
    def read_rows_after_rowid(self, table: str, last_rowid: int, limit: int) -> list[tuple[Any, ...]]:
        """Return up to *limit* rows of *table* with a rowid above *last_rowid*.

        Each row is ``(rowid, *columns)``, in rowid order, the columns those
        of get_columns - not the row id column, which is the rowid. Reading
        a table in blocks by "the rowid after the last one seen" rather than
        by OFFSET keeps every block as cheap as the first - the way the table
        preview and the table editor page through a table of any size.
        """
        assert self._con is not None
        columns = "".join(f", {_quote_ident(name)}" for name in self.get_columns(table))
        cursor = self._con.execute(
            f"SELECT rowid{columns} FROM {_quote_ident(table)} WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (int(last_rowid), int(limit)),
        )
        return [tuple(row) for row in cursor.fetchall()]

    @ensure_connection_wrapper
    def row_count(self, table: str) -> int:
        """Return row count for a table (or 0 on error)."""
        assert self._con is not None
        try:
            row = self._con.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()
            return int(row[0]) if row else 0
        except READ_FAILURES:
            return 0

    @ensure_connection_wrapper
    def get_table_link(self, table: str) -> dict[str, Any] | None:
        """Get the import link for a table, or None if not found."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT id, source_path, settings_json FROM __import_links__ WHERE table_name = ?",
            (table,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "source_path": str(row["source_path"]),
            "settings": _loads_json(row["settings_json"]),
        }

    @ensure_connection_wrapper
    def query_df(self, sql: str, params: tuple[Any, ...] | None = None, *, row_id: bool = False) -> pd.DataFrame:
        """Execute SQL and return DataFrame (or empty DF for non-SELECT).

        Detects SELECT/WITH/PRAGMA/EXPLAIN and returns data.
        For DDL/DML (CREATE/INSERT/UPDATE/DELETE), executes and returns empty DF.
        Rows come without the tables' row id column (ROW_ID) unless *row_id*.
        """
        sql_text = (sql or "").strip()
        if not sql_text or self._con is None:
            return pd.DataFrame()
        assert self._con is not None

        # SQL the opening scan found would write is refused, whatever it is.
        self.refuse_if_blocked(sql_text)
        if _RETURNS_ROWS_RE.match(sql_text):
            frame = pd.read_sql_query(sql_text, self._con, params=params or ())
            if not row_id and ROW_ID in frame.columns:
                frame = frame.loc[:, frame.columns != ROW_ID]
            return frame
        else:
            self._con.execute(sql_text, params or ())
            return pd.DataFrame()

    # =====================================================================
    # DataFrame import
    # =====================================================================
    @ensure_connection_wrapper
    def import_dataframe(
        self,
        df: pd.DataFrame,
        *,
        table_name: str,
        normalize_columns: bool = True,
        dtype_overrides: DtypeArg | None = None,
    ) -> int:
        """Import DataFrame to SQLite table.

        Args:
            df: DataFrame to import
            table_name: destination table name
            normalize_columns: if True, convert column names to lowercase with underscores
            dtype_overrides: optional dtype mappings for columns

        Returns:
            Number of rows imported (or 0 on error)
        """
        table = table_name.strip()
        table_q = _quote_ident(table)

        # Normalize column names if requested
        if normalize_columns:
            df = df.copy()
            df.columns = [
                str(c).strip().lower().replace(" ", "_") for c in df.columns
            ]

        assert self._con is not None

        # Drop existing table to avoid conflicts
        self._con.execute(f"DROP TABLE IF EXISTS {table_q}")

        if dtype_overrides is not None or not self._write_frame_directly(df, table_q):
            # Import using pandas.to_sql
            df.to_sql(
                table,
                self._con,
                if_exists="append",
                index=False,
                dtype=dtype_overrides,
            )
        self.ensure_row_id(table)
        applogger.info(f"Imported {df.shape[0]} rows to {table_name}")
        return int(df.shape[0])

    #: What a column of plain Python values may hold for the direct write.
    _PLAIN_TYPES = (str, int, float, bool, type(None))

    def _write_frame_directly(self, df: pd.DataFrame, table_q: str) -> bool:
        """Write *df* with one ``executemany`` of column lists; False when it should go through pandas.

        ``DataFrame.to_sql`` converts every cell on its way in and was ten
        times slower than SQLite itself on a file (0.28 s against 0.03 s for
        20,000 rows of 8 columns): the cost of every Preview of a data
        operation, which writes its result table each time. This writes the
        same table - INTEGER for integers and booleans, REAL for floats, TEXT
        for the rest, NaN as NULL - for the frames operations produce, and
        leaves what it does not know (dates and times, extension types,
        values that are not plain numbers or text) to pandas, which is also
        where it falls back if the write fails part way (nothing is left behind).
        """
        assert self._con is not None
        if not len(df.columns) or df.columns.has_duplicates or not all(isinstance(c, str) for c in df.columns):
            return False
        try:
            columns, declared = self._frame_columns(df)
        except (TypeError, ValueError):
            return False
        if columns is None:
            return False
        definition = ", ".join(f"{_quote_ident(str(name))} {kind}" for name, kind in zip(df.columns, declared))
        if ROW_ID not in df.columns:
            # The row id from the start: no rebuild after (ensure_row_id).
            definition = f"{_quote_ident(ROW_ID)} INTEGER PRIMARY KEY, {definition}"
        names = ", ".join(_quote_ident(str(name)) for name in df.columns)
        try:
            self._con.execute(f"CREATE TABLE {table_q} ({definition})")
            if len(df):
                if not self._con.in_transaction:
                    self._con.execute("BEGIN")
                self._con.executemany(
                    f"INSERT INTO {table_q} ({names}) VALUES ({', '.join('?' * len(columns))})", zip(*columns)
                )
        except sqlite3.Error:
            # Not inside a dialog's Preview: a rollback there would end its
            # savepoint too, and the DROP below is the undo needed anyway.
            if self._con.in_transaction and not self._preview_savepoint_name:
                self._con.rollback()
            self._con.execute(f"DROP TABLE IF EXISTS {table_q}")
            return False
        # Like to_sql, which commits at its end - but through _commit, which
        # leaves a Preview's savepoint open: a plain commit ended it, and the
        # dialog's rollback then found it gone ("already gone during
        # rollback") with the preview's writes already kept.
        self._commit()
        return True

    def _frame_columns(self, df: pd.DataFrame) -> tuple[list[list[Any]] | None, list[str]]:
        """Each column as a list of plain Python values, with its SQL type; (None, []) if one is not plain."""
        columns: list[list[Any]] = []
        declared: list[str] = []
        for name in df.columns:
            series = df[name]
            dtype = series.dtype
            if not isinstance(dtype, np.dtype):
                if not pd.api.types.is_string_dtype(dtype):
                    return None, []  # nullable integers, categories, ...
                values = series.tolist()
                missing = series.isna().to_numpy()
                if missing.any():
                    values = [None if gone else item for item, gone in zip(values, missing)]
                kind = "TEXT"
            elif dtype.kind == "f":
                array = series.to_numpy(dtype=float)
                missing = np.isnan(array)
                if missing.any():
                    boxed = array.astype(object)
                    boxed[missing] = None  # NaN is NULL
                    values = boxed.tolist()
                else:
                    values = array.tolist()
                kind = "REAL"
            elif dtype.kind in "iu":
                values, kind = series.tolist(), "INTEGER"
            elif dtype.kind == "b":
                values, kind = [int(item) for item in series.tolist()], "INTEGER"
            elif dtype.kind == "O":
                values = series.tolist()
                if not set(map(type, values)) <= set(self._PLAIN_TYPES):
                    return None, []
                if bool(series.isna().to_numpy().any()):  # NaN in an object column is NULL too
                    values = [None if item != item else item for item in values]
                kind = "TEXT"
            else:
                return None, []  # datetime, timedelta, ...
            columns.append(values)
            declared.append(kind)
        return columns, declared

    # =====================================================================
    # Import links management
    # =====================================================================
    @ensure_connection_wrapper
    def upsert_link(
        self, *, table_name: str, source_path: str, settings: dict[str, Any]
    ) -> int | None:
        """Insert or update import link; return link id (or None on error)."""
        assert self._con is not None

        self._con.execute(
            """
            INSERT INTO __import_links__ (table_name, source_path, settings_json)
            VALUES (?, ?, ?)
            ON CONFLICT(table_name) DO UPDATE SET
                source_path = excluded.source_path,
                settings_json = excluded.settings_json
            """,
            (table_name, source_path, _dumps_json(settings)),
        )
        row = self._con.execute(
            "SELECT id FROM __import_links__ WHERE table_name = ?",
            (table_name,),
        ).fetchone()

        if row is None:
            applogger.error("Failed to retrieve link after upsert")
            return None
        self._commit()
        return int(row["id"])

    @ensure_connection_wrapper
    def get_import_link(self, link_id: int) -> dict[str, Any]:
        """Fetch import link by id. Raises KeyError if not found."""
        assert self._con is not None

        row = self._con.execute(
            "SELECT * FROM __import_links__ WHERE id = ?",
            (int(link_id),),
        ).fetchone()
        if row is None:
            applogger.error(f"Link not found: {link_id}")
            return {}
        return {
            "id": int(row["id"]),
            "table_name": str(row["table_name"]),
            "source_path": str(row["source_path"]),
            "settings": _loads_json(row["settings_json"]),
        }

    @ensure_connection_wrapper
    def ensure_column(self, table_name: str, col_name:str, col_type:str="INTEGER")-> None:
        """Add *col_name* to a table unless it is there already."""
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        if not self.has_column(table_name, col_name):
            # An INTEGER column is a flag or an id and starts at 0; any other
            # kind starts empty. (The conditional used to bind to the whole
            # expression, so every non-INTEGER column ran an empty statement
            # and was never added.)
            default = " NOT NULL DEFAULT 0" if col_type == "INTEGER" else ""
            self._con.execute(
                f"ALTER TABLE {table_sql} ADD COLUMN {_quote_ident(col_name)} {col_type}{default}"
            )
            self._commit()

    @ensure_connection_wrapper
    def rename_table_column(self, table_name: str, old_name: str, new_name: str) -> None:
        """Rename one column of a user table."""
        assert self._con is not None
        if ROW_ID in (old_name, new_name):
            raise ValueError(f"'{ROW_ID}' is the table's row id")
        self._con.execute(
            f"ALTER TABLE {_quote_ident(table_name)} "
            f"RENAME COLUMN {_quote_ident(old_name)} TO {_quote_ident(new_name)}"
        )
        self._commit()

    @ensure_connection_wrapper
    def snapshot_column(self, table_name: str, col_name: str, backup_name: str) -> bool:
        """Move a column aside under ``backup_name`` so it can be restored.

        Returns True when a snapshot was taken, False when the column did not
        exist (in which case restoring means simply dropping whatever replaced
        it).

        Why rename instead of copying the values: a rename is O(1) metadata and
        cannot run out of space or time on a large table, and it guarantees the
        restored column is byte-for-byte the original rather than a re-inserted
        approximation of it.
        """
        assert self._con is not None

        # A leftover backup means a previous preview never finished cleaning
        # up; the live column is the newer truth, so drop the stale copy.
        if self.has_column(table_name, backup_name):
            applogger.warning(
                "Dropping a stale column snapshot %s.%s left by an earlier preview.",
                table_name,
                backup_name,
            )
            self._drop_column(table_name, backup_name)

        if not self.has_column(table_name, col_name):
            return False

        self.rename_table_column(table_name, col_name, backup_name)
        return True

    @ensure_connection_wrapper
    def restore_column_snapshot(
        self,
        table_name: str,
        col_name: str,
        backup_name: str,
    ) -> None:
        """Undo ``snapshot_column``: drop the live column, restore the backup."""
        assert self._con is not None

        if self.has_column(table_name, col_name):
            self._drop_column(table_name, col_name)

        if self.has_column(table_name, backup_name):
            self.rename_table_column(table_name, backup_name, col_name)

    @ensure_connection_wrapper
    def discard_column_snapshot(self, table_name: str, backup_name: str) -> None:
        """Drop a snapshot after the change it protected has been committed."""
        assert self._con is not None
        if self.has_column(table_name, backup_name):
            self._drop_column(table_name, backup_name)

    def query_source_table(self, sql_query: str) -> str:
        """Best-effort extraction of the first table name after FROM."""
        match = re.search(
            r'\bfrom\s+(?:"([^"]+)"|\'([^\']+)\'|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))',
            sql_query,
            flags=re.IGNORECASE,
        )
        if match is None:
            applogger.error("Cannot identify source table from SQL query.")
            return ""
        mg=match.groups()
        if mg is None:
            return ""
        return next(part for part in match.groups() if part)

    @staticmethod
    def is_table_backed_sql(sql_query: str) -> bool:
        """True when a series query selects directly from a named table.

        The Hide machinery flips a flag on real rows and finds them by rowid,
        so it only applies to a plain table.  A series over a saved query reads
        from a subquery, which has neither a Hide column nor a rowid - adding
        the filter there produces "no such column: Hide" and loses the series.
        """
        sql = str(sql_query or "").strip()
        # A WITH query reads from its own named subqueries, not a table.
        if re.match(r"with\b", sql, flags=re.IGNORECASE):
            return False
        match = top_level_from(sql)
        return match is not None and sql[match.end():].lstrip()[:1] not in ("(", "")

    @ensure_connection_wrapper
    def column_series_sql(self, source_name: str, column: str, alias: str = "value") -> str:
        """SELECT one column of a table or saved query as *alias*, skipping hidden rows.

        What a chart made straight from one column reads - the preview's
        "Histogram and statistics" - written the way New plot writes it.
        """
        source = self.get_data_source(source_name)
        if source is None:
            raise ValueError(f"No table or saved query named {source_name!r}.")
        sql = f"SELECT {quote_identifier(column)} AS {quote_identifier(alias)} FROM {source.from_clause()}"
        if self.source_has_hide_column(source_name):
            sql = self.sql_with_hide_filter(sql)
        return sql

    @staticmethod
    def sql_without_hide_filter(sql_query: str) -> str:
        """The series SQL without the Hide filter sql_with_hide_filter adds."""
        hide = r'"?Hide"?\s*=\s*0\b'
        sql = str(sql_query or "").strip().rstrip(";")
        sql = re.sub(rf"\s+AND\s+{hide}", "", sql, flags=re.IGNORECASE)
        sql = re.sub(rf"\bWHERE\s+{hide}\s+AND\s+", "WHERE ", sql, flags=re.IGNORECASE)
        return re.sub(rf"\s+WHERE\s+{hide}", "", sql, flags=re.IGNORECASE)

    def sql_with_hide_filter(self, sql_query: str) -> str:
        """Return SQL with a Hide=False filter inserted, where that applies."""
        sql = str(sql_query or "").strip().rstrip(";")
        if not sql:
            return sql
        if not self.is_table_backed_sql(sql):
            return sql
        # "Hide" quoted too: the filter this adds is quoted, and the check
        # missing it appended one more on every run.
        if re.search(r'"?\bhide\b"?\s*(?:=\s*0|is\s+false)', sql, flags=re.IGNORECASE):
            return sql
        return self._sql_with_where_clause(sql, '"Hide" = 0')

    def sql_with_row_id(self, sql_query: str) -> str:
        """*sql_query* selecting its table's row id too, as ROW_ID - where a point is one row.

        For a series' query (todo R-07): a point drawn from it then names
        its table row. Left as it is when its rows are not rows of one table
        - a WITH, a join, a union, a grouping or an aggregate - and when it
        selects the row id already. A series over a saved query reads it as a
        subquery (FROM (...) AS q): the row id is taken inside and passed out.
        """
        sql = str(sql_query or "").strip().rstrip(";")
        if has_projection_alias(sql, ROW_ID):
            return sql
        span = self.from_subquery(sql)
        if span is not None:
            return self._subquery_with_row_id(sql, span)
        if not self.is_table_backed_sql(sql) or not self._one_row_per_row(sql):
            return sql
        column = self.row_id_column(self.query_source_table(sql))
        if column is None:
            return sql
        expression = _quote_ident(ROW_ID) if column == ROW_ID else f"{_quote_ident(column)} AS {_quote_ident(ROW_ID)}"
        return sql_insert_select_expression(sql, expression)

    @staticmethod
    def _select_items(sql: str) -> list[str] | None:
        start = re.match(r"(?is)^\s*select\s+", sql)
        from_match = top_level_from(sql)
        if start is None or from_match is None:
            return None
        return split_top_level_select_items(sql[start.end():from_match.start()])

    def _selects_row_id(self, sql: str) -> bool:
        """Whether *sql* returns ROW_ID: the column itself, or another named so."""
        items = self._select_items(sql) or []
        return has_projection_alias(sql, ROW_ID) or any(item.strip('"`[]') == ROW_ID for item in items)

    def _one_row_per_row(self, sql: str) -> bool:
        """Whether each row *sql* returns is one row of what it reads: no grouping, join, union, DISTINCT,
        aggregate or * - and it does not select the row id already."""
        if top_level_match(sql, r"\bgroup\s+by\b|\bjoin\b|\bunion\b|\bintersect\b|\bexcept\b"):
            return False
        items = self._select_items(sql)
        from_match = top_level_from(sql)
        start = re.match(r"(?is)^\s*select\s+", sql)
        if items is None or from_match is None or start is None or re.match(r"(?i)distinct\b", sql[start.end():]):
            return False
        aggregate = re.compile(r"(?i)^(?:count|sum|avg|min|max|total|group_concat)\s*\(")
        if any(item == "*" or item.endswith(".*") or item.strip('"`[]') == ROW_ID or aggregate.match(item)
               for item in items):
            return False
        return not top_level_match(sql[from_match.end():], r",")  # FROM a, b: a join

    @staticmethod
    def from_subquery(sql: str) -> tuple[int, int] | None:
        """Where the subquery *sql* reads FROM is - (start, end) of its text inside the parentheses - or None."""
        match = top_level_from(sql)
        if match is None:
            return None
        start = match.end() + len(sql[match.end():]) - len(sql[match.end():].lstrip())
        if sql[start:start + 1] != "(":
            return None
        depth, quote = 0, ""
        for index in range(start, len(sql)):
            char = sql[index]
            if quote:
                quote = "" if char == quote else quote
            elif char in "'\"`":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return start + 1, index
        return None

    def _subquery_with_row_id(self, sql: str, span: tuple[int, int]) -> str:
        inner = sql[span[0]:span[1]]
        with_id = self.sql_with_row_id(inner)
        everything = self._select_items(inner) == ["*"] and self.is_table_backed_sql(inner) \
            and self.row_id_column(self.query_source_table(inner)) == ROW_ID
        if not self._selects_row_id(with_id) and not everything:
            return sql
        rebuilt = sql[:span[0]] + with_id + sql[span[1]:]
        if self._select_items(rebuilt) == ["*"]:
            return rebuilt
        if not self._one_row_per_row(rebuilt):
            return sql
        return sql_insert_select_expression(rebuilt, _quote_ident(ROW_ID))

    @staticmethod
    def _sql_with_where_clause(sql: str, clause: str) -> str:
        """*sql* with *clause* joined to its outer WHERE, or as a new WHERE."""
        # The outer query's own clauses only: a WHERE inside a subquery in
        # the select list is not one this filter can join with AND.
        insert_before = top_level_match(sql, r"\b(?:order\s+by|group\s+by|limit|offset)\b")
        addition = (
            f" AND {clause}"
            if top_level_match(sql, r"\bwhere\b")
            else f" WHERE {clause}"
        )
        if insert_before is None:
            return sql + addition
        index = insert_before.start()
        return sql[:index].rstrip() + addition + " " + sql[index:].lstrip()

    @staticmethod
    def sql_literal(value: Any) -> str:
        """*value* written as an SQL literal: a number, a quoted text, or NULL."""
        if hasattr(value, "item"):
            value = value.item()  # a numpy scalar, as pandas hands them out
        if value is None or (isinstance(value, float) and value != value):
            return "NULL"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return repr(value)
        return "'" + str(value).replace("'", "''") + "'"

    def source_has_hide_column(self, source_name: str) -> bool:
        """True when *source_name* is a table with a Hide column."""
        source = self.get_data_source(source_name)
        if source is None or source.is_query:
            return False
        return "Hide" in self.query_df(f"SELECT * FROM {source.from_clause()} LIMIT 0").columns

    @ensure_connection_wrapper
    def distinct_values(self, source_name: str, column: str, *, limit: int) -> list[Any]:
        """The distinct values of *column* in rows not hidden, in order - at
        most *limit* + 1 of them, so a caller can tell there were too many."""
        source = self.get_data_source(source_name)
        if source is None:
            raise ValueError(f"No table or saved query named {source_name!r}.")
        sql = f"SELECT DISTINCT {quote_identifier(column)} AS value FROM {source.from_clause()}"
        if self.source_has_hide_column(source_name):
            sql = self.sql_with_hide_filter(sql)
        frame = self.query_df(f"{sql} ORDER BY 1 LIMIT {int(limit) + 1}")
        return list(frame["value"])

    def sql_with_value_filter(self, sql_query: str, column: str, value: Any) -> str:
        """*sql_query* keeping only the rows whose *column* is *value*.

        What "Group by" in New plot makes of one series: one copy per value,
        each with its own condition. Only a plain SELECT ... FROM can take
        one; anything else is refused rather than rewritten.
        """
        sql = str(sql_query or "").strip().rstrip(";")
        if not self.is_table_backed_sql(sql):
            raise ValueError("Only a plain SELECT ... FROM query can be split into groups.")
        literal = self.sql_literal(value)
        condition = (
            f"{quote_identifier(column)} IS NULL" if literal == "NULL"
            else f"{quote_identifier(column)} = {literal}"
        )
        return self._sql_with_where_clause(sql, condition)

    @ensure_connection_wrapper
    def update_series_hide_filter(self, series_id: int, sql_query: str | None = None) -> None:
        """Update a series descriptor query so it excludes Hide=True rows."""
        current_sql = sql_query if sql_query is not None else self.get_series_sql_query(series_id)
        if current_sql is None:
            return
        filtered_sql = self.sql_with_hide_filter(current_sql)
        table_name = self.query_source_table(filtered_sql)
        self.ensure_hide_column(table_name)
        self.update_series_sql_query(int(series_id), filtered_sql)

    @ensure_connection_wrapper
    def list_table_names(self) -> list[str]:
        """Return every user table name, excluding this application's own."""
        assert self._con is not None
        rows = self._con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' "
            "  AND name NOT LIKE '__%__' ESCAPE '_' "
            "  AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return [str(row[0]) for row in rows]

    # Table management (rename/delete with propagation)
    # =====================================================================
    @ensure_connection_wrapper
    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table and propagate references in series queries."""
        assert self._con is not None

        old = (old_name or "").strip()
        new = (new_name or "").strip()

        if not old or not new:
            applogger.error("Missing old_name/new_name")
        if old == new:
            return
        if not _is_ident(old) or not _is_ident(new):
            applogger.error(f"Invalid table name(s): {old!r} -> {new!r}")

        old_q = _quote_ident(old)
        new_q = _quote_ident(new)

        with self.transaction(immediate=True):
            # Verify source exists
            if not self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                (old,),
            ).fetchone():
                applogger.error(f"Table not found for renaming: {old}")
                return

            # Verify target doesn't exist
            if self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                (new,),
            ).fetchone():
                applogger.error(f"Target table already exists: {new}")
                return

            # Rename the table; its notes and information follow it.
            self._con.execute(f"ALTER TABLE {old_q} RENAME TO {new_q}")
            self._con.execute(
                "UPDATE __table_descriptors__ SET name = ? WHERE name = ?", (new, old)
            )
            # Update all series that reference this table
            self._propagate_table_name(old, new, mode="rename")

    @ensure_connection_wrapper
    def check_if_table_exists(self, table_name: str) -> bool:
        """Check if a user table exists."""
        assert self._con is not None
        name = (table_name or "").strip()
        if not name:
            return False
        row = self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    @ensure_connection_wrapper
    def delete_table(self, table_name: str) -> None:
        """Drop a user table and remove related series descriptors."""
        table = (table_name or "").strip()
        if not table:
            return
        assert self._con is not None

        # Before the transaction, not inside it: the undo store attaches its
        # own database, which SQLite does not allow mid-transaction. Both
        # tables, because this deletes both - the data, and the series
        # descriptors that drew it.
        self.snapshot_for_undo(
            [table, "__series_descriptors__", "__table_descriptors__", "__import_links__"],
            label=f"Delete table '{table}'",
        )

        with self.transaction(immediate=True):
            self._con.execute(f"DROP TABLE IF EXISTS {_quote_ident(table)}")
            self._con.execute("DELETE FROM __table_descriptors__ WHERE name = ?", (table,))
            # The link to the file or database the table was read from goes with
            # it: left behind it points at a table that no longer exists, which
            # the database check reports as a problem.
            self._con.execute("DELETE FROM __import_links__ WHERE table_name = ?", (table,))
            self._propagate_table_name(table, None, mode="delete")

    @ensure_connection_wrapper
    def _propagate_table_name(
        self,
        table_name: str,
        new_name: str | None,
        mode: Literal["rename", "delete"],
    ) -> None:
        """Update series queries after table rename/delete.

        On rename: update FROM clauses in sql_query.
        On delete: remove affected series descriptors.

        Called within a transaction; no explicit commit.
        """
        assert self._con is not None

        for s in self.list_series_dict():
            sql = str(s.get("sql_query", "") or "")
            # Its id: series_index is a place within an axis, the same on
            # every chart - deleting a table deleted every chart's series in
            # that place, renaming one rewrote theirs.
            series_id = s.get("id")
            if series_id is None:
                continue

            # Extract table name from FROM clause
            parts = re.split(r"\bFROM\b", sql, flags=re.IGNORECASE)
            if len(parts) < 2:
                continue

            from_token = parts[1].strip().split()[0]
            from_token = from_token.strip('"`[]')

            if from_token != table_name:
                continue

            if mode == "rename":
                # The table's name where it stands as a name, quoted or bare -
                # not inside a longer name or a column ("sales" in "sales_2023").
                repl = re.sub(
                    rf'"{re.escape(table_name)}"|(?<![\w"]){re.escape(table_name)}(?![\w"])',
                    lambda _m: f'"{new_name}"' if new_name else "", sql,
                )
                self._con.execute(
                    "UPDATE __series_descriptors__ SET sql_query = ? WHERE id = ?",
                    (repl, series_id),
                )
            else:
                # Delete series referencing deleted table
                self._con.execute(
                    "DELETE FROM __series_descriptors__ WHERE id = ?",
                    (series_id,),
                )

    @ensure_connection_wrapper
    def import_into_sqlite(
        self, table: str, df: pd.DataFrame, types: dict[str, str]
    ) -> None:
        """Import DataFrame with explicit type declarations.

        Uses executemany with value normalization to handle Excel bytes/memoryview.
        """
        if not table:
            return
        assert self._con is not None

        tq = _quote_ident(table)

        # Drop existing table if present
        if self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone():
            self._con.execute(f"DROP TABLE {tq}")

        # Build CREATE TABLE statement with types
        cols_sql = [
            f"{_quote_ident(col)} {types.get(col, 'TEXT')}" for col in df.columns
        ]
        if ROW_ID not in df.columns:
            cols_sql.insert(0, f"{_quote_ident(ROW_ID)} INTEGER PRIMARY KEY")
        self._con.execute(f"CREATE TABLE {tq} ({', '.join(cols_sql)})")

        # Prepare batch insert
        cols = list(df.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(_quote_ident(c) for c in cols)
        sql = f"INSERT INTO {tq} ({col_list}) VALUES ({placeholders})"
        decls = [types.get(c, "TEXT") for c in cols]

        # Normalize values and insert
        base_iter = df.where(pd.notna(df), None).itertuples(
            index=False, name=None
        )
        records = (
            tuple(
                self._normalize_sqlite_value(row[i], decls[i])
                for i in range(len(cols))
            )
            for row in base_iter
        )
        self._con.executemany(sql, records)
        self._commit()
        self.ensure_row_id(table)

    @staticmethod
    def _normalize_sqlite_value(val: Any, decl: str) -> Any:
        """Normalize Python values for SQLite binding.

        Handles:
          - numpy scalars -> Python native
          - pandas Timestamp -> datetime
          - Excel bytes/memoryview -> numeric (IEEE754 unpack or decimal parse)
        """
        # None/NaN early return
        if val is None or pd.isna(val):
            return None

        # Convert numpy scalar to Python native
        if isinstance(val, np.generic):
            val = val.item()

        # Convert pandas Timestamp to datetime
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime()

        decl_u = (decl or "").upper()

        # Handle bytes/memoryview for numeric types
        if decl_u in ("REAL", "INTEGER", "NUMERIC", "DATE", "TIME", "DATETIME", "TEXT"):
            if isinstance(val, memoryview):
                val = val.tobytes()

            if isinstance(val, (bytes, bytearray)):
                b = bytes(val)

                # Try IEEE754 unpack (common Excel export format)
                try:
                    if len(b) == 8:
                        return float(struct.unpack("<d", b)[0])
                    if len(b) == 4:
                        return float(struct.unpack("<f", b)[0])
                except struct.error:
                    pass

                # Try UTF-8 decode and numeric conversion
                try:
                    s = b.decode("utf-8", errors="strict").strip()
                    if decl_u == "INTEGER":
                        return int(float(s))
                    if decl_u in ("REAL", "NUMERIC"):
                        return float(s)
                    return s
                except ValueError:  # undecodable bytes, or text that is not a number
                    if decl_u in ("REAL", "INTEGER", "NUMERIC"):
                        return None  # Can't convert to number
                    return b.decode("utf-8", errors="replace")

        # BLOB: convert memoryview to bytes
        if decl_u == "BLOB":
            if isinstance(val, memoryview):
                return val.tobytes()
            return val

        return val

    def delete_table_column(
        self, table_name: str, column_name: str, *, undo_entry: int | None = None
    ) -> None:
        """Delete a column from a user table."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        if column_name.lower() == "rowid" or column_name == ROW_ID:
            raise ValueError(f"'{column_name}' is the table's row id and cannot be deleted")
        if column_name == "Hide":
            applogger.error(
                "Column 'Hide' is managed by %s and cannot be deleted.", APP_NAME
            )

        # A dropped column takes its data with it and SQLite has no way back,
        # which is what makes this worth a snapshot of the whole table.
        self.snapshot_for_undo(
            [table_name],
            label=f"Delete column '{column_name}' from '{table_name}'",
            entry_id=undo_entry,
        )

        self._drop_column(table_name, column_name)

    @ensure_connection_wrapper
    def _drop_column(self, table_name: str, column_name: str) -> None:
        """Drop a column without recording an undo entry.

        For the column snapshots below, which are bookkeeping rather than
        something the user did: they run while a preview SAVEPOINT is open,
        where attaching the undo database is not allowed, and a preview is
        rolled back anyway, so it must never reach the undo history.
        """
        assert self._con is not None
        self._con.execute(
            f"ALTER TABLE {_quote_ident(table_name)} DROP COLUMN {_quote_ident(column_name)}"
        )
        self._commit()

    def supports_sql_math(self) -> bool:
        """Whether this SQLite can evaluate cos(), radians() and friends.

        Only builds with SQLITE_ENABLE_MATH_FUNCTIONS have them - most now,
        not all - and a query written with them fails outright elsewhere.
        """
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        try:
            self._con.execute("SELECT cos(radians(0.0))").fetchone()
        except sqlite3.OperationalError:
            return False
        return True

    def preview_expression(
        self,
        table_name: str,
        expression: str,
        limit: int = 10,
    ) -> list[Any]:
        """Evaluate *expression* on the first *limit* rows, writing nothing.

        What the computed-column form shows while the expression is typed,
        and the check add_column_from_expression makes before it alters the
        table: an expression naming a column that does not exist fails here,
        with SQLite's own message, instead of after the new column is added.
        """
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        expr = str(expression or "").strip()
        check_sql_expression(expr)
        with read_only(self._con) as con:
            rows = con.execute(
                f"SELECT {expr} FROM {_quote_ident(table_name)} ORDER BY rowid LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [row[0] for row in rows]

    def add_column_from_expression(
        self,
        table_name: str,
        column_name: str,
        expression: str,
        column_type: str | None = None,
        *,
        undo_entry: int | None = None,
    ) -> None:
        """Add a column and populate it from a SQL expression evaluated per row.

        *column_type* is the declared type (REAL, INTEGER, TEXT); None leaves
        it undeclared, so each value keeps the type the expression gave it.
        """
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        name = str(column_name or "").strip()
        if not name:
            raise ValueError("a column needs a name")
        if not _is_ident(name):
            raise ValueError(f"'{name}' is not a usable column name")
        if name in set(self.get_columns(table_name)):
            raise ValueError(f"'{name}' is already a column of this table")
        declared = str(column_type or "").strip().upper()
        if declared not in ("", "REAL", "INTEGER", "TEXT"):
            raise ValueError(f"'{column_type}' is not a column type")
        expr = str(expression or "").strip()
        # Spliced into an UPDATE below: one expression, nothing that writes -
        # and evaluated once read-only first, so a typo fails before ALTER.
        self.preview_expression(table_name, expr, limit=1)

        self.snapshot_for_undo(
            [table_name],
            label=f"Add column '{name}' to '{table_name}'",
            entry_id=undo_entry,
        )
        table_sql = _quote_ident(table_name)
        column_sql = _quote_ident(name)
        self._con.execute(f"ALTER TABLE {table_sql} ADD COLUMN {column_sql} {declared}".rstrip())
        self._con.execute(f"UPDATE {table_sql} SET {column_sql} = {expr}")
        self._commit()

    # ==================================================================
    # Flags: Hide, Selected and ClusterId: the 0/1 (or id) columns kept on a table.
    # ==================================================================
    @ensure_connection_wrapper
    def clear_integer_column(self, table_name: str, col_name:str) -> None:
        """Reset all Col values to False/0 for one user data table."""
        assert self._con is not None
        self.ensure_column(table_name=table_name,col_name=col_name,col_type="INTEGER")
        self._con.execute(f"UPDATE {_quote_ident(table_name)} SET {_quote_ident(col_name)} = 0")
        self._commit()

    @ensure_connection_wrapper
    def invert_flag(self, table_name: str, flag: str) -> int:
        """Ensure the *flag* column exists and swap its 0s and 1s."""
        assert self._con is not None
        self.ensure_column(table_name=table_name, col_name=flag, col_type="INTEGER")
        column = _quote_ident(flag)
        cur = self._con.execute(
            f'UPDATE {_quote_ident(table_name)} '
            f'SET {column} = CASE WHEN COALESCE({column}, 0) = 0 THEN 1 ELSE 0 END'
        )
        self._commit()
        return int(cur.rowcount or 0)

    @ensure_connection_wrapper
    def flag_rows_by_value(
        self,
        table_name: str,
        flag: str,
        column_name: str,
        operator: str,
        value: Any,
    ) -> int:
        """Set *flag* = 1 where *column_name* compares to a user-provided value."""
        assert self._con is not None
        self.ensure_column(table_name=table_name, col_name=flag, col_type="INTEGER")
        op_map = {"=": "=", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
        sql_op = op_map.get(str(operator).strip())
        if sql_op is None:
            applogger.error(f"Unsupported operator: {operator}")
        cur = self._con.execute(
            f'UPDATE {_quote_ident(table_name)} SET {_quote_ident(flag)} = 1 '
            f'WHERE {_quote_ident(column_name)} {sql_op} ?',
            (value,),
        )
        self._commit()
        return int(cur.rowcount or 0)

    @ensure_connection_wrapper
    def flag_rows_special(self, table_name: str, flag: str, column_name: str, mode: str) -> int:
        """Set *flag* = 1 using a predefined special predicate."""
        assert self._con is not None
        self.ensure_column(table_name=table_name, col_name=flag, col_type="INTEGER")
        column_sql = _quote_ident(column_name)
        if mode == "null_or_empty":
            predicate = f"{column_sql} IS NULL OR TRIM(CAST({column_sql} AS TEXT)) = ''"
        else:
            applogger.error(f"Unsupported mode: {mode}")
            return 0
        cur = self._con.execute(
            f'UPDATE {_quote_ident(table_name)} SET {_quote_ident(flag)} = 1 WHERE {predicate}'
        )
        self._commit()
        return int(cur.rowcount or 0)

    def row_table(self, sql_query: str) -> str | None:
        """The table whose row ids *sql_query* returns as ROW_ID - through a saved query too - or None."""
        sql = str(sql_query or "").strip()
        if not sql:
            return None
        span = self.from_subquery(sql)
        if span is not None:  # a saved query read as a subquery: its own table
            return self.row_table(sql[span[0]:span[1]])
        name = self.query_source_table(sql) if self.is_table_backed_sql(sql) else ""
        source = self.get_data_source(name) if name else None
        if source is not None and source.is_query:
            inner = str(source.sql or "")
            name = self.query_source_table(inner) if self.is_table_backed_sql(inner) else ""
            source = self.get_data_source(name) if name else None
        if source is None or source.is_query or self.row_id_column(source.name) is None:
            return None
        return source.name

    @ensure_connection_wrapper
    def series_rows(self, sql_query: str, roles: Mapping[str, Any]) -> pd.DataFrame:
        """A series' rows with their table row id: ROW_ID, ``x``, ``y`` as stored; empty when it has no ids.

        Read through the series' own query, so only the rows it draws. Not
        forced to numbers: dates and categories are put in a chart's units by
        its axis (the chart panel), as its points were.
        """
        empty = pd.DataFrame(columns=[ROW_ID, "x", "y"])
        try:
            frame = self.query_df(f"SELECT * FROM ({str(sql_query).strip().rstrip(';')})", row_id=True)
        except READ_FAILURES:
            return empty
        x_col = resolve_role_column(frame.columns, roles, "x")
        y_col = resolve_role_column(frame.columns, roles, "y")
        if ROW_ID not in frame.columns or not x_col or not y_col or frame.columns.duplicated().any():
            return empty
        return pd.DataFrame({
            ROW_ID: to_numbers(column_of(frame, ROW_ID)),
            "x": frame[x_col],
            "y": frame[y_col],
        }).dropna()

    #: The table setting that makes every chart of the table draw only its Selected rows (R-07's data filter).
    SELECTED_ONLY_KEY = "show_selected_only"

    def selected_only(self, table_name: str) -> bool:
        """Whether every chart of *table_name* draws only its rows marked Selected (Show only Selected)."""
        return bool(self.get_table_info(table_name).get(self.SELECTED_ONLY_KEY))

    def set_selected_only(self, table_name: str, on: bool) -> None:
        self.set_table_info(table_name, self.SELECTED_ONLY_KEY, True if on else None)

    def sql_selected_only(self, sql_query: str) -> str:
        """*sql_query* keeping only its Selected rows when its table shows only those; else as it is.

        By the row id the series' query names (sql_with_row_id): the rows stay where they are, out of view -
        nothing is Hidden, and the analyses read every row as before.
        """
        sql = str(sql_query or "").strip().rstrip(";")
        table = self.row_table(sql)
        if not table or not self.selected_only(table):
            return sql
        chosen = (f'SELECT rowid FROM {_quote_ident(table)} WHERE "Selected" = 1'
                  if self.has_column(table, "Selected") else "SELECT NULL WHERE 0")
        return f"SELECT * FROM ({sql}) WHERE {_quote_ident(ROW_ID)} IN ({chosen})"

    @ensure_connection_wrapper
    def select_rows(self, table_name: str, row_ids: Sequence[int], *, add: bool = False) -> int:
        """Mark exactly *row_ids* Selected - with *add*, as well as those already; how many are."""
        assert self._con is not None
        self.ensure_column(table_name=table_name, col_name="Selected", col_type="INTEGER")
        table_sql = _quote_ident(table_name)
        if not add:
            self._con.execute(f'UPDATE {table_sql} SET "Selected" = 0 WHERE COALESCE("Selected", 0) <> 0')
        self._con.executemany(
            f'UPDATE {table_sql} SET "Selected" = 1 WHERE rowid = ?', [(int(i),) for i in row_ids]
        )
        self._commit()
        return len(self.selected_row_ids(table_name))

    @ensure_connection_wrapper
    def row_position(self, table_name: str, row_id: int) -> int:
        """Where row *row_id* is in *table_name*'s row order, from 0 - its row in the table preview."""
        assert self._con is not None
        row = self._con.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE rowid < ?", (int(row_id),)
        ).fetchone()
        return int(row[0]) if row else 0

    @ensure_connection_wrapper
    def selected_row_ids(self, table_name: str) -> list[int]:
        """The row ids of the rows of *table_name* marked Selected."""
        assert self._con is not None
        if not self.has_column(table_name, "Selected"):
            return []
        rows = self._con.execute(
            f'SELECT rowid FROM {_quote_ident(table_name)} WHERE "Selected" = 1 ORDER BY rowid'
        ).fetchall()
        return [int(row[0]) for row in rows]

    @ensure_connection_wrapper
    def set_rows_hidden(self, table_name: str, row_ids: Sequence[int], hidden: bool) -> int:
        """Hide (Hide = 1) or show again (Hide = 0) exactly *row_ids*; how many rows changed."""
        assert self._con is not None
        self.ensure_hide_column(table_name)
        value = 1 if hidden else 0
        before = self._con.total_changes
        self._con.executemany(
            f'UPDATE {_quote_ident(table_name)} SET "Hide" = ? WHERE rowid = ? AND COALESCE("Hide", 0) <> ?',
            [(value, int(row_id), value) for row_id in row_ids],
        )
        changed = self._con.total_changes - before
        self._commit()
        return changed

    def ensure_hide_column(self, table_name: str) -> None:
        """Ensure a boolean-compatible Hide column exists on a user data table."""
        self.ensure_column(table_name=table_name,col_name="Hide",col_type="INTEGER")

    def clear_hide_column(self, table_name: str) -> None:
        """Reset all Hide values to False/0 for one user data table."""
        self.clear_integer_column(table_name=table_name,col_name="Hide")

    @ensure_connection_wrapper
    def count_hidden_rows(self, table_name: str) -> int:
        """Return count of rows where Hide=1."""
        assert self._con is not None
        self.ensure_hide_column(table_name)
        row = self._con.execute(
            f'SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE "Hide" = 1'
        ).fetchone()
        return int(row[0]) if row else 0

    @ensure_connection_wrapper
    def hidden_rowids(self, table: str) -> list[int]:
        """Return the rowids of the rows of *table* marked Hide."""
        self.ensure_hide_column(table)
        assert self._con is not None
        rows = self._con.execute(f'SELECT rowid FROM {_quote_ident(table)} WHERE "Hide" = 1').fetchall()
        return [int(row[0]) for row in rows]

    @ensure_connection_wrapper
    def mark_hide_rowids(
        self,
        *,
        table_name: str,
        rowids: Sequence[int],
        clear_existing: bool = False,
        scope_rowids: Sequence[int] | None = None,
    ) -> int:
        """Set Hide=True/1 for exact SQLite rowids and report matched totals.

        *scope_rowids* are shown again first: the rows of the series being
        processed, so a re-run replaces its own earlier choice and leaves
        other series' hidden rows alone. ``clear_existing`` shows every row
        of the table again instead.
        """
        assert self._con is not None
        if clear_existing:
            self.clear_hide_column(table_name)
        else:
            self.ensure_hide_column(table_name)
        if scope_rowids is not None:
            self._con.executemany(
                f'UPDATE {_quote_ident(table_name)} SET "Hide" = 0 WHERE rowid = ?',
                [(int(rowid),) for rowid in scope_rowids],
            )
        ids = [int(rowid) for rowid in rowids]
        before = self._con.total_changes
        self._con.executemany(
            f'UPDATE {_quote_ident(table_name)} SET "Hide" = 1 WHERE rowid = ?',
            [(rowid,) for rowid in ids],
        )
        updated_count = self._con.total_changes - before
        self._commit()
        hidden_count = self.count_hidden_rows(table_name)
        applogger.info(
            "Outlier Hide update table=%s requested=%d matched=%d hidden_total=%d",
            table_name,
            len(ids),
            updated_count,
            hidden_count,
        )
        return hidden_count

    @ensure_connection_wrapper
    def query_series_frame_for_hide(
        self,
        *,
        sql_query: str,
        roles: Mapping[str, Any],
    ) -> pd.DataFrame:
        """The rows a series draws, with their source rowid: ``__rowid__``, ``x``, ``y``.

        Read through the series' own SQL, so its WHERE clause holds - one
        ticker of three in the table, one species - and the roles name the
        columns it returns, aliases included. The Hide filter is left out:
        rows an earlier run hid are looked at again, since the new run
        decides afresh which of the series' rows to hide.

        It used to read the whole table and quote the role names as columns:
        another series' rows were searched (and could be hidden), and a role
        naming an alias ("date AS x") was read by SQLite as the text 'x', so
        no point was usable and nothing was hidden (todo O-01).
        """
        assert self._con is not None
        base = self.sql_without_hide_filter(sql_query)
        if re.search(r"\bgroup\s+by\b|\bselect\s+distinct\b", base, flags=re.IGNORECASE):
            raise ValueError(
                "this series aggregates its rows (GROUP BY or DISTINCT), so a point "
                "it draws is not one row of the table that could be hidden"
            )
        table_name = self.query_source_table(base)
        self.ensure_hide_column(table_name)
        frame = self.query_df(sql_insert_select_expression(base, 'rowid AS "__rowid__"'))

        x_col = resolve_role_column(frame.columns, roles, "x") or ""
        y_col = resolve_role_column(frame.columns, roles, "y") or ""
        missing = [role for role, column in (("x", x_col), ("y", y_col)) if not column]
        if missing:
            raise ValueError(
                f"the series' query returns no column for its {' and '.join(missing)} role "
                f"(it returns {', '.join(str(c) for c in frame.columns if c != '__rowid__')})"
            )
        return pd.DataFrame(
            {"__rowid__": frame["__rowid__"], "x": frame[x_col], "y": frame[y_col]}
        )

    @ensure_connection_wrapper
    def ensure_preview_state_columns(self, table_name: str) -> None:
        """Create/update temporary preview state columns for Hide preview.

        ``Hide`` is the editable runtime column used by chart filtering.
        ``__ChartLibrePreviewHide`` stores the pre-preview Hide state so Preview can
        be rolled back without relying on UI-side SQL or direct connection use.
        """
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        preview_col = _quote_ident("__ChartLibrePreviewHide")
        self.ensure_hide_column(table_name)
        self.ensure_column(
            table_name=table_name,
            col_name="__ChartLibrePreviewHide",
            col_type="INTEGER",
        )
        self._con.execute(
            f'UPDATE {table_sql} SET {preview_col} = COALESCE("Hide", 0)'
        )
        self._commit()

    @ensure_connection_wrapper
    def restore_preview_state_columns(self, table_name: str) -> None:
        """Restore Hide values from the temporary preview state column."""
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        preview_col = _quote_ident("__ChartLibrePreviewHide")
        if not self.has_column(table_name, "__ChartLibrePreviewHide"):
            return
        self.ensure_hide_column(table_name)
        self._con.execute(
            f'UPDATE {table_sql} SET "Hide" = COALESCE({preview_col}, 0)'
        )
        self._commit()

    @ensure_connection_wrapper
    def drop_preview_state_columns(self, table_name: str) -> None:
        """Drop temporary preview state columns created for Hide preview."""
        assert self._con is not None
        if not self.has_column(table_name, "__ChartLibrePreviewHide"):
            return
        self._con.execute(
            f'ALTER TABLE {_quote_ident(table_name)} '
            f'DROP COLUMN {_quote_ident("__ChartLibrePreviewHide")}'
        )
        self._commit()

    @ensure_connection_wrapper
    def colour_snapshot(self, table_name: str, column: str) -> dict[int, str] | None:
        """The colours held in *column* of a table, by rowid; None when there is no such column."""
        assert self._con is not None
        if column not in self.get_columns(table_name):
            return None
        rows = self._con.execute(
            f"SELECT rowid, {_quote_ident(column)} FROM {_quote_ident(table_name)} "
            f"WHERE {_quote_ident(column)} IS NOT NULL AND {_quote_ident(column)} != ''"
        ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}

    @ensure_connection_wrapper
    def set_row_colours(self, table_name: str, column: str, rowids: Sequence[int], colour: str) -> int:
        """Give exactly *rowids* the text *colour* in *column*, and every other row none.

        The column is created (TEXT) when it is not there. The chart reads
        it as a per-point colour: a row with none is drawn in the series' own.
        Returns how many rows were coloured.
        """
        assert self._con is not None
        self.ensure_column(table_name, column, "TEXT")
        table_sql, column_sql = _quote_ident(table_name), _quote_ident(column)
        self._con.execute(f"UPDATE {table_sql} SET {column_sql} = NULL")
        coloured = 0
        for rowid in rowids:
            cursor = self._con.execute(
                f"UPDATE {table_sql} SET {column_sql} = ? WHERE rowid = ?", (str(colour), int(rowid))
            )
            coloured += max(0, int(cursor.rowcount or 0))
        self._commit()
        return coloured

    @ensure_connection_wrapper
    def restore_row_colours(self, table_name: str, column: str, snapshot: dict[int, str] | None) -> None:
        """Put *column* back as a :meth:`colour_snapshot` found it: None drops the column."""
        assert self._con is not None
        if snapshot is None:
            if column in self.get_columns(table_name):
                self._con.execute(
                    f"ALTER TABLE {_quote_ident(table_name)} DROP COLUMN {_quote_ident(column)}"
                )
                self._commit()
            return
        self.ensure_column(table_name, column, "TEXT")
        table_sql, column_sql = _quote_ident(table_name), _quote_ident(column)
        self._con.execute(f"UPDATE {table_sql} SET {column_sql} = NULL")
        for rowid, colour in snapshot.items():
            self._con.execute(f"UPDATE {table_sql} SET {column_sql} = ? WHERE rowid = ?", (colour, int(rowid)))
        self._commit()

    def set_ClusterId(self,source_table,source_x_column, x_values,cluster_values) -> None:  # noqa: N802 - the column's name
        assert self._con is not None
        quoted_table = _quote_ident(source_table)
        quoted_x = _quote_ident(source_x_column)
        for x_value, cluster_value in zip(x_values, cluster_values, strict=False):
            if pd.notna(x_value) and pd.notna(cluster_value):
                cluster_id:int = int(cluster_value)
                self._con.execute(
                    f'UPDATE {quoted_table} SET "ClusterId" = ? WHERE {quoted_x} = ?',
                    (cluster_id, x_value),
                )
        self._commit()

    # ==================================================================
    # Editing: Cell, row and column edits made by hand.
    # ==================================================================
    @ensure_connection_wrapper
    def execute_change(self, sql: str, *, tables: Sequence[str], label: str) -> int:
        """Run one statement that changes the data - written by the assistant, confirmed by the user - as one
        Undo step over *tables*; the rows it changed (-1 when SQLite does not say)."""
        assert self._con is not None
        self.snapshot_for_undo(list(tables), label=label)
        cursor = self._con.execute(sql)
        self._commit()
        self.invalidate_series_cache()
        return int(cursor.rowcount)

    def has_integer_primary_key(self, table_name: str) -> bool:
        """Say whether this table's rowid is a column the user can see.

        ``INTEGER PRIMARY KEY`` makes a column *be* the rowid rather than
        shadow it. Renumbering rowids to reorder rows would then be silently
        rewriting the user's own key values, so the positional insert below
        refuses on such a table rather than doing that quietly. The row id
        this application gives every table (ROW_ID) is not one: renumbering
        it is what it is there for.
        """
        return self.row_id_column(table_name) not in (None, ROW_ID)

    @ensure_connection_wrapper
    def update_table_cell(
        self,
        table_name: str,
        rowid: int,
        column_name: str,
        value: Any,
        *,
        undo_entry: int | None = None,
    ) -> None:
        """Write one cell, addressed by rowid."""
        assert self._con is not None
        if column_name.lower() == "rowid":
            raise ValueError("rowid is not editable")
        self.snapshot_for_undo(
            [table_name],
            label=f"Edit '{column_name}' in '{table_name}'",
            entry_id=undo_entry,
        )
        self._con.execute(
            f"UPDATE {_quote_ident(table_name)} "
            f"SET {_quote_ident(column_name)} = ? WHERE rowid = ?",
            (value, int(rowid)),
        )
        self._commit()

    @ensure_connection_wrapper
    def append_table_row(self, table_name: str, *, undo_entry: int | None = None) -> int:
        """Add one empty row at the end, and return its rowid."""
        assert self._con is not None
        self.snapshot_for_undo(
            [table_name], label=f"Add a row to '{table_name}'", entry_id=undo_entry
        )
        table_sql = _quote_ident(table_name)
        columns = self.get_columns(table_name)
        if columns:
            names = ", ".join(_quote_ident(name) for name in columns)
            holes = ", ".join("NULL" for _unused in columns)
            cursor = self._con.execute(
                f"INSERT INTO {table_sql} ({names}) VALUES ({holes})"
            )
        else:
            cursor = self._con.execute(f"INSERT INTO {table_sql} DEFAULT VALUES")
        self._commit()
        return int(cursor.lastrowid or 0)

    @ensure_connection_wrapper
    def insert_table_row_before(
        self, table_name: str, rowid: int, *, undo_entry: int | None = None
    ) -> int:
        """Add one empty row immediately above *rowid*, and return its rowid.

        A preview lists rows in rowid order, so "above" means "at a lower
        rowid", and every rowid from there up has to move to make space. The
        shift goes through negatives (``rowid -> -(rowid + 1)``, then back)
        because a single ``rowid + 1`` pass collides with the row it is
        about to move onto the moment SQLite reaches it.
        """
        assert self._con is not None
        if self.has_integer_primary_key(table_name):
            raise ValueError(
                "this table's rowid is one of its own columns, so its rows "
                "cannot be renumbered to make space - add the row at the end "
                "instead"
            )
        self.snapshot_for_undo(
            [table_name],
            label=f"Insert a row into '{table_name}'",
            entry_id=undo_entry,
        )
        table_sql = _quote_ident(table_name)
        target = int(rowid)
        self._con.execute(
            f"UPDATE {table_sql} SET rowid = -(rowid + 1) WHERE rowid >= ?", (target,)
        )
        self._con.execute(f"UPDATE {table_sql} SET rowid = -rowid WHERE rowid < 0")

        columns = self.get_columns(table_name)
        if columns:
            names = ", ".join(_quote_ident(name) for name in columns)
            holes = ", ".join("NULL" for _unused in columns)
            self._con.execute(
                f"INSERT INTO {table_sql} (rowid, {names}) VALUES (?, {holes})",
                (target,),
            )
        else:
            self._con.execute(f"INSERT INTO {table_sql} (rowid) VALUES (?)", (target,))
        self._commit()
        return target

    @ensure_connection_wrapper
    def delete_table_rows(
        self, table_name: str, rowids: Sequence[int], *, undo_entry: int | None = None
    ) -> int:
        """Delete the named rows, and return how many went."""
        assert self._con is not None
        wanted = [int(value) for value in rowids]
        if not wanted:
            return 0
        self.snapshot_for_undo(
            [table_name],
            label=f"Delete {len(wanted)} row(s) from '{table_name}'",
            entry_id=undo_entry,
        )
        holes = ", ".join("?" for _unused in wanted)
        cursor = self._con.execute(
            f"DELETE FROM {_quote_ident(table_name)} WHERE rowid IN ({holes})", wanted
        )
        self._commit()
        return int(cursor.rowcount or 0)

    @ensure_connection_wrapper
    def insert_table_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str = "REAL",
        *,
        before: str | None = None,
        undo_entry: int | None = None,
    ) -> None:
        """Add a column, at the end or before an existing one.

        ``ALTER TABLE ... ADD COLUMN`` can only append, which is fine until
        someone wants the new column beside the one it belongs with. Any
        other position means rebuilding the table in the new column order
        and copying the rows across, so that is what this does - but only
        when a position is actually asked for.

        The rebuild keeps the rows and their rowids. It does not carry
        indexes, triggers or constraints across; the tables this edits are
        imported data, which have none.
        """
        assert self._con is not None
        name = str(column_name or "").strip()
        if not name:
            raise ValueError("a column needs a name")
        if not _is_ident(name):
            raise ValueError(f"'{name}' is not a usable column name")

        table_sql = _quote_ident(table_name)
        existing = self.get_columns(table_name, with_id=True)
        if name in existing:
            raise ValueError(f"'{name}' is already a column of this table")

        declared = str(column_type or "REAL").strip().upper() or "REAL"
        if declared not in ("REAL", "INTEGER", "TEXT", "BLOB", "NUMERIC"):
            declared = "REAL"

        self.snapshot_for_undo(
            [table_name],
            label=f"Add column '{name}' to '{table_name}'",
            entry_id=undo_entry,
        )

        if before is None or before not in existing:
            self._con.execute(
                f"ALTER TABLE {table_sql} ADD COLUMN {_quote_ident(name)} {declared}"
            )
            self._commit()
            return

        types = dict(self._schema(table_name))
        types[name] = declared
        order = list(existing)
        order.insert(order.index(before), name)

        temporary = f"__rebuild_{table_name}"
        columns_sql = ", ".join(
            f"{_quote_ident(column)} {types.get(column, '')}".strip() for column in order
        )
        copied = ", ".join(_quote_ident(column) for column in existing)
        self._con.execute(f"DROP TABLE IF EXISTS {_quote_ident(temporary)}")
        self._con.execute(f"CREATE TABLE {_quote_ident(temporary)} ({columns_sql})")
        self._con.execute(
            f"INSERT INTO {_quote_ident(temporary)} (rowid, {copied}) "
            f"SELECT rowid, {copied} FROM {table_sql}"
        )
        self._con.execute(f"DROP TABLE {table_sql}")
        self._con.execute(f"ALTER TABLE {_quote_ident(temporary)} RENAME TO {table_sql}")
        self._commit()
