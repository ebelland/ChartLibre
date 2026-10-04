"""The data itself: tables, what is in them, and how it gets there.

Everything that reads or writes a *user* table rather than a descriptor -
listing them, describing their columns, running queries against them,
importing a DataFrame into one, the import links that say where a table
came from, the Hide column the chart filters on, and renaming or deleting
one (which has to reach into the series that draw it, hence
_propagate_table_name).

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import re
import sqlite3
import struct
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

from app import APP_NAME
from app.data.data_source import DataSource, quote_identifier
from app.data.select_sql import top_level_from, top_level_match
from app.data.repo._common import (
    READ_FAILURES,
    RepoHost,
    _RETURNS_ROWS_RE,
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    ensure_connection_wrapper,
    check_sql_expression,
    is_read_only_select,
    read_only,
)
from app.logs.logger import applogger


class TablesMixin(RepoHost):
    """User tables: listing, reading, importing, renaming, deleting."""

    __slots__ = ()

    # =====================================================================
    # Table listing and introspection
    # =====================================================================


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

    # =====================================================================
    # Data sources: physical tables and saved queries
    # =====================================================================

    def list_data_sources(self) -> pd.DataFrame:
        """List tables and saved queries as one addressable list.

        Same columns as ``list_user_tables`` plus ``kind``, so the table list
        can render both without a second code path.  Saved queries have no
        import link and no source file; their SQL goes in ``source_path`` so it
        can be shown as a tooltip.
        """
        tables = self.list_user_tables()
        tables = tables.assign(kind="table")

        queries = self.list_queries()
        if not queries:
            return tables

        query_frame = pd.DataFrame(
            {
                "Table": [query.name for query in queries],
                "has_link": [False] * len(queries),
                "Notes": [None] * len(queries),
                "source_path": [query.sql for query in queries],
                "kind": ["query"] * len(queries),
            }
        )

        combined = pd.concat([tables, query_frame], ignore_index=True)
        return combined.sort_values("Table", key=lambda s: s.str.lower(), ignore_index=True)

    @ensure_connection_wrapper
    def get_data_source(self, name: str) -> DataSource | None:
        """Resolve a name to a table or a saved query.

        Tables win over queries when a name is used twice: the physical object
        is the one the rest of SQLite would resolve, so shadowing it here would
        make the preview and a hand-written query disagree.
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
            return [str(item[0]) for item in cursor.description if item and item[0]]
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

    def get_columns(self, table: str) -> list[str]:
        """A table's column names, in order; empty when there is no such table."""
        return [str(row[1]) for row in self.table_info(table)]

    def has_column(self, table_name: str, col_name: str) -> bool:
        """True when the table has a column with this name."""
        return col_name in self.get_columns(table_name)


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

    # =====================================================================
    # Query execution
    # =====================================================================


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

        Each row is ``(rowid, *columns)``, in rowid order. Reading a table in
        blocks by "the rowid after the last one seen" rather than by OFFSET
        keeps every block as cheap as the first - the way the table preview
        and the table editor page through a table of any size.
        """
        assert self._con is not None
        cursor = self._con.execute(
            f"SELECT rowid, * FROM {_quote_ident(table)} WHERE rowid > ? ORDER BY rowid LIMIT ?",
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
    def query_df(self, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
        """Execute SQL and return DataFrame (or empty DF for non-SELECT).
        
        Detects SELECT/WITH/PRAGMA/EXPLAIN and returns data.
        For DDL/DML (CREATE/INSERT/UPDATE/DELETE), executes and returns empty DF.
        """
        sql_text = (sql or "").strip()
        if not sql_text or self._con is None:
            return pd.DataFrame()
        assert self._con is not None

        # SQL the opening scan found would write is refused, whatever it is.
        self.refuse_if_blocked(sql_text)
        if _RETURNS_ROWS_RE.match(sql_text):
            return pd.read_sql_query(sql_text, self._con, params=params or ())
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

        # Import using pandas.to_sql
        df.to_sql(
            table,
            self._con,
            if_exists="append",
            index=False,
            dtype=dtype_overrides,
        )
        applogger.info(f"Imported {df.shape[0]} rows to {table_name}")
        return int(df.shape[0])


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


    # =====================================================================
    # Columns
    # =====================================================================

               

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
                show_dialog=False,
                raise_error=False,
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

            # Rename the table
            self._con.execute(f"ALTER TABLE {old_q} RENAME TO {new_q}")
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
            [table, "__series_descriptors__"], label=f"Delete table '{table}'"
        )

        with self.transaction(immediate=True):
            self._con.execute(f"DROP TABLE IF EXISTS {_quote_ident(table)}")
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
            series_id = s.get("series_index")
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
                # Replace old table name with new
                repl = sql.replace(table_name, new_name or "")
                self._con.execute(
                    "UPDATE __series_descriptors__ SET sql_query = ? WHERE series_index = ?",
                    (repl, series_id),
                )
            else:
                # Delete series referencing deleted table
                self._con.execute(
                    "DELETE FROM __series_descriptors__ WHERE series_index = ?",
                    (series_id,),
                )

    # =====================================================================
    # Advanced data import (with type coercion)
    # =====================================================================

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
    


    # =====================================================================
    # Runtime-attached table preview context-menu operations
    # =====================================================================
    # These are attached after the SqliteRepo class definition so the file remains
    # drop-in even if the class layout changes.

    def delete_table_column(
        self, table_name: str, column_name: str, *, undo_entry: int | None = None
    ) -> None:
        """Delete a column from a user table."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        if column_name.lower() == "rowid":
            applogger.error("Cannot delete rowid.")
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