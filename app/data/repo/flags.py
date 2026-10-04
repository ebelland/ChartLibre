"""The columns the application keeps on a user table: Hide, Selected, ClusterId.

Hide is the one every chart skips (a series' query carries
``WHERE "Hide" = 0``); Selected marks rows for the user's own queries and
operations, and the same tools act on it; ClusterId is what Cluster writes.
Each is an INTEGER column, 0 or 1 (an id for ClusterId), added to a table
the first time it is wanted. Outliers also keeps a per-row colour column,
and a copy of Hide while its preview is open, so Cancel can put it back.

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pandas as pd

from app.data.data_source import resolve_role_column
from app.data.repo._common import RepoHost, _quote_ident, ensure_connection_wrapper
from app.data.select_sql import sql_insert_select_expression
from app.logs.logger import applogger


class FlagsMixin(RepoHost):
    """Hide, Selected and ClusterId: the 0/1 (or id) columns kept on a table."""

    __slots__ = ()

    #: The 0/1 columns the application maintains on a table: Hide, which
    #: every chart skips, and Selected, which marks rows for the user's own
    #: queries and operations. The same tools act on either.
    FLAG_COLUMNS: tuple[str, ...] = ("Hide", "Selected")

    # ------------------------------------------------------------------
    # Any flag
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Hide
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Outliers' preview: Hide kept aside, and per-row colours
    # ------------------------------------------------------------------

    @ensure_connection_wrapper
    def ensure_preview_state_columns(self, table_name: str) -> None:
        """Create/update temporary preview state columns for Hide preview.

        ``Hide`` is the editable runtime column used by chart filtering.
        ``__DataHubPreviewHide`` stores the pre-preview Hide state so Preview can
        be rolled back without relying on UI-side SQL or direct connection use.
        """
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        preview_col = _quote_ident("__DataHubPreviewHide")
        self.ensure_hide_column(table_name)
        self.ensure_column(
            table_name=table_name,
            col_name="__DataHubPreviewHide",
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
        preview_col = _quote_ident("__DataHubPreviewHide")
        if not self.has_column(table_name, "__DataHubPreviewHide"):
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
        if not self.has_column(table_name, "__DataHubPreviewHide"):
            return
        self._con.execute(
            f'ALTER TABLE {_quote_ident(table_name)} '
            f'DROP COLUMN {_quote_ident("__DataHubPreviewHide")}'
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

    # ------------------------------------------------------------------
    # ClusterId
    # ------------------------------------------------------------------

    @ensure_connection_wrapper
    def set_ClusterId(self,source_table,source_x_column, x_values,cluster_values) -> None:
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
