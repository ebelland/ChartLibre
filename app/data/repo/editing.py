"""Hand editing of a user table: its cells, its rows, its columns.

The other parts of the repository serve the application: an import writes
a table, an operation writes a result, a renderer reads one. These serve a
person with the table in front of them and a value that is wrong - the
operations behind ``TableEditorDialog``.

That difference is why every one of them records an undo snapshot. An
import can be run again and an operation recomputed; a hand edit has no
source to be re-read from, so the previous state has to be kept or it is
gone.
"""
from __future__ import annotations

from typing import Any, Sequence

from app.data.repo._common import _is_ident, _quote_ident


class EditingMixin:
    """Cell, row and column edits made by hand."""

    __slots__ = ()

    # Each of these records an undo snapshot, unlike the import and
    # operation paths: this is somebody editing their own data by hand, and
    # there is no source to re-read it from when it turns out to have been
    # the wrong cell.
    #
    # ``undo_entry`` is how they stay usable on a table of any size. A
    # snapshot copies the whole table, so one per cell edit would copy a
    # hundred thousand rows every time a number was corrected. Passing the
    # id of an entry already open adds to it instead, and UndoStore skips a
    # table that entry already holds - so an editing session costs one copy,
    # taken before its first change, and undoes in one step. Left as None,
    # each call opens an entry of its own, which is what a single edit from
    # anywhere else should do.

    def _connected(self) -> None:
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None

    def _table_columns(self, table_name: str) -> list[str]:
        assert self._con is not None
        return [
            str(row[1])
            for row in self._con.execute(
                f"PRAGMA table_info({_quote_ident(table_name)})"
            ).fetchall()
        ]

    def has_integer_primary_key(self, table_name: str) -> bool:
        """Say whether this table's rowid is a column the user can see.

        ``INTEGER PRIMARY KEY`` makes a column *be* the rowid rather than
        shadow it. Renumbering rowids to reorder rows would then be silently
        rewriting the user's own key values, so the positional insert below
        refuses on such a table rather than doing that quietly.
        """
        self._connected()
        assert self._con is not None
        rows = self._con.execute(
            f"PRAGMA table_info({_quote_ident(table_name)})"
        ).fetchall()
        for row in rows:
            if int(row[5] or 0) == 1 and str(row[2] or "").strip().upper() == "INTEGER":
                return True
        return False

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
        self._connected()
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

    def append_table_row(self, table_name: str, *, undo_entry: int | None = None) -> int:
        """Add one empty row at the end, and return its rowid."""
        self._connected()
        assert self._con is not None
        self.snapshot_for_undo(
            [table_name], label=f"Add a row to '{table_name}'", entry_id=undo_entry
        )
        table_sql = _quote_ident(table_name)
        columns = self._table_columns(table_name)
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
        self._connected()
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

        columns = self._table_columns(table_name)
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

    def delete_table_rows(
        self, table_name: str, rowids: Sequence[int], *, undo_entry: int | None = None
    ) -> int:
        """Delete the named rows, and return how many went."""
        self._connected()
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
        self._connected()
        assert self._con is not None
        name = str(column_name or "").strip()
        if not name:
            raise ValueError("a column needs a name")
        if not _is_ident(name):
            raise ValueError(f"'{name}' is not a usable column name")

        table_sql = _quote_ident(table_name)
        existing = self._table_columns(table_name)
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

        types = {
            str(row[1]): str(row[2] or "")
            for row in self._con.execute(f"PRAGMA table_info({table_sql})").fetchall()
        }
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
