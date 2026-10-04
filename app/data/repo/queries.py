"""Saved queries, and the indexes made for them.

A saved query is a SELECT kept under a name so a chart can be built on it
like any table (see ``list_data_sources``, which lists the two together).
That makes it the one place the application stores SQL a person wrote, so
it is also where the read-only check lives - ``is_read_only_select`` in
``_common``, which is stricter than "starts with SELECT" for the reason
written there.

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

from typing import Any, Mapping

from app.data.repo._common import (
    RepoHost,
    SavedQuery,
    _dumps_json,
    _loads_json,
    _quote_ident,
    ensure_connection_wrapper,
    ensure_read_only_select,
)
from app.logs.logger import applogger


class QueriesMixin(RepoHost):
    """Saved queries and explicit indexing."""

    __slots__ = ()

    # =====================================================================
    # Saved queries and explicit indexing
    # =====================================================================

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


