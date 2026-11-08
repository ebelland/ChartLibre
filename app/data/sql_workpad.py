"""The SQL workpad's session: SQL typed against the project, writing only on a copy until it is kept.

Reading is free and runs on the project itself. The first statement that writes copies the whole project with
SQLite's backup API into a temporary file, and from then on every statement runs there: Apply puts the copy back
(one Undo entry takes it all back), Cancel deletes it and the project was never touched.

What a statement may do is checked twice: on its text (one statement, none of the refused words), and by an
authorizer that SQLite itself consults for every action - so a statement the text check misread is still stopped:
    always refused   ATTACH, DETACH, PRAGMA that sets something, load_extension, writing sqlite_master or any of
                     the application's own tables (names starting "__")
    guard on         reading only (SELECT, WITH)
    guard off        also INSERT, UPDATE, DELETE, CREATE / DROP / ALTER of the user's own tables
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from collections.abc import Callable
from typing import Any

from app.data.repo._common import (
    _READING_PRAGMAS,
    _WRITE_ACTIONS,
    _sql_without_comments_and_literals,
    is_read_only_select,
    sql_guard_enabled,
)

MAX_ROWS = 10_000
TIME_LIMIT = 15.0  # seconds a statement may run


class RefusedStatement(ValueError):
    """A statement the workpad does not run, with why."""


def _own_table(name: Any) -> bool:
    # sqlite_master is written by every CREATE and DROP, so it cannot be refused here; writing it directly is
    # refused by SQLite itself (writable_schema stays off: PRAGMA that sets something never runs).
    return str(name or "").startswith("__")


def check(sql: str) -> bool:
    """Raise RefusedStatement for what is never run; return whether *sql* only reads."""
    text = str(sql or "").strip()
    if not text:
        raise RefusedStatement("The statement is empty.")
    if not sqlite3.complete_statement(text if text.endswith(";") else text + ";"):
        raise RefusedStatement("The statement is not complete (an open quote or bracket?).")
    code = _sql_without_comments_and_literals(text).strip().rstrip(";").rstrip()
    if ";" in code:
        raise RefusedStatement("One statement at a time.")
    words = code.lower().split()
    if words and words[0] in {"attach", "detach", "vacuum", "pragma", "begin", "commit", "rollback", "savepoint",
                              "release", "reindex", "analyze"}:
        raise RefusedStatement(f"{words[0].upper()} is not run in the workpad.")
    reads, _reason = is_read_only_select(text)
    if not reads and sql_guard_enabled():
        raise RefusedStatement("The SQL guard is on: only SELECT and WITH run. Writing needs the guard off "
                               "(Settings > Advanced) - and happens on a copy until Apply.")
    return reads


def _authorizer(writing: bool) -> Callable[..., int]:
    def decide(action: int, arg1: Any, arg2: Any, _db: Any, _source: Any) -> int:
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            return sqlite3.SQLITE_OK if arg2 is None and str(arg1 or "").lower() in _READING_PRAGMAS \
                else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(arg2 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        if action in _WRITE_ACTIONS:
            if not writing:
                return sqlite3.SQLITE_DENY
            table = arg2 if action == sqlite3.SQLITE_ALTER_TABLE else arg1
            if _own_table(table) or action in (sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_DROP_TRIGGER,
                                               sqlite3.SQLITE_CREATE_VTABLE, sqlite3.SQLITE_DROP_VTABLE):
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return decide


class SqlSession:
    """SQL on the project *repo()*; writes on a copy until apply()."""

    def __init__(self, repo: Callable[[], Any]) -> None:
        self._repo = repo
        self._copy_path: str | None = None
        self._copy: sqlite3.Connection | None = None
        self._copy_of: Any = None  # the project the copy was made of: another one opened throws it away

    # ------------------------------------------------------------------ the copy
    @property
    def on_copy(self) -> bool:
        return self._copy is not None

    def _project(self) -> sqlite3.Connection:
        repo = self._repo()
        if self._copy is not None and repo is not self._copy_of:
            self.cancel()
        if repo is None:
            raise RefusedStatement("No project is open.")
        if getattr(repo, "_con", None) is None:
            repo._connect()
        return repo._con

    def _make_copy(self) -> sqlite3.Connection:
        source = self._project()
        handle, path = tempfile.mkstemp(prefix="chartlibre-sql-", suffix=".sqlite")
        os.close(handle)
        copy = sqlite3.connect(path)
        source.backup(copy)
        self._copy_path, self._copy, self._copy_of = path, copy, self._repo()
        return copy

    def user_tables(self, con: sqlite3.Connection) -> set[str]:
        return {name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
                if not _own_table(name) and not str(name).startswith("sqlite_")}

    def apply(self) -> bool:
        """Put the copy back into the project, as one Undo entry; False when there was no copy."""
        project = self._project()
        if self._copy is None:
            return False
        repo = self._repo()
        self._copy.commit()
        tables = sorted(self.user_tables(project) | self.user_tables(self._copy))
        project.commit()
        repo.snapshot_for_undo(tables, label="SQL workpad")
        project.execute("BEGIN")
        try:
            for table in tables:  # table by table, so the application's own tables and the undo stay as they are
                quoted = '"' + table.replace('"', '""') + '"'
                project.execute(f"DROP TABLE IF EXISTS main.{quoted}")
                ddl = self._copy.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                                         (table,)).fetchone()
                if ddl is None:
                    continue
                project.execute(ddl[0])
                cursor = self._copy.execute(f"SELECT * FROM {quoted}")
                width = len(cursor.description)
                project.executemany(f"INSERT INTO main.{quoted} VALUES ({', '.join('?' * width)})", cursor)
            project.commit()
        except Exception:
            project.rollback()
            raise
        if hasattr(repo, "invalidate_series_cache"):
            repo.invalidate_series_cache()
        self.cancel()
        return True

    def cancel(self) -> None:
        """Throw the copy away; the project was never touched."""
        if self._copy is not None:
            self._copy.close()
        if self._copy_path:
            try:
                os.remove(self._copy_path)
            except OSError:
                pass
        self._copy = self._copy_path = self._copy_of = None

    # ------------------------------------------------------------------ running
    def run(self, sql: str) -> tuple[list[str], list[tuple], str]:
        """Run one statement: (columns, rows - at most MAX_ROWS -, a message)."""
        reads = check(sql)
        self._project()  # another project opened since: its copy goes
        con = self._copy if self._copy is not None else (self._project() if reads else self._make_copy())
        deadline = time.monotonic() + TIME_LIMIT
        con.set_authorizer(_authorizer(writing=not reads))
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cursor = con.execute(sql.strip().rstrip(";"))
            if cursor.description is None:
                count = cursor.rowcount
                return [], [], (f"{count} rows changed." if count >= 0 else "Done.") + " (on the copy)"
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
            more = len(rows) > MAX_ROWS
            rows = [tuple(row) for row in rows[:MAX_ROWS]]
            return columns, rows, f"{len(rows)} rows" + (f" (the first {MAX_ROWS})" if more else "") + "."
        except sqlite3.DatabaseError as error:
            if "interrupted" in str(error):
                raise RefusedStatement(f"Stopped after {TIME_LIMIT:.0f} seconds.") from error
            if "not authorized" in str(error):
                raise RefusedStatement("Not allowed here: the application's own tables, ATTACH, PRAGMA "
                                       "settings and extensions are never changed from the workpad.") from error
            raise
        finally:
            con.set_authorizer(None)
            con.set_progress_handler(None, 0)
