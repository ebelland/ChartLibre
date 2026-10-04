"""What a project says about itself: its author, when it was created, notes.

Kept in the project, in ``__project_info__``: one row per entry, a key and a
text value - the shape that takes a new entry (a title, a licence) without
a change of schema, and that any SQLite tool reads back as it is. Like the
other ``__..__`` tables it never shows among the user's tables.

``created`` is written when the table is first made: the moment a new
project is created, or - for a project made before this table existed -
the date the file itself was created, which is the nearest thing to it.

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from app.data.repo._common import RepoHost, ensure_connection_wrapper

PROJECT_INFO_TABLE = "__project_info__"


def parse_references(text: str | None) -> list[dict[str, str]]:
    """The ``references`` entry as a list; [] when empty or not a list.

    A plain string in the list - a reference written by hand - is taken as
    its citation.
    """
    try:
        value = json.loads(text) if text else []
    except ValueError:
        return []
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append({"citation": item.strip()})
        elif isinstance(item, dict) and str(item.get("citation") or "").strip():
            result.append({key: str(item[key]).strip() for key in ("citation", "doi", "url") if str(item.get(key) or "").strip()})
    return result


def dump_references(references: list[dict[str, str]]) -> str | None:
    """*references* as the entry's JSON, or None when there are none."""
    kept = [entry for entry in references if str(entry.get("citation") or "").strip()]
    return json.dumps(kept, ensure_ascii=False) if kept else None


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


def read_project_info(path: Path) -> dict[str, str]:
    """The project information of the .dhub at *path*, without opening it.

    Read-only, through SQLite's own read-only mode: no -wal or undo file is
    made, nothing is migrated - this is for showing a project that is not
    the one open (Open recent, Load demo). Empty when the file is missing,
    unreadable, or predates the information.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as con:
            rows = con.execute(f"SELECT key, value FROM {PROJECT_INFO_TABLE}").fetchall()
    except sqlite3.Error:
        return {}
    return {str(key): str(value) for key, value in rows}


class ProjectInfoMixin(RepoHost):
    """The ``__project_info__`` entries."""

    __slots__ = ()

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
