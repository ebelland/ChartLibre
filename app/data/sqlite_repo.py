"""SQLite repository for ChartLibre: the connection, and everything on it.

``SqliteRepo`` is the one class the application talks to - every caller
writes ``from app.data.sqlite_repo import SqliteRepo``. This module holds:

* the dataclass and its fields, including the connection itself;
* connecting, the pragmas, and closing;
* the **series DataFrame cache**, the single biggest lever on render time -
  without it every property tweak re-runs every series query against
  SQLite (see ``series_frame`` for the invalidation contract);
* preview savepoints and the transaction helper every writer uses;
* the undo hooks, which need the connection to attach their own database;
* the figure, axis and series descriptors.

The user tables (with their flags and hand edits) are TablesMixin in
``app/data/repo/tables.py``; table tools, saved queries, maintenance, the
operations record and the project info are ProjectMixin in
``app/data/repo/project.py``. SqliteRepo inherits both.

Errors, reports and SQL guards shared with the rest of the application are
in ``app/data/repo/_common.py``; the SQL functions registered on each
connection in ``app/data/repo/function_sql.py``.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import app.data.descriptors

# Re-exported, not just imported: "from app.data.sqlite_repo import
# DatabaseReport" is what the rest of the application writes, and the split
# of its helpers into app/data/repo/ is meant to be invisible from outside (todo.txt N-5).
from app.data.repo._common import (  # noqa: F401 - re-export
    _RETURNS_ROWS_RE,
    READ_FAILURES,
    ROW_ID,
    DatabaseError,
    DatabasePragmaInfo,
    DatabaseReport,
    SavedQuery,
    StoredFunction,
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    check_sql_expression,
    descriptor_write_wrapper,
    ensure_connection_wrapper,
    ensure_read_only_select,
    is_read_only_select,
    read_only,
    sql_guard_enabled,
)
from app.data.repo.function_sql import register as register_function_sql
from app.data.repo.project import PROJECT_INFO_TABLE, PROJECT_PREVIEW_TABLE, ProjectMixin, file_created_at
from app.data.repo.tables import TablesMixin
from app.data.select_sql import (
    has_projection_alias,
)
from app.data.series_frame import SeriesFrame
from app.data.undo_store import UndoEntry, UndoStore
from app.logs.logger import applogger
from app.utils.config import get_constant, load_config

# Series cache defaults, overridable from config.json under "series_cache".
_SERIES_CACHE_DEFAULT_ENABLED = True
_SERIES_CACHE_DEFAULT_MAX_ENTRIES = get_constant("series_cache_entries", 64)


class _PreviewSafeConnection(sqlite3.Connection):
    """A connection whose commit() waits while a Preview's savepoint is open (``holding``).

    Whatever commits on its own - pandas' to_sql does, after every write it
    makes - ended the savepoint, and the Preview the user then cancelled
    stayed in the project: its tables, and its figure, which the window went
    on showing after the rollback. The repository commits for real when the
    savepoint is released.
    """

    holding: bool = False

    def commit(self) -> None:
        if not self.holding:
            super().commit()


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


def read_project_preview(path: Path) -> bytes | None:
    """The preview PNG embedded in the .dhub at *path*, read without opening it; None if there is none."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as con:
            row = con.execute(f"SELECT png FROM {PROJECT_PREVIEW_TABLE} WHERE id = 1").fetchone()
    except sqlite3.Error:
        return None
    return bytes(row[0]) if row and row[0] else None


@dataclass(slots=True)
class SqliteRepo(TablesMixin, ProjectMixin):
    """SQLite repository for ChartLibre.

    Manages:
      - DataFrame queries and imports
      - Saved import links (source path + settings)
      - Figure/axis/series descriptor schemas
      - The series DataFrame cache used by the render pipeline

    Optimizations:
      - Persistent connection with WAL mode
      - Memory-mapped I/O and large cache
      - Prepared statements (row_factory cached)
      - Bounded LRU cache of series frames (see ``series_frame``)
    """

    db_path: Path
    _con: sqlite3.Connection | None = None
    _is_connected: bool = False
    _preview_savepoint_name: str | None = None

    # --- series cache -----------------------------------------------------
    # Holds SeriesFrame, not DataFrame (todo.txt P2-17); .to_pandas() on a
    # returned frame for a caller that wants one.
    _series_cache: OrderedDict[str, SeriesFrame] = field(default_factory=OrderedDict)
    _series_cache_stamp: tuple[int, int, int] | None = None
    _series_cache_enabled: bool = _SERIES_CACHE_DEFAULT_ENABLED
    _series_cache_max_entries: int = _SERIES_CACHE_DEFAULT_MAX_ENTRIES
    _series_cache_hits: int = 0
    _series_cache_misses: int = 0
    # Accumulated by @descriptor_write_wrapper: total_changes contributed by
    # writes that only touch descriptor tables, subtracted back out in
    # _database_stamp() below so an axis/figure/series edit does not clear
    # every other series's cached frame along with its own.
    _series_cache_metadata_changes: int = 0
    # Reentrancy depth for @descriptor_write_wrapper - see its docstring on
    # why only the outermost nested call may add to the counter above.
    _series_cache_metadata_write_depth: int = 0

    # Row counts behind series_row_count(), invalidated the same way and at
    # the same time as _series_cache - see that field's docstring.
    _series_row_count_cache: dict[str, int] = field(default_factory=dict)
    #: SQL texts found to write when this project was opened (scan_user_sql);
    #: they are refused wherever they would run, until edited.
    _blocked_sql: set[str] = field(default_factory=set)

    # Built on first use rather than in __post_init__: a repository is
    # created for a great many things that never change a table, and the
    # store is only a path until one of them does.
    _undo_store: UndoStore | None = None


    # =====================================================================
    # Path helpers
    # =====================================================================

    @staticmethod
    def ensure_dhub_extension(path: Path) -> Path:
        """Enforce .dhub extension; use existing path if found, else add .dhub."""
        p = path.expanduser().resolve()
        if p.exists():
            return p
        return p if p.suffix.lower() == ".dhub" else p.with_suffix(".dhub")

    @classmethod
    def create_empty(cls, path: Path) -> Path:
        """Create an empty database at *path* and return the file written.

        Connecting is what creates a database - the file, the pragmas and the
        system tables all come from ``_connect`` - so this exists to make that
        happen *now*, at a point where a failure can still be handled, rather
        than on the first query.  Startup uses it when the remembered database
        has been moved or deleted: an empty project the user can import into
        beats a file dialog they have nothing to pick in.

        The connection is closed again, because the caller opens its own.
        """
        repo = cls(db_path=cls.ensure_dhub_extension(Path(path)))
        try:
            repo._connect()
        finally:
            repo.close()
        applogger.info("Created empty database: %s", repo.db_path)
        return repo.db_path

    # =====================================================================
    # Connection lifecycle
    # =====================================================================

    def _connect(self):
        """Establish SQLite connection with performance pragmas.

        Enables WAL mode, memory-mapped I/O, and large cache for speed.
        Creates required system tables if missing.
        """
        # Close existing connection if open
        if self._con is not None:
            self._con.close()
            del self._con

        db_path = self.ensure_dhub_extension(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Establish connection
        self._con = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
            factory=_PreviewSafeConnection,
        )
        self._is_connected = False

        try:
            # Performance pragmas (must be outside transaction)
            self._con.execute("PRAGMA journal_mode = WAL;")
            self._con.execute("PRAGMA synchronous = NORMAL;")  # Fast + safe with WAL
            self._con.execute("PRAGMA foreign_keys = ON;")
            self._con.execute("PRAGMA busy_timeout = 30000;")

            # In-transaction pragmas
            with self._con:
                self._con.execute("PRAGMA temp_store = MEMORY;")
                self._con.execute("PRAGMA cache_size = -65536;")  # 64 MiB
                self._con.execute("PRAGMA mmap_size = 268435456;")  # 256 MiB
                self._con.execute("PRAGMA wal_autocheckpoint = 1000;")
                self._con.execute("PRAGMA journal_size_limit = 67108864;")
                self._con.execute("PRAGMA optimize;")
        except sqlite3.Error:
            applogger.error("Failed to apply performance pragmas", exc_info=True)
        finally:
            applogger.debug(f"Connected to {db_path}")

        self._is_connected = True
        # Cache row_factory for performance
        self._con.row_factory = sqlite3.Row
        # A function source's query evaluates its stored function, at each read.
        register_function_sql(self._con)

        # A new connection means new counters; nothing cached is trustworthy.
        self.invalidate_series_cache()
        self._load_series_cache_settings()

        # Create system tables
        self._create_system_tables()
        # A project from before R-07: its tables get their row id column once,
        # and its series' queries select it.
        self.ensure_row_ids()
        try:
            self.add_row_ids_to_series()
        except sqlite3.Error:
            applogger.exception("Could not add the row id to the series queries")
        try:
            self.repair_misplaced_hide_filters()
        except sqlite3.Error:
            applogger.exception("Could not check the series queries")

    # =====================================================================
    # Series DataFrame cache
    # =====================================================================

    def _load_series_cache_settings(self) -> None:
        """Read the optional ``series_cache`` section from config.json.

        Example::

            "series_cache": { "enabled": true, "max_entries": 64 }

        Why: the cache is the one optimisation that can produce wrong output if
        invalidation ever misses a write, so it must be switchable off without
        touching code.
        """
        self._series_cache_enabled = _SERIES_CACHE_DEFAULT_ENABLED
        self._series_cache_max_entries = _SERIES_CACHE_DEFAULT_MAX_ENTRIES

        try:
            section = load_config().get("series_cache", {})
        except (AttributeError, TypeError, ValueError):  # a config that is not an object
            applogger.exception("Failed to read series_cache config; using defaults")
            return

        if not isinstance(section, Mapping):
            return

        self._series_cache_enabled = bool(section.get("enabled", _SERIES_CACHE_DEFAULT_ENABLED))
        try:
            self._series_cache_max_entries = max(1, int(section.get("max_entries", _SERIES_CACHE_DEFAULT_MAX_ENTRIES)))
        except (TypeError, ValueError):
            applogger.warning(
                "Invalid series_cache.max_entries=%r; using %d",
                section.get("max_entries"),
                _SERIES_CACHE_DEFAULT_MAX_ENTRIES,
            )

    def _database_stamp(self) -> tuple[int, int, int]:
        """Return a triple that changes whenever query results could change.

        - ``Connection.total_changes`` counts every row inserted, updated or
          deleted through *this* connection, so it catches writes that
          ``PRAGMA data_version`` deliberately ignores.
        - ``PRAGMA data_version`` changes when *another* connection commits.
        - ``PRAGMA schema_version`` changes on any DDL (CREATE / DROP / ALTER),
          which neither of the other two reports.

        Together they need no manual bookkeeping on write paths, which removes
        the whole class of "invalidation missed a writer" bugs.  Cost is ~8 µs
        against the ~35 ms the cache saves per series.

        ``total_changes`` is reduced by ``_series_cache_metadata_changes``
        first: that counter (kept by ``@descriptor_write_wrapper``) is every
        change so far attributable to a write that only touched a descriptor
        table - a figure/axis/series options edit - which cannot itself
        change what any series query returns. Subtracting it is additive
        only: a write nothing has marked as descriptor-only still counts
        fully here, so this can only make the stamp change *less* often than
        raw ``total_changes`` would, never miss a real data write.
        """
        con = self._con
        if con is None:
            return (0, 0, 0)

        data_version = int(con.execute("PRAGMA data_version").fetchone()[0])
        schema_version = int(con.execute("PRAGMA schema_version").fetchone()[0])
        changes = int(con.total_changes) - self._series_cache_metadata_changes
        return (changes, data_version, schema_version)

    def invalidate_series_cache(self) -> None:
        """Drop every cached series DataFrame."""
        self._series_cache.clear()
        self._series_row_count_cache.clear()
        self._series_cache_stamp = None

    @property
    def series_cache_stats(self) -> dict[str, int | bool]:
        """Return cache counters; used by tests and the benchmark."""
        return {
            "enabled": self._series_cache_enabled,
            "entries": len(self._series_cache),
            "max_entries": self._series_cache_max_entries,
            "hits": self._series_cache_hits,
            "misses": self._series_cache_misses,
        }

    @ensure_connection_wrapper
    def series_frame(self, sql: str) -> SeriesFrame:
        """Return the SeriesFrame for a series query, cached per database state.

        This is the render path (see ``render_figure._load_series_df``):
        SQLite's own cursor, typed straight into numpy arrays - no
        ``pd.read_sql_query`` between the database and the cache. A caller
        that wants a DataFrame calls ``.to_pandas()`` on the result.

        The cache is keyed by SQL text and wholesale invalidated whenever
        ``_database_stamp`` changes, so a hit can only ever be served for the
        exact database state that produced it.

        Contract: the returned frame is a copy. Mutating it - adding a
        column, changing one in place - never reaches the cached entry.
        """
        sql_text = (sql or "").strip()
        if not sql_text:
            return SeriesFrame()

        assert self._con is not None

        if not self._series_cache_enabled:
            return self._read_series_frame(sql_text)

        stamp = self._database_stamp()
        if stamp != self._series_cache_stamp:
            self._series_cache.clear()
            self._series_cache_stamp = stamp

        cached = self._series_cache.get(sql_text)
        if cached is not None:
            self._series_cache_hits += 1
            self._series_cache.move_to_end(sql_text)
            return cached.copy()

        self._series_cache_misses += 1
        frame = self._read_series_frame(sql_text)
        self._series_cache[sql_text] = frame

        # Bound the cache; the oldest entry is the least recently used one.
        while len(self._series_cache) > self._series_cache_max_entries:
            self._series_cache.popitem(last=False)

        return frame.copy()

    def _read_series_frame(self, sql_text: str) -> SeriesFrame:
        assert self._con is not None
        self.refuse_if_blocked(sql_text)
        cursor = self._con.execute(sql_text)
        names = tuple(str(d[0]) for d in cursor.description or ())
        rows = cursor.fetchall()
        return SeriesFrame.from_rows(names, rows)

    # =====================================================================
    # SQL guard (todo N-05): only at open and where SQL is written by hand
    # =====================================================================
    def refuse_if_blocked(self, sql: str | None) -> None:
        """Raise ValueError for SQL that the opening scan found would write."""
        if self._blocked_sql and str(sql or "").strip() in self._blocked_sql:
            raise ValueError("This query would change the database and has been blocked.")

    def scan_user_sql(self) -> list[str]:
        """Check every saved query and series SQL; block and return the unsafe ones.

        Run once when a project is opened: a file made elsewhere can carry a
        query that writes. Returns "kind 'name': reason" lines for a message;
        nothing is blocked when the guard is switched off in user.json.
        """
        self._blocked_sql.clear()
        if not sql_guard_enabled():
            return []
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        found: list[str] = []
        candidates: list[tuple[str, str, str]] = []
        for table, kind, name_col in (("__queries__", "Query", "name"), ("__series_descriptors__", "Series", "name")):
            exists = self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                continue
            column = "sql" if table == "__queries__" else "sql_query"
            try:
                for name, sql in self._con.execute(f'SELECT "{name_col}", "{column}" FROM "{table}"'):
                    candidates.append((kind, str(name or ""), str(sql or "").strip()))
            except sqlite3.Error:
                applogger.exception("Could not read %s for the SQL check.", table)
        for kind, name, sql in candidates:
            if not sql or " " not in sql:
                continue  # empty, or a bare table name
            ok, reason = is_read_only_select(sql)
            if not ok:
                self._blocked_sql.add(sql)
                found.append(f"{kind} '{name}': {reason}")
        return found


    @ensure_connection_wrapper
    def series_row_count(self, sql: str) -> int:
        """Return how many rows *sql* would produce, without building them.

        Downsampling needs the total row count before it can size a stride -
        a single ``COUNT(*)`` rather than reading the whole result into a
        DataFrame just to call ``len()`` on it, which would defeat the point
        of downsampling in the first place. Cached the same way and
        invalidated by the same database-state stamp as ``series_frame``.
        """
        sql_text = (sql or "").strip()
        if not sql_text:
            return 0

        assert self._con is not None

        stamp = self._database_stamp()
        if stamp != self._series_cache_stamp:
            self._series_cache.clear()
            self._series_row_count_cache.clear()
            self._series_cache_stamp = stamp

        cached = self._series_row_count_cache.get(sql_text)
        if cached is not None:
            return cached

        self.refuse_if_blocked(sql_text)
        row = self._con.execute(f"SELECT COUNT(*) FROM ({sql_text})").fetchone()
        count = int(row[0]) if row and row[0] is not None else 0
        self._series_row_count_cache[sql_text] = count
        return count

    @ensure_connection_wrapper
    def downsampled_series_frame(self, sql: str, *, threshold: int) -> SeriesFrame:
        """Return *sql*'s result, decimated to roughly ``threshold`` rows.

        Decided and applied entirely inside SQLite: one ``COUNT(*)``
        (``series_row_count``, itself cached) to size a stride, and - only
        when the query actually exceeds ``threshold`` - one row kept out of
        every stride via ``ROW_NUMBER()``, in the query's own order (a
        curve's ``ORDER BY x``); a row with a colour of its own (an outlier
        coloured, a cluster) is always kept. The full
        result is never read only to be thinned out afterwards - that would
        keep the exact cost this exists to avoid.

        ``threshold <= 0`` disables downsampling and reads the plain query.
        """
        sql_text = (sql or "").strip()
        if not sql_text or threshold <= 0:
            return self.series_frame(sql_text)

        total = self.series_row_count(sql_text)
        if total <= threshold:
            return self.series_frame(sql_text)

        stride = max(1, -(-total // threshold))  # ceil division
        # OVER () numbers the rows in the query's own order - a curve's ORDER
        # BY x. "ORDER BY 1" inside a window is the constant 1, not the first
        # column, and kept an arbitrary set of rows.
        keep = f"(__chartlibre_rn__ - 1) % {stride} = 0"
        if has_projection_alias(sql_text, "color"):
            # A row given a colour of its own - an outlier, a cluster - is
            # what the chart is there to show: never thinned out.
            keep += " OR (\"color\" IS NOT NULL AND \"color\" <> '')"
        wrapped = (
            "SELECT * FROM ("
            f"SELECT *, ROW_NUMBER() OVER () AS __chartlibre_rn__ FROM ({sql_text})"
            f") WHERE {keep}"
        )
        return self.series_frame(wrapped).drop_column("__chartlibre_rn__")


    @ensure_connection_wrapper
    def _create_figure_touch_triggers(self) -> None:
        """Triggers keeping __figure_descriptors__.last_modified current."""
        assert self._con
        now = "strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')"
        touch_figure = "UPDATE __figure_descriptors__ SET last_modified = " + now + " WHERE id = {figure}"
        axis_of_series = "(SELECT figure_id FROM __axis_descriptors__ WHERE id = {series}.axis_id)"
        statements = [
            # Not on last_modified itself, nor on created: those are the record.
            "CREATE TRIGGER IF NOT EXISTS trg_figure_touched AFTER UPDATE OF name, nrows, ncols, options_json, note "
            "ON __figure_descriptors__ BEGIN " + touch_figure.format(figure="NEW.id") + "; END",
        ]
        for event in ("INSERT", "UPDATE", "DELETE"):
            row = "OLD" if event == "DELETE" else "NEW"
            statements.append(
                f"CREATE TRIGGER IF NOT EXISTS trg_axis_{event.lower()}_touches_figure AFTER {event} "
                f"ON __axis_descriptors__ BEGIN " + touch_figure.format(figure=f"{row}.figure_id") + "; END"
            )
            statements.append(
                f"CREATE TRIGGER IF NOT EXISTS trg_series_{event.lower()}_touches_figure AFTER {event} "
                f"ON __series_descriptors__ BEGIN "
                + touch_figure.format(figure=axis_of_series.format(series=row)) + "; END"
            )
        for statement in statements:
            self._con.execute(statement)

    @ensure_connection_wrapper
    def _create_system_tables(self):
        """Create required system tables if missing."""

        # Import links: maps table names to source + settings
        assert self._con
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __import_links__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name    TEXT NOT NULL UNIQUE,
                source_path   TEXT NOT NULL,
                settings_json TEXT NOT NULL
            );
            """
        )

        # Figure descriptors: grid layout and options
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __figure_descriptors__ (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                nrows       INTEGER NOT NULL DEFAULT 1,
                ncols       INTEGER NOT NULL DEFAULT 1,
                options_json TEXT
            );
            """
        )

        # Ensure options_json exists on older databases (schema migration)
        cols = {
            row[1] for row in self._con.execute(
                "PRAGMA table_info(__figure_descriptors__)"
            ).fetchall()
        }
        if "options_json" not in cols:
            self._con.execute(
                "ALTER TABLE __figure_descriptors__ ADD COLUMN options_json TEXT"
            )
        # A figure's notes and dates (older projects gain the columns here;
        # their figures are dated by the project file, the nearest there is).
        added_dates = False
        for column in ("note", "created", "last_modified"):
            if column not in cols:
                self._con.execute(f"ALTER TABLE __figure_descriptors__ ADD COLUMN {column} TEXT")
                added_dates = added_dates or column != "note"
        if added_dates:
            born = file_created_at(Path(str(self.db_path))) if self.db_path else None
            stamp = (born or datetime.now(timezone.utc)).isoformat(timespec="seconds")
            self._con.execute(
                "UPDATE __figure_descriptors__ SET created = COALESCE(created, ?), "
                "last_modified = COALESCE(last_modified, ?)",
                (stamp, stamp),
            )

        # Axis descriptors: chart type and labels
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __axis_descriptors__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                figure_id     INTEGER NOT NULL,
                axis_index    INTEGER NOT NULL,
                chart_type    TEXT NOT NULL,
                title         TEXT,
                x_label       TEXT,
                y_label       TEXT,
                z_label       TEXT,
                options_json  TEXT,
                FOREIGN KEY(figure_id) REFERENCES __figure_descriptors__(id)
                    ON DELETE CASCADE,
                UNIQUE(figure_id, axis_index)
            );
            """
        )

        # Series descriptors: data queries and styling
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __series_descriptors__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                axis_id       INTEGER NOT NULL,
                series_index  INTEGER NOT NULL,
                name          TEXT,
                sql_query     TEXT NOT NULL,
                roles         TEXT,
                style_json    TEXT,
                FOREIGN KEY(axis_id) REFERENCES __axis_descriptors__(id)
                    ON DELETE CASCADE,
                UNIQUE(axis_id, series_index)
            );
            """
        )

        # Table descriptors: user table metadata
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __table_descriptors__ (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                notes       TEXT,
                info_json   TEXT
            );
            """
        )

        # Saved queries: see app/data/repo/queries.py.
        self.create_queries_table()
        self.create_sympy_table()

        # A figure's last_modified follows every change to it, its axes or
        # their series, whichever code makes it.
        self._create_figure_touch_triggers()

        # What was applied, and how: see app/data/repo/operations.py.
        self.create_operations_table()

        # Author, creation date, notes: see app/data/repo/project_info.py.
        # After the descriptor tables, which it looks at to tell a new
        # project from an old one.
        self.create_project_info_table()

        # No indexes of its own: each lookup goes through a UNIQUE constraint
        # above (axis by figure, series by axis, query, link and table
        # descriptor by name), and SQLite indexes every one of those.


    # =====================================================================
    # Preview savepoints
    # =====================================================================

    @property
    def preview_savepoint_active(self) -> bool:
        """True when a UI preview SAVEPOINT is currently open."""
        return bool(self._preview_savepoint_name)

    @ensure_connection_wrapper
    def begin_preview_savepoint(self, name: str | None = None) -> str:
        """Start a rollback-able SAVEPOINT for dialog Preview changes.

        Repository methods often call commit() after writes. During an active
        preview savepoint, those commits are intentionally suppressed by
        ``_commit()`` so Close/Cancel can roll back all preview changes at once.
        """
        assert self._con is not None
        if self._preview_savepoint_name:
            self.rollback_preview_savepoint()
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", str(name or f"preview_{uuid.uuid4().hex}"))
        if not safe_name or safe_name[0].isdigit():
            safe_name = f"preview_{safe_name}"
        self._con.execute(f"SAVEPOINT {safe_name}")
        self._preview_savepoint_name = safe_name
        self._hold_commits(True)
        applogger.info("Started preview savepoint: %s", safe_name)
        return safe_name

    @ensure_connection_wrapper
    def release_preview_savepoint(self) -> bool:
        """Commit the active Preview SAVEPOINT.

        Returns True when SQLite released the savepoint, False when Python state
        was stale and the savepoint was already gone.
        """
        assert self._con is not None
        sp = self._preview_savepoint_name
        if not sp:
            return False
        released = False
        try:
            self._con.execute(f"RELEASE SAVEPOINT {sp}")
            released = True
            applogger.info("Released preview savepoint: %s", sp)
        except sqlite3.OperationalError as exc:
            if "no such savepoint" not in str(exc).lower():
                raise
            applogger.warning(
                "Preview savepoint %s was already gone during release; clearing stale state.",
                sp,
            )
        finally:
            self._preview_savepoint_name = None
            self._hold_commits(False)
        if released:
            self._con.commit()
        return released

    @ensure_connection_wrapper
    def rollback_preview_savepoint(self) -> bool:
        """Undo the active Preview SAVEPOINT.

        Returns True when SQLite rolled back the savepoint, False when Python
        state was stale and the savepoint was already gone. Callers can use the
        False result to perform explicit temporary-artifact cleanup.
        """
        assert self._con is not None
        sp = self._preview_savepoint_name
        if not sp:
            return False
        rolled_back = False
        try:
            self._con.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            self._con.execute(f"RELEASE SAVEPOINT {sp}")
            rolled_back = True
            applogger.info("Rolled back preview savepoint: %s", sp)
        except sqlite3.OperationalError as exc:
            if "no such savepoint" not in str(exc).lower():
                raise
            applogger.warning(
                "Preview savepoint %s was already gone during rollback; clearing stale state.",
                sp,
            )
        finally:
            self._preview_savepoint_name = None
            self._hold_commits(False)
        return rolled_back

    def _hold_commits(self, holding: bool) -> None:
        """Make the connection's commit() wait (or not) - see _PreviewSafeConnection."""
        if isinstance(self._con, _PreviewSafeConnection):
            self._con.holding = holding

    def _commit(self) -> None:
        """Commit unless a UI preview savepoint is open.

        Savepoint-backed Preview must survive repository helper methods that
        normally call commit() after each write. While preview is active, writes
        remain inside the savepoint and are either released on Apply/OK or rolled
        back on Close/Cancel.
        """
        if self._con is None:
            return
        if self._preview_savepoint_name:
            return
        self._con.commit()

    # =====================================================================
    # Transactions
    # =====================================================================

    @contextmanager
    @ensure_connection_wrapper
    def transaction(self, *, immediate: bool = False) -> Iterator[None]:
        """Context manager for transactions with SAVEPOINT support.

        - No active tx: start new transaction (optionally IMMEDIATE), then commit/rollback.
        - Active tx: use SAVEPOINT for nested safety.
        """

        if not self._con.in_transaction:  # pyright: ignore[reportOptionalMemberAccess]
            if immediate:
                self._con.execute("BEGIN IMMEDIATE")  # pyright: ignore[reportOptionalMemberAccess]
                try:
                    yield
                except Exception:  # noqa: BLE001 - roll back, then re-raise whatever it was
                    self._con.rollback()  # pyright: ignore[reportOptionalMemberAccess]
                    raise
                else:
                    self._commit()  # pyright: ignore[reportOptionalMemberAccess]
            else:
                with self._con:  # pyright: ignore[reportOptionalContextManager]
                    yield
            return

        # Nested transaction: use SAVEPOINT
        sp = f"sp_{uuid.uuid4().hex}"
        self._con.execute(f"SAVEPOINT {sp}")  # pyright: ignore[reportOptionalMemberAccess]
        try:
            yield
        except Exception:  # noqa: BLE001 - roll back, then re-raise whatever it was
            self._con.execute(f"ROLLBACK TO {sp}")  # pyright: ignore[reportOptionalMemberAccess]
            self._con.execute(f"RELEASE {sp}")  # pyright: ignore[reportOptionalMemberAccess]
            raise
        else:
            self._con.execute(f"RELEASE {sp}")  # pyright: ignore[reportOptionalMemberAccess]

    # =====================================================================
    # Undo
    # =====================================================================
    @property
    def undo_store(self) -> UndoStore:
        """The per-project snapshot store, made on first use.

        Keyed to this repository's own file, so two projects open one after
        the other never see each other's history.
        """
        if self._undo_store is None or self._undo_store.path.parent != Path(
            self.db_path
        ).parent:
            self._undo_store = UndoStore(Path(self.db_path))
        return self._undo_store

    #: The tables that hold what the properties panels and the operations
    #: edit: the figures, their axes, their series, and the per-table notes
    #: and links. Small enough that snapshotting all four is cheaper than
    #: working out which of them an edit will reach.
    DESCRIPTOR_TABLES: ClassVar[tuple[str, ...]] = (
        "__figure_descriptors__",
        "__axis_descriptors__",
        "__series_descriptors__",
        "__table_descriptors__",
    )

    def snapshot_for_undo(
        self,
        tables: Sequence[str],
        *,
        label: str,
        entry_id: int | None = None,
    ) -> int | None:
        """Record the state of *tables* before changing them.

        Call before opening the transaction that does the work: attaching
        the undo database is not allowed inside one. Returns None when
        nothing was recorded, which is not a failure the caller has to
        handle - the action goes ahead either way, it just cannot be taken
        back.

        Pass the returned id back as *entry_id* to add more tables to the
        same entry when what an action touches is only known in stages -
        an operation that creates the axis it then writes a result table
        to. One action, one entry, one step to undo it.
        """
        if self._con is None:
            return None
        return self.undo_store.snapshot(
            self._con, tables, label=label, entry_id=entry_id
        )

    def undo_entries(self) -> list[UndoEntry]:
        """What can be undone, most recent first."""
        if self._con is None:
            return []
        return self.undo_store.entries(self._con)

    def undo_last(self) -> UndoEntry | None:
        """Undo the most recent recorded action and return what it was.

        Everything cached about the database is dropped afterwards: the
        rows behind a series may have just been replaced wholesale, and a
        cache that survived that would serve the state the user has
        undone.
        """
        if self._con is None:
            return None
        entry = self.undo_store.undo(self._con)
        if entry is not None:
            self.invalidate_series_cache()
        return entry

    def undo_entry(self, entry_id: int) -> UndoEntry | None:
        """Undo one particular recorded action, wherever it sits in the list.

        ``undo_last`` is what the Undo command uses; this is for a caller
        that knows which entry is its own - the table editor, whose Cancel
        has to put back the state from before it was opened and nothing
        else. Same restore, same cache invalidation, chosen by id.
        """
        if self._con is None:
            return None
        entry = self.undo_store.undo(self._con, int(entry_id))
        if entry is not None:
            self.invalidate_series_cache()
        return entry

    def close(self) -> None:
        """Close the database connection (call on app shutdown)."""
        con = self._con
        if con is not None:
            try:
                con.commit()
            except sqlite3.Error:
                pass  # nothing left to keep; closing still has to happen
            try:
                con.close()
            except sqlite3.Error:
                pass
        self._con = None
        self._is_connected = False
        self.invalidate_series_cache()

    # ==================================================================
    # Descriptors: Figure, axis and series descriptors, and their stored options.
    # ==================================================================
    # =====================================================================
    # Figure/axis/series deletion
    # =====================================================================
    @ensure_connection_wrapper
    def get_figure_descriptor(self, figure_id: int) -> app.data.descriptors.FigureDescriptor|None:
        """Get figure descriptor as dict (never None)."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT * FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        if row is None:
            return None
        return   app.data.descriptors.FigureDescriptor(
            id=int(row["id"]),
            name=str(row["name"]),
            nrows=int(row["nrows"]),
            ncols=int(row["ncols"]),
            options=_loads_json(row["options_json"]) if "options_json" in row.keys() else {},
            axes=[],
            note=str(row["note"] or "") if "note" in row.keys() else "",
            created=str(row["created"] or "") if "created" in row.keys() else "",
            last_modified=str(row["last_modified"] or "") if "last_modified" in row.keys() else "",
        )

    def load_figure_descriptor(self, figure_id: int) -> app.data.descriptors.FigureDescriptor | None:
        """Load a full figure descriptor tree: figure, axes, and their series.

        Returns None only when the figure itself does not exist.  A figure with
        no axes is a valid, empty figure and is returned as such.
        """
        fig = self.get_figure_descriptor(figure_id)
        if fig is None:
            return None

        axis_rows = self.get_axes(figure_id) or []
        series_by_axis = self.get_series_for_axes([int(a["id"]) for a in axis_rows])

        for a in axis_rows:
            axis = app.data.descriptors.AxisDescriptor(
                id=int(a["id"]),
                parent_id=int(a["figure_id"]),
                index=int(a["axis_index"]),
                name=str(a["chart_type"]),
                title=str(a["title"] or ""),
                x_label=str(a["x_label"] or ""),
                y_label=str(a["y_label"] or ""),
                z_label=str(a["z_label"] or "") if "z_label" in a.keys() else "",
                options=_loads_json(a["options_json"]) if a["options_json"] is not None else None,
                series=[
                    app.data.descriptors.SeriesDescriptor(
                        id=int(s["id"]),
                        parent_id=int(s["axis_id"]),
                        index=int(s["series_index"]),
                        name=str(s["name"] or f"Series {s['series_index']}"),
                        sql_query=str(s["sql_query"]),
                        roles=s["roles"],
                        options=_loads_json(s["style_json"]) if s["style_json"] is not None else None,
                    )
                    for s in series_by_axis.get(int(a["id"]), [])
                ],
            )

            if fig.axes is None:
                fig.axes = []
            fig.axes.append(axis)

        # Apply persisted ordering.
        if fig.axes:
            fig_opts = fig.options if isinstance(fig.options, dict) else {}
            order_ids = [int(x) for x in fig_opts.get("axes_order", []) if str(x).isdigit()]
            if order_ids:
                by_id = {ax.id: ax for ax in fig.axes}
                ordered = [by_id[i] for i in order_ids if i in by_id]
                tail = [ax for ax in fig.axes if ax.id not in set(order_ids)]
                fig.axes = ordered + tail

            for ax in fig.axes:
                opts = ax.options if isinstance(ax.options, dict) else {}
                s_order = [int(x) for x in opts.get("series_order", []) if str(x).isdigit()]
                if s_order and ax.series is not None:
                    s_by = {s.id: s for s in ax.series}
                    s_ord = [s_by[i] for i in s_order if i in s_by]
                    s_tail = [s for s in ax.series if s.id not in set(s_order)]
                    ax.series = s_ord + s_tail

        return fig

    @ensure_connection_wrapper
    def set_figure_note(self, figure_id: int, note: str) -> None:
        """Replace the figure's notes (its last_modified follows, by trigger)."""
        assert self._con is not None
        self._con.execute(
            "UPDATE __figure_descriptors__ SET note = ? WHERE id = ?",
            (str(note or "").strip() or None, int(figure_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    def get_figure_title(self, figure_id: int) -> str | None:
        """Get the name/title of a figure descriptor."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT name FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        return str(row[0]) if row else None

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def delete_figure(self, figure_id: int) -> None:
        """Delete figure and all associated axes and series (cascade)."""
        assert self._con is not None

        with self.transaction(immediate=True):
            # Foreign key constraint handles cascade automatically
            axes= self.get_axes(figure_id)
            if axes is not None:
                for ax in axes:
                    self.delete_axis(axis_id=int(ax[0]))
            self._con.execute(
                "DELETE FROM __figure_descriptors__ WHERE id = ?", (figure_id,)
            )

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def delete_axis(self, axis_id: int) -> None:
        """Delete axis and all associated series (cascade via FK)."""
        assert self._con is not None
        series=self.get_series(axis_id)
        if series is not None:
            for s in series:
                self.delete_series(s[0])
        self._con.execute(
            "DELETE FROM __axis_descriptors__ WHERE id = ?", (int(axis_id),)
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def delete_series(self, series_id: int) -> None:
        """Delete series from database."""
        assert self._con is not None
        self._con.execute("DELETE FROM __series_descriptors__ WHERE id = ?", (int(series_id),))
        self._commit()

    # =====================================================================
    # UI options persistence
    # =====================================================================
    @ensure_connection_wrapper
    def get_figure_options(self, figure_id: int) -> dict[str, Any]:
        """Return figure UI options as a dictionary."""
        assert self._con is not None

        row = self._con.execute(
            "SELECT options_json FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        return _loads_json(row["options_json"]) if row else {}

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def set_figure_grid(self, figure_id: int, *, nrows: int, ncols: int) -> None:
        """Persist figure grid layout."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __figure_descriptors__ SET nrows = ?, ncols = ? WHERE id = ?",
            (int(nrows), int(ncols), int(figure_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def set_figure_options(self, figure_id: int, options: dict[str, Any]) -> None:
        """Persist figure UI options."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __figure_descriptors__ SET options_json = ? WHERE id = ?",
        (_dumps_json(options), int(figure_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def set_figure_properties(self, figure_id: int, nrows:int, ncols:int,name:str, options: dict[str, Any]) -> None:
        """Persist figure UI options."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __figure_descriptors__ SET options_json = ?, nrows = ?, ncols = ?, name= ? WHERE id = ?",
        (_dumps_json(options), nrows,ncols, name, figure_id),
        )
        self._commit()

    @ensure_connection_wrapper
    def get_axis_options(self, axis_id: int) -> dict[str, Any]:
        """Return axis UI options as a dictionary."""
        assert self._con is not None

        row = self._con.execute(
            "SELECT options_json FROM __axis_descriptors__ WHERE id = ?",
            (int(axis_id),),
        ).fetchone()
        return _loads_json(row["options_json"]) if row else {}

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def set_axis_options(self, axis_id: int, options: dict[str, Any]) -> None:
        """Persist axis UI options."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __axis_descriptors__ SET options_json = ? WHERE id = ?",
            (_dumps_json(options), int(axis_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def update_series_style(self, series_id: int, style: dict[str, Any]) -> None:
        """Persist series style_json."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __series_descriptors__ SET style_json = ? WHERE id = ?",
            (_dumps_json(style), int(series_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def update_series_sql_query(self, series_id: int, sql_query: str) -> None:
        """Persist the SQL query for one series descriptor."""
        assert self._con is not None

        self._con.execute(
            """
            UPDATE __series_descriptors__
            SET sql_query = ?
            WHERE id = ?
            """,
            (str(sql_query or "").strip(), int(series_id)),
        )
        self._commit()

    # =====================================================================
    # Series management
    # =====================================================================
    @ensure_connection_wrapper
    def list_series_dict(self) -> list[dict[str, Any]]:
        """Return all series descriptors as dicts."""
        assert self._con is not None
        rows = self._con.execute(
            "SELECT id, series_index, name, sql_query FROM __series_descriptors__"
        ).fetchall()
        return [
            {
                "id": int(r[0]),
                "series_index": int(r[1]),
                "name": str(r[2]),
                "sql_query": str(r[3]),
            }
            for r in rows
        ]

    @ensure_connection_wrapper
    def get_series_sql_query(self, series_id: int) -> str | None:
        """Return the stored SQL of one series, or None when it does not exist."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT sql_query FROM __series_descriptors__ WHERE id = ?",
            (int(series_id),),
        ).fetchone()
        return None if row is None else str(row["sql_query"] or "")

    @ensure_connection_wrapper
    def get_series(self, axis_id: int) -> list[sqlite3.Row]:
        """Return every series descriptor row for one axis, ordered by index."""
        assert self._con is not None
        return self._con.execute(
            "SELECT * FROM __series_descriptors__ WHERE axis_id = ? ORDER BY series_index",
            (axis_id,),
        ).fetchall()

    @ensure_connection_wrapper
    def get_series_for_axes(self, axis_ids: Sequence[int]) -> dict[int, list[sqlite3.Row]]:
        """Return series rows for many axes in a single query, grouped by axis id.

        Why: loading a figure used to issue one ``get_series`` per axis, so the
        cost grew with the number of axes for no reason.  Axes with no series
        are still present in the result, mapped to an empty list.
        """
        assert self._con is not None

        ids = [int(axis_id) for axis_id in axis_ids]
        grouped: dict[int, list[sqlite3.Row]] = {axis_id: [] for axis_id in ids}
        if not ids:
            return grouped

        placeholders = ",".join("?" * len(ids))
        rows = self._con.execute(
            f"SELECT * FROM __series_descriptors__ "
            f"WHERE axis_id IN ({placeholders}) ORDER BY axis_id, series_index",
            tuple(ids),
        ).fetchall()

        for row in rows:
            grouped[int(row["axis_id"])].append(row)
        return grouped

    @ensure_connection_wrapper
    def next_axis_index(self, figure_id: int) -> int:
        """Get next available axis_index for a figure."""
        assert self._con is not None
        try:
            row = self._con.execute(
                "SELECT COALESCE(MAX(axis_index), -1) + 1 "
                "FROM __axis_descriptors__ WHERE figure_id = ?",
                (int(figure_id),),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except READ_FAILURES:
            return 0

    @ensure_connection_wrapper
    def next_series_index(self, axis_id: int) -> int:
        """Get next available series_index for an axis."""
        assert self._con is not None
        try:
            row = self._con.execute(
                "SELECT COALESCE(MAX(series_index), -1) + 1 "
                "FROM __series_descriptors__ WHERE axis_id = ?",
                (int(axis_id),),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except READ_FAILURES:
            return 0

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def update_axis_chart_type(self, *, axis_id: int, chart_type: str) -> None:
        """Update chart_type for an existing axis."""
        assert self._con is not None
        self._con.execute(
            "UPDATE __axis_descriptors__ SET chart_type = ? WHERE id = ?",
            (str(chart_type), int(axis_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def update_axis_descriptor(
        self,
        *,
        axis_id: int,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None,
    ) -> None:
        """Update the labels of an existing axis.

        Only the fields that are given are written, so a caller that knows the
        y unit but not the title does not have to invent one.
        """
        assert self._con is not None

        updates = {
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
        }
        assignments = {
            column: value for column, value in updates.items() if value is not None
        }
        if not assignments:
            return

        clause = ", ".join(f"{column} = ?" for column in assignments)
        self._con.execute(
            f"UPDATE __axis_descriptors__ SET {clause} WHERE id = ?",
            (*[str(value) for value in assignments.values()], int(axis_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def create_figure_descriptor(
        self,
        *,
        name: str,
        nrows: int = 1,
        ncols: int = 1,
        options: Mapping[str, Any] | None = None,
    ) -> int:
        """Create figure descriptor and return its id.

        Args:
            name: Figure name/title
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            layout: Optional layout configuration dict

        Returns:
            Primary key id of created figure (or 0 on error)
        """
        assert self._con is not None
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self._con.execute(
            "INSERT INTO __figure_descriptors__ "
            "(name, nrows, ncols, options_json, created, last_modified) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(name),
                int(nrows),
                int(ncols),
                _dumps_json(options or {}) if options else None,
                now,
                now,
            ),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    def load_figures_from_db(self) -> list[tuple[int, str]]:
        """Load all figures from database as (id, name) tuples."""
        df = self.query_df("SELECT id, name FROM __figure_descriptors__ ORDER BY id")
        if df is None or df.empty:
            return []
        return [
            (int(fig_id), str(name))
            for fig_id, name in df.itertuples(index=False, name=None)
        ]

    @ensure_connection_wrapper
    def get_axes(self, figure_id:int)-> list[sqlite3.Row]|None:
        assert self._con is not None
        axis_rows = self._con.execute(
            "SELECT * FROM __axis_descriptors__ WHERE figure_id = ? ORDER BY axis_index",
            (figure_id,),
        ).fetchall()
        return axis_rows

    @ensure_connection_wrapper
    def list_axes_for_figure(self, figure_id: int) -> list[tuple[int, int, str]]:
        """Get axes for a figure as (axis_id, axis_index, title) tuples."""
        assert self._con is not None

        rows = self._con.execute(
            "SELECT id, axis_index, COALESCE(title, '') AS title "
            "FROM __axis_descriptors__ WHERE figure_id = ? ORDER BY axis_index",
            (int(figure_id),),
        ).fetchall()
        return [
            (int(r["id"]), int(r["axis_index"]), str(r["title"] or ""))
            for r in rows
        ]

    @ensure_connection_wrapper
    def create_axis_descriptor(
        self,
        *,
        figure_id: int,
        axis_index: int,
        chart_type: str,
        title: str,
        x_label: str,
        y_label: str,
        z_label: str | None = None,
        options: Mapping[str, Any] | None,
    ) -> int:
        """Create axis descriptor and return its id.

        Args:
            figure_id: Parent figure id
            axis_index: Position in subplot grid
            chart_type: Type of chart (scatter, line, etc.)
            title: Axis/subplot title
            x_label: X-axis label
            y_label: Y-axis label
            z_label: Z-axis label
            options: Optional configuration dict

        Returns:
            Primary key id of created axis (or 0 on error)
        """
        assert self._con is not None
        cur = self._con.execute(
            "INSERT INTO __axis_descriptors__ "
            "(figure_id, axis_index, chart_type, title, x_label, y_label, z_label, options_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(figure_id),
                int(axis_index),
                str(chart_type),
                str(title),
                str(x_label),
                str(y_label),
                str(z_label) if z_label is not None else "",
                _dumps_json(options or {}) if options else None,
            ),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def create_series_descriptor(
        self,
        *,
        axis_id: int,
        series_index: int,
        name: str,
        sql_query: str,
        roles: Mapping[str,str]|None = None,
        style: Mapping[str, Any] | None = None,
    ) -> int:
        """Create series descriptor and return its id.

        Args:
            axis_id: Parent axis id
            series_index: Position within axis
            name: Series name
            sql_query: SELECT query returning data for this series
            columns: column descriptios dic
            style: Style configuration dict (optional)

        Returns:
            Primary key id of created series (or 0 on error)
        """
        assert self._con is not None
        filtered_sql_query = self.sql_with_row_id(self.sql_with_hide_filter(str(sql_query)))

        # Only a table-backed series gets a Hide column; a series over a saved
        # query reads from a subquery that has no row identity to hide.
        if self.is_table_backed_sql(filtered_sql_query):
            source_table = self.query_source_table(filtered_sql_query)
            if source_table:
                self.ensure_hide_column(source_table)
        cur = self._con.execute(
            "INSERT INTO __series_descriptors__ "
            "(axis_id, series_index, name, sql_query, roles, style_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                int(axis_id),
                int(series_index),
                str(name),
                filtered_sql_query,
                _dumps_json(roles or {}) if roles else None,
                _dumps_json(style or {}) if style else None,
            ),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def ensure_figure_grid_capacity(self, *, figure_id: int, needed_axes: int) -> None:
        """Ensure figure grid has enough cells for needed_axes subplots.

        Dynamically increases nrows if necessary (keeps ncols fixed).
        """
        assert self._con is not None

        fig = self._con.execute(
            "SELECT nrows, ncols FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        if fig is None:
            return

        nrows = max(1, int(fig["nrows"]))
        ncols = max(1, int(fig["ncols"]))
        cap = nrows * ncols

        if needed_axes <= cap:
            return

        # Increase nrows while keeping ncols fixed
        import math
        new_nrows = int(math.ceil(float(needed_axes) / float(ncols)))
        self._con.execute(
            "UPDATE __figure_descriptors__ SET nrows = ? WHERE id = ?",
            (int(new_nrows), int(figure_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    @descriptor_write_wrapper
    def swap_axis_indexes(
        self,
        *,
        figure_id: int,
        first_axis_id: int,
        second_axis_id: int,
    ) -> None:
        """Swap two axis_index values for axes in one figure.

        The temporary index avoids violating the UNIQUE(figure_id, axis_index)
        constraint while the two rows are exchanged.
        """
        assert self._con is not None
        rows = self.list_axes_for_figure(int(figure_id))
        by_id = {int(axis_id): int(axis_index) for axis_id, axis_index, _title in rows}
        first_id = int(first_axis_id)
        second_id = int(second_axis_id)
        if first_id not in by_id or second_id not in by_id:
            applogger.error("Cannot swap axis indexes: axis id not found.")
            return
        temporary_index = min(by_id.values()) - 1
        with self.transaction(immediate=True):
            self._con.execute(
                "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                (temporary_index, first_id),
            )
            self._con.execute(
                "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                (by_id[first_id], second_id),
            )
            self._con.execute(
                "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                (by_id[second_id], first_id),
            )

    @descriptor_write_wrapper
    def apply_axis_layout(
        self,
        *,
        figure_id: int,
        nrows: int,
        ncols: int,
        placements: list[tuple[int, int, dict[str, Any]]],
    ) -> None:
        """Rewrite a figure's grid size and every axis's position/options at
        once - what a layout preset applies in one click.

        *placements* is ``(axis_id, axis_index, option_overrides)`` for every
        axis of the figure, in any order. An override value of ``None``
        removes that key from the axis's options rather than storing null,
        the same convention ``set_axis_options`` callers already follow
        elsewhere; anything else in the axis's existing options - a colour,
        a title - is left alone.

        Every axis_index is rewritten together, not one at a time: the
        UNIQUE(figure_id, axis_index) constraint would reject a plan that
        gives one axis the index another axis is about to give up, in
        whichever order a loop happened to touch them. Two passes side-step
        it the same way :meth:`swap_axis_indexes` does for two axes - here
        for however many a whole layout touches.
        """
        assert self._con is not None
        if not placements:
            return

        with self.transaction(immediate=True):
            self._con.execute(
                "UPDATE __figure_descriptors__ SET nrows = ?, ncols = ? WHERE id = ?",
                (int(nrows), int(ncols), int(figure_id)),
            )

            for offset, (axis_id, _axis_index, _overrides) in enumerate(placements):
                self._con.execute(
                    "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                    (-(offset + 1), int(axis_id)),
                )

            for axis_id, axis_index, overrides in placements:
                row = self._con.execute(
                    "SELECT options_json FROM __axis_descriptors__ WHERE id = ?",
                    (int(axis_id),),
                ).fetchone()
                options = _loads_json(row["options_json"]) if row else {}
                for key, value in overrides.items():
                    if value is None:
                        options.pop(key, None)
                    else:
                        options[key] = value
                self._con.execute(
                    "UPDATE __axis_descriptors__ SET axis_index = ?, options_json = ? WHERE id = ?",
                    (int(axis_index), _dumps_json(options), int(axis_id)),
                )


    # =====================================================================
