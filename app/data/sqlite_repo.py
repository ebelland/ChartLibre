"""SQLite repository for ChartLibre: the connection, and everything on it.

``SqliteRepo`` is the one class the application talks to - every caller
writes ``from app.data.sqlite_repo import SqliteRepo``. It is one class in
one module, in sections:

* the dataclass and its fields, including the connection itself;
* connecting, the pragmas, and closing;
* the **series DataFrame cache**, the single biggest lever on render time -
  without it every property tweak re-runs every series query against
  SQLite (see ``series_frame`` for the invalidation contract);
* preview savepoints and the transaction helper every writer uses;
* the undo hooks, which need the connection to attach their own database;
* then, by subject: user tables, flags, editing, table tools, descriptors,
  saved queries, maintenance, the operations record and the project info.

Errors, reports and SQL guards shared with the rest of the application are
in ``app/data/repo/_common.py``; the SQL functions registered on each
connection in ``app/data/repo/function_sql.py``.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

import app.data.descriptors
from app import APP_NAME
from app.data.data_source import DataSource, quote_identifier, resolve_role_column

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
from app.data.select_sql import (
    has_projection_alias,
    split_top_level_select_items,
    sql_insert_select_expression,
    top_level_from,
    top_level_match,
)
from app.data.series_frame import SeriesFrame
from app.data.undo_store import UndoEntry, UndoStore
from app.logs.logger import applogger
from app.utils.coercion import to_numbers
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
class SqliteRepo:
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
        keep = f"(__dhub_rn__ - 1) % {stride} = 0"
        if has_projection_alias(sql_text, "color"):
            # A row given a colour of its own - an outlier, a cluster - is
            # what the chart is there to show: never thinned out.
            keep += " OR (\"color\" IS NOT NULL AND \"color\" <> '')"
        wrapped = (
            "SELECT * FROM ("
            f"SELECT *, ROW_NUMBER() OVER () AS __dhub_rn__ FROM ({sql_text})"
            f") WHERE {keep}"
        )
        return self.series_frame(wrapped).drop_column("__dhub_rn__")


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
    def transaction(self, *, immediate: bool = False):
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
                "Table": [query.name for query in queries] + [row["function_name"] for row in stored],
                "has_link": [False] * (len(queries) + len(stored)),
                "Notes": [None] * len(queries) + [row["description"] or None for row in stored],
                "source_path": [query.sql for query in queries] + [row["code_string"] for row in stored],
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
                return DataSource.function(stored["function_name"], source_sql(stored))
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
            ROW_ID: pd.to_numeric(frame[ROW_ID], errors="coerce"),
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

    @ensure_connection_wrapper
    def set_integer_by_row_id(self, table_name: str, col_name: str, values: Mapping[int, int]) -> None:
        """*col_name* (an INTEGER column, made if missing, 0 elsewhere) set row by row: {_id: value}."""
        assert self._con is not None
        self.clear_integer_column(table_name, col_name)
        self._con.executemany(
            f"UPDATE {_quote_ident(table_name)} SET {_quote_ident(col_name)} = ? WHERE {_quote_ident(ROW_ID)} = ?",
            [(int(value), int(row)) for row, value in values.items()],
        )
        self._commit()

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

    # =====================================================================
