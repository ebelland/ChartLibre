"""The operations applied to a project, kept in it (todo R-03).

A result table says what was computed, not how. Each Apply of a series
operation adds a row to ``__operations__``: which operation, every entry of
its dialog, the series it read and the tables it wrote, when, and with
which version of ChartLibre - the record JMP keeps as a script beside each
report and SPSS as syntax. It is what a history of a result, a
"recalculate when the data change" and an exported script are built on.

Written inside the same transaction as the results, and snapshotted with
them for undo: an operation undone leaves no record behind.

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.data.repo._common import RepoHost, ensure_connection_wrapper

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


class OperationsMixin(RepoHost):
    """The ``__operations__`` record."""

    __slots__ = ()

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
