"""Undo, as table snapshots in a database of its own.

The alternative was a copy of the whole .dhub before every change, and it
is worse twice over: it costs the size of the project to take back a
change to one table, and it reverts everything else that happened since,
which is not what undo means. What is copied here is the table an action
is about to change - either kind, the imported data or the chart settings
in the ``__…__`` descriptor tables - into ``<project>.undo.db``.

The properties worth pinning are the ones that are silently wrong if the
implementation drifts: that a restored table has its *own* schema back and
not an approximation of it, that a table the action created is dropped
rather than left behind, and that one entry covers every table one action
touched.
"""
from __future__ import annotations

from collections.abc import Iterator

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data.undo_store import MAX_UNDO_ENTRIES, UndoStore


@pytest.fixture
def connection(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A project with one data table and one descriptor table."""
    con = sqlite3.connect(str(tmp_path / "project.dhub"), isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, v REAL NOT NULL)")
    con.execute("CREATE INDEX idx_data_v ON data(v)")
    con.executemany("INSERT INTO data (id, v) VALUES (?, ?)", [(1, 1.5), (2, 2.5)])
    con.execute(
        "CREATE TABLE __series_descriptors__ ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
    )
    con.execute("INSERT INTO __series_descriptors__ (name) VALUES ('a')")
    yield con
    con.close()


@pytest.fixture
def store(tmp_path: Path) -> UndoStore:
    return UndoStore(tmp_path / "project.dhub")


def _tables(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


# ----------------------------------------------------------------------
# Taking a change back
# ----------------------------------------------------------------------
def test_a_dropped_table_comes_back_with_its_rows(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Delete table 'data'")
    connection.execute("DROP TABLE data")

    store.undo(connection)

    assert connection.execute("SELECT * FROM data ORDER BY id").fetchall() == [
        (1, 1.5),
        (2, 2.5),
    ]


def test_a_deleted_column_comes_back(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Delete column 'v'")
    # SQLite refuses to drop a column an index covers, so a caller has to
    # remove the index first - which is another reason the snapshot keeps
    # the index DDL and puts it back.
    connection.execute("DROP INDEX idx_data_v")
    connection.execute("ALTER TABLE data DROP COLUMN v")
    assert "v" not in [row[1] for row in connection.execute("PRAGMA table_info(data)")]

    store.undo(connection)

    assert [row[1] for row in connection.execute("PRAGMA table_info(data)")] == [
        "id",
        "v",
    ]


def test_a_table_the_action_created_is_dropped_again(connection, store) -> None:
    """Undoing an operation that *wrote* a result means removing it. Without
    this the commonest change of all would silently survive its own undo."""
    store.snapshot(connection, ["result"], label="Apply operation")
    connection.execute("CREATE TABLE result (x REAL)")
    connection.execute("INSERT INTO result VALUES (1.0)")

    store.undo(connection)

    assert "result" not in _tables(connection)


# ----------------------------------------------------------------------
# The stack
# ----------------------------------------------------------------------
def test_the_most_recent_change_is_undone_first(connection, store) -> None:
    store.snapshot(connection, ["data"], label="First")
    connection.execute("DELETE FROM data WHERE id = 1")
    store.snapshot(connection, ["data"], label="Second")
    connection.execute("DELETE FROM data WHERE id = 2")

    assert [entry.label for entry in store.entries(connection)] == ["Second", "First"]

    store.undo(connection)
    assert connection.execute("SELECT count(*) FROM data").fetchone()[0] == 1

    store.undo(connection)
    assert connection.execute("SELECT count(*) FROM data").fetchone()[0] == 2


def test_the_stack_is_capped(connection, store) -> None:
    for index in range(MAX_UNDO_ENTRIES + 3):
        store.snapshot(connection, ["data"], label=f"Change {index}")

    entries = store.entries(connection)
    assert len(entries) == MAX_UNDO_ENTRIES
    assert entries[0].label == f"Change {MAX_UNDO_ENTRIES + 2}"


# ----------------------------------------------------------------------
# Not costing the user their action
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Through the repository, where the call sites are
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Descriptor edits and applied operations (todo.txt P2-11)
# ----------------------------------------------------------------------


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    """A window on a figure with one axis and one series."""
    import numpy as np

    from app.main_window.main_window import MainWindow
    from app.logs.logger import applogger

    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    series_id = int(
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=0, name="s",
            sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
        )
    )
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    built._properties_figure_id = figure_id
    yield built, axis_id, series_id
    built.close()
    applogger.set_status_bar(None)
    repo.undo_store.discard_file()


def test_a_deleted_series_comes_back(window) -> None:
    built, axis_id, series_id = window
    built._on_series_delete_requested(series_id)
    assert len(built._repo.get_series(axis_id)) == 0

    built._on_undo()

    assert len(built._repo.get_series(axis_id)) == 1


def test_a_deleted_figure_comes_back(window) -> None:
    built, axis_id, _series_id = window
    panel = built._current_chart_panel()
    figure_id = int(panel.figure_id)

    panel.close()
    assert built._repo.load_figure_descriptor(figure_id) is None

    built._on_undo()

    assert built._repo.load_figure_descriptor(figure_id) is not None
    assert len(built._repo.get_series(axis_id)) == 1


def test_an_axis_edit_comes_back(window) -> None:
    built, axis_id, _series_id = window
    built._on_axis_options_requested({"axis_id": axis_id, "x_scale": "log"})
    assert (built._repo.get_axis_options(axis_id) or {})["x_scale"] == "log"

    built._on_undo()

    assert (built._repo.get_axis_options(axis_id) or {}).get("x_scale") != "log"


