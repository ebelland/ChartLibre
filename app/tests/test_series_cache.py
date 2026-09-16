"""Tests for the series DataFrame cache in SqliteRepo.

The cache is the one optimisation that can return *wrong* data if invalidation
misses a write, so every write shape gets its own test: same-connection DML,
DDL, and a commit from a second connection.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo

SQL = "SELECT x, y FROM t_cache ORDER BY x"


def _make_repo(tmp_db_path: Path) -> SqliteRepo:
    """Return a repo with a small populated table to query."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS t_cache")
    repo.query_df("CREATE TABLE t_cache (x INTEGER, y REAL)")
    repo.query_df("INSERT INTO t_cache (x, y) VALUES (1, 1.0), (2, 2.0)")
    return repo


def test_second_read_is_served_from_cache(tmp_db_path: Path) -> None:
    repo = _make_repo(tmp_db_path)

    first = repo.series_df(SQL)
    hits_before = repo.series_cache_stats["hits"]
    second = repo.series_df(SQL)

    pd.testing.assert_frame_equal(first, second)
    assert repo.series_cache_stats["hits"] == hits_before + 1
    repo.close()


def test_returned_frame_is_a_shallow_copy(tmp_db_path: Path) -> None:
    """Adding a column to a returned frame must not corrupt the cache."""
    repo = _make_repo(tmp_db_path)

    first = repo.series_df(SQL)
    first["injected"] = 0

    second = repo.series_df(SQL)
    assert "injected" not in second.columns
    repo.close()


def test_insert_on_same_connection_invalidates(tmp_db_path: Path) -> None:
    """PRAGMA data_version does not move for same-connection writes."""
    repo = _make_repo(tmp_db_path)

    assert len(repo.series_df(SQL)) == 2
    repo.query_df("INSERT INTO t_cache (x, y) VALUES (3, 3.0)")
    assert len(repo.series_df(SQL)) == 3
    repo.close()


def test_ddl_invalidates(tmp_db_path: Path) -> None:
    """Pure DDL changes no rows, so only schema_version catches it."""
    repo = _make_repo(tmp_db_path)

    assert list(repo.series_df("SELECT * FROM t_cache").columns) == ["x", "y"]
    repo.query_df("ALTER TABLE t_cache ADD COLUMN z REAL")
    assert list(repo.series_df("SELECT * FROM t_cache").columns) == ["x", "y", "z"]
    repo.close()


def test_external_connection_write_invalidates(tmp_db_path: Path) -> None:
    """A commit from another connection must be picked up."""
    repo = _make_repo(tmp_db_path)
    assert len(repo.series_df(SQL)) == 2

    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        con.execute("INSERT INTO t_cache (x, y) VALUES (4, 4.0)")
        con.commit()

    assert len(repo.series_df(SQL)) == 3
    repo.close()


def test_cache_is_bounded(tmp_db_path: Path) -> None:
    """The LRU must never grow past max_entries."""
    repo = _make_repo(tmp_db_path)
    repo._series_cache_max_entries = 3

    for i in range(10):
        repo.series_df(f"SELECT x + {i} AS x FROM t_cache")

    assert repo.series_cache_stats["entries"] == 3
    repo.close()


def test_cache_can_be_disabled(tmp_db_path: Path) -> None:
    """With the cache off, every read is a straight SQL round trip."""
    repo = _make_repo(tmp_db_path)
    repo._series_cache_enabled = False

    repo.series_df(SQL)
    repo.series_df(SQL)

    assert repo.series_cache_stats["hits"] == 0
    assert repo.series_cache_stats["entries"] == 0
    repo.close()


def _make_axis(repo: SqliteRepo) -> int:
    """A figure/axis pair to edit descriptor options on."""
    figure_id = repo.create_figure_descriptor(name="f")
    return int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="t",
            x_label="x",
            y_label="y",
            options={},
        )
    )


def test_editing_axis_options_does_not_evict_other_cached_series(
    tmp_db_path: Path,
) -> None:
    """The bug this guards: an axis/figure/series options edit used to clear
    the *whole* series cache, because it goes through the same connection
    that ``_database_stamp`` reads ``total_changes`` from - so every other
    axis's cached frame paid for an edit that never touched its data."""
    repo = _make_repo(tmp_db_path)
    axis_id = _make_axis(repo)

    repo.series_df(SQL)
    misses_before = repo.series_cache_stats["misses"]

    repo.set_axis_options(axis_id, {"grid": True})

    repo.series_df(SQL)
    assert repo.series_cache_stats["misses"] == misses_before
    repo.close()


def test_a_real_write_still_invalidates_after_descriptor_writes(
    tmp_db_path: Path,
) -> None:
    """Excluding descriptor writes from the stamp must not swallow a real
    one: it has to still catch a write to the data table itself."""
    repo = _make_repo(tmp_db_path)
    axis_id = _make_axis(repo)
    repo.set_axis_options(axis_id, {"grid": True})
    repo.set_axis_options(axis_id, {"grid": False})

    assert len(repo.series_df(SQL)) == 2
    repo.query_df("INSERT INTO t_cache (x, y) VALUES (5, 5.0)")
    assert len(repo.series_df(SQL)) == 3
    repo.close()


def test_nested_descriptor_writes_do_not_double_count(tmp_db_path: Path) -> None:
    """delete_figure cascades into delete_axis, which cascades into
    delete_series - all three wrapped. If each nested call folded its own
    slice of total_changes into the counter instead of only the outermost
    call doing so once, the counter would double-count the cascaded rows and
    could out-grow total_changes itself - at which point _database_stamp's
    "changes" figure could repeat a value already sitting in the cache, and
    a fresh read would be served as if it were the same old one."""
    repo = _make_repo(tmp_db_path)
    axis_id = _make_axis(repo)
    figure_id = int(
        repo.query_df(
            "SELECT figure_id FROM __axis_descriptors__ WHERE id = ?", (axis_id,)
        )["figure_id"].iloc[0]
    )
    for index in range(3):
        repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=index,
            name=f"s{index}",
            sql_query=SQL,
            roles={},
            style={},
        )

    assert repo._con is not None
    repo.delete_figure(figure_id)

    assert repo._series_cache_metadata_changes <= repo._con.total_changes
    repo.close()
