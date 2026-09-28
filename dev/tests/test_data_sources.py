"""Tests for saved queries as first-class data sources.

The contract under test is that a caller can build ``SELECT <cols> FROM
{source.from_clause()}`` and stop caring whether it got a physical table or a
saved query - and that a saved query is *executed*, never materialised, so
editing it changes every chart built on it.
"""
from __future__ import annotations

from collections.abc import Iterator

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> Iterator[SqliteRepo]:
    """A repo with one table and one saved query over it."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS measurements")
    repo.query_df("CREATE TABLE measurements (x REAL, y REAL, grp TEXT)")
    repo.query_df(
        "INSERT INTO measurements (x, y, grp) VALUES "
        "(1.0, 10.0, 'a'), (2.0, 20.0, 'a'), (3.0, 30.0, 'b'), (4.0, 40.0, 'b')"
    )
    repo.save_query("group_a", "SELECT x, y FROM measurements WHERE grp = 'a'")
    yield repo
    repo.close()


# ----------------------------------------------------------------------
# The FROM fragment
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------
def test_a_table_resolves_to_a_table_source(repo: SqliteRepo) -> None:
    source = repo.get_data_source("measurements")
    assert source is not None and not source.is_query


def test_a_saved_query_resolves_to_a_query_source(repo: SqliteRepo) -> None:
    source = repo.get_data_source("group_a")
    assert source is not None and source.is_query
    assert "measurements" in source.sql


# ----------------------------------------------------------------------
# Reading through a source
# ----------------------------------------------------------------------


def test_new_rows_appear_in_a_saved_query(repo: SqliteRepo) -> None:
    repo.query_df("INSERT INTO measurements (x, y, grp) VALUES (5.0, 50.0, 'a')")
    assert repo.data_source_row_count(repo.get_data_source("group_a")) == 3


# ----------------------------------------------------------------------
# Listing
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_a_syntax_error_is_reported_not_raised(repo: SqliteRepo) -> None:
    ok, message = repo.validate_query("SELECT FROM WHERE")
    assert not ok and message


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "DELETE FROM measurements",
        "UPDATE measurements SET x = 1",
        "DROP TABLE measurements",
    ],
)
def test_only_row_returning_statements_are_accepted(repo: SqliteRepo, sql: str) -> None:
    """A saved query is read on every render; a write would run every time."""
    ok, _message = repo.validate_query(sql)
    assert not ok


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_a_saved_query_survives_a_reopen(tmp_db_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
    repo.save_query("q1", "SELECT a FROM t")
    repo.close()

    reopened = SqliteRepo(db_path=tmp_db_path)
    saved = reopened.get_query("q1")
    assert saved is not None and saved.sql == "SELECT a FROM t"
    reopened.close()


# ----------------------------------------------------------------------
# A new, unsaved query
# ----------------------------------------------------------------------


