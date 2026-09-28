"""The table tools: each one does its job, and each one can be undone."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture

def _con(repo: SqliteRepo):
    assert repo._con is not None
    return repo._con

def repo(tmp_path) -> SqliteRepo:
    repo = SqliteRepo(db_path=tmp_path / "tools.dhub")
    repo.import_dataframe(
        pd.DataFrame({"name": ["b", "a", None, "c"], "x": [2.0, None, 3.0, 5.0], "code": ["1", "2", "x", "4"]}),
        table_name="t",
    )
    return repo


def column(repo: SqliteRepo, name: str, table: str = "t") -> list:
    return [row[0] for row in _con(repo).execute(f'SELECT "{name}" FROM "{table}" ORDER BY rowid')]


def test_duplicate_copies_rows_and_undo_drops_the_copy(repo) -> None:
    copy = repo.duplicate_table("t")
    assert copy == "t_copy"
    assert column(repo, "x", copy) == column(repo, "x")
    assert repo.duplicate_table("t") == "t_copy_2"
    repo.undo_last()
    repo.undo_last()
    assert "t_copy" not in repo.list_table_names()


def test_sort_puts_empty_cells_last_and_can_descend(repo) -> None:
    repo.sort_table("t", "x")
    assert column(repo, "x") == [2.0, 3.0, 5.0, None]
    repo.sort_table("t", "x", descending=True)
    assert column(repo, "x") == [5.0, 3.0, 2.0, None]


def test_cast_converts_and_counts_what_would_not(repo) -> None:
    failed = repo.cast_column("t", "code", "INTEGER")
    assert failed == 1
    assert column(repo, "code") == [1, 2, None, 4]


def test_stats_describe_a_numeric_column(repo) -> None:
    stats = repo.column_stats("t", "x")
    assert stats["rows"] == 4 and stats["empty"] == 1
    assert stats["min"] == 2.0 and stats["max"] == 5.0 and stats["median"] == 3.0


@pytest.mark.parametrize(
    ("method", "expected"),
    [("mean", 10.0 / 3), ("median", 3.0), ("previous", 2.0), ("linear", 2.5)],
)
def test_fill_missing(repo, method, expected) -> None:
    assert repo.fill_missing("t", "x", method) == 1
    assert column(repo, "x")[1] == pytest.approx(expected)


def test_find_replace_inside_text_and_whole_cells(repo) -> None:
    assert repo.find_replace("t", "a", "A", columns=["name"]) == 1
    assert column(repo, "name") == ["b", "A", None, "c"]
    assert repo.find_replace("t", "5", "50", columns=["x"], whole_cell=True) == 1
    assert column(repo, "x")[3] == 50


def test_group_aggregate_writes_a_new_table(repo) -> None:
    repo.import_dataframe(pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]}), table_name="g")
    name = repo.group_aggregate("g", ["k"], "SUM", "v")
    assert repo.query_df(f'SELECT * FROM "{name}"').values.tolist() == [["a", 4], ["b", 2]]
