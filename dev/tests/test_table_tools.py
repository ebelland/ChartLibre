"""The table tools: each one does its job, and each one can be undone."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
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


def test_fill_missing_reads_the_codes_too(repo) -> None:
    """-999 as a number, NA and n/a as text whatever their case: filled like the empty cells, and left out of
    the mean."""
    repo.import_dataframe(pd.DataFrame({"v": [1.0, -999.0, 3.0, None, 5.0]}), table_name="codes")
    repo.import_dataframe(pd.DataFrame({"s": ["a", "na", "N/A", "", "b"]}), table_name="words")
    assert repo.fill_missing("codes", "v", "mean", codes=["-999"]) == 2
    assert column(repo, "v", table="codes") == [1.0, 3.0, 3.0, 3.0, 5.0]
    assert repo.fill_missing("words", "s", "constant", "?", codes=["NA", "n/a"]) == 3
    assert column(repo, "s", table="words") == ["a", "?", "?", "?", "b"]


def test_fill_missing_with_nothing_makes_the_codes_empty_cells(repo) -> None:
    repo.import_dataframe(pd.DataFrame({"v": [1.0, -999.0, None, 4.0], "s": ["x", "NA", " ", "y"]}),
                          table_name="raw")
    assert repo.fill_missing("raw", "v", "null", codes=["-999"]) == 1  # the NULL is one already
    assert column(repo, "v", table="raw") == [1.0, None, None, 4.0]
    assert repo.fill_missing("raw", "s", "null", codes=["na"]) == 2  # "NA" and the blank text
    assert column(repo, "s", table="raw") == ["x", None, None, "y"]
    repo.undo_last()
    assert column(repo, "s", table="raw") == ["x", "NA", " ", "y"]


def test_find_replace_inside_text_and_whole_cells(repo) -> None:
    assert repo.find_replace("t", "a", "A", columns=["name"]) == 1
    assert column(repo, "name") == ["b", "A", None, "c"]
    assert repo.find_replace("t", "5", "50", columns=["x"], whole_cell=True) == 1
    assert column(repo, "x")[3] == 50


def test_group_aggregate_writes_a_new_table(repo) -> None:
    repo.import_dataframe(pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]}), table_name="g")
    name = repo.group_aggregate("g", ["k"], "SUM", "v")
    assert repo.query_df(f'SELECT * FROM "{name}"').values.tolist() == [["a", 4], ["b", 2]]


def _con(repo: SqliteRepo):
    assert repo._con is not None
    return repo._con


def test_the_fill_dialog_offers_the_imports_codes_and_nothing(qapp) -> None:
    from app.dialogs.table_tools_dialogs import FillMissingDialog
    from app.utils.import_cleaning import DEFAULT_MISSING

    dialog = FillMissingDialog("x")
    try:
        assert dialog.missing_codes() == list(DEFAULT_MISSING)  # the import's, so a code means the same
        dialog.codes.setText("-999, n/a ,, NA")
        assert dialog.missing_codes() == ["-999", "n/a", "NA"]
        dialog.method.setCurrentIndex(dialog.method.findData("null"))
        assert not dialog.value.isEnabled()
    finally:
        dialog.close()
