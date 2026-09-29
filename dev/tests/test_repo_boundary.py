"""SQL stays in app/data (todo R-11): widgets ask the repository.

The project's rule is that only app/data talks to the database. Two things
are checked: the repository methods the table views and the outlier dialog
now use instead of reaching for the connection, and - so the rule stays
kept - that no other source file touches the repository's connection.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo

APP_DIR = Path(__file__).resolve().parents[2] / "app"

#: Outside app/data, only these talk to a database of their own: they read
#: the servers and files a user imports *from*, never the project.
_OWN_CONNECTIONS = {"utils/data_sources.py"}


def test_no_widget_reaches_for_the_repositorys_connection() -> None:
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        relative = path.relative_to(APP_DIR).as_posix()
        if relative.startswith("data/") or relative in _OWN_CONNECTIONS:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\b_con\b|\.connect\(\)\s+as\b|import sqlite3", code):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert offenders == []


@pytest.fixture
def numbers(repo: SqliteRepo) -> SqliteRepo:
    repo.import_dataframe(
        pd.DataFrame({"a": range(1, 8), "b": [x * 10 for x in range(1, 8)]}),
        table_name="numbers", normalize_columns=False,
    )
    return repo


def test_rows_are_read_in_blocks_after_the_last_rowid(numbers: SqliteRepo) -> None:
    first = numbers.read_rows_after_rowid("numbers", 0, 3)
    assert [row[0] for row in first] == [1, 2, 3]
    assert first[0] == (1, 1, 10)  # rowid, then the columns

    second = numbers.read_rows_after_rowid("numbers", first[-1][0], 3)
    assert [row[0] for row in second] == [4, 5, 6]
    assert numbers.read_rows_after_rowid("numbers", 7, 3) == []


def test_hidden_rowids_lists_only_the_hidden_rows(numbers: SqliteRepo) -> None:
    assert numbers.hidden_rowids("numbers") == []  # adds the Hide column, hides nothing
    numbers.mark_hide_rowids(table_name="numbers", rowids=[2, 5], clear_existing=True)
    assert numbers.hidden_rowids("numbers") == [2, 5]


def test_table_frame_returns_every_row(numbers: SqliteRepo) -> None:
    frame = numbers.table_frame("numbers")
    assert list(frame["a"]) == list(range(1, 8))


def test_is_open_follows_the_connection(numbers: SqliteRepo) -> None:
    assert numbers.is_open
    numbers.close()
    assert not numbers.is_open
