"""import_dataframe writes plain frames directly and leaves the rest to pandas.

DataFrame.to_sql was ten times slower than SQLite on a file, and every Preview
of a data operation writes its result table. The direct write has to leave
the same table behind - same columns, SQL types and values - as to_sql did.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> Iterator[SqliteRepo]:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def _table(repo: SqliteRepo, name: str) -> tuple[list[tuple[str, str]], list[tuple]]:
    con = repo._con
    schema = [(row[1], row[2]) for row in con.execute(f'PRAGMA table_info("{name}")')]
    return schema, [tuple(row) for row in con.execute(f'SELECT * FROM "{name}" ORDER BY rowid')]


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [0.0, 1.5, np.nan, 3.0],
        "n": [1, 2, 3, 4],
        "flag": [True, False, True, True],
        "label": ["a", None, "c", "d"],
        "mixed": [1, "two", 3.5, None],
    })


def test_a_plain_frame_gives_the_table_to_sql_gave(repo: SqliteRepo) -> None:
    frame = _frame()
    repo.import_dataframe(frame, table_name="direct", normalize_columns=False)
    repo.import_dataframe(frame, table_name="pandas", normalize_columns=False, dtype_overrides={})  # {} is the to_sql path
    assert _table(repo, "direct") == _table(repo, "pandas")
    schema, rows = _table(repo, "direct")
    assert dict(schema) == {"x": "REAL", "n": "INTEGER", "flag": "INTEGER", "label": "TEXT", "mixed": "TEXT"}
    assert rows[2][0] is None and rows[1][3] is None  # NaN and None are NULL


def test_dates_go_through_pandas(repo: SqliteRepo) -> None:
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-02-01"]), "v": [1.0, 2.0]})
    repo.import_dataframe(frame, table_name="dates", normalize_columns=False)
    schema, rows = _table(repo, "dates")
    assert dict(schema)["when"] == "TIMESTAMP" and len(rows) == 2


def test_an_empty_frame_and_a_replaced_table(repo: SqliteRepo) -> None:
    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0]}), table_name="t", normalize_columns=False)
    repo.import_dataframe(pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)}),
                          table_name="t", normalize_columns=False)
    schema, rows = _table(repo, "t")
    assert [name for name, _kind in schema] == ["a", "b"] and rows == []


def test_values_that_are_not_plain_fall_back_and_still_import(repo: SqliteRepo) -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [b"xy", b"z"]})
    assert repo.import_dataframe(frame, table_name="odd", normalize_columns=False) == 2
    assert len(_table(repo, "odd")[1]) == 2


def test_the_direct_write_is_much_faster_than_to_sql(repo: SqliteRepo) -> None:
    n = 20_000
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({name: rng.random(n) for name in ("x", "y", "fit", "residual", "ci_low", "ci_high")})
    frame["source"] = "y0"
    start = time.perf_counter()
    repo.import_dataframe(frame, table_name="fast", normalize_columns=False)
    direct = time.perf_counter() - start
    start = time.perf_counter()
    repo.import_dataframe(frame, table_name="slow", normalize_columns=False, dtype_overrides={})
    pandas_way = time.perf_counter() - start
    assert direct < pandas_way / 2, (direct, pandas_way)
