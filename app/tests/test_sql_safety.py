"""todo N-05: SQL a person types can read, and nothing else."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils.data_sources import read_sqlite_query


@pytest.fixture
def repo(tmp_path) -> SqliteRepo:
    repo = SqliteRepo(db_path=tmp_path / "safe.dhub")
    repo.import_dataframe(pd.DataFrame({"a": [1, 2, 3]}), table_name="t", normalize_columns=False)
    return repo


def rows(repo: SqliteRepo) -> int:
    return int(repo._con.execute('SELECT COUNT(*) FROM "t"').fetchone()[0])


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (SELECT 1) DELETE FROM t",
        "SELECT 1; DELETE FROM t",
        "SELECT load_extension('x')",
    ],
)
def test_a_series_query_that_writes_is_refused(repo, sql) -> None:
    with pytest.raises(Exception):
        repo.series_frame(sql)
    assert rows(repo) == 3


def test_query_df_reads_but_cannot_write_through_a_select(repo) -> None:
    assert repo.query_df("SELECT SUM(a) AS s FROM t")["s"].iat[0] == 6
    with pytest.raises(Exception):
        repo.query_df("WITH x AS (SELECT 1) DELETE FROM t")
    assert rows(repo) == 3


def test_a_saved_query_must_only_read(repo) -> None:
    with pytest.raises(ValueError):
        repo.save_query("bad", "DELETE FROM t")
    assert repo.save_query("good", "SELECT a FROM t") > 0


def test_an_expression_cannot_smuggle_a_statement(repo) -> None:
    with pytest.raises(ValueError):
        repo.add_column_from_expression("t", "b", "1; DROP TABLE t")
    repo.add_column_from_expression("t", "c", "a * 2")
    assert repo.query_df("SELECT c FROM t")["c"].tolist() == [2, 4, 6]


def test_a_query_on_another_sqlite_file_is_read_only(tmp_path) -> None:
    other = tmp_path / "other.db"
    con = sqlite3.connect(other)
    con.execute("CREATE TABLE z (v)")
    con.execute("INSERT INTO z VALUES (1)")
    con.commit()
    con.close()
    assert read_sqlite_query(str(other), "SELECT v FROM z")["v"].tolist() == [1]
    with pytest.raises(ValueError):
        read_sqlite_query(str(other), "DELETE FROM z")
