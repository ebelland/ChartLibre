"""todo N-05: the SQL guard - where a person writes SQL, and when a project opens.

It never runs while charts are drawn; user.json "sql_guard": false turns
it off.
"""
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


def guard(monkeypatch, on: bool) -> None:
    import app.utils.config as config

    real = config.get_value
    monkeypatch.setattr(config, "get_value", lambda name, default=None: on if name == "sql_guard" else real(name, default))


def test_a_saved_query_that_writes_is_refused(repo) -> None:
    with pytest.raises(ValueError):
        repo.save_query("bad", "WITH x AS (SELECT 1) DELETE FROM t")
    assert repo.save_query("good", "SELECT a FROM t") > 0


def test_opening_a_project_blocks_sql_that_would_write(repo) -> None:
    repo._con.execute(
        "CREATE TABLE IF NOT EXISTS __queries__ (id INTEGER PRIMARY KEY, name TEXT UNIQUE, sql TEXT, settings_json TEXT)"
    )
    repo._con.execute("INSERT INTO __queries__ (name, sql) VALUES ('evil', 'DELETE FROM t')")
    found = repo.scan_user_sql()
    assert found and "evil" in found[0]
    with pytest.raises(ValueError):
        repo.query_df("DELETE FROM t")
    assert rows(repo) == 3


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


def test_the_guard_can_be_switched_off(repo, monkeypatch) -> None:
    guard(monkeypatch, False)
    repo.save_query("allowed", "SELECT a FROM t WHERE a > 1")
    repo._con.execute("UPDATE __queries__ SET sql = 'DELETE FROM t' WHERE name = 'allowed'")
    assert repo.scan_user_sql() == []
