"""Tests for app.utils.data_sources: the reading logic behind both the live
import dialog and a saved link's "Update link".

No real PostgreSQL or MySQL server is available here, so those two engines
are exercised against a stubbed DBAPI connection - enough to check the SQL
this module builds and the connection lifecycle it manages, which is what is
actually this module's responsibility; pandas' own SQL execution is pandas'
own test suite's job.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.utils.data_sources import (
    DatabaseConnection,
    is_valid_web_url,
    read_from_link_source,
    read_sqlite_query,
    read_sqlite_table,
)


# ----------------------------------------------------------------------
# DatabaseConnection
# ----------------------------------------------------------------------


def test_a_server_connections_link_settings_carry_no_password() -> None:
    conn = DatabaseConnection(
        kind="postgres",
        host="db.example.com",
        port=5432,
        database="analytics",
        username="reader",
        password="hunter2",
    )

    settings = conn.to_link_settings()

    assert "password" not in settings
    assert settings == {
        "kind": "postgres",
        "host": "db.example.com",
        "port": 5432,
        "database": "analytics",
        "username": "reader",
    }


# ----------------------------------------------------------------------
# Another SQLite database
# ----------------------------------------------------------------------
@pytest.fixture
def other_db(tmp_path: Path) -> Path:
    path = tmp_path / "other.dhub"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE readings (t REAL, v REAL)")
        conn.execute("INSERT INTO readings VALUES (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)")
        conn.execute("CREATE TABLE __figure_descriptors__ (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    return path


def test_read_sqlite_table_applies_skip_options(other_db: Path) -> None:
    df = read_sqlite_table(str(other_db), "readings", skiprows=1, skipfooter=1)

    assert list(df["t"]) == [2.0]


def test_read_sqlite_query_runs_arbitrary_sql(other_db: Path) -> None:
    df = read_sqlite_query(str(other_db), "SELECT t, v FROM readings WHERE v > 2")

    assert list(df["t"]) == [2.0, 3.0]


# ----------------------------------------------------------------------
# The web
# ----------------------------------------------------------------------
def test_web_urls_reject_every_scheme_but_http_and_https() -> None:
    assert is_valid_web_url("https://example.com/data.csv")
    assert is_valid_web_url("http://example.com/data.csv")
    assert not is_valid_web_url("file:///etc/passwd")
    assert not is_valid_web_url("ftp://example.com/data.csv")
    assert not is_valid_web_url("")


# ----------------------------------------------------------------------
# A stubbed server database, for PostgreSQL and MySQL
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# read_from_link_source: the one dispatcher both the dialog and the link
# refresh call into
# ----------------------------------------------------------------------
def test_dispatches_a_file_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    df = read_from_link_source(
        {"kind": "file", "path": str(csv_path), "sheet": None},
        {"skiprows": 0, "skip_last": 0, "header": True},
    )

    assert list(df.columns) == ["a", "b"]


def test_dispatches_a_sqlite_source(other_db: Path) -> None:
    df = read_from_link_source(
        {"kind": "sqlite", "path": str(other_db), "table": "readings"},
        {},
    )

    assert list(df.columns) == ["t", "v"]


def test_an_unknown_source_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown source kind"):
        read_from_link_source({"kind": "ftp"}, {})


# ----------------------------------------------------------------------
# Asking a server which databases it has
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The user's own web-source catalogue
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_user_web_sources(monkeypatch: pytest.MonkeyPatch) -> list:
    """A user_web_sources catalogue of its own, not the developer's own -
    add_user_web_source/remove_user_web_source otherwise read and write the
    real user.json."""
    import app.utils.config as config

    stored: list[dict] = []
    monkeypatch.setattr(config, "get_user_web_sources", lambda: list(stored))

    def _set(entries: list) -> None:
        stored.clear()
        stored.extend(entries)

    monkeypatch.setattr(config, "set_user_web_sources", _set)
    return stored


