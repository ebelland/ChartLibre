"""SQL workpad: the guard, the authorizer, the copy, Apply and Undo."""
import os

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data import sql_workpad as sql_engine
from app.data.sql_workpad import RefusedStatement, SqlSession


@pytest.fixture
def session(tmp_path, monkeypatch):
    repo = SqliteRepo(db_path=tmp_path / "t.dhub")
    repo._connect()
    repo._con.execute("CREATE TABLE T (a INTEGER, b TEXT)")
    repo._con.executemany("INSERT INTO T VALUES (?, ?)", [(1, "x;y"), (2, "z")])
    repo._con.execute("CREATE TABLE __own__ (k)")
    repo._con.commit()
    guard = {"on": True}
    monkeypatch.setattr(sql_engine, "sql_guard_enabled", lambda: guard["on"])
    pad = SqlSession(lambda: repo)
    yield pad, repo, guard
    pad.cancel()
    repo.close()


def count(repo):
    return repo._con.execute("SELECT COUNT(*) FROM T").fetchone()[0]


def test_reading_and_the_text_guard(session):
    pad, _repo, _guard = session
    columns, rows, _ = pad.run("SELECT a, b FROM T WHERE b = 'x;y';")
    assert columns == ["a", "b"] and rows == [(1, "x;y")]
    for sql in ("SELECT 1; DELETE FROM T", "DELETE FROM T", "ATTACH 'x' AS y", "PRAGMA writable_schema = 1",
                "SELECT 'open"):
        with pytest.raises(RefusedStatement):
            pad.run(sql)
    assert not pad.on_copy


def test_writing_happens_on_a_copy(session):
    pad, repo, guard = session
    guard["on"] = False
    pad.run("DELETE FROM T WHERE a = 1")
    assert pad.on_copy and count(repo) == 2
    assert pad.run("SELECT COUNT(*) FROM T")[1] == [(1,)]
    path = pad._copy_path
    pad.cancel()
    assert not os.path.exists(path) and count(repo) == 2


def test_the_authorizer_keeps_the_own_tables(session):
    pad, _repo, guard = session
    guard["on"] = False
    for sql in ("DELETE FROM __own__", "DROP TABLE __own__", "UPDATE sqlite_master SET sql = ''",
                "CREATE TRIGGER t AFTER INSERT ON T BEGIN SELECT 1; END"):
        with pytest.raises((RefusedStatement, Exception)):
            pad.run(sql)
    assert pad.run("SELECT COUNT(*) FROM __own__")[1] == [(0,)]


def test_apply_and_undo(session):
    pad, repo, guard = session
    guard["on"] = False
    pad.run("DELETE FROM T WHERE a = 1")
    pad.run("CREATE TABLE N AS SELECT a * 10 AS c FROM T")
    assert pad.apply() and not pad.on_copy
    assert count(repo) == 1 and repo._con.execute("SELECT c FROM N").fetchone()[0] == 20
    assert repo.undo_last() is not None
    assert count(repo) == 2
    assert not repo._con.execute("SELECT 1 FROM sqlite_master WHERE name='N'").fetchone()


def test_panel_copy_and_save_query(session, qapp, monkeypatch):
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QInputDialog

    from app.widgets.sql_workpad_panel import SqlWorkpadPanel

    _pad, repo, _guard = session
    panel = SqlWorkpadPanel(None, repo=lambda: repo)
    panel.run("SELECT a FROM T ORDER BY a")
    panel.copy_rows()
    assert QGuiApplication.clipboard().text() == "a\n1\n2"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Qa", True))
    panel.save_as_query()
    assert any(q.name == "Qa" for q in repo.list_queries())
    panel._session.cancel()
