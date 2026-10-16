"""Deleting a table takes its import link with it, and Undo brings both back."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def _repo(tmp_path: Path) -> SqliteRepo:
    repo = SqliteRepo(db_path=tmp_path / "d.dhub")
    repo.import_dataframe(pd.DataFrame({"x": [1.0, 2.0]}), table_name="t1", normalize_columns=False)
    repo.upsert_link(table_name="t1", source_path="/x.csv",
                     settings={"source": {}, "read": {}, "destination": {"table": "t1"}, "columns": {}})
    return repo


def test_a_deleted_table_leaves_no_link_and_the_database_check_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    try:
        repo.delete_table("t1")
        assert repo.query_df("SELECT * FROM __import_links__").empty
        assert repo.optimize_db().is_healthy
    finally:
        repo.close()


def test_undo_restores_the_table_and_its_link(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    try:
        repo.delete_table("t1")
        assert repo.undo_last() is not None
        assert "t1" in list(repo.list_user_tables()["Table"])
        assert list(repo.query_df("SELECT table_name FROM __import_links__")["table_name"]) == ["t1"]
        assert repo.optimize_db().is_healthy
    finally:
        repo.close()
