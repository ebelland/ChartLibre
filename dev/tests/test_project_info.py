"""A project's record of itself: author, creation date, notes (__project_info__)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.data.repo.project_info import PROJECT_INFO_TABLE
from app.data.sqlite_repo import SqliteRepo


def test_a_new_project_is_dated_and_keeps_what_it_is_told(tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "new.dhub")
    info = repo.project_info()
    assert set(info) == {"created"}
    assert abs((datetime.fromisoformat(info["created"]) - datetime.now().astimezone()).total_seconds()) < 60

    repo.set_project_info({"author": "  Ada Lovelace ", "notes": "Line one\nline two"})
    assert repo.project_info()["author"] == "Ada Lovelace"
    repo.set_project_info({"notes": ""})  # emptied: gone
    assert "notes" not in repo.project_info() and repo.project_info()["created"] == info["created"]
    repo.close()

    reopened = SqliteRepo(db_path=tmp_path / "new.dhub")
    assert reopened.project_info()["author"] == "Ada Lovelace"
    reopened.close()


def test_an_older_project_gets_the_table_on_opening_and_hides_it(tmp_path: Path) -> None:
    path = tmp_path / "old.dhub"
    repo = SqliteRepo(db_path=path)
    repo.import_dataframe(pd.DataFrame({"a": [1, 2]}), table_name="data", normalize_columns=False)
    repo.close()
    with sqlite3.connect(path) as con:  # as a project made before the table existed
        con.execute(f"DROP TABLE {PROJECT_INFO_TABLE}")

    repo = SqliteRepo(db_path=path)
    assert "created" in repo.project_info()
    assert repo.list_table_names() == ["data"]
    assert list(repo.list_user_tables()["Table"]) == ["data"]
    repo.close()


def test_the_window_saves_the_entries_and_remembers_the_author(qapp, tmp_path: Path) -> None:
    from app.dialogs import project_info_dialog
    from app.utils.config import get_value

    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    dialog = project_info_dialog.ProjectInfoDialog(repo)
    assert dialog.created_label.text() != ""
    dialog.author_edit.setText("Grace Hopper")
    dialog.notes_edit.setPlainText("Calibration runs, March")
    dialog.accept()
    assert repo.project_info()["notes"] == "Calibration runs, March"
    assert get_value(project_info_dialog.AUTHOR_SETTING) == "Grace Hopper"
    assert project_info_dialog.default_author() == "Grace Hopper"
    repo.close()


def test_the_report_names_the_author_and_carries_the_notes(qapp, tmp_path: Path) -> None:
    from app.utils.project_report import build_project_report

    repo = SqliteRepo(db_path=tmp_path / "r.dhub")
    repo.set_project_info({"author": "Rosalind Franklin", "notes": "Photo 51 & friends"})
    report = build_project_report(repo, include_tables=False, include_history=False)
    assert "Rosalind Franklin" in report.html and "Photo 51 &amp; friends" in report.html
    repo.close()
