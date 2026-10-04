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


def test_references_round_trip_and_a_bare_string_is_a_citation() -> None:
    from app.data.repo.project_info import dump_references, parse_references

    entries = [{"citation": "Anscombe 1973", "doi": "10.1080/00031305.1973.10478966"}, {"citation": "  "}]
    text = dump_references(entries)
    assert parse_references(text) == [entries[0]]
    assert parse_references('["Tukey 1977"]') == [{"citation": "Tukey 1977"}]
    assert parse_references("not json") == [] and dump_references([]) is None


def test_saving_draws_the_first_figure_into_the_preview(qapp, tmp_path: Path) -> None:
    from PIL import Image

    from app.utils.project_preview import preview_file_for, resolve_preview, write_project_preview

    repo = SqliteRepo(db_path=tmp_path / "My study.dhub")
    assert write_project_preview(repo) is None and "preview_path" not in repo.project_info()
    repo.import_dataframe(pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0]}), table_name="t",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="first"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="first", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query="SELECT x, y FROM t",
                                  roles={"x": "x", "y": "y"}, style={})
    written = write_project_preview(repo)
    assert written == preview_file_for(tmp_path / "My study.dhub") == tmp_path / "My study.preview.png"
    assert repo.project_info()["preview_path"] == "My study.preview.png"
    assert resolve_preview(tmp_path / "My study.dhub", "My study.preview.png") == written
    with Image.open(written) as picture:
        assert picture.size == (1280, 800)
    repo.close()


def test_a_figure_keeps_notes_and_its_dates_follow_its_changes(tmp_path: Path) -> None:
    import time

    repo = SqliteRepo(db_path=tmp_path / "f.dhub")
    figure_id = int(repo.create_figure_descriptor(name="f"))
    made = repo.get_figure_descriptor(figure_id)
    assert made is not None and made.created and made.created == made.last_modified and made.note == ""
    time.sleep(1.1)
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="", x_label="", y_label="", options={}))
    after_axis = repo.get_figure_descriptor(figure_id)
    assert after_axis is not None and after_axis.last_modified > made.last_modified
    time.sleep(1.1)
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query="SELECT 1 AS x, 2 AS y",
                                  roles={"x": "x", "y": "y"}, style={})
    after_series = repo.get_figure_descriptor(figure_id)
    assert after_series is not None and after_series.last_modified > after_axis.last_modified
    repo.set_figure_note(figure_id, "  Calibrated on 3 March  ")
    noted = repo.get_figure_descriptor(figure_id)
    assert noted is not None and noted.note == "Calibrated on 3 March" and noted.created == made.created
    repo.close()


def test_selected_is_kept_like_hide(tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "s.dhub")
    repo.import_dataframe(pd.DataFrame({"v": [1, 5, 9, None]}), table_name="t", normalize_columns=False)
    assert repo.flag_rows_by_value("t", "Selected", "v", ">", 4) == 2
    assert repo.flag_rows_special("t", "Selected", "v", "null_or_empty") == 1
    flags = list(repo.query_df('SELECT COALESCE("Selected", 0) AS s FROM t ORDER BY rowid')["s"])
    assert flags == [0, 1, 1, 1]
    repo.invert_flag("t", "Selected")
    assert list(repo.query_df('SELECT "Selected" AS s FROM t ORDER BY rowid')["s"]) == [1, 0, 0, 0]
    repo.clear_integer_column("t", "Selected")
    assert repo.query_df('SELECT SUM("Selected") AS s FROM t')["s"][0] == 0
    assert "Hide" not in repo.get_columns("t")  # its own column, Hide untouched
    repo.close()
