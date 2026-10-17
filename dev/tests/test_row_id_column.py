"""R-07: every user table has a row id column, "_id" INTEGER PRIMARY KEY, hidden from what a person sees."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def _schema(repo: SqliteRepo, table: str) -> list[tuple[str, str, int]]:
    return [(str(r[1]), str(r[2]), int(r[5])) for r in repo.table_info(table)]


def _ids(repo: SqliteRepo, table: str, column: str) -> list[tuple]:
    frame = repo.query_df(f'SELECT "_id", "{column}" FROM "{table}" ORDER BY rowid', row_id=True)
    return [tuple(row) for row in frame.itertuples(index=False)]


def test_an_import_gives_the_table_its_row_id_hidden_from_columns_and_frames(tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    try:
        repo.import_dataframe(pd.DataFrame({"x": [3.0, 1.0, 2.0]}), table_name="t", normalize_columns=False)
        assert _schema(repo, "t")[0] == ("_id", "INTEGER", 1)
        assert repo.row_id_column("t") == "_id" and repo.has_column("t", "_id")
        assert repo.get_columns("t") == ["x"] and repo.get_columns("t", with_id=True) == ["_id", "x"]
        assert list(repo.query_df('SELECT * FROM "t"').columns) == ["x"]
        assert list(repo.query_df('SELECT * FROM "t"', row_id=True).columns) == ["_id", "x"]
        assert not repo.has_integer_primary_key("t")  # ours: rows may be renumbered
    finally:
        repo.close()


def test_a_project_from_before_gets_row_ids_once_when_it_opens(tmp_path: Path) -> None:
    path = tmp_path / "old.dhub"
    SqliteRepo(db_path=path).close()
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "plain" (x REAL, label TEXT)')
    con.executemany('INSERT INTO "plain" VALUES (?, ?)', [(1.0, "a"), (2.0, "b"), (3.0, "c")])
    con.execute('DELETE FROM "plain" WHERE x = 2.0')
    con.execute('CREATE TABLE "own_key" (id INTEGER PRIMARY KEY, v REAL)')
    con.execute('CREATE TABLE "mongo" ("_id" TEXT, v REAL)')
    con.executemany('INSERT INTO "mongo" VALUES (?, ?)', [("5f1a", 1.0), ("5f1b", 2.0)])
    con.commit()
    con.close()

    repo = SqliteRepo(db_path=path)
    try:
        # Numbered in row order; a gap stays a gap.
        assert _ids(repo, "plain", "label") == [(1, "a"), (3, "c")]
        assert _schema(repo, "plain")[0] == ("_id", "INTEGER", 1)
        # A key of its own is its row id already: left alone.
        assert repo.row_id_column("own_key") == "id" and not repo.has_column("own_key", "_id")
        # Someone's own "_id" is kept, under another name.
        assert repo.get_columns("mongo") == ["_id_2", "v"]
        assert repo.query_df('SELECT "_id_2" FROM "mongo"')["_id_2"].tolist() == ["5f1a", "5f1b"]
    finally:
        repo.close()


def test_copies_keep_the_row_ids_and_a_sort_numbers_the_rows_anew(tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    try:
        repo.import_dataframe(pd.DataFrame({"x": [3.0, 1.0, 2.0], "n": ["3", "1", "2"]}),
                              table_name="t", normalize_columns=False)
        copy = repo.duplicate_table("t")
        assert _schema(repo, copy)[0] == ("_id", "INTEGER", 1) and _ids(repo, copy, "x") == _ids(repo, "t", "x")
        repo.cast_column("t", "n", "INTEGER")
        assert _schema(repo, "t")[0] == ("_id", "INTEGER", 1) and _ids(repo, "t", "n") == [(1, 3), (2, 1), (3, 2)]
        repo.insert_table_column("t", "z", before="x")
        assert repo.get_columns("t", with_id=True) == ["_id", "z", "x", "n"]
        assert _schema(repo, "t")[0] == ("_id", "INTEGER", 1) and _ids(repo, "t", "x")[0] == (1, 3.0)
        repo.sort_table("t", "x")
        assert _ids(repo, "t", "x") == [(1, 1.0), (2, 2.0), (3, 3.0)]
        assert repo.append_table_row("t") == 4
    finally:
        repo.close()


def test_a_table_saved_from_a_query_or_a_grouping_has_its_row_ids(tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    try:
        repo.import_dataframe(pd.DataFrame({"g": ["a", "b", "a"], "v": [1.0, 2.0, 3.0]}),
                              table_name="t", normalize_columns=False)
        # SELECT * brings the source's ids along: kept, they are distinct integers.
        saved = repo.create_table_from_query("picked", 'SELECT * FROM "t" WHERE g = \'a\'')
        assert _schema(repo, saved)[0] == ("_id", "INTEGER", 1) and _ids(repo, saved, "v") == [(1, 1.0), (3, 3.0)]
        grouped = repo.group_aggregate("t", ["g"], measures=[("SUM", "v")], new_name="sums")
        assert repo.row_id_column(grouped) == "_id" and repo.get_columns(grouped) == ["g", "sum_v"]
    finally:
        repo.close()


def test_a_series_query_selects_the_row_id_and_the_renderers_never_see_it(tmp_path: Path) -> None:
    from app.data.descriptors import AxisDescriptor
    from app.plugin_api.charts.render_figure import _build_series_data_list

    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    try:
        repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
                              table_name="t", normalize_columns=False)
        figure_id = repo.create_figure_descriptor(name="f")
        axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="", x_label="", y_label="", options={})
        series_id = repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s",
                                                  sql_query='SELECT "a" AS x, "b" AS y FROM "t"',
                                                  roles={"x": "x", "y": "y"})
        assert '"_id"' in repo.get_series_sql_query(series_id)
        # Where a point is not one row, nothing is added.
        assert "_id" not in repo.sql_with_row_id('SELECT a AS x, COUNT(*) AS y FROM "t" GROUP BY a')
        assert "_id" not in repo.sql_with_row_id('SELECT AVG(a) AS y FROM "t"')

        axis = next(a for a in repo.load_figure_descriptor(figure_id).axes if a.id == axis_id)
        (data,) = _build_series_data_list(repo=repo, axis_desc=axis)
        assert "_id" not in data.df.columns and list(data.ids) == [1, 2, 3]

        # A series made before R-07 gets it when the project opens.
        repo.update_series_sql_query(series_id, 'SELECT "a" AS x, "b" AS y FROM "t"')
        assert repo.add_row_ids_to_series() == 1 and '"_id"' in repo.get_series_sql_query(series_id)
    finally:
        repo.close()
