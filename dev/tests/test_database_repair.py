"""Optimize DB fixes what can be fixed: stray descriptors, series and links to a missing table, the app's leftover tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def test_repair_drops_what_is_broken_or_left_over_and_never_a_table_of_data(tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "r.dhub")
    try:
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        for name in ("data", "unused_data", "gone", "_feeds_a_chart_", "_left_over_"):
            repo.import_dataframe(frame, table_name=name, normalize_columns=False)
        figure_id = repo.create_figure_descriptor(name="f")
        axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="", x_label="", y_label="", options={})
        for index, table in enumerate(("data", "_feeds_a_chart_", "gone")):
            repo.create_series_descriptor(axis_id=axis_id, series_index=index, name=table,
                                          sql_query=f'SELECT x, y FROM "{table}"', roles={"x": "x", "y": "y"})
        repo.upsert_link(table_name="gone", source_path=str(tmp_path / "gone.csv"), settings={})
        repo.delete_table("gone")  # its series and link would stay behind, as from an older version
        con = repo._con
        con.execute("PRAGMA foreign_keys = OFF")
        con.execute("INSERT INTO __series_descriptors__ (axis_id, series_index, name, sql_query) "
                    "VALUES (999, 0, 'orphan', 'SELECT x, y FROM \"data\"')")
        con.execute("INSERT INTO __import_links__ (table_name, source_path, settings_json) VALUES ('gone', 'x', '{}')")
        con.execute('INSERT INTO __series_descriptors__ (axis_id, series_index, name, sql_query) '
                    'VALUES (?, 5, \'reads gone\', \'SELECT x, y FROM "gone"\')', (axis_id,))
        con.commit()

        found = repo.check_database()
        assert any("_left_over_" in line for line in found.fixable)
        assert not any("_feeds_a_chart_" in line for line in found.fixable), found.fixable
        assert found.dangling_series and found.dangling_links and found.orphan_descriptors

        done = repo.repair_database()
        assert done
        after = repo.check_database()
        assert after.fixable == []
        tables = set(repo.list_table_names()) | {str(r[0]) for r in con.execute("SELECT name FROM sqlite_master")}
        assert {"data", "unused_data", "_feeds_a_chart_"} <= tables and "_left_over_" not in tables
        assert "unused_data" in after.unreferenced_tables  # data: told, never dropped

        repo.undo_last()
        assert "_left_over_" in {str(r[0]) for r in repo._con.execute("SELECT name FROM sqlite_master")}
    finally:
        repo.close()


def test_deleting_or_renaming_a_table_touches_its_own_series_only(tmp_path: Path) -> None:
    """The series were found by series_index - their place in an axis, the same on every chart."""
    repo = SqliteRepo(db_path=tmp_path / "s.dhub")
    try:
        frame = pd.DataFrame({"x": [1.0], "y": [2.0]})
        for name in ("a", "b", "sales", "sales_2023"):
            repo.import_dataframe(frame, table_name=name, normalize_columns=False)
        ids = {}
        for table in ("a", "b", "sales", "sales_2023"):
            figure_id = repo.create_figure_descriptor(name=table)
            axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                                  title="", x_label="", y_label="", options={})
            ids[table] = repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=table,
                                                       sql_query=f'SELECT x, y FROM "{table}"',
                                                       roles={"x": "x", "y": "y"})
        repo.delete_table("a")
        assert repo.get_series_sql_query(ids["a"]) is None
        assert repo.get_series_sql_query(ids["b"]) is not None  # same place on another chart: kept
        repo.rename_table("sales", "revenue")
        assert '"revenue"' in repo.get_series_sql_query(ids["sales"])
        assert '"sales_2023"' in repo.get_series_sql_query(ids["sales_2023"])  # not "revenue_2023"
        assert '"b"' in repo.get_series_sql_query(ids["b"])
    finally:
        repo.close()
