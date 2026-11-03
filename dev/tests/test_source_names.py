"""Tables, saved queries and stored functions share one namespace: a name is one source's."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


def test_a_name_belongs_to_one_source(tmp_path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "names.dhub")
    try:
        repo.import_dataframe(pd.DataFrame({"x": [1, 2]}), table_name="Query 1")
        repo.save_sympy_function("Function1", "expression", "a", "a*x")
        repo.save_query("Means", 'SELECT * FROM "Query 1"')

        assert (repo.source_kind("query 1"), repo.source_kind("MEANS"), repo.source_kind("function1")) == \
            ("table", "query", "function")
        assert repo.next_query_name() == "Query 2"  # a table holds "Query 1"
        assert repo.next_source_name("Function", separator="") == "Function2"

        with pytest.raises(ValueError, match="table"):
            repo.save_query("query 1", "SELECT 1")
        with pytest.raises(ValueError, match="query"):
            repo.save_sympy_function("means", "expression", "", "x")
        with pytest.raises(ValueError, match="function"):
            repo.save_query("Function1", "SELECT 1")
        repo.save_query("Means", "SELECT 2")  # the same source again: an update
        assert repo.get_query("Means").sql == "SELECT 2"
    finally:
        repo.close()


def test_a_query_and_a_function_are_renamed_with_their_charts(tmp_path) -> None:
    from app.data.repo.function_sql import GridRange, function_sql
    from app.data_operations import sympy_store

    repo = SqliteRepo(db_path=tmp_path / "rename.dhub")
    try:
        repo.save_query("Means", "SELECT 1 AS x")
        repo.rename_query("Means", "Averages")
        assert repo.get_query("Means") is None and repo.get_query("Averages").sql == "SELECT 1 AS x"

        repo.save_sympy_function("Bump", "code", "h", "def Bump(h):\n    return h * x\n")
        repo.save_sympy_function("Twice", "expression", "a", "2*Bump(a)")
        sympy_store.trust([sympy_store.get(repo, "Bump")])
        figure = repo.create_figure_descriptor(name="F")
        axis = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Scatter Plot", title="",
                                           x_label="x", y_label="y", options={})
        sql = f'SELECT "x" AS "x", "y" AS "y" FROM ({function_sql("Bump", {"h": 1.0}, GridRange(0, 1, 2))}) AS "_s"'
        series = repo.create_series_descriptor(axis_id=axis, series_index=0, name="s", roles={"x": "x", "y": "y"},
                                               sql_query=sql, style={})

        with pytest.raises(ValueError, match="Twice"):  # another function calls it
            sympy_store.rename(repo, "Bump", "Hump")
        repo.delete_sympy_function("Twice")
        assert sympy_store.rename(repo, "Bump", "Hump") == 1
        renamed = sympy_store.get(repo, "Hump")
        assert renamed is not None and renamed["code_string"].startswith("def Hump(h):")
        assert sympy_store.is_trusted(renamed)  # trusted before, trusted after
        assert "chartlibre_function('Hump'" in repo.get_series_sql_query(series)
        with pytest.raises(ValueError, match="query"):
            sympy_store.rename(repo, "Hump", "averages")
    finally:
        repo.close()


def test_deleting_a_function_names_the_charts_it_empties(tmp_path) -> None:
    from app.data.repo.function_sql import GridRange, function_sql

    repo = SqliteRepo(db_path=tmp_path / "delete.dhub")
    try:
        repo.save_sympy_function("Line", "expression", "a", "a*x")
        assert repo.charts_drawing_function("Line") == []
        figure = repo.create_figure_descriptor(name="Calibration")
        axis = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Scatter Plot", title="",
                                           x_label="x", y_label="y", options={})
        sql = f'SELECT "x" AS "x", "y" AS "y" FROM ({function_sql("Line", {"a": 1.0}, GridRange(0, 1, 2))}) AS "_s"'
        repo.create_series_descriptor(axis_id=axis, series_index=0, name="model", roles={"x": "x", "y": "y"},
                                      sql_query=sql, style={})
        assert repo.charts_drawing_function("Line") == ["Calibration: model"]
    finally:
        repo.close()
