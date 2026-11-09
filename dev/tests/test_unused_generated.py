"""Unused generated tables: found (nothing names them) and deleted with Undo."""
import pandas as pd


def test_unused_generated_tables(repo):
    for name in ("_fit_1", "_fit_2", "data"):
        repo.import_dataframe(pd.DataFrame({"a": [1, 2]}), table_name=name, normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="t", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query='SELECT a AS x FROM "_fit_1"',
                                  roles={"x": "x"}, style={})
    assert repo.unused_generated_tables() == ["_fit_2"]
    repo.delete_table("_fit_2")
    assert "_fit_2" not in repo.list_table_names()
    repo.undo_last()
    assert "_fit_2" in repo.list_table_names()
