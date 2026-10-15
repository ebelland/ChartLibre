"""AN-11: the Tables operations' SQL, run on SQLite and checked against hand-worked and pandas answers."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from app.data import sql_builders as sb


@pytest.fixture
def con() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    pd.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "x": [1.0, 2.0, None, 4.0]}).to_sql(
        "left_t", con, index=False)
    pd.DataFrame({"id": [2, 3, 5], "name": ["B", "C", "E"], "score": [20, 30, 50]}).to_sql(
        "right_t", con, index=False)
    pd.DataFrame({"subject": ["s1", "s1", "s2", "s2"], "time": ["t1", "t2", "t1", "t2"],
                  "y": [1.0, 2.0, 3.0, 4.0]}).to_sql("long_t", con, index=False)
    return con


def _run(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)


@pytest.mark.parametrize("kind,ids", [("INNER", [2, 3]), ("LEFT", [1, 2, 3, 4]), ("RIGHT", [2, 3, 5]),
                                      ("FULL OUTER", [1, 2, 3, 4, 5])])
def test_join_keeps_the_rows_each_kind_promises(con, kind, ids) -> None:
    sql = sb.join("left_t", ["id", "name", "x"], "right_t", ["id", "name", "score"], [("id", "id")], kind=kind,
                  match_column=True)
    frame = _run(con, sql)
    assert sorted(frame["id"].astype(int)) == ids
    assert list(frame.columns) == ["id", "name", "x", "name_2", "score", "Match"]
    match = dict(zip(frame["id"].astype(int), frame["Match"]))
    assert match.get(2) == "Both" and match.get(1, "Left only") == "Left only" and match.get(5, "Right only") == "Right only"


def test_union_matches_columns_by_name_and_says_where_rows_came_from(con) -> None:
    frame = _run(con, sb.union({"left_t": ["id", "name", "x"], "right_t": ["id", "name", "score"]}))
    assert list(frame.columns) == ["Source", "id", "name", "x", "score"]
    assert len(frame) == 7 and frame["Source"].tolist().count("right_t") == 3
    assert frame.loc[frame["Source"] == "right_t", "x"].isna().all()


def test_stack_is_pandas_melt_in_row_order(con) -> None:
    frame = _run(con, sb.stack("left_t", ["x", "name"], ["id"], label="Column", value="Value"))
    assert len(frame) == 8
    assert frame["id"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert frame["Column"].tolist()[:2] == ["x", "name"]
    dropped = _run(con, sb.stack("left_t", ["x"], ["id"], drop_missing=True))
    assert dropped["id"].tolist() == [1, 2, 4]


def test_split_undoes_a_stack(con) -> None:
    frame = _run(con, sb.split("long_t", "time", "y", ["t1", "t2"], ["subject"]))
    expected = pd.DataFrame({"subject": ["s1", "s2"], "t1": [1.0, 3.0], "t2": [2.0, 4.0]})
    pd.testing.assert_frame_equal(frame, expected)


def test_transpose_by_a_label_column_and_by_row_number(con) -> None:
    frame = _run(con, sb.transpose("left_t", ["x", "id"], label_column="name", labels=["a", "b", "c", "d"]))
    assert list(frame.columns) == ["Label", "a", "b", "c", "d"]
    assert frame.iloc[1, 1:].tolist() == [1, 2, 3, 4]
    numbered = _run(con, sb.transpose("left_t", ["id"], rows=4))
    assert list(numbered.columns) == ["Label", "Row 1", "Row 2", "Row 3", "Row 4"]
    with pytest.raises(ValueError):
        sb.transpose("t", ["x"], label_column="g", labels=["a", "a"])


def test_recode_maps_the_values_given_and_keeps_the_rest(con) -> None:
    frame = _run(con, sb.recode("left_t", ["id", "name", "x"], "name", {"a": "first", "b": "2"},
                                new_column="name_recoded"))
    assert frame["name_recoded"].tolist() == ["first", 2, "c", "d"]
    in_place = _run(con, sb.recode("left_t", ["id", "name", "x"], "id", {"1": "10"}))
    assert in_place["id"].tolist() == [10, 2, 3, 4]


def test_bins_close_on_the_left_and_keep_the_maximum(con) -> None:
    cuts = sb.bin_edges(np.array([1.0, 2.0, 4.0]), "equal width", 3)
    np.testing.assert_allclose(cuts, [1, 2, 3, 4])
    frame = _run(con, sb.bin_column("left_t", ["id", "x"], "x", cuts, new_column="x_bin"))
    assert [v if isinstance(v, str) else None for v in frame["x_bin"]] == ["[1, 2)", "[2, 3)", None, "[3, 4]"]
    numbered = _run(con, sb.bin_column("left_t", ["x"], "x", cuts, new_column="b", style="numbers"))
    assert numbered["b"].tolist()[:2] == [1, 2]
    assert len(sb.bin_edges(np.arange(100.0), "quantiles", 4)) == 5


def test_names_and_text_are_quoted(con) -> None:
    pd.DataFrame({"a b": ["it's"]}).to_sql('odd "t"', con, index=False)
    frame = _run(con, sb.recode('odd "t"', ["a b"], "a b", {"it's": "ok"}))
    assert frame["a b"].tolist() == ["ok"]


def test_split_without_groups_lines_the_values_up_by_order(con) -> None:
    frame = _run(con, sb.split("long_t", "time", "y", ["t1", "t2"], []))
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"t1": [1.0, 3.0], "t2": [2.0, 4.0]}))


def test_every_build_operation_writes_sql_that_runs(qapp, repo) -> None:
    from app.dialogs.query_builder_dialog import QueryBuilderDialog

    repo.import_dataframe(pd.DataFrame({"g": ["a", "b", "a", "b"], "t": ["x", "x", "y", "y"], "v": [1.0, 2.0, 3.0, 4.0]}),
                          table_name="facts", normalize_columns=False)
    repo.import_dataframe(pd.DataFrame({"g": ["a", "b"], "label": ["first", "second"]}), table_name="names",
                          normalize_columns=False)
    dialog = QueryBuilderDialog(repo)
    try:
        for index, form in enumerate(dialog._operation_forms):
            dialog._operation_combo.setCurrentIndex(index)
            if type(form).__name__ == "RecodeForm":
                form.mapping.item(0, 1).setText("A")
            assert dialog.write_operation_sql(), (type(form).__name__, dialog._status.text())
        assert repo.create_table_from_query("grouped", dialog.sql) == "grouped"
        assert len(repo.query_df('SELECT * FROM "grouped"')) == 1
        with pytest.raises(ValueError):
            repo.create_table_from_query("grouped", dialog.sql)
    finally:
        dialog.close()
