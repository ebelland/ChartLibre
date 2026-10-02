"""Outliers hide the series' own rows (todo O-01).

The detection used to read the whole source table and quote the roles as
column names: a series filtered to one ticker searched all of them, and a
role naming an alias ("price AS y") came back as the text 'y', so nothing
was usable and nothing was hidden - the report "the outliers are not
hidden". It reads through the series' own SQL now.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

import app.series_operations.outlier_dialog as outlier_module
from app.analysis.outliers import OUTLIER_ZSCORE
from app.data.select_sql import sql_insert_select_expression
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.outlier_dialog import ACTION_HIDE, SeriesOutlierDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    clear_state("SeriesOutlierDialog")
    told: list[str] = []
    monkeypatch.setattr(outlier_module, "show_message", lambda _parent, key, **_kw: told.append(key))
    yield told
    clear_state("SeriesOutlierDialog")


def _prices(repo: SqliteRepo) -> None:
    rng = np.random.default_rng(3)
    frames = []
    for ticker, level in (("AAA", 10.0), ("BBB", 50.0), ("CCC", 200.0)):
        price = level + rng.normal(0.0, 0.2, 40)
        price[20] = level * 3  # one spike per ticker, at rowid 20, 60, 100 + 1
        frames.append(pd.DataFrame({"day": np.arange(40.0), "ticker": ticker, "price": price}))
    repo.import_dataframe(pd.concat(frames, ignore_index=True), table_name="prices", normalize_columns=False)


def _figure(repo: SqliteRepo, series: list[tuple[str, str]]) -> int:
    figure_id = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Line Plot",
        title="t", x_label="x", y_label="y", options={},
    ))
    for index, (name, sql) in enumerate(series):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=name, sql_query=sql,
            roles={"x": "x", "y": "y"}, style={},
        )
    return figure_id


def _hide(repo: SqliteRepo, figure_id: int) -> SeriesOutlierDialog:
    dialog = SeriesOutlierDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    dialog.model_combo.setCurrentText(OUTLIER_ZSCORE)
    dialog._parameter_form_spec.set_values({"action": ACTION_HIDE, "threshold": 3.0})
    dialog.ok()
    return dialog


def _hidden(repo: SqliteRepo, table: str) -> list[int]:
    return sorted(int(r) for r in repo.query_df(f'SELECT rowid AS r FROM "{table}" WHERE "Hide" = 1')["r"])


def test_aliased_roles_on_a_filtered_series_hide_that_series_spike(qapp, repo: SqliteRepo, _quiet: list[str]) -> None:
    _prices(repo)
    figure_id = _figure(repo, [("AAA", "SELECT day AS x, price AS y FROM prices WHERE ticker = 'AAA'")])
    _hide(repo, figure_id)

    assert _quiet == []
    assert _hidden(repo, "prices") == [21]  # AAA's spike only, not BBB's or CCC's
    [row] = repo.get_series(int(repo.query_df("SELECT id FROM __axis_descriptors__")["id"][0]))
    shown = repo.query_df(str(dict(row)["sql_query"]))
    assert len(shown) == 39 and shown["y"].max() < 11


def test_series_on_the_same_rows_hide_the_union(qapp, repo: SqliteRepo) -> None:
    x = np.arange(60.0)
    raw = np.sin(x / 5.0)
    raw[10] = 9.0
    smooth = np.cos(x / 5.0)
    smooth[40] = -9.0
    repo.import_dataframe(pd.DataFrame({"t": x, "raw": raw, "smooth": smooth}), table_name="m", normalize_columns=False)
    figure_id = _figure(repo, [
        ("raw", "SELECT t AS x, raw AS y FROM m"),
        ("smooth", "SELECT t AS x, smooth AS y FROM m"),
    ])
    _hide(repo, figure_id)
    # The second series used to show the first one's outlier again.
    assert _hidden(repo, "m") == [11, 41]


def test_a_rerun_keeps_other_series_hidden_rows_and_one_filter(qapp, repo: SqliteRepo) -> None:
    _prices(repo)
    sql_a = "SELECT day AS x, price AS y FROM prices WHERE ticker = 'AAA'"
    sql_b = "SELECT day AS x, price AS y FROM prices WHERE ticker = 'BBB'"
    figure_a = _figure(repo, [("AAA", sql_a)])
    figure_b = _figure(repo, [("BBB", sql_b)])
    _hide(repo, figure_b)
    _hide(repo, figure_a)
    _hide(repo, figure_a)  # again: replaces its own choice only

    assert _hidden(repo, "prices") == [21, 61]
    sqls = [str(sql) for sql in repo.query_df("SELECT sql_query FROM __series_descriptors__")["sql_query"]]
    assert all(sql.count('"Hide" = 0') == 1 for sql in sqls), sqls


def test_an_aggregated_series_is_refused_with_a_reason(repo: SqliteRepo) -> None:
    _prices(repo)
    with pytest.raises(ValueError, match="aggregates"):
        repo.query_series_frame_for_hide(
            sql_query="SELECT ticker AS x, AVG(price) AS y FROM prices GROUP BY ticker",
            roles={"x": "x", "y": "y"},
        )


def test_the_rowid_goes_into_the_outer_query_not_a_subquery() -> None:
    sql = "SELECT a.m AS x, (SELECT COUNT(*) FROM e b WHERE b.m <= a.m) AS y FROM e a"
    assert sql_insert_select_expression(sql, 'rowid AS "__rowid__"') == (
        "SELECT a.m AS x, (SELECT COUNT(*) FROM e b WHERE b.m <= a.m) AS y, "
        'rowid AS "__rowid__" FROM e a'
    )


def test_the_hide_filter_comes_off_for_reading(repo: SqliteRepo) -> None:
    strip = repo.sql_without_hide_filter
    assert strip('SELECT x, y FROM t WHERE a = 1 AND "Hide" = 0 ORDER BY x') == "SELECT x, y FROM t WHERE a = 1 ORDER BY x"
    assert strip('SELECT x, y FROM t WHERE "Hide" = 0 AND a = 1') == "SELECT x, y FROM t WHERE a = 1"
    assert strip('SELECT x, y FROM t WHERE "Hide" = 0') == "SELECT x, y FROM t"
    assert repo.sql_with_hide_filter('SELECT x FROM t WHERE "Hide" = 0') == 'SELECT x FROM t WHERE "Hide" = 0'


def test_a_kept_outlier_run_is_in_the_project_history(qapp, repo: SqliteRepo) -> None:
    """Todo R-03: Outlier keeps its preview on OK, outside the common Apply path."""
    _prices(repo)
    figure_id = _figure(repo, [("AAA", "SELECT day AS x, price AS y FROM prices WHERE ticker = 'AAA'")])
    _hide(repo, figure_id)
    [record] = repo.operations()
    assert "utlier" in record.operation  # the window title, translated
    assert record.results == [{"table": "prices", "name": "rows hidden", "rows": 1}]
    assert [source["name"] for source in record.sources] == ["AAA"]
    assert repo.operations(table="prices") == [record]
