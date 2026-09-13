"""The series checklist every operation dialog shares: colour swatches.

Every series-operation dialog picks its series from AxisSeriesSelector's
checklist, and until now every row was plain text - a series' own plotted
colour, already visible on the chart the dialog was opened from, was
nowhere in the list that picks among them. The one behaviour worth pinning:
an explicit colour is used as-is, and a series left on the style cycle gets
a colour that actually varies by position, so two cycle-coloured series
never render identical swatches.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cycler import cycler
from matplotlib import rcParams

from app.data.sqlite_repo import SqliteRepo
from app.widgets.axis_series_selector import AxisSeriesSelector, _color_swatch_icon


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("CREATE TABLE t (x REAL, y REAL)")
    repo.query_df("INSERT INTO t (x, y) VALUES (1.0, 2.0)")

    figure_id = repo.create_figure_descriptor(name="fig", nrows=1, ncols=1)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="t",
        x_label="x",
        y_label="y",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="explicit",
        sql_query='SELECT x AS x, y AS y FROM "t"',
        style={"color": "#ff0000"},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=1,
        name="cycled",
        sql_query='SELECT x AS x, y AS y FROM "t"',
        style={},
    )
    yield repo
    repo.close()


@pytest.fixture(autouse=True)
def _restore_prop_cycle():
    saved = rcParams["axes.prop_cycle"]
    yield
    rcParams["axes.prop_cycle"] = saved


def test_an_explicit_colour_is_used_as_is(qapp, repo: SqliteRepo) -> None:
    figure_id = repo.get_figures()[0][0]
    selector = AxisSeriesSelector(repo, figure_id)

    row = repo.get_series(selector.selected_axis_id())[0]
    assert selector._series_color(row, 0) == "#ff0000"


def test_two_cycle_coloured_series_get_different_swatches(qapp, repo: SqliteRepo) -> None:
    rcParams["axes.prop_cycle"] = cycler(color=["#111111", "#222222"])
    figure_id = repo.get_figures()[0][0]
    selector = AxisSeriesSelector(repo, figure_id)

    rows = repo.get_series(selector.selected_axis_id())
    cycled = next(row for row in rows if row["name"] == "cycled")

    assert selector._series_color(cycled, 0) == "#111111"
    assert selector._series_color(cycled, 1) == "#222222"


def test_every_list_row_carries_an_icon(qapp, repo: SqliteRepo) -> None:
    figure_id = repo.get_figures()[0][0]
    selector = AxisSeriesSelector(repo, figure_id)

    assert selector.series_list.count() == 2
    for row in range(selector.series_list.count()):
        assert not selector.series_list.item(row).icon().isNull()


def test_the_icon_function_falls_back_on_an_invalid_colour(qapp) -> None:
    assert not _color_swatch_icon("not-a-real-color").isNull()
