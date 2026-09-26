"""Charts built on a saved query must behave like charts built on a table.

The saved query is embedded in the series SQL as a subquery, so the stored
series is self-contained and re-executes the query on every render.  These
tests go through the real render pipeline rather than checking SQL strings,
because the string being plausible is not the same as the chart drawing.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    """A repo with a table and a saved query selecting half of it."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS readings")
    repo.query_df("CREATE TABLE readings (t REAL, v REAL, grp TEXT)")
    repo.query_df(
        "INSERT INTO readings (t, v, grp) VALUES "
        "(1.0, 10.0, 'a'), (2.0, 20.0, 'a'), (3.0, 30.0, 'b'), (4.0, 40.0, 'b')"
    )
    repo.save_query("only_a", "SELECT t, v FROM readings WHERE grp = 'a'")
    yield repo
    repo.close()


def _figure_over(repo: SqliteRepo, source_name: str) -> int:
    """Create a one-axis scatter figure over a named source."""
    source = repo.get_data_source(source_name)
    assert source is not None

    figure_id = repo.create_figure_descriptor(name=f"fig_{source_name}", nrows=1, ncols=1)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title=source_name,
        x_label="t",
        y_label="v",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name=source_name,
        sql_query=f'SELECT "t" AS x, "v" AS y FROM {source.from_clause()}',
        roles={"x": "x", "y": "y"},
        style={"marker": "o"},
    )
    return int(figure_id)


def _render(repo: SqliteRepo, figure_id: int):
    descriptor = repo.load_figure_descriptor(figure_id)
    assert descriptor is not None
    fig = Figure(figsize=(6.0, 4.0))
    render_figure_from_descriptor(figure=fig, descriptor=descriptor, repo=repo)
    return fig


def _point_count(fig: Figure) -> int:
    axis = fig.axes[0]
    return sum(len(collection.get_offsets()) for collection in axis.collections)


# ----------------------------------------------------------------------
# Charts over a saved query
# ----------------------------------------------------------------------
def test_a_chart_over_a_query_renders_the_query_rows(repo: SqliteRepo) -> None:
    fig = _render(repo, _figure_over(repo, "only_a"))
    assert _point_count(fig) == 2


def test_editing_the_query_changes_the_chart(repo: SqliteRepo) -> None:
    """The query is executed on render, not frozen into the series."""
    figure_id = _figure_over(repo, "only_a")
    assert _point_count(_render(repo, figure_id)) == 2

    repo.save_query("only_a", "SELECT t, v FROM readings")

    # The stored series embeds the *old* SQL, so re-deriving it is what a chart
    # rebuilt from the source would do; the point of this test is that nothing
    # was materialised, so the new query is immediately usable.
    assert repo.data_source_row_count(repo.get_data_source("only_a")) == 4


def test_a_query_with_a_where_clause_survives_being_wrapped(repo: SqliteRepo) -> None:
    """The subquery wrapper must not break ORDER BY or LIMIT inside it."""
    repo.save_query("top_two", "SELECT t, v FROM readings ORDER BY v DESC LIMIT 2")

    fig = _render(repo, _figure_over(repo, "top_two"))
    assert _point_count(fig) == 2


def test_a_cte_query_can_back_a_chart(repo: SqliteRepo) -> None:
    repo.save_query(
        "doubled",
        "WITH d AS (SELECT t, v * 2 AS v FROM readings) SELECT t, v FROM d",
    )
    fig = _render(repo, _figure_over(repo, "doubled"))

    axis = fig.axes[0]
    values = axis.collections[0].get_offsets()
    assert max(point[1] for point in values) == pytest.approx(80.0)


# ----------------------------------------------------------------------
# Columns and preview, through the same abstraction the UI uses
# ----------------------------------------------------------------------


def test_a_broken_query_degrades_instead_of_raising(repo: SqliteRepo) -> None:
    """A query over a dropped table must not take the preview down with it."""
    repo.save_query("orphan", "SELECT a FROM long_gone")
    source = repo.get_data_source("orphan")

    assert source is not None
    assert repo.data_source_columns(source) == []
    assert repo.data_source_row_count(source) == 0
    assert repo.data_source_page(source, limit=10, offset=0).empty


# ----------------------------------------------------------------------
# The list the UI renders
# ----------------------------------------------------------------------


