"""Rubber-band box selection on the chart (todo.txt N-20).

A rectangle drawn while "Select points" is on picks out the plotted points
it covers, grouped by the series they belong to (matched by the artist's
own label, same identity _on_pick and the legend toggle already use). Both
actions on the result wrap the source series' own SQL as a subquery, the
way a series-over-a-series already works elsewhere - "New series from
selection" adds a new series keeping only the rows inside the rectangle;
"Hide selected points" rewrites the existing series' query to keep only
the rows outside it. Neither needs the series to read a plain table (an
earlier, rowid-based Hide-column version of "Hide selected points" did,
and turned out unreliable in practice - see todo.txt P-02).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel


@dataclass
class _FakeEvent:
    inaxes: Any
    xdata: float | None
    ydata: float | None
    button: int = 1


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    built = ChartPanel(repo, figure_id)
    built.resize(600, 400)
    built.show()
    qapp.processEvents()
    yield built, axis_id
    built.close()
    repo.undo_store.discard_file()


def _drag(built: ChartPanel, axes: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    built._on_box_select_press(_FakeEvent(inaxes=axes, xdata=x0, ydata=y0))
    built._on_box_select_motion(_FakeEvent(inaxes=axes, xdata=x1, ydata=y1))
    built._on_box_select_release(_FakeEvent(inaxes=axes, xdata=x1, ydata=y1))


# ----------------------------------------------------------------------
# Making a selection
# ----------------------------------------------------------------------
def test_dragging_while_off_selects_nothing(panel) -> None:
    built, _axis_id = panel
    axes = built._figure.axes[0]

    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    assert built._last_selection is None


def test_dragging_a_rectangle_selects_the_points_inside_it(panel) -> None:
    built, axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]

    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    selection = built._last_selection
    assert selection is not None
    assert selection["axis_id"] == axis_id
    points = selection["series"]["s"]
    assert sorted(points["x"].tolist()) == [3.0, 4.0, 5.0, 6.0]


def test_a_click_without_dragging_selects_nothing(panel) -> None:
    built, _axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]

    _drag(built, axes, 3.0, 3.0, 3.0, 3.0)

    assert built._last_selection is None


def test_a_rectangle_over_empty_space_selects_nothing(panel) -> None:
    built, _axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]

    _drag(built, axes, 20.0, 20.0, 25.0, 25.0)

    assert built._last_selection is None


def test_turning_the_tool_off_discards_a_pending_drag(panel) -> None:
    built, _axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]
    built._on_box_select_press(_FakeEvent(inaxes=axes, xdata=2.0, ydata=2.0))

    built._on_toggle_select_mode(False)

    assert built._selection_start is None
    assert built._selection_rect_artist is None


def test_select_mode_disables_annotation_dragging(panel) -> None:
    """The two tools are mutually exclusive - see _on_annotation_drag_press."""
    built, axis_id = panel
    options = dict(built._repo.get_axis_options(axis_id) or {})
    options["annotations"] = [{"x": 5.0, "y": 5.0, "type": "text", "text": "hi", "kwargs": {}}]
    built._repo.set_axis_options(axis_id, options)
    built.reload()
    axes = built._figure.axes[0]
    built._select_mode_enabled = True

    built._on_annotation_drag_press(_FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0))

    assert built._dragging_annotation is None


# ----------------------------------------------------------------------
# Hide selected points
# ----------------------------------------------------------------------
def test_hiding_the_selection_removes_those_rows_from_the_chart(panel) -> None:
    built, axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]
    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    built._hide_selection()

    assert built._last_selection is None
    remaining_axes = built._figure.axes[0]
    collection = remaining_axes.collections[0]
    x_data = np.asarray(collection.get_offsets())[:, 0]
    assert sorted(float(v) for v in x_data) == [0.0, 1.0, 2.0, 7.0, 8.0, 9.0]
    # The original table is untouched - only the series' own query changed.
    assert built._repo.query_df("SELECT COUNT(*) AS n FROM w")["n"].iloc[0] == 10


def test_hiding_the_selection_rewrites_the_series_query(panel) -> None:
    built, axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]
    original_sql = built._repo.get_series(axis_id)[0]["sql_query"]
    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    built._hide_selection()

    new_sql = built._repo.get_series(axis_id)[0]["sql_query"]
    assert new_sql != original_sql
    frame = built._repo.query_df(f"SELECT x FROM ({new_sql})")
    assert sorted(frame["x"].tolist()) == [0.0, 1.0, 2.0, 7.0, 8.0, 9.0]


def test_hiding_the_selection_can_be_undone(panel) -> None:
    built, axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]
    original_sql = built._repo.get_series(axis_id)[0]["sql_query"]
    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    built._hide_selection()
    assert built._repo.get_series(axis_id)[0]["sql_query"] != original_sql

    built._repo.undo_last()

    assert built._repo.get_series(axis_id)[0]["sql_query"] == original_sql


def test_hiding_an_empty_selection_does_nothing(panel) -> None:
    built, axis_id = panel
    built._last_selection = None
    original_sql = built._repo.get_series(axis_id)[0]["sql_query"]

    built._hide_selection()  # must not raise

    assert built._repo.get_series(axis_id)[0]["sql_query"] == original_sql


# ----------------------------------------------------------------------
# New series from selection
# ----------------------------------------------------------------------
def test_new_series_from_selection_adds_a_series_with_only_those_rows(panel) -> None:
    built, axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]
    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    built._create_series_from_selection()

    assert built._last_selection is None
    series = built._repo.get_series(axis_id)
    assert len(series) == 2
    new_series = series[-1]
    assert "selection" in str(new_series["name"])
    frame = built._repo.query_df(f"SELECT x FROM ({new_series['sql_query']})")
    assert sorted(frame["x"].tolist()) == [3.0, 4.0, 5.0, 6.0]
    # The original series is untouched - the selection only added, never removed.
    assert built._repo.query_df("SELECT COUNT(*) AS n FROM w")["n"].iloc[0] == 10


def test_new_series_from_selection_can_be_undone(panel) -> None:
    built, axis_id = panel
    built._select_mode_enabled = True
    axes = built._figure.axes[0]
    _drag(built, axes, 2.5, 2.5, 6.5, 6.5)

    built._create_series_from_selection()
    assert len(built._repo.get_series(axis_id)) == 2

    built._repo.undo_last()

    assert len(built._repo.get_series(axis_id)) == 1
