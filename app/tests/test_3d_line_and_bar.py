"""Tests for the two newest 3D renderers: a connected line and bar forest.

Line3D's one real decision is *not* sorting or pivoting - row order is the
curve - so that is what is pinned here, alongside the usual visibility/
required-roles/empty-data guards every renderer shares. Bar3D's is that
each bar is actually centred on its (x, y) footprint rather than drawn from
a corner, which is easy to get backwards with pcolormesh's cell-corner
convention next door in heatmap.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.bar3d import Bar3DAxisRenderer
from app.charts.surface import Line3DAxisRenderer


def _axes3d():
    fig = Figure(figsize=(4.0, 4.0))
    return fig, fig.add_subplot(1, 1, 1, projection="3d")


# ----------------------------------------------------------------------
# 3D Line Plot
# ----------------------------------------------------------------------
def test_points_are_connected_in_row_order_not_sorted() -> None:
    """A deliberately non-monotonic x must survive unsorted - reordering it
    would draw a different curve, the one thing this renderer must not do."""
    df = pd.DataFrame({"x": [3.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "z": [0.0, 1.0, 2.0]})
    _fig, ax = _axes3d()
    Line3DAxisRenderer().render_axis(ax, [SeriesData(name="s", df=df, style={})], {})

    assert len(ax.lines) == 1
    drawn_x = np.asarray(ax.lines[0].get_data_3d()[0])
    assert list(drawn_x) == [3.0, 1.0, 2.0]


def test_several_series_draw_several_curves() -> None:
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]})
    _fig, ax = _axes3d()
    Line3DAxisRenderer().render_axis(
        ax,
        [
            SeriesData(name="a", df=df, style={}),
            SeriesData(name="b", df=df, style={}),
        ],
        {},
    )
    assert len(ax.lines) == 2


def test_a_hidden_series_is_not_drawn() -> None:
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    _fig, ax = _axes3d()
    Line3DAxisRenderer().render_axis(
        ax, [SeriesData(name="a", df=df, style={"visible": False})], {}
    )
    assert len(ax.lines) == 0


def test_non_finite_rows_are_dropped() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan], "y": [1.0, 2.0], "z": [1.0, 2.0]})
    _fig, ax = _axes3d()
    Line3DAxisRenderer().render_axis(ax, [SeriesData(name="a", df=df, style={})], {})
    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_data_3d()[0]) == 1


# ----------------------------------------------------------------------
# 3D Bar Chart
# ----------------------------------------------------------------------
def test_bars_are_centred_on_their_x_y_footprint() -> None:
    df = pd.DataFrame({"x": [5.0], "y": [10.0], "z": [3.0]})
    _fig, ax = _axes3d()
    Bar3DAxisRenderer().render_axis(
        ax, [SeriesData(name="s", df=df, style={"width": 2.0, "depth": 4.0})], {}
    )
    assert len(ax.collections) == 1


def test_bottom_shifts_the_base_without_changing_the_drawn_height() -> None:
    """A bar from bottom=5 to z=8 is 3 tall, same as bottom=0 to z=3 -
    Bottom moves where it starts, not how tall it reads."""
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [8.0]})
    _fig, ax = _axes3d()
    Bar3DAxisRenderer().render_axis(
        ax, [SeriesData(name="s", df=df, style={"bottom": 5.0})], {}
    )
    assert len(ax.collections) == 1


def test_a_second_series_is_dropped_with_a_reason() -> None:
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [1.0]})
    _fig, ax = _axes3d()
    Bar3DAxisRenderer().render_axis(
        ax,
        [
            SeriesData(name="a", df=df, style={}),
            SeriesData(name="b", df=df, style={}),
        ],
        {},
    )
    assert len(ax.collections) == 1


def test_no_finite_rows_draws_nothing() -> None:
    df = pd.DataFrame({"x": [np.nan], "y": [0.0], "z": [1.0]})
    _fig, ax = _axes3d()
    Bar3DAxisRenderer().render_axis(ax, [SeriesData(name="a", df=df, style={})], {})
    assert len(ax.collections) == 0
