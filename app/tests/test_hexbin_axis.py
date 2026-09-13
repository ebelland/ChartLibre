"""Tests for the Hexbin renderer.

The one behaviour worth pinning beyond "it draws something": mapping the
optional Value role switches the hexagons from a plain count to an
aggregate of that column, through whichever Reduce function was chosen -
that switch, not the binning itself (Matplotlib's job), is what this
renderer actually decides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.hexbin import HexbinAxisRenderer

RNG = np.random.default_rng(11)


def _frame(n: int = 500, with_value: bool = False) -> pd.DataFrame:
    data = {"x": RNG.normal(0, 1, n), "y": RNG.normal(0, 1, n)}
    if with_value:
        data["value"] = RNG.uniform(0.0, 1.0, n)
    return pd.DataFrame(data)


def _render(df: pd.DataFrame, style: dict | None = None):
    fig = Figure(figsize=(4.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    HexbinAxisRenderer().render_axis(
        ax, [SeriesData(name="s", df=df, style=style or {})], {}
    )
    return fig, ax


def test_x_y_alone_draws_one_hexbin_collection() -> None:
    _fig, ax = _render(_frame())
    assert len(ax.collections) == 1


def test_mapping_value_changes_what_each_hexagon_holds() -> None:
    """With no Value role every cell is a plain count; mapping it and
    picking 'max' must actually change the drawn array, not just accept
    the option and ignore it."""
    df = _frame(with_value=True)

    _fig, ax_count = _render(df.drop(columns=["value"]))
    count_values = np.asarray(ax_count.collections[0].get_array())

    _fig, ax_max = _render(df, {"reduce_function": "max"})
    max_values = np.asarray(ax_max.collections[0].get_array())

    assert not np.array_equal(np.sort(count_values), np.sort(max_values))


def test_too_few_finite_points_draws_nothing() -> None:
    df = pd.DataFrame({"x": [1.0], "y": [np.nan]})
    _fig, ax = _render(df)
    assert len(ax.collections) == 0


def test_colorbar_adds_a_second_axes_only_when_asked() -> None:
    fig, _ax = _render(_frame())
    assert len(fig.axes) == 1

    fig, _ax = _render(_frame(), {"colorbar": True})
    assert len(fig.axes) == 2


def test_a_second_series_is_dropped_with_a_reason() -> None:
    frame = _frame()
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    HexbinAxisRenderer().render_axis(
        ax,
        [
            SeriesData(name="a", df=frame, style={}),
            SeriesData(name="b", df=frame, style={}),
        ],
        {},
    )
    assert len(ax.collections) == 1
