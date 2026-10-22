"""Heatmap clustered (G-14): rows and columns reordered so alike ones sit together, the trees beside them."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from app.charts.heatmap import HeatmapAxisRenderer
from app.plugin_api.charts.base import SeriesData


def _grid() -> pd.DataFrame:
    """Rows y=0..5 and columns x=0..3; rows 0, 2, 4 are high on x 0 and 2, rows 1, 3, 5 on x 1 and 3."""
    rows = []
    for y in range(6):
        for x in range(4):
            high = (y % 2) == (x % 2)
            rows.append({"x": float(x), "y": float(y), "z": (10.0 if high else 0.0) + 0.1 * y + 0.01 * x})
    return pd.DataFrame(rows)


def _draw(options: dict) -> Figure:
    figure = Figure()
    ax = figure.add_subplot()
    HeatmapAxisRenderer().render_axis(ax, [SeriesData(name="g", df=_grid(), style={})], options)
    return figure


def test_both_ways_the_alike_rows_and_columns_are_neighbours_and_two_trees_are_drawn() -> None:
    figure = _draw({"cluster": "both"})
    ax = figure.axes[0]
    rows = [label.get_text() for label in ax.get_yticklabels()]
    columns = [label.get_text() for label in ax.get_xticklabels()]
    parity = lambda labels: [int(float(text)) % 2 for text in labels]  # noqa: E731
    assert parity(rows) in ([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0])
    assert parity(columns) in ([0, 0, 1, 1], [1, 1, 0, 0])
    assert len(ax.child_axes) == 2


def test_unclustered_by_default() -> None:
    ax = _draw({}).axes[0]
    assert not ax.child_axes
