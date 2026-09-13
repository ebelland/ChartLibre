"""Tests for the Heatmap renderer.

The one thing worth pinning beyond "it draws something": Heatmap must show
the raw per-cell value with no interpolation between cells, which is the
entire reason it exists next to Contour Plot - so most of this is about the
mesh actually carrying the grid's own values, and about the same "is this
even a grid" refusal Contour Plot already has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.heatmap import HeatmapAxisRenderer


def _grid_frame() -> pd.DataFrame:
    x = np.repeat([1.0, 2.0, 3.0], 3)
    y = np.tile([10.0, 20.0, 30.0], 3)
    z = x * 10 + y
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _render(df: pd.DataFrame, style: dict | None = None):
    fig = Figure(figsize=(4.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    HeatmapAxisRenderer().render_axis(
        ax, [SeriesData(name="field", df=df, style=style or {})], {}
    )
    return fig, ax


def test_a_complete_grid_draws_one_mesh_with_every_cell_value() -> None:
    _fig, ax = _render(_grid_frame())

    assert len(ax.collections) == 1
    mesh = ax.collections[0]
    drawn = np.asarray(mesh.get_array()).ravel()
    expected = np.sort((_grid_frame()["z"]).to_numpy())
    assert np.array_equal(np.sort(drawn), expected)


def test_an_incomplete_grid_is_refused_not_guessed() -> None:
    """One missing (x, y) pair - the same test Contour Plot's grid check
    already has, applied to the renderer that shares it."""
    frame = _grid_frame().iloc[:-1]
    _fig, ax = _render(frame)
    assert len(ax.collections) == 0


def test_annotate_writes_one_label_per_cell() -> None:
    frame = _grid_frame()
    _fig, ax = _render(frame, {"annotate": True})
    assert len(ax.texts) == len(frame)


def test_annotate_is_skipped_past_the_cell_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.charts.heatmap as heatmap_module

    monkeypatch.setattr(heatmap_module, "MAX_ANNOTATED_CELLS", 1)
    _fig, ax = _render(_grid_frame(), {"annotate": True})
    assert len(ax.texts) == 0


def test_colorbar_adds_a_second_axes_only_when_asked() -> None:
    fig, _ax = _render(_grid_frame())
    assert len(fig.axes) == 1

    fig, _ax = _render(_grid_frame(), {"colorbar": True, "colorbar_label": "Value"})
    assert len(fig.axes) == 2


def test_a_second_series_is_dropped_with_a_reason(caplog: pytest.LogCaptureFixture) -> None:
    frame = _grid_frame()
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    HeatmapAxisRenderer().render_axis(
        ax,
        [
            SeriesData(name="a", df=frame, style={}),
            SeriesData(name="b", df=frame, style={}),
        ],
        {},
    )
    assert len(ax.collections) == 1
