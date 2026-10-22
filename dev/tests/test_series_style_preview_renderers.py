"""S-01: the live style preview reaches Stairs, 3D Line, 3D Scatter and ECDF, drawn by their own renderers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch

from app.charts.area import StairsAxisRenderer
from app.charts.ecdf import EcdfAxisRenderer
from app.charts.surface import Line3DAxisRenderer, Scatter3DAxisRenderer
from app.plugin_api.charts.base import SeriesData
from app.widgets.chart_panel.picking import PickingMixin


def _panel(axes: Any) -> SimpleNamespace:
    return SimpleNamespace(_canvas=SimpleNamespace(figure=axes.figure, draw_idle=lambda: None))


def _series(name: str = "s", **columns: Any) -> SeriesData:
    frame = pd.DataFrame(columns or {"x": np.arange(8.0), "y": np.arange(8.0) ** 2})
    return SeriesData(name=name, df=frame, style={})


def _draw(renderer: Any, series: list[SeriesData], *, projection: str | None = None) -> Any:
    axes = Figure().add_subplot(projection=projection)
    axes._dhub_axis_id = 1
    renderer.render_axis(axes, series, {})
    return axes


def test_stairs_takes_a_colour_and_a_line_style_without_a_render() -> None:
    axes = _draw(StairsAxisRenderer(), [_series()])
    (patch,) = [a for a in axes.get_children() if isinstance(a, StepPatch)]
    assert patch._dhub_series == "s"
    panel = _panel(axes)
    assert PickingMixin.preview_series_style(panel, 1, "s", {"color": "red"}) is True
    assert patch.get_edgecolor()[:3] == (1.0, 0.0, 0.0)
    assert PickingMixin.preview_series_style(panel, 1, "s", {"linestyle": "--"}) is True
    assert patch.get_linestyle() == "--"
    # a marker or a size is not a stairs setting: a render says so
    assert PickingMixin.preview_series_style(panel, 1, "s", {"marker": "o"}) is False


def test_3d_line_takes_colour_and_line_style_without_a_render() -> None:
    frame = {"x": np.arange(6.0), "y": np.arange(6.0), "z": np.arange(6.0) ** 2}
    axes = _draw(Line3DAxisRenderer(), [_series(**frame)], projection="3d")
    (line,) = [a for a in axes.get_children() if isinstance(a, Line2D) and getattr(a, "_dhub_series", None) == "s"]
    panel = _panel(axes)
    assert PickingMixin.preview_series_style(panel, 1, "s", {"color": "green"}) is True
    assert line.get_color() == "green"
    assert PickingMixin.preview_series_style(panel, 1, "s", {"linestyle": ":"}) is True
    assert line.get_linestyle() == ":"


def test_3d_scatter_takes_a_colour_and_a_size_without_a_render() -> None:
    frame = {"x": np.arange(6.0), "y": np.arange(6.0), "z": np.arange(6.0) ** 2}
    axes = _draw(Scatter3DAxisRenderer(), [_series(**frame)], projection="3d")
    panel = _panel(axes)
    assert PickingMixin.preview_series_style(panel, 1, "s", {"color": "red"}) is True
    assert PickingMixin.preview_series_style(panel, 1, "s", {"markersize": 3.0}) is True


def test_3d_scatter_coloured_by_a_column_is_left_to_the_render() -> None:
    frame = {"x": np.arange(6.0), "y": np.arange(6.0), "z": np.arange(6.0) ** 2, "color": np.arange(6.0)}
    axes = _draw(Scatter3DAxisRenderer(), [_series(**frame)], projection="3d")
    assert PickingMixin.preview_series_style(_panel(axes), 1, "s", {"color": "red"}) is False


def test_ecdf_is_a_scatter_underneath_and_already_follows_the_preview() -> None:
    axes = _draw(EcdfAxisRenderer(), [_series(value=np.random.default_rng(0).normal(size=30))])
    tagged = [a for a in axes.get_children() if getattr(a, "_dhub_series", None) == "s"]
    assert tagged, "the ECDF's points carry the series name"
    assert PickingMixin.preview_series_style(_panel(axes), 1, "s", {"color": "red"}) is True
