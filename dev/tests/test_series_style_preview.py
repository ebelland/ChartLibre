"""The no-render preview of a series style (S-01) must not report as shown what it did not draw.

Series properties, line style: None -> Default -> Solid on a series drawn as points left the chart
without its line, because the preview had nothing to restyle and said it had succeeded. "Default"
is the style sheet's own value, so it is never decided here: the chart is rendered.
"""
from __future__ import annotations

from types import SimpleNamespace

from matplotlib.figure import Figure

from app.widgets.chart_panel.picking import PickingMixin, _style_steps


def _points_only():
    axes = Figure().subplots()
    cloud = axes.scatter([0, 1, 2], [0, 1, 4])
    return axes, cloud


def test_a_line_style_that_would_draw_a_line_is_left_to_the_render_on_a_points_only_series() -> None:
    _axes, cloud = _points_only()
    for value in ("-", "--", "default"):
        assert _style_steps(cloud, {"linestyle": value}, has_points=True, sizes=True, has_lines=False)[0] is False


def test_no_line_on_points_only_is_nothing_to_draw() -> None:
    _axes, cloud = _points_only()
    assert _style_steps(cloud, {"linestyle": "none"}, has_points=True, sizes=True, has_lines=False) == (True, [])


def test_a_series_with_a_line_and_points_keeps_the_line_style_for_its_line() -> None:
    _axes, cloud = _points_only()
    assert _style_steps(cloud, {"linestyle": "-"}, has_points=True, sizes=True, has_lines=True) == (True, [])


def _panel(axes) -> SimpleNamespace:
    canvas = SimpleNamespace(figure=axes.figure, draw_idle=lambda: None)
    return SimpleNamespace(_canvas=canvas)


def test_the_preview_declines_a_line_the_series_does_not_have() -> None:
    axes, cloud = _points_only()
    axes._dhub_axis_id = 1
    cloud._dhub_series = "s"
    panel = _panel(axes)
    assert PickingMixin.preview_series_style(panel, 1, "s", {"linestyle": "-"}) is False  # the chart is rendered
    assert PickingMixin.preview_series_style(panel, 1, "s", {"linestyle": "default"}) is False
    assert PickingMixin.preview_series_style(panel, 1, "s", {"markersize": 0.5}) is True  # points take a size


def test_the_preview_still_restyles_an_existing_line() -> None:
    axes = Figure().subplots()
    (line,) = axes.plot([0, 1, 2], [0, 1, 4], linestyle="-")
    axes._dhub_axis_id = 1
    line._dhub_series = "s"
    assert PickingMixin.preview_series_style(_panel(axes), 1, "s", {"linestyle": "--"}) is True
    assert line.get_linestyle() == "--"
    # "default" is not a Matplotlib style: the style sheet decides, so the render does.
    assert PickingMixin.preview_series_style(_panel(axes), 1, "s", {"linestyle": "default"}) is False
