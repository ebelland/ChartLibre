"""x labels that would overlap are wrapped or turned, and set back when there is room again."""
from __future__ import annotations

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from app.plugin_api.charts.render_figure import _fit_x_tick_labels

_CAUSES = ["Solder bridge", "Missing component", "Misalignment", "Cold joint", "Wrong value",
           "Scratched surface", "Other"]


def _figure(width: float, labels: list[str]) -> tuple[Figure, object]:
    figure = Figure(figsize=(width, 4), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    ax.bar(labels, range(len(labels), 0, -1))
    return figure, ax


def _texts(ax) -> list[str]:
    return [label.get_text() for label in ax.get_xticklabels()]


def test_room_enough_leaves_them_level() -> None:
    figure, ax = _figure(14, _CAUSES)
    _fit_x_tick_labels(figure, ax, {})
    assert _texts(ax) == _CAUSES and not any(label.get_rotation() for label in ax.get_xticklabels())


def test_wrapped_at_the_spaces_then_turned_when_narrower() -> None:
    figure, ax = _figure(9.5, _CAUSES)
    _fit_x_tick_labels(figure, ax, {})
    assert "Solder\nbridge" in _texts(ax) and not any(label.get_rotation() for label in ax.get_xticklabels())
    figure, ax = _figure(4.5, _CAUSES)
    _fit_x_tick_labels(figure, ax, {})
    assert {label.get_rotation() for label in ax.get_xticklabels()} == {30.0}
    assert _texts(ax)[0] == "Solder bridge"


def test_a_resize_fits_them_again() -> None:
    figure, ax = _figure(4.5, _CAUSES)
    _fit_x_tick_labels(figure, ax, {})
    figure.set_size_inches(14, 4)
    from matplotlib.backend_bases import ResizeEvent
    ResizeEvent("resize_event", figure.canvas)._process()
    assert not any(label.get_rotation() for label in ax.get_xticklabels())


def test_a_chosen_angle_is_kept() -> None:
    figure, ax = _figure(4.5, _CAUSES)
    _fit_x_tick_labels(figure, ax, {"x_tick_rotation": 45})
    assert not any(label.get_rotation() for label in ax.get_xticklabels())
