"""Markers of one's own: a character, mathtext, (sides, style, angle)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.plugin_api.charts.markers import is_custom, to_matplotlib


def test_forms():
    assert to_matplotlib("o") == "o"
    assert to_matplotlib("5") == 5 and to_matplotlib("1") == "1" and to_matplotlib("8") == "8"
    assert to_matplotlib("★") == "$★$"
    assert to_matplotlib("$") == r"$\$$"
    assert to_matplotlib("(5, 1, 0)") == (5, 1, 0)
    assert to_matplotlib(r"$\clubsuit$") == r"$\clubsuit$"
    assert to_matplotlib("中") is None  # not in the shipped font: would differ between systems
    assert to_matplotlib("(1, 9)") is None and to_matplotlib("abc") is None
    assert is_custom("A") and not is_custom("s")


def test_drawn():
    fig, ax = plt.subplots()
    for marker in ("★", "$", "(6, 2, 30)", r"$\clubsuit$"):
        ax.plot([1, 2], [1, 2], marker=to_matplotlib(marker))
    fig.canvas.draw()
    plt.close(fig)


def test_combo_keeps_a_custom_marker(qapp):
    from app.widgets.drop_down_controls.marker_combo import MarkerStyleCombo

    combo = MarkerStyleCombo()
    assert combo.set_current_marker("★")
    assert combo.current_marker() == "★"


def test_a_series_line_pattern_is_drawn_as_its_tuple():
    from app.plugin_api.charts.base import BaseAxisRenderer

    value, draw = BaseAxisRenderer.series_linestyle(None, {"linestyle": "---.--."}, {})
    assert draw and isinstance(value, tuple)


def test_marker_every():
    from app.charts.time_series import _marker_every

    assert _marker_every(None) == 1 and _marker_every("x") == 1 and _marker_every(0) == 1 and _marker_every(5) == 5
