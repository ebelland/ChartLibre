"""Broken axis (G-17): the gap between two values takes no room, its ticks are dropped, marks are drawn."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from app.plugin_api.charts.broken_axis import apply_breaks, parse_break


def test_the_ends_are_read_in_either_spelling() -> None:
    assert parse_break("10, 90") == (10.0, 90.0)
    assert parse_break("1,5; 2,5") == (1.5, 2.5)
    assert parse_break("90, 10") is None and parse_break("") is None and parse_break("x") is None


def test_the_gap_takes_no_room_and_has_no_ticks() -> None:
    figure = Figure()
    ax = figure.add_subplot()
    ax.plot([0, 1, 2, 3], [1.0, 5.0, 101.0, 105.0])
    apply_breaks(ax, {"y_break": "10, 100"})
    figure.canvas.draw()
    to_pixels = ax.transData.transform
    # 5 -> 101 is as far as 1 -> 5 plus 1: the 90 between them is gone.
    gap = to_pixels((0, 101))[1] - to_pixels((0, 5))[1]
    unit = (to_pixels((0, 5))[1] - to_pixels((0, 1))[1]) / 4
    assert abs(gap - 6 * unit) < 1.0
    ticks = ax.yaxis.get_majorticklocs()
    assert not np.any((ticks > 10) & (ticks < 100))
    assert len(ax.lines) == 2  # the data and the marks


def test_a_log_axis_is_left_unbroken() -> None:
    figure = Figure()
    ax = figure.add_subplot()
    ax.plot([1, 2], [1.0, 100.0])
    ax.set_yscale("log")
    apply_breaks(ax, {"y_break": "10, 50", "y_scale": "log"})
    assert ax.get_yscale() == "log"
