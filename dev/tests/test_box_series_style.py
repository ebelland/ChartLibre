"""A box plot series: its Line style draws the box's lines, its Marker style the outliers."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from app.charts.box import BoxAxisRenderer
from app.plugin_api.charts.kwarg_spec import DEFAULT


def _box():
    ax = Figure().add_subplot()
    return ax.boxplot([np.r_[np.arange(20.0), 60.0]], patch_artist=True)


def test_line_style_and_marker_reach_the_box_and_the_outliers() -> None:
    result = _box()
    BoxAxisRenderer._style_box(BoxAxisRenderer.__new__(BoxAxisRenderer), result, "", 1, None, None, None,
                               linestyle="--", marker="s", markersize=7)
    assert all(line.get_linestyle() == "--" for line in result["whiskers"] + result["medians"])
    assert result["boxes"][0].get_linestyle() == "--"
    assert result["fliers"][0].get_marker() == "s" and result["fliers"][0].get_markersize() == 7


def test_default_leaves_the_style_sheet_alone() -> None:
    result = _box()
    before = result["fliers"][0].get_marker()
    BoxAxisRenderer._style_box(BoxAxisRenderer.__new__(BoxAxisRenderer), result, "", 1, None, None, None,
                               linestyle=DEFAULT, marker=DEFAULT)
    assert result["fliers"][0].get_marker() == before and result["whiskers"][0].get_linestyle() == "-"
