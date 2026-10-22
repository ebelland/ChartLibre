"""The 1/x axis scale (issue #9)."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from app.plugin_api.charts import render_figure as rf


def test_reciprocal_is_a_supported_scale_with_a_readable_label() -> None:
    assert "reciprocal" in rf.SUPPORTED_AXIS_SCALES
    assert rf.AXIS_SCALE_LABELS["reciprocal"] == "1/x"


def test_the_x_axis_maps_values_to_their_reciprocals_and_back() -> None:
    ax = Figure().subplots()
    ax.plot([1.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    rf._apply_scales(ax, {"x_scale": "reciprocal"})
    transform = ax.xaxis.get_transform()
    np.testing.assert_allclose(transform.transform([1.0, 2.0, 4.0]), [1.0, 0.5, 0.25])
    np.testing.assert_allclose(transform.inverted().transform([1.0, 0.5, 0.25]), [1.0, 2.0, 4.0])


def test_zero_is_not_drawn_and_nothing_raises() -> None:
    ax = Figure().subplots()
    ax.plot([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    rf._apply_scales(ax, {"x_scale": "reciprocal", "y_scale": "reciprocal"})
    ax.figure.canvas.draw()  # a zero x or y must not break the draw


def test_the_other_scales_are_unchanged() -> None:
    ax = Figure().subplots()
    rf._apply_scales(ax, {"x_scale": "log", "y_scale": "symlog"})
    assert ax.get_xscale() == "log" and ax.get_yscale() == "symlog"
