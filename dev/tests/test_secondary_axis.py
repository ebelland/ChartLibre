"""A secondary axis as the primary one in other units (G-17)."""
from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from app.plugin_api.charts import secondary_axis


@pytest.mark.parametrize("key", list(secondary_axis.CONVERSIONS))
def test_every_conversion_round_trips(key: str) -> None:
    _label, forward, inverse, _caption = secondary_axis.CONVERSIONS[key]
    values = np.array([1.0, 20.0, 300.0])
    assert np.allclose(inverse(forward(values)), values)


def test_known_conversions() -> None:
    forward = secondary_axis.CONVERSIONS["degC_degF"][1]
    assert forward(100.0) == pytest.approx(212.0)
    assert secondary_axis.CONVERSIONS["nm_eV"][1](620.0) == pytest.approx(2.0, rel=1e-3)
    assert secondary_axis.CONVERSIONS["degC_K"][1](0.0) == pytest.approx(273.15)


def test_applied_once_per_render_and_replaced_on_a_redraw() -> None:
    ax = Figure().add_subplot()
    ax.plot([0, 100], [0, 1])
    options = {"secondary_x": "degC_degF", "secondary_y": "kg_lb"}
    secondary_axis.apply_secondary_axes(ax, options)
    secondary_axis.apply_secondary_axes(ax, options)
    assert len(ax._dhub_secondary_axes) == 2
    assert len([child for child in ax.child_axes]) == 2
    secondary_axis.apply_secondary_axes(ax, {})
    assert ax._dhub_secondary_axes == [] and not ax.child_axes


def test_a_polar_axes_refuses_and_is_left_alone() -> None:
    ax = Figure().add_subplot(projection="polar")
    secondary_axis.apply_secondary_axes(ax, {"secondary_x": "degC_degF"})
    assert ax._dhub_secondary_axes == []
