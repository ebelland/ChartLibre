"""Axis properties: tabs Properties, X, Y, Z, Kwargs; one coordinate page shown in X, Y and Z."""
from __future__ import annotations

from app.charts import axis_options
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget


def _select(widget: AxisPropertiesWidget, axis: str) -> None:
    widget._tabs.setCurrentIndex(1 + axis_options.AXES.index(axis))


def test_each_coordinate_tab_edits_its_own_coordinate(qapp) -> None:
    widget = AxisPropertiesWidget()
    widget._load_extended_axis_options({"y_scale": "log", axis_options.grid_key("x", "major"): "on"})
    _select(widget, "x")
    assert widget._targets() == ["x"] and widget._grid_combos["major"].currentData() == "on"

    _select(widget, "y")
    assert widget._scale_combo.currentData() == "log"  # the page shows Y
    widget._coordinate_label.setText("speed")
    assert widget._coordinates["y"]["label"] == "speed" and widget._coordinates["x"]["label"] == ""
    payload = widget._extended_axis_options_payload()
    assert payload["y_scale"] == "log" and payload["x_scale"] == "linear"


def test_copy_gives_x_y_and_z_the_shown_ticks_and_grid(qapp) -> None:
    widget = AxisPropertiesWidget()
    _select(widget, "y")
    widget._grid_combos["minor"].setCurrentIndex(widget._grid_combos["minor"].findData("on"))
    widget._minor_tick_length_spin.setValue(2.0)
    widget._coordinate_label.setText("only y")
    widget._copy_ticks_and_grid()
    payload = widget._extended_axis_options_payload()
    assert {payload[axis_options.grid_key(a, "minor")] for a in "xyz"} == {"on"}
    assert {payload[f"{a}_minor_tick_length"] for a in "xyz"} == {2.0}
    assert widget._coordinates["x"]["label"] == ""  # only ticks and grid are copied


def test_share_is_one_box_per_coordinate_in_properties(qapp) -> None:
    widget = AxisPropertiesWidget()
    assert set(widget._share_checks) == {"x", "y", "z"}


def test_minor_ticks_have_their_own_length_and_rotation(qapp) -> None:
    widget = AxisPropertiesWidget()
    _select(widget, "y")
    widget._minor_tick_length_spin.setValue(1.5)
    widget._tick_rotation_spin.setValue(30)
    payload = widget._extended_axis_options_payload()
    assert payload["y_minor_tick_length"] == 1.5 and payload["y_tick_rotation"] == 30.0
    assert payload["x_minor_tick_length"] is None

    from matplotlib.figure import Figure

    from app.charts.render_figure import _apply_tick_sizes

    ax = Figure().add_subplot()
    ax.minorticks_on()
    _apply_tick_sizes(ax, payload)
    assert ax.yaxis.get_minor_ticks()[0].tick1line.get_markersize() == 1.5


def test_the_scale_box_is_the_base_for_log_and_the_linear_zone_for_symlog(qapp) -> None:
    widget = AxisPropertiesWidget()
    widget._scale_combo.setCurrentIndex(widget._scale_combo.findData("symlog"))
    assert widget._scale_parameter.currentWidget() is widget._linthresh_spin
    widget._scale_combo.setCurrentIndex(widget._scale_combo.findData("log"))
    assert widget._scale_parameter.currentWidget() is widget._scale_base_spin


def test_the_renderer_is_found_by_its_own_name_not_the_translated_one(qapp) -> None:
    """An axis saved with options["renderer"] = "Diagramma di Pareto" still finds its renderer."""
    from types import SimpleNamespace

    widget = AxisPropertiesWidget()
    axis = SimpleNamespace(name="Pareto Chart", id=1, options={"renderer": "Diagramma di Pareto"})
    assert widget._axis_renderer(axis, axis.options) == "Pareto Chart"
