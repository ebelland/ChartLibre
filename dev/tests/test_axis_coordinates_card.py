"""Axis properties: one coordinate card, applied to the X, Y, Z ticked."""
from __future__ import annotations

from app.charts import axis_options
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget


def _select(widget: AxisPropertiesWidget, *axes: str) -> None:
    for axis in axes:
        widget._target_checks[axis].setChecked(True)
    for axis, check in widget._target_checks.items():
        if axis not in axes:
            check.setChecked(False)


def test_an_edit_goes_to_the_ticked_coordinates_only(qapp) -> None:
    widget = AxisPropertiesWidget()
    widget._load_extended_axis_options({"y_scale": "log", axis_options.grid_key("x", "major"): "on"})
    assert widget._target_checks["x"].isChecked() and widget._grid_combos["major"].currentData() == "on"

    _select(widget, "y")
    assert widget._scale_combo.currentData() == "log"  # the card shows Y
    widget._coordinate_label.setText("speed")
    widget._share_check.setChecked(True)
    assert widget._coordinates["y"]["label"] == "speed" and widget._coordinates["x"]["label"] == ""

    _select(widget, "x", "y")
    widget._grid_combos["minor"].setCurrentIndex(widget._grid_combos["minor"].findData("on"))
    payload = widget._extended_axis_options_payload()
    assert payload[axis_options.grid_key("x", "minor")] == payload[axis_options.grid_key("y", "minor")] == "on"
    assert payload[axis_options.grid_key("z", "minor")] == axis_options.AUTO
    assert payload["y_scale"] == "log" and payload["x_scale"] == "linear"


def test_the_last_ticked_coordinate_cannot_be_unticked(qapp) -> None:
    widget = AxisPropertiesWidget()
    _select(widget, "z")
    widget._target_checks["z"].setChecked(False)
    assert widget._targets() == ["z"]


def test_the_renderer_is_found_by_its_own_name_not_the_translated_one(qapp) -> None:
    """An axis saved with options["renderer"] = "Diagramma di Pareto" still finds its renderer."""
    from types import SimpleNamespace

    widget = AxisPropertiesWidget()
    axis = SimpleNamespace(name="Pareto Chart", id=1, options={"renderer": "Diagramma di Pareto"})
    assert widget._axis_renderer(axis, axis.options) == "Pareto Chart"
