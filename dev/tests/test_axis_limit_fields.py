"""Axis limits are typed in plain fields, with an Auto check box beside each."""
from __future__ import annotations

import pytest

from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget, parse_limit


def test_typed_limits_are_read_as_numbers() -> None:
    assert parse_limit("0.5") == 0.5 and parse_limit("1e-3") == 0.001 and parse_limit("1,5") == 1.5
    assert parse_limit("-2") == -2.0
    assert parse_limit("") is None and parse_limit("abc") is None and parse_limit("inf") is None


@pytest.fixture
def widget(qapp):
    built = AxisPropertiesWidget()
    yield built
    built.deleteLater()


def test_auto_empties_and_disables_the_field_and_a_typed_value_is_kept(widget) -> None:
    edit, auto = widget._limit_edits["max"], widget._limit_auto_checks["max"]
    assert auto.text() == "Auto" and auto.isChecked() and not edit.isEnabled()
    auto.setChecked(False)
    assert edit.isEnabled()
    edit.setText("12.5")
    edit.editingFinished.emit()
    assert widget._coordinates[widget._coordinate_axis]["max"] == 12.5
    edit.setText("twelve")  # not a number: the limit it had stays
    edit.editingFinished.emit()
    assert edit.text() == "12.5" and widget._coordinates[widget._coordinate_axis]["max"] == 12.5
    auto.setChecked(True)
    assert edit.text() == "" and widget._read_field("max") is None
