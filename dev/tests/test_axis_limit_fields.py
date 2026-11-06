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


def test_min_max_and_interval_are_three_rows_each_with_its_auto(widget) -> None:
    from PySide6.QtWidgets import QGridLayout, QLabel

    rows: dict[str, int] = {}
    for grid in widget.findChildren(QGridLayout):  # the Scale card's grid: the one that holds these labels
        for index in range(grid.count()):
            item = grid.itemAt(index).widget()
            if isinstance(item, QLabel) and item.text() in ("Min", "Max", "Interval"):
                rows[item.text()] = grid.getItemPosition(index)[0]
    assert rows["Max"] == rows["Min"] + 1 and rows["Interval"] == rows["Max"] + 1
    for edge in ("min", "max", "interval"):
        assert widget._limit_auto_checks[edge].isChecked()  # all automatic at first


def test_an_interval_is_kept_sent_and_refused_when_not_a_step(widget) -> None:
    from app.plugin_api.charts import axis_options

    edit, auto = widget._limit_edits["interval"], widget._limit_auto_checks["interval"]
    auto.setChecked(False)
    edit.setText("2,5")
    edit.editingFinished.emit()
    assert widget._coordinates[widget._coordinate_axis]["interval"] == 2.5
    assert widget._extended_axis_options_payload()[axis_options.interval_key(widget._coordinate_axis)] == 2.5
    edit.setText("0")  # no step: the interval it had stays
    edit.editingFinished.emit()
    assert edit.text() == "2.5" and widget._coordinates[widget._coordinate_axis]["interval"] == 2.5
    widget._scale_combo.setCurrentIndex(widget._scale_combo.findData("log"))
    assert not auto.isEnabled() and not edit.isEnabled()  # a log scale keeps its own ticks
    widget._scale_combo.setCurrentIndex(widget._scale_combo.findData("linear"))
    assert auto.isEnabled() and edit.isEnabled()


def test_a_figure_opens_with_its_interval(widget) -> None:
    widget._load_extended_axis_options({"x_interval": 10, "y_interval": "-1"})
    assert widget._coordinates["x"]["interval"] == 10.0
    assert widget._coordinates["y"]["interval"] is None  # no step: automatic
