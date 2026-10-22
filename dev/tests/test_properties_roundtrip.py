"""Opening a series or an axis in the properties panels and changing one field
must not change the others (P0-13, P0-14).

The panels show what a combo can show and write back what they show, so every
value the combo could not represent - a chosen "None", a dash pattern, "C0", an
auto label held in the descriptor's column - was replaced on the next edit.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget
from app.widgets.chart_properties.series_properties import SeriesPropertiesWidget


@pytest.fixture
def series_widget(qapp: Any) -> SeriesPropertiesWidget:
    return SeriesPropertiesWidget()


@pytest.fixture
def axis_widget(qapp: Any) -> AxisPropertiesWidget:
    return AxisPropertiesWidget()


def _series_payload(widget: SeriesPropertiesWidget, style: dict[str, Any]) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []
    widget.series_options_requested.connect(seen.append)
    widget._load_series_descriptor(SimpleNamespace(id=1, options=style, sql_query=""))
    widget._emit_series_options_requested()
    return seen[-1]


@pytest.mark.parametrize("linestyle", ["", "none", "None", "--", ":", "default", (0, (1, 1))])
@pytest.mark.parametrize("marker", ["", "None", "o", "1", "default"])
def test_line_style_and_marker_survive_an_edit_of_another_field(
    series_widget: SeriesPropertiesWidget, linestyle: Any, marker: str
) -> None:
    payload = _series_payload(series_widget, {"linestyle": linestyle, "marker": marker})
    assert payload["linestyle"] == linestyle
    assert payload["marker"] == marker


@pytest.mark.parametrize("color", ["", "red", "#ff0000", "tab:blue", "C0", "C3", "0.5", "#abc", "#ff000080"])
def test_colour_survives_an_edit_of_another_field(series_widget: SeriesPropertiesWidget, color: str) -> None:
    assert _series_payload(series_widget, {"color": color})["color"] == color


def test_a_large_markersize_is_not_clipped(series_widget: SeriesPropertiesWidget) -> None:
    assert _series_payload(series_widget, {"markersize": 150})["markersize"] == 150


def test_touching_the_combo_writes_the_new_choice(series_widget: SeriesPropertiesWidget) -> None:
    seen: list[dict[str, Any]] = []
    series_widget.series_options_requested.connect(seen.append)
    series_widget._load_series_descriptor(SimpleNamespace(id=1, options={"marker": "", "linestyle": ""}, sql_query=""))
    series_widget._marker_combo.setCurrentIndex(series_widget._marker_combo.findData("o"))
    series_widget._linestyle_combo.setCurrentIndex(series_widget._linestyle_combo.findData("--"))
    series_widget._emit_series_options_requested()
    assert (seen[-1]["marker"], seen[-1]["linestyle"]) == ("o", "--")


def _axis_payload(widget: AxisPropertiesWidget, options: dict[str, Any], **columns: str) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []
    widget.axis_options_requested.connect(seen.append)
    descriptor = SimpleNamespace(
        id=1, chart_type="Scatter Plot", title="T", options=options,
        **{"x_label": "", "y_label": "", "z_label": "", **columns},
    )
    widget._load_axis_descriptor(descriptor)
    widget._emit_axis_options_requested()
    return seen[-1]


def test_a_label_held_in_the_descriptor_column_is_not_blanked(axis_widget: AxisPropertiesWidget) -> None:
    """AMC Growth: the x label is the data column's name, in the column only."""
    payload = _axis_payload(axis_widget, {}, x_label="Time_min", y_label="ThicknessChange_nm")
    assert (payload["x_label"], payload["y_label"]) == ("Time_min", "ThicknessChange_nm")


def test_an_option_label_wins_over_the_column_and_an_empty_one_stays_empty(axis_widget: AxisPropertiesWidget) -> None:
    payload = _axis_payload(axis_widget, {"x_label": "Mine", "y_label": ""}, x_label="Time_min", y_label="Thickness")
    assert (payload["x_label"], payload["y_label"]) == ("Mine", "")


def test_a_series_without_a_line_style_key_stays_without_one(series_widget: SeriesPropertiesWidget) -> None:
    """Absent means "no line" (a scatter's markers); "default" means the style's line (issue #6)."""
    payload = _series_payload(series_widget, {"label": "Signal"})
    assert payload["linestyle"] is None  # the handler removes a None key: nothing is written
    assert payload["marker"] is None

