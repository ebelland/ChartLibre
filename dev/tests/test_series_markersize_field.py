"""Series properties: the Marker size field reads and writes the series' markersize."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.widgets.chart_properties.series_properties import SeriesPropertiesWidget


@pytest.fixture
def widget(qapp: Any) -> SeriesPropertiesWidget:
    return SeriesPropertiesWidget()


def _payload(widget: SeriesPropertiesWidget) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []
    widget.series_options_requested.connect(seen.append)
    widget._emit_series_options_requested()
    return seen[-1]


def test_the_field_shows_the_series_markersize(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"markersize": 6.5}, sql_query=""))
    assert widget._markersize_spin.value() == 6.5
    assert _payload(widget)["markersize"] == 6.5


def test_a_series_without_one_shows_default_and_stores_nothing(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={}, sql_query=""))
    assert widget._markersize_spin.value() == 0.0
    assert widget._markersize_spin.text() == "Default"
    assert _payload(widget)["markersize"] is None  # None removes the key from the style


def test_changing_it_applies_the_series_style(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={}, sql_query=""))
    widget._markersize_spin.setValue(9.0)
    assert _payload(widget)["markersize"] == 9.0
