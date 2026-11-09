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


def test_marker_every_is_read_and_written(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"markevery": 5}, sql_query=""))
    assert widget._markevery_spin.value() == 5 and _payload(widget)["markevery"] == 5
    widget._markevery_spin.setValue(1)
    assert _payload(widget)["markevery"] is None


def test_sort_by_x_is_the_querys_order_by() -> None:
    from app.widgets.chart_properties.series_properties import is_sorted, with_sort, without_sort

    sql = 'SELECT a AS x, b AS y FROM t WHERE a IN (SELECT a FROM u ORDER BY a)'
    assert not is_sorted(sql)
    sorted_sql = with_sort(sql, "x")
    assert sorted_sql.endswith('ORDER BY "x"') and is_sorted(sorted_sql)
    assert without_sort(sorted_sql) == sql


def test_the_sort_box_edits_the_query_and_the_marker_colour(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"markercolor": "#ff0000"},
                                                   sql_query="SELECT a AS x FROM t", roles={"x": "x"}))
    assert not widget._marker_same_colour.isChecked()
    assert _payload(widget)["markercolor"].lower() == "#ff0000"
    widget._sort_x_check.setChecked(True)
    assert widget._sql_query_edit.toPlainText().endswith('ORDER BY "x"')
    widget._marker_same_colour.setChecked(True)
    assert _payload(widget)["markercolor"] is None


def test_default_groups_and_line_width(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"linestyle": "default", "marker": "default"},
                                                   sql_query=""))
    widget._set_editing_enabled(True) if hasattr(widget, "_set_editing_enabled") else None
    assert widget._line_default.isChecked() and widget._marker_default.isChecked()
    payload = _payload(widget)
    assert payload["linestyle"] == "default" and payload["marker"] == "default" and payload["linewidth"] is None
    widget._line_default.setChecked(False)
    widget._linewidth_spin.setValue(2.5)
    assert _payload(widget)["linewidth"] == 2.5


def test_unticking_default_stores_a_style_of_its_own(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"linestyle": "default", "marker": "default"},
                                                   sql_query=""))
    widget._line_default.setChecked(False)
    widget._marker_default.setChecked(False)
    payload = _payload(widget)
    assert payload["linestyle"] not in ("default", "") and payload["marker"] not in ("default", "")


def test_unticking_same_colour_stores_a_colour(widget: SeriesPropertiesWidget) -> None:
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"marker": "o"}, sql_query=""))
    widget._marker_same_colour.setChecked(False)
    assert _payload(widget)["markercolor"]
