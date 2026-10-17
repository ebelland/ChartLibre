"""A click on the chart selects its figure, axis or series in Chart properties."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseEvent

from app.data.sqlite_repo import SqliteRepo
from app.main_window.main_window import MainWindow


def _spin(qapp, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)


def _click(panel, x_px: float, y_px: float) -> None:
    canvas = panel.canvas
    for name, handler in (("button_press_event", panel._on_object_click_press),
                          ("button_release_event", panel._on_object_click_release)):
        handler(MouseEvent(name, canvas, x_px, y_px, button=1))


def test_clicks_select_the_figure_the_axis_and_the_series(qapp, tmp_path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "c.dhub")
    x = np.linspace(0.0, 10.0, 11)
    repo.import_dataframe(pd.DataFrame({"x": x, "low": x, "high": x + 20.0}), table_name="t",
                          normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="clicks")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="t", x_label="x", y_label="y", options={})
    for index, column in enumerate(("low", "high")):
        repo.create_series_descriptor(axis_id=axis_id, series_index=index, name=column,
                                      sql_query=f'SELECT "x", "{column}" AS y FROM "t"',
                                      roles={"x": "x", "y": "y"}, style={})
    window = MainWindow(repo, tmp_path / "c.dhub")
    try:
        window.resize(1200, 760)
        window.show()
        window._left_panel.select("chart_properties")
        _spin(qapp, 0.6)
        panel = window._current_chart_panel()
        control = window._properties_control
        axes = panel._figure.axes[0]

        # On a point of the second series: the Series page, on that series.
        x_px, y_px = axes.transData.transform((5.0, 25.0))
        _click(panel, x_px, y_px)
        assert control.tabText(control.currentIndex()) == "Series"
        assert window._series_widget._series_combo.currentText().startswith("high")

        # Inside the axis, on nothing: the Axis page.
        x_px, y_px = axes.transData.transform((9.0, 2.0))
        _click(panel, x_px, y_px)
        assert control.tabText(control.currentIndex()) == "Axis"
        assert window._axis_widget.current_axis_id() == axis_id

        # Outside every axis: the Figure page.
        _click(panel, 2.0, 2.0)
        assert control.tabText(control.currentIndex()) == "Figure"

        # A colour edit in Series properties shows on the drawn points at once,
        # and is stored - with no render of the chart; a label edit renders it.
        from matplotlib.colors import to_hex

        series = window._series_widget
        series.select_series_named("high")
        stored = window._properties_series_descriptor(series._current_series_id)
        payload = {"series_id": stored.id, "color": "#ff0000", "sql_query": stored.sql_query}
        window._on_series_options_requested(payload)
        assert not window._properties_redraw_timer.isActive()
        points = next(a for a in axes.collections if getattr(a, "_dhub_series", None) == "high")
        assert to_hex(points.get_facecolor()[0]) == "#ff0000"
        assert window._properties_series_descriptor(series._current_series_id).options["color"] == "#ff0000"
        window._on_series_options_requested({**payload, "label": "High"})
        assert window._properties_redraw_timer.isActive()
    finally:
        window.close()
        repo.close()
