"""Switching chart tabs never leaves a canvas showing a draw made at another size.

The Figure properties panel writes the figure's export size and dpi onto the
figure it connects to on every tab switch; changing the dpi queues a draw at
once, and the chart panel used to put its own size back only on the next turn
of the event loop - after that draw. The canvas then showed a buffer of the
export size, anchored at a corner, until something else redrew it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.main_window.main_window import MainWindow


def _spin(qapp, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)


def test_switching_tabs_redraws_at_the_canvas_size(qapp, tmp_path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "t.dhub")
    repo.import_dataframe(pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0) ** 2}),
                          table_name="t", normalize_columns=False)
    for name in ("one", "two"):
        figure_id = repo.create_figure_descriptor(name=name)
        axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title=name, x_label="x", y_label="y", options={})
        repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=name,
                                      sql_query='SELECT "x", "y" FROM "t"', roles={"x": "x", "y": "y"}, style={})
    window = MainWindow(repo, tmp_path / "t.dhub")
    try:
        window.resize(1100, 700)
        window.show()
        _spin(qapp, 0.5)
        for index in range(window._tabs.count()):
            window._tabs.setCurrentIndex(index)
            _spin(qapp, 0.2)
            window._current_chart_panel().set_resize_mode("FIT")
            _spin(qapp, 0.3)
        for index in (1, 0, 1):
            window._tabs.setCurrentIndex(index)
            _spin(qapp, 0.4)
            canvas = window._current_chart_panel().canvas
            ratio = canvas.device_pixel_ratio
            shown = canvas.renderer.buffer_rgba().shape[1::-1]
            assert shown == (round(canvas.width() * ratio), round(canvas.height() * ratio))
    finally:
        window.close()
        repo.close()
