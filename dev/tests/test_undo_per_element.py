"""S-01: an edit in Chart properties is an Undo step of its own element and property, named for them - and an
Undo that takes back a colour, a line or a marker is shown on the chart's own lines, with no render."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from app.data.sqlite_repo import SqliteRepo
from app.main_window.main_window import MainWindow


def _spin(qapp, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)


def test_each_element_and_property_is_its_own_step_and_a_colour_comes_back_live(qapp, tmp_path,
                                                                               monkeypatch) -> None:
    repo = SqliteRepo(db_path=tmp_path / "u.dhub")
    x = np.linspace(0.0, 10.0, 11)
    repo.import_dataframe(pd.DataFrame({"x": x, "low": x, "high": x + 20.0}), table_name="t", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="undo")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="levels", x_label="x", y_label="y", options={})
    for index, column in enumerate(("low", "high")):
        repo.create_series_descriptor(axis_id=axis_id, series_index=index, name=column,
                                      sql_query=f'SELECT "x", "{column}" AS y FROM "t"',
                                      roles={"x": "x", "y": "y"}, style={})
    window = MainWindow(repo, tmp_path / "u.dhub")
    try:
        window.resize(1200, 760)
        window.show()
        window._left_panel.select("chart_properties")
        _spin(qapp, 0.6)
        panel = window._current_chart_panel()

        def edit(name: str, **changes) -> None:
            stored = next(window._properties_series_descriptor(i) for i, d in window._series_widget._series_map.items()
                          if d.name == name)
            window._on_series_options_requested({"series_id": stored.id, "sql_query": stored.sql_query, **changes})

        def labels() -> list[str]:
            return [entry.label for entry in repo.undo_entries()]

        edit("high", color="#ff0000")
        edit("high", color="#ee0000")  # the same property a moment later: the same step
        edit("high", markersize=9)
        edit("low", color="#ff0000")
        assert labels()[:3] == ["Colour of 'low'", "Marker size of 'high'", "Colour of 'high'"]

        window._last_snapshot_label = ""  # a while later: a step of its own
        edit("low", color="#0000ff")
        reloads = []
        monkeypatch.setattr(window, "_reload_tabs", lambda *args, **kwargs: reloads.append(args))
        window._on_undo()
        points = next(a for a in panel._canvas.figure.axes[0].collections if getattr(a, "_dhub_series", None) == "low")
        assert to_hex(points.get_facecolor()[0]) == "#ff0000" and not reloads
        assert window._current_chart_panel() is panel
    finally:
        window.close()
        repo.close()
