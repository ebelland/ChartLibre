"""A data operation's Draw on changed after a Preview: what was drawn for the old choice - a new figure, a new axis -
is taken back at once, and the next Preview draws where now asked."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def test_switching_from_a_new_figure_takes_it_back(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.interpolate_dialog import SeriesInterpolateDialog

    x = np.linspace(0.0, 10.0, 15)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": np.sin(x)}), table_name="t", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="t", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query='SELECT x, y FROM "t"',
                                  roles={"x": "x", "y": "y"}, style={})
    before = {figure for figure, _name in repo.load_figures_from_db()}
    dialog = SeriesInterpolateDialog(repo=repo, figure_id=figure_id)
    shown: list[int] = []
    dialog.preview_drawn.connect(shown.append)
    try:
        dialog.series_selector.reload(select_all_series=True)
        combo = dialog.destination_combo

        def preview() -> None:
            dialog.preview()
            deadline = time.monotonic() + 30
            while dialog.job_running and time.monotonic() < deadline:
                qapp.processEvents()
                time.sleep(0.01)

        combo.setCurrentIndex(combo.findData(dialog.DEST_NEW_FIGURE))
        preview()
        made = {figure for figure, _name in repo.load_figures_from_db()} - before
        assert len(made) == 1 and shown[-1] == next(iter(made))
        combo.setCurrentIndex(combo.findData(dialog.DEST_NEW_AXIS))
        assert {figure for figure, _name in repo.load_figures_from_db()} == before and shown[-1] == -1
        preview()
        assert len(repo.load_figure_descriptor(figure_id).axes) == 2  # a new axis beside the data
    finally:
        dialog.close()
