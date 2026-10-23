"""A data operation that fails says why in its Results pane, not only in the log (an X-bar chart on 9 points)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def test_a_failed_preview_says_why_in_the_results(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.control_chart_dialog import SeriesControlChartDialog

    repo.import_dataframe(pd.DataFrame({"x": np.arange(9.0), "y": np.arange(9.0) % 4}), table_name="t",
                          normalize_columns=False)
    figure = repo.create_figure_descriptor(name="f")
    axis = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Scatter Plot", title="",
                                       x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis, series_index=0, name="Readings", sql_query='SELECT x, y FROM "t"',
                                  roles={"x": "x", "y": "y"})
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure, parent=None)
    try:
        dialog.series_selector.select_all_series()
        dialog.model_combo.setCurrentText("X-bar and R")
        dialog.preview()
        deadline = time.monotonic() + 10
        while dialog.evaluating and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()
        label = dialog._results_label
        text = label.toPlainText() if hasattr(label, "toPlainText") else label.text()
        assert "at least 2 are needed" in text
    finally:
        dialog.close()
