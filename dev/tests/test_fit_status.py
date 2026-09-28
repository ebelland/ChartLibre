"""After Fit, OK saves the optimum - and its report says it is a fit."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QTableWidgetItem

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.fit_dialog import SeriesFitDialog

X = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.5, 6.0, 8.0])
#: A logistic curve: bottom 0, span 10, steepness 1, midpoint 3.
Y = 10.0 / (1.0 + np.exp(-(X - 3.0)))


@pytest.fixture
def dialog(qapp, tmp_db_path: Path) -> Iterator[SeriesFitDialog]:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.import_dataframe(pd.DataFrame({"x": X, "y": Y}), table_name="points", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="points", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="points",
        sql_query='SELECT "x", "y" FROM "points"', roles={"x": "x", "y": "y"}, style={},
    )
    built = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    built.series_selector.reload(select_all_series=True)
    yield built
    built.close()
    repo.close()
    applogger.set_status_bar(None)


def _fit(dialog: SeriesFitDialog) -> None:
    assert dialog.select_model("Logistic 4P")
    dialog.on_estimate_initial_values()
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)


def test_ok_after_fit_draws_the_optimum_and_reports_the_fit(dialog: SeriesFitDialog) -> None:
    _fit(dialog)
    result = dialog.compute_results()[0]
    np.testing.assert_allclose(result.params, [0.0, 10.0, 1.0, 3.0], atol=1e-6)
    assert result.message == dialog._last_fit[2]
    assert "Evaluated" not in result.message


def test_a_hand_edited_parameter_is_reported_as_an_evaluation(dialog: SeriesFitDialog) -> None:
    _fit(dialog)
    dialog._params_table.setItem(3, 1, QTableWidgetItem("2.5"))
    result = dialog.compute_results()[0]
    assert "Evaluated" in result.message
