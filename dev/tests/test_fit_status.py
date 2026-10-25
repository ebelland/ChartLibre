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
from app.data_operations.fit_dialog import SeriesFitDialog

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


def test_the_curve_is_drawn_smooth_and_past_the_data(dialog: SeriesFitDialog) -> None:
    """Fitted curve > Points and Extend: rows of the curve alone, the measured ones untouched."""
    dialog.chart_form.set_values({"curve_points": 101, "curve_extend": 10.0})
    _fit(dialog)
    frame = dialog.compute_results()[0].frame
    curve = frame[frame["y"].isna()]
    assert len(curve) == 101 and len(frame) - len(curve) == len(X)
    assert curve["x"].min() < X.min() and curve["x"].max() > X.max()
    np.testing.assert_allclose(curve["y_fit"], 10.0 / (1.0 + np.exp(-(curve["x"] - 3.0))), atol=1e-6)


def test_draw_on_offers_the_axis_a_new_axis_and_a_new_figure(dialog: SeriesFitDialog) -> None:
    combo = dialog.destination_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [
        dialog.DEST_SAME_AXIS, dialog.DEST_NEW_AXIS, dialog.DEST_NEW_FIGURE]
    assert combo.currentData() == dialog.DEST_SAME_AXIS


def test_a_fit_drawn_on_a_new_figure_takes_its_accessory_charts_and_cancel_takes_both(dialog: SeriesFitDialog) -> None:
    repo = dialog._repo
    before = {figure_id for figure_id, _name in repo.load_figures_from_db()}
    combo = dialog.destination_combo
    combo.setCurrentIndex(combo.findData(dialog.DEST_NEW_FIGURE))
    dialog.chart_form.set_values({"residual_chart": True, "fit_vs_measured_chart": True})
    _fit(dialog)
    made = {figure_id for figure_id, _name in repo.load_figures_from_db()} - before
    assert len(made) == 1
    titles = [axis.title for axis in repo.load_figure_descriptor(made.pop()).axes]
    assert "Residuals" in titles and "Measured vs. fit" in titles
    assert len(repo.load_figure_descriptor(dialog._figure_id).axes) == 1  # the source figure as it was
    dialog.reject()
    assert {figure_id for figure_id, _name in repo.load_figures_from_db()} == before


def test_cancel_takes_the_accessory_axes_even_when_their_ids_were_lost(dialog: SeriesFitDialog) -> None:
    dialog.chart_form.set_values({"residual_chart": True, "fit_vs_measured_chart": True})
    _fit(dialog)
    assert len(dialog._repo.load_figure_descriptor(dialog._figure_id).axes) == 3
    dialog._residual_axis_id = dialog._fit_vs_measured_axis_id = None  # as after Cancel left two empty axes
    dialog.reject()
    figure = dialog._repo.load_figure_descriptor(dialog._figure_id)
    assert len(figure.axes) == 1 and (figure.nrows, figure.ncols) == (1, 1)
