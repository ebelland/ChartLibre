"""Fit weights from the series' error column (R-05)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.data_operations.fit_dialog import SeriesFitDialog
from app.data_operations.fit_dialog.model_data import _merge_points_with_errors

RNG = np.random.default_rng(3)
X = np.linspace(0.0, 10.0, 30)
SIGMA = 0.1 + 0.3 * X
Y = 2.0 * X + 1.0 + RNG.normal(size=X.size) * SIGMA


@pytest.fixture
def dialog(qapp, tmp_db_path: Path) -> Iterator[SeriesFitDialog]:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.import_dataframe(pd.DataFrame({"x": X, "y": Y, "e": SIGMA}), table_name="points", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="points", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="points",
        sql_query='SELECT "x", "y", "e" FROM "points"', roles={"x": "x", "y": "y", "yerr": "e"}, style={},
    )
    built = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    built.series_selector.reload(select_all_series=True)
    yield built
    built.close()
    repo.close()
    applogger.set_status_bar(None)


def test_error_bars_weigh_the_fit_as_curve_fit_does(dialog: SeriesFitDialog) -> None:
    assert dialog.select_model("Linear")
    dialog._combo_weights.setCurrentIndex(dialog._combo_weights.findData("errors"))
    dialog.on_estimate_initial_values()
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)
    result = dialog._last_result
    expected, covariance = curve_fit(lambda x, a, b: a * x + b, X, Y, sigma=SIGMA)
    np.testing.assert_allclose(sorted(result.params), sorted(expected), rtol=1e-6)
    np.testing.assert_allclose(sorted(result.param_std), sorted(np.sqrt(np.diag(covariance))), rtol=1e-4)
    assert "chi2_reduced" in result.metrics


def test_repeated_x_merge_into_their_weighted_mean() -> None:
    x, y, sigma = _merge_points_with_errors(
        np.array([1.0, 0.0, 1.0]), np.array([4.0, 9.0, 2.0]), np.array([1.0, 1.0, 2.0]), "s"
    )
    np.testing.assert_allclose(x, [0.0, 1.0])
    np.testing.assert_allclose(y, [9.0, (4.0 + 2.0 / 4.0) / 1.25])
    np.testing.assert_allclose(sigma, [1.0, 1.0 / np.sqrt(1.25)])
