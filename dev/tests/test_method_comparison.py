"""ID12 Method Comparison: Bland-Altman agreement, Deming and Passing-Bablok regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import method_comparison as mc
from app.data.sqlite_repo import SqliteRepo
from dev.tests._helpers import wait_for_job


def _methods(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1, 100, n)
    return truth + rng.normal(0, 2, n), 2.0 + 1.05 * truth + rng.normal(0, 2, n)


def test_bland_altman_bias_limits_and_their_intervals() -> None:
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = x + np.array([1.0, 2.0, 0.0, 1.0, 1.0])
    a = mc.bland_altman(y, x)
    assert a.bias == pytest.approx(1.0) and a.sd == pytest.approx(np.sqrt(0.5))
    assert (a.lower, a.upper) == pytest.approx((1 - 1.959964 * np.sqrt(0.5), 1 + 1.959964 * np.sqrt(0.5)))
    t = 2.776445  # t(0.975, 4)
    assert a.bias_ci == pytest.approx((1 - t * np.sqrt(0.5) / np.sqrt(5), 1 + t * np.sqrt(0.5) / np.sqrt(5)))


def test_deming_with_equal_errors_is_the_orthogonal_regression() -> None:
    """With an error ratio of 1 Deming is the first principal axis of the (x, y) cloud."""
    x, y = _methods()
    line = mc.deming(x, y, ratio=1.0)
    _values, vectors = np.linalg.eigh(np.cov(x, y))
    assert line.slope == pytest.approx(vectors[1, 1] / vectors[0, 1])
    assert line.intercept == pytest.approx(np.mean(y) - line.slope * np.mean(x))
    assert line.slope_ci[0] < 1.05 < line.slope_ci[1]


def test_passing_bablok_recovers_a_line_through_an_outlier() -> None:
    x = np.arange(1.0, 31.0)
    y = 3.0 + 2.0 * x
    y[5] = 500.0  # one gross error: ordinary least squares would follow it
    line = mc.passing_bablok(x, y)
    assert line.slope == pytest.approx(2.0) and line.intercept == pytest.approx(3.0)
    noisy_x, noisy_y = _methods()
    noisy = mc.passing_bablok(noisy_x, noisy_y)
    assert noisy.slope_ci[0] < 1.05 < noisy.slope_ci[1] and not noisy.row()["Slope 1 inside"]


def test_the_operation_reports_and_draws_both_plots(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.method_comparison_dialog import MethodComparisonDialog

    x, y = _methods(45)
    repo.import_dataframe(pd.DataFrame({"reference": x, "new": y}), table_name="glucose", normalize_columns=False)
    dialog = MethodComparisonDialog(repo=repo, table="glucose", parent=None)
    try:
        dialog.roles_widget.set_casting({"response": ["new"], "factor": ["reference"]})
        dialog.ok()
        wait_for_job(qapp, dialog)
        (figure_id,) = dialog.created_figure_ids
    finally:
        dialog.close()
    titles = [axis.title for axis in repo.load_figure_descriptor(figure_id).axes]
    assert titles == ["Bland-Altman", "new against reference"]
    assert {"_glucose_comparison", "_glucose_comparison_lines"} <= set(repo.list_table_names())
