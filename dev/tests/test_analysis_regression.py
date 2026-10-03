"""Regression (app.analysis.regression): robust and non-parametric fits.

A straight line with a few wild points: the robust fits must ignore the wild
points and an ordinary least-squares line must not. Isotonic must be
monotone, the forest and boosting must follow a curve. No dialog, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import regression as rg
from dev.tests._cases import check_all

RNG = np.random.default_rng(2)
X = np.linspace(0.0, 10.0, 120)
LINE = 2.0 * X + 1.0
WILD = LINE + RNG.normal(0.0, 0.2, X.size)
WILD[[10, 50, 90]] += 40.0  # three points far above the line


def _slope(fit: rg.Regression) -> float:
    return float(np.polyfit(fit.x, fit.y, 1)[0])


def test_the_fit_is_predicted_over_a_dense_grid_across_the_x_range() -> None:
    fit = rg.fit_regression(rg.KIND_RANSAC, X, WILD)
    assert fit.x.size == rg.GRID_POINTS
    assert fit.x[0] == X.min() and fit.x[-1] == X.max()
    assert fit.y.shape == fit.x.shape


_CASES_ROBUST_FITS_IGNORE_WILD_POINTS_BUT_LEAST_SQUARES_DOES_NOT = [(case,) for case in [rg.KIND_RANSAC, rg.KIND_HUBER]]


def _robust_fits_ignore_wild_points_but_least_squares_does_not(kind: str) -> None:
    fit = rg.fit_regression(kind, X, WILD)
    assert _slope(fit) == pytest.approx(2.0, abs=0.1)
    # Level of the line: robust stays on the true one, least squares is
    # pulled up by 3 points x 40 / 120 = 1.0.
    truth = float(np.mean(LINE))
    assert float(np.mean(fit.y)) == pytest.approx(truth, abs=0.3)
    assert float(np.mean(np.polyval(np.polyfit(X, WILD, 1), X))) == pytest.approx(truth + 1.0, abs=0.1)


def test_robust_fits_ignore_wild_points_but_least_squares_does_not() -> None:
    check_all(_robust_fits_ignore_wild_points_but_least_squares_does_not, _CASES_ROBUST_FITS_IGNORE_WILD_POINTS_BUT_LEAST_SQUARES_DOES_NOT)


def test_ransac_reports_its_inliers_and_huber_its_outliers() -> None:
    inliers = rg.fit_regression(rg.KIND_RANSAC, X, WILD).details["inliers"]
    kept, total = (int(part) for part in inliers.split("/"))
    assert total == X.size and 100 <= kept <= X.size - 3
    assert rg.fit_regression(rg.KIND_HUBER, X, WILD).details["outliers"] >= 3


def test_isotonic_is_monotone_and_can_be_forced_decreasing() -> None:
    noisy = X + RNG.normal(0.0, 1.5, X.size)
    up = rg.fit_regression(rg.KIND_ISOTONIC, X, noisy)
    assert np.all(np.diff(up.y) >= -1e-12)
    down = rg.fit_regression(rg.KIND_ISOTONIC, X, noisy, rg.RegressionSettings(increasing="false"))
    assert np.all(np.diff(down.y) <= 1e-12)
    assert "r2" not in up.details  # isotonic reports no score


_CASES_TREES_FOLLOW_A_CURVE_AND_REPORT_R2 = [(case,) for case in [rg.KIND_RANDOM_FOREST, rg.KIND_GRADIENT_BOOSTING]]


def _trees_follow_a_curve_and_report_r2(kind: str) -> None:
    curve = np.sin(X) * 3.0 + RNG.normal(0.0, 0.1, X.size)
    fit = rg.fit_regression(kind, X, curve, rg.RegressionSettings(n_estimators=60))
    assert fit.details["r2"] > 0.95
    truth = np.sin(fit.x) * 3.0
    assert np.mean((fit.y - truth) ** 2) < 0.15


def test_trees_follow_a_curve_and_report_r2() -> None:
    check_all(_trees_follow_a_curve_and_report_r2, _CASES_TREES_FOLLOW_A_CURVE_AND_REPORT_R2)


def test_the_tree_depth_setting_is_used() -> None:
    curve = np.sin(X) * 3.0
    shallow = rg.fit_regression(rg.KIND_RANDOM_FOREST, X, curve, rg.RegressionSettings(max_depth=1, n_estimators=20))
    deep = rg.fit_regression(rg.KIND_RANDOM_FOREST, X, curve, rg.RegressionSettings(max_depth=0, n_estimators=20))
    assert shallow.details["r2"] < deep.details["r2"]


def test_the_fit_is_repeatable() -> None:
    first = rg.fit_regression(rg.KIND_RANSAC, X, WILD)
    second = rg.fit_regression(rg.KIND_RANSAC, X, WILD)
    np.testing.assert_array_equal(first.y, second.y)


def test_an_unknown_kind_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported regression"):
        rg.fit_regression("nope", X, WILD)
