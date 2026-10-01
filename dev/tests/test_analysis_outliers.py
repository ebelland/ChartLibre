"""Outlier detectors (app.analysis.outliers): planted outliers, found."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import outliers as ol

RNG = np.random.default_rng(4)
#: x and y on comparable scales: the shape-aware detectors measure distances
#: in the (x, y) plane as given, without rescaling either axis.
X = np.linspace(0.0, 5.0, 100)
Y = 2.0 + RNG.normal(0.0, 0.1, X.size)
PLANTED = [17, 52, 88]
Y[PLANTED] += [3.0, -3.5, 2.5]


@pytest.mark.parametrize("method", [ol.OUTLIER_ZSCORE, ol.OUTLIER_IQR, ol.OUTLIER_MAD, ol.OUTLIER_ROLLING])
def test_the_y_detectors_find_exactly_the_planted_points(method: str) -> None:
    settings = ol.OutlierSettings(threshold=4.0, iqr_factor=3.0)
    assert np.flatnonzero(ol.outlier_mask(method, X, Y, settings)).tolist() == PLANTED


@pytest.mark.parametrize("method", sorted(ol.SHAPE_AWARE))
def test_the_shape_aware_detectors_flag_the_planted_points(method: str) -> None:
    # nu bounds the One-Class SVM's error fraction from above: kept small, so
    # the planted points are what it gives up.
    settings = ol.OutlierSettings(contamination=0.03, nu=0.01, n_neighbors=10)
    found = set(np.flatnonzero(ol.outlier_mask(method, X, Y, settings)).tolist())
    assert len(found & set(PLANTED)) >= 2
    assert len(found) <= 8


def test_a_flat_series_has_no_outliers() -> None:
    flat = np.full(30, 2.0)
    for method in (ol.OUTLIER_ZSCORE, ol.OUTLIER_IQR, ol.OUTLIER_MAD, ol.OUTLIER_ROLLING):
        assert not ol.outlier_mask(method, np.arange(30.0), flat).any()


def test_the_threshold_decides() -> None:
    loose = ol.outlier_mask(ol.OUTLIER_ZSCORE, X, Y, ol.OutlierSettings(threshold=1.0)).sum()
    strict = ol.outlier_mask(ol.OUTLIER_ZSCORE, X, Y, ol.OutlierSettings(threshold=5.0)).sum()
    assert loose > strict


def test_the_rolling_window_is_odd_and_fitted_to_the_series() -> None:
    assert ol.odd_window(10, 100) == 11
    assert ol.odd_window(51, 20) == 19
    assert ol.odd_window(1, 100) == 3


def test_a_shape_aware_detector_catches_a_point_off_the_curve_that_y_alone_misses() -> None:
    x = np.linspace(0, 10, 60)
    y = x.copy()
    y[30] = 2.0  # an ordinary y value, far from the line at that x
    assert not ol.outlier_mask(ol.OUTLIER_ZSCORE, x, y).any()
    lof = ol.outlier_mask(ol.OUTLIER_LOCAL_OUTLIER_FACTOR, x, y, ol.OutlierSettings(contamination=0.02, n_neighbors=5))
    assert lof[30]


def test_an_unknown_method_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported outlier"):
        ol.outlier_mask("nope", X, Y)
