"""Baseline correction (app.analysis.baseline): AsLS and rubber band.

Both estimators are checked against a series built from a known baseline
plus known peaks: the estimate must track the baseline where there is no
peak, and must not be pulled up towards a peak where there is one.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import baseline as bl
from app.analysis.baseline import asls_baseline, rubber_band_baseline

X = np.linspace(0.0, 100.0, 400)
TRUE_BASELINE = 5.0 + 0.02 * X + 3.0 * np.sin(X / 30.0)
PEAKS = 10.0 * np.exp(-((X - 30.0) ** 2) / 4.0) + 15.0 * np.exp(-((X - 70.0) ** 2) / 8.0)
NO_PEAK = PEAKS < 0.05


def test_asls_tracks_a_wandering_baseline_away_from_peaks() -> None:
    rng = np.random.default_rng(0)
    y = TRUE_BASELINE + PEAKS + 0.05 * rng.standard_normal(X.size)
    fitted = asls_baseline(y, lam=1e5, p=0.01, iterations=10)
    assert fitted[NO_PEAK] == pytest.approx(TRUE_BASELINE[NO_PEAK], abs=0.3)


def test_asls_stays_below_the_peaks_it_is_meant_to_ignore() -> None:
    y = TRUE_BASELINE + PEAKS
    fitted = asls_baseline(y, lam=1e5, p=0.01)
    assert fitted[np.argmax(PEAKS)] < TRUE_BASELINE[np.argmax(PEAKS)] + 2.0  # not pulled up to the 15-high peak


def test_a_stiffer_baseline_bends_less() -> None:
    y = TRUE_BASELINE + PEAKS
    supple = asls_baseline(y, lam=1e2, p=0.01)
    stiff = asls_baseline(y, lam=1e9, p=0.01)
    assert np.std(np.diff(stiff, 2)) < np.std(np.diff(supple, 2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"lam": 0.0, "p": 0.01}, "lambda"), ({"lam": 1e5, "p": 0.0}, "between 0 and 1"),
     ({"lam": 1e5, "p": 1.0}, "between 0 and 1")],
)
def test_asls_refuses_bad_parameters(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        asls_baseline(np.arange(20.0), **kwargs)


def test_asls_needs_five_points() -> None:
    with pytest.raises(ValueError, match="at least 5 points"):
        asls_baseline(np.arange(4.0), lam=1e5, p=0.01)


def test_rubber_band_matches_a_flat_baseline_exactly() -> None:
    flat = 5.0 + 0.02 * X  # itself convex-hull-flat, so the hull is exact
    baseline = rubber_band_baseline(X, flat + PEAKS)
    assert baseline[NO_PEAK] == pytest.approx(flat[NO_PEAK], abs=1e-6)


def test_the_rubber_band_never_rises_above_the_data() -> None:
    y = TRUE_BASELINE + PEAKS + 0.05 * np.random.default_rng(1).standard_normal(X.size)
    assert np.all(rubber_band_baseline(X, y) <= y + 1e-9)


def test_correct_baseline_subtracts_it_and_reports_its_area() -> None:
    y = 4.0 + PEAKS
    result = bl.correct_baseline(bl.METHOD_RUBBER_BAND, X, y)
    np.testing.assert_allclose(result.corrected, y - result.baseline)
    # A flat baseline of 4 over a width of 100 has an area of 400.
    assert result.area == pytest.approx(400.0, rel=0.02)


def test_an_unknown_method_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported baseline method"):
        bl.correct_baseline("nope", X, X)
