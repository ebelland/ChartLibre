"""Peak finding (app.analysis.peaks): peaks with known positions and widths.

A Gaussian of sigma s has a full width at half prominence of 2*sqrt(2*ln 2)*s
(about 2.355 s), so a measured width is a number to compare with, not just
"looks plausible". No dialog, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import peaks as pk

X = np.linspace(0.0, 100.0, 1001)


def _gauss(centre: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-((X - centre) ** 2) / (2 * sigma**2))


THREE = _gauss(20, 2, 1.0) + _gauss(50, 3, 2.0) + _gauss(80, 1.5, 1.5)
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def test_three_known_peaks_are_found_where_they_are() -> None:
    peaks = pk.find_peaks_1d(X, THREE, pk.MAXIMA)
    assert [round(peak.x) for peak in peaks] == [20, 50, 80]
    assert [peak.y for peak in peaks] == pytest.approx([1.0, 2.0, 1.5], abs=0.01)
    assert not any(peak.is_minimum for peak in peaks)


def test_the_width_is_the_width_at_half_prominence_and_its_bounds_bracket_the_peak() -> None:
    peak = pk.find_peaks_1d(X, THREE, pk.MAXIMA)[1]  # sigma 3
    assert peak.width == pytest.approx(FWHM * 3.0, rel=0.02)
    assert peak.left_x == pytest.approx(50 - FWHM * 3.0 / 2, abs=0.2)
    assert peak.right_x == pytest.approx(50 + FWHM * 3.0 / 2, abs=0.2)
    assert peak.prominence == pytest.approx(2.0, abs=0.01)


def test_minima_are_the_maxima_of_the_inverted_signal_and_both_come_back_in_x_order() -> None:
    signal = _gauss(30, 2, 1.0) - _gauss(60, 2, 1.2)  # one hill, one pit
    both = pk.find_peaks_1d(X, signal, pk.BOTH)
    assert [(round(peak.x), peak.is_minimum) for peak in both] == [(30, False), (60, True)]
    only_minima = pk.find_peaks_1d(X, signal, pk.MINIMA)
    assert [round(peak.x) for peak in only_minima] == [60]
    assert only_minima[0].y == pytest.approx(-1.2, abs=0.01)  # reported on the real signal, not the inverted one


def test_a_flat_series_has_no_peaks() -> None:
    assert pk.find_peaks_1d(X, np.ones_like(X), pk.BOTH) == []


def test_prominence_keeps_a_peak_on_a_rising_background_that_a_height_cut_loses() -> None:
    sloping = THREE + 0.01 * X  # the first peak's top (1.2) is low against the range (2.5)
    by_prominence = pk.find_peaks_1d(X, sloping, pk.MAXIMA, pk.PeakSettings(threshold=0.2))
    assert [round(peak.x) for peak in by_prominence] == [20, 50, 80]
    by_height = pk.find_peaks_1d(X, sloping, pk.MAXIMA, pk.PeakSettings(filter_by="height", threshold=0.6))
    assert [round(peak.x) for peak in by_height] == [50, 80]  # it compares against the floor, not the local ground


def test_a_higher_threshold_drops_the_small_peaks() -> None:
    # 80 % of the range (2.0): only the tallest, 2.0 high, stays.
    settings = pk.PeakSettings(threshold=0.8)
    assert [round(peak.x) for peak in pk.find_peaks_1d(X, THREE, pk.MAXIMA, settings)] == [50]


def test_the_minimum_width_drops_the_narrow_peaks() -> None:
    narrow = _gauss(30, 0.3, 1.0) + _gauss(70, 4.0, 1.0)
    wide_only = pk.find_peaks_1d(X, narrow, pk.MAXIMA, pk.PeakSettings(min_width=20.0))  # in samples: 10 units
    assert [round(peak.x) for peak in wide_only] == [70]


def test_the_distance_merges_close_neighbours_and_the_limit_keeps_the_most_prominent() -> None:
    twin = _gauss(40, 1.0, 1.0) + _gauss(43, 1.0, 0.9)
    assert len(pk.find_peaks_1d(X, twin, pk.MAXIMA, pk.PeakSettings(threshold=0.01))) == 2
    assert len(pk.find_peaks_1d(X, twin, pk.MAXIMA, pk.PeakSettings(threshold=0.01, distance=100))) == 1

    kept = pk.find_peaks_1d(X, THREE, pk.MAXIMA, pk.PeakSettings(limit=1))
    assert [round(peak.x) for peak in kept] == [50]  # the most prominent, not the first in x


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
GX, GY = np.meshgrid(np.linspace(-5, 5, 61), np.linspace(-5, 5, 61))


def test_a_bump_has_one_maximum_at_its_centre_and_no_width() -> None:
    bump = np.exp(-((GX - 1.0) ** 2 + (GY + 2.0) ** 2) / 2.0)
    peaks = pk.find_peaks_2d(GX, GY, bump, pk.MAXIMA, pk.PeakSettings(distance=5))
    assert len(peaks) == 1
    peak = peaks[0]
    assert (peak.x, peak.y) == pytest.approx((1.0, -2.0), abs=0.1)
    assert peak.z == pytest.approx(1.0, abs=0.01)
    assert np.isnan(peak.width) and np.isnan(peak.left_x)


def test_a_pit_is_a_minimum_and_two_bumps_give_two_maxima() -> None:
    pit = -np.exp(-(GX**2 + GY**2) / 2.0)
    assert [(p.x, p.y, p.is_minimum) for p in pk.find_peaks_2d(GX, GY, pit, pk.MINIMA, pk.PeakSettings(distance=5))] == [(0.0, 0.0, True)]
    two = np.exp(-((GX + 3) ** 2 + GY**2) / 2.0) + np.exp(-((GX - 3) ** 2 + GY**2) / 2.0)
    found = pk.find_peaks_2d(GX, GY, two, pk.MAXIMA, pk.PeakSettings(distance=5))
    assert [round(p.x) for p in found] == [-3, 3]


def test_cells_that_are_not_numbers_are_never_peaks_and_a_neighbouring_gap_disqualifies() -> None:
    bump = np.exp(-(GX**2 + GY**2) / 2.0)
    holed = bump.copy()
    holed[30, 30] = np.nan  # the very top of the bump
    assert pk.find_peaks_2d(GX, GY, holed, pk.MAXIMA, pk.PeakSettings(distance=5)) == []
    assert pk.find_peaks_2d(GX, GY, np.full_like(GX, np.nan), pk.MAXIMA) == []


def test_a_flat_surface_has_no_peaks_and_the_limit_applies_to_surfaces_too() -> None:
    assert pk.find_peaks_2d(GX, GY, np.ones_like(GX), pk.BOTH) == []
    rng = np.random.default_rng(3)
    noisy = rng.normal(size=GX.shape)
    assert len(pk.find_peaks_2d(GX, GY, noisy, pk.MAXIMA, pk.PeakSettings(threshold=0.01, limit=5))) == 5
