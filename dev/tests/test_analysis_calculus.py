"""Calculus (app.analysis.calculus): derivatives and integrals with exact answers."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import calculus as calc
from dev.tests._cases import check_all

X = np.linspace(0.0, 2.0 * np.pi, 201)


_CASES_THE_DERIVATIVE_OF_SINE_IS_COSINE = [(case,) for case in [calc.DERIVATIVE_SAVGOL, calc.DERIVATIVE_GRADIENT, calc.DERIVATIVE_SPLINE]]


def _the_derivative_of_sine_is_cosine(method: str) -> None:
    derivative = calc.differentiate(method, X, np.sin(X), window=11, polyorder=3)
    np.testing.assert_allclose(derivative.values[10:-10], np.cos(X[10:-10]), atol=2e-3)


def test_the_derivative_of_sine_is_cosine() -> None:
    check_all(_the_derivative_of_sine_is_cosine, _CASES_THE_DERIVATIVE_OF_SINE_IS_COSINE)


def test_the_second_derivative_of_a_parabola_is_constant() -> None:
    x = np.linspace(-3, 3, 121)
    second = calc.differentiate(calc.DERIVATIVE_SAVGOL, x, 4.0 * x**2, order=2, window=9, polyorder=3)
    np.testing.assert_allclose(second.values[5:-5], 8.0, atol=1e-6)


def test_finite_differences_are_exact_on_a_line_even_when_unevenly_sampled() -> None:
    x = np.sort(np.random.default_rng(0).uniform(0, 10, 50))
    derivative = calc.differentiate(calc.DERIVATIVE_GRADIENT, x, 3.0 * x - 1.0)
    np.testing.assert_allclose(derivative.values, 3.0)


def test_the_savgol_window_is_fitted_to_the_series() -> None:
    assert calc.savgol_window(available=7, window=21, polyorder=3) == (7, 3)
    assert calc.savgol_window(available=8, window=21, polyorder=3) == (7, 3)  # odd, and no longer than the data
    assert calc.savgol_window(available=100, window=5, polyorder=9) == (5, 4)  # order below the window
    assert calc.savgol_window(available=100, window=1, polyorder=0) == (3, 1)


def test_a_series_with_no_spacing_says_so() -> None:
    spacing, note = calc.uniform_spacing(np.array([1.0, 1.0, 1.0]))
    assert spacing == 1.0 and "assuming 1" in note
    assert calc.uniform_spacing(np.array([0.0, 0.5, 1.0])) == (0.5, "")


def test_the_area_under_sine_over_half_a_period_is_two() -> None:
    x = np.linspace(0, np.pi, 101)  # odd: Simpson applies
    assert calc.definite_integral(x, np.sin(x), simpson_rule=True).total == pytest.approx(2.0, abs=1e-7)
    assert calc.definite_integral(x, np.sin(x)).total == pytest.approx(2.0, abs=2e-4)


def test_simpson_on_an_even_count_falls_back_to_the_trapezoid_and_says_so() -> None:
    x = np.linspace(0, np.pi, 100)
    area = calc.definite_integral(x, np.sin(x), simpson_rule=True)
    assert area.rule == "trapezoidal" and "odd number of points" in area.notes[0]


def test_the_running_integral_starts_at_zero_and_ends_at_the_area() -> None:
    x = np.linspace(0, 3, 301)
    running = calc.cumulative_integral(x, x**2)
    assert running[0] == 0.0 and running.size == x.size
    assert running[-1] == pytest.approx(9.0, abs=1e-3)


def test_baselines() -> None:
    x = np.linspace(0, 10, 11)
    peak = np.exp(-((x - 5) ** 2))
    sloped = peak + 2.0 + 0.3 * x
    corrected, detail = calc.subtract_baseline(x, sloped, calc.BASELINE_ENDPOINTS)
    assert detail == "endpoint line subtracted"
    np.testing.assert_allclose(corrected, peak - (peak[0] + (peak[-1] - peak[0]) * x / 10), atol=1e-12)
    lifted, detail = calc.subtract_baseline(x, peak + 4.0, calc.BASELINE_MINIMUM)
    assert lifted.min() == pytest.approx(0.0) and detail.startswith("minimum")
    assert calc.subtract_baseline(x, peak, calc.BASELINE_NONE)[1] == "none"


def test_dated_x_is_expressed_in_the_unit_of_its_spacing() -> None:
    daily = 1.7e9 + np.arange(30) * 86400.0
    assert calc.time_unit_of(daily) == ("day", 86400.0)
    assert calc.pick_time_unit(3600.0 * 0.6) == ("hour", 3600.0)
    assert calc.pick_time_unit(0.0) == ("second", 1.0)


def test_the_gradient_of_a_plane_is_its_slope_on_every_cell() -> None:
    gx, gy = np.meshgrid(np.linspace(0, 4, 21), np.linspace(-1, 1, 11))
    gradient = calc.surface_gradient(gx, gy, 2.0 * gx + 3.0 * gy)
    np.testing.assert_allclose(gradient.dz_dx, 2.0)
    np.testing.assert_allclose(gradient.dz_dy, 3.0)
    np.testing.assert_allclose(gradient.magnitude, np.sqrt(13.0))


def test_the_volume_under_a_plane_and_the_missing_cells() -> None:
    gx, gy = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 1, 11))
    assert calc.surface_volume(gx, gy, np.full_like(gx, 3.0)).total == pytest.approx(6.0)
    holed = np.full_like(gx, 3.0)
    holed[0, 0] = np.nan
    volume = calc.surface_volume(gx, gy, holed)
    assert volume.missing_cells == 1 and volume.total < 6.0


def test_an_unknown_derivative_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported derivative"):
        calc.differentiate("nope", X, X)
