"""The smoothing engine (app.analysis.smoothing): answers known in advance.

No dialog, no Qt: each method is fed a signal whose smoothed form is known -
a ramp a moving average must leave alone, a parabola Savitzky-Golay must
reproduce exactly, a slow tone kept and a fast one removed - so a wrong
argument or a swapped axis shows up as a wrong number.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import smoothing as sm

RNG = np.random.default_rng(11)
X = np.linspace(0.0, 10.0, 201)


def _smooth(method: str, y: np.ndarray, **params) -> np.ndarray:
    return sm.smooth_1d(X, y, method, params)


# ----------------------------------------------------------------------
# 1D
# ----------------------------------------------------------------------
def test_a_moving_average_leaves_a_ramp_alone_away_from_the_ends() -> None:
    ramp = 3.0 * X + 1.0
    out = _smooth(sm.SMOOTH_MOVING_AVERAGE, ramp, window=7, centered=True)
    np.testing.assert_allclose(out[5:-5], ramp[5:-5])


def test_savitzky_golay_reproduces_a_parabola_and_its_derivative() -> None:
    parabola = 2.0 * X**2 - X + 4.0
    out = _smooth(sm.SMOOTH_SAVGOL, parabola, window=11, polyorder=2)
    np.testing.assert_allclose(out, parabola, atol=1e-8)

    step = float(X[1] - X[0])
    slope = _smooth(sm.SMOOTH_SAVGOL, parabola, window=11, polyorder=2, deriv=1, delta=step)
    np.testing.assert_allclose(slope[10:-10], 4.0 * X[10:-10] - 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "method",
    [sm.SMOOTH_GAUSSIAN, sm.SMOOTH_MEDIAN, sm.SMOOTH_MOVING_AVERAGE, sm.SMOOTH_KALMAN,
     sm.SMOOTH_FFT, sm.SMOOTH_WHITTAKER, sm.SMOOTH_SPLINE],
)
def test_a_constant_stays_constant(method: str) -> None:
    out = _smooth(method, np.full(X.size, 7.5))
    np.testing.assert_allclose(out, 7.5, atol=1e-6)


def test_a_median_filter_removes_a_single_spike() -> None:
    y = np.zeros(X.size)
    y[100] = 50.0
    out = _smooth(sm.SMOOTH_MEDIAN, y, kernel=5)
    assert out[100] == 0.0


def test_gaussian_smoothing_lowers_the_noise() -> None:
    noise = RNG.normal(0.0, 1.0, X.size)
    out = _smooth(sm.SMOOTH_GAUSSIAN, noise, sigma=3.0)
    assert np.std(out) < 0.4 * np.std(noise)


def test_kalman_smoother_gets_closer_to_the_truth_than_the_noisy_input() -> None:
    truth = np.sin(X)
    noisy = truth + RNG.normal(0.0, 0.3, X.size)
    out = _smooth(sm.SMOOTH_KALMAN, noisy, kalman_process_variance=1e-3, kalman_measurement_variance=0.09)
    assert np.mean((out - truth) ** 2) < 0.5 * np.mean((noisy - truth) ** 2)


def test_lowess_reproduces_a_straight_line() -> None:
    if not sm.is_available(sm.SMOOTH_LOWESS):
        pytest.skip("statsmodels not installed")
    line = 2.0 * X + 3.0
    np.testing.assert_allclose(_smooth(sm.SMOOTH_LOWESS, line, lowess_frac=0.3), line, atol=1e-6)


def test_the_hodrick_prescott_trend_of_a_line_is_the_line() -> None:
    if not sm.is_available(sm.SMOOTH_HP):
        pytest.skip("statsmodels not installed")
    line = 0.5 * X + 2.0
    np.testing.assert_allclose(_smooth(sm.SMOOTH_HP, line, hp_lambda=1600.0), line, atol=1e-6)


def test_whittaker_with_no_penalty_is_the_identity_and_a_huge_one_is_a_line() -> None:
    y = np.sin(X) + RNG.normal(0.0, 0.1, X.size)
    np.testing.assert_allclose(_smooth(sm.SMOOTH_WHITTAKER, y, whittaker_lambda=0.0), y)

    slope, intercept = np.polyfit(X, y, 1)
    stiff = _smooth(sm.SMOOTH_WHITTAKER, y, whittaker_lambda=1e12, whittaker_order=2)
    np.testing.assert_allclose(stiff, slope * X + intercept, atol=1e-3)


@pytest.mark.parametrize("method", [sm.SMOOTH_FFT, sm.SMOOTH_BUTTERWORTH])
def test_a_low_pass_keeps_the_slow_tone_and_drops_the_fast_one(method: str) -> None:
    n = 2000
    t = np.arange(n) / n  # one unit long; fs = n
    slow = np.sin(2 * np.pi * 3.0 * t)
    fast = 0.5 * np.sin(2 * np.pi * 200.0 * t)
    out = sm.smooth_1d(
        t, slow + fast, method,
        {"fft_cutoff_ratio": 0.1, "butter_fs": float(n), "butter_cutoff": 50.0, "butter_order": 4},
    )
    core = slice(200, -200)
    np.testing.assert_allclose(out[core], slow[core], atol=0.06)


def test_wavelet_denoising_lowers_the_error() -> None:
    if not sm.is_available(sm.SMOOTH_WAVELET):
        pytest.skip("PyWavelets not installed")
    truth = np.sin(X)
    noisy = truth + RNG.normal(0.0, 0.3, X.size)
    out = _smooth(sm.SMOOTH_WAVELET, noisy, wavelet="db4", wavelet_level=0, wavelet_threshold_factor=1.0)
    assert np.mean((out - truth) ** 2) < np.mean((noisy - truth) ** 2)


# ----------------------------------------------------------------------
# Bad input is a ValueError, not a dialog
# ----------------------------------------------------------------------
def test_too_few_points_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="3 finite points"):
        sm.clean_xy([1.0, 2.0], [1.0, 2.0])


def test_the_points_are_sorted_by_x_and_the_non_finite_ones_dropped() -> None:
    x, y = sm.clean_xy([3.0, 1.0, np.nan, 2.0, 4.0], [30.0, 10.0, 99.0, 20.0, 40.0])
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(y, [10.0, 20.0, 30.0, 40.0])


def test_x_and_y_of_different_lengths_are_a_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        sm.clean_xy([1.0, 2.0, 3.0], [1.0, 2.0])


def test_an_unknown_method_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported 1D"):
        sm.smooth_1d(X, X, "No Such Method", {})


# ----------------------------------------------------------------------
# 2D and 3D
# ----------------------------------------------------------------------
def test_a_2d_gaussian_leaves_a_plane_alone_inside() -> None:
    gx, gy = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    plane = 2.0 * gx - 3.0 * gy + 1.0
    _x, _y, z = sm.smooth_2d(gx, gy, plane, sm.SMOOTH2D_GAUSSIAN, {"sigma": 1.0})
    # The kernel is cut at 4 sigma, so a plane comes back to about 1e-5, not exactly.
    np.testing.assert_allclose(z[5:-5, 5:-5], plane[5:-5, 5:-5], atol=1e-4)


def test_a_gridded_only_method_refuses_scattered_data() -> None:
    x = RNG.uniform(size=50)
    y = RNG.uniform(size=50)
    with pytest.raises(ValueError, match="gridded"):
        sm.smooth_2d(x, y, x + y, sm.SMOOTH2D_GAUSSIAN, {})


@pytest.mark.parametrize(
    ("dimension", "method"),
    [(2, sm.SMOOTH2D_TV), (3, sm.SMOOTH3D_FFT), (3, sm.SMOOTH3D_TV)],
)
def test_gridded_only_methods_refuse_scattered_points(dimension: int, method: str) -> None:
    """These used to report the problem and then smooth the points as if they
    were one long vector - a result with no meaning, drawn as if it had."""
    if method in (sm.SMOOTH2D_TV, sm.SMOOTH3D_TV) and not sm.is_available(method):
        pytest.skip("scikit-image not installed")
    x, y, z = (RNG.uniform(size=120) for _ in range(3))
    values = np.sin(3 * x)
    with pytest.raises(ValueError, match="gridded"):
        if dimension == 2:
            sm.smooth_2d(x, y, values, method, {})
        else:
            sm.smooth_3d(x, y, z, values, method, {})


def test_a_3d_gaussian_keeps_a_constant_volume() -> None:
    grid = np.linspace(0, 1, 8)
    vol = np.full((8, 8, 8), 4.0)
    _x, _y, _z, values = sm.smooth_3d(grid, grid, grid, vol, sm.SMOOTH3D_GAUSSIAN, {"sigma": 1.0})
    np.testing.assert_allclose(values, 4.0)


# ----------------------------------------------------------------------
# The one entry point
# ----------------------------------------------------------------------
def test_smooth_series_sorts_a_1d_series_and_returns_its_x() -> None:
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    y = 2.0 * x
    out = sm.smooth_series(1, sm.SMOOTH_MOVING_AVERAGE, {"window": 1}, x=x, y=y)
    np.testing.assert_array_equal(out.x, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(out.y, [2.0, 4.0, 6.0, 8.0, 10.0])
    assert out.z is None and out.values is None


def test_smooth_series_says_what_a_2d_or_3d_series_lacks() -> None:
    with pytest.raises(ValueError, match="no Z values"):
        sm.smooth_series(2, sm.SMOOTH2D_GAUSSIAN, {}, x=X, y=X)
    with pytest.raises(ValueError, match="no 3D values"):
        sm.smooth_series(3, sm.SMOOTH3D_GAUSSIAN, {}, x=X, y=X, z=X)


def test_availability_follows_the_optional_packages() -> None:
    assert sm.is_available(sm.SMOOTH_GAUSSIAN)  # SciPy only: always
    assert sm.is_available(sm.SMOOTH_LOWESS) == (sm.sm_lowess is not None)
