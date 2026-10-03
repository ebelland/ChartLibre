"""Gaussian process regression (app.analysis.gaussian_process).

A smooth curve with a little noise: the posterior mean must follow the curve,
the band must be wider where there are no data than where there are, and
contain the truth most of the time. No dialog, no Qt.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from app.analysis import gaussian_process as gp
from dev.tests._cases import check_all

RNG = np.random.default_rng(6)
X = np.linspace(0.0, 10.0, 60)
TRUTH = np.sin(X)
Y = TRUTH + RNG.normal(0.0, 0.1, X.size)


@pytest.fixture(autouse=True)
def _quiet_convergence_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


_CASES_EVERY_KERNEL_FOLLOWS_A_SMOOTH_CURVE = [(case,) for case in [gp.KERNEL_RBF, gp.KERNEL_MATERN_32, gp.KERNEL_MATERN_52, gp.KERNEL_RATIONAL_QUADRATIC]]


def _every_kernel_follows_a_smooth_curve(kernel: str) -> None:
    fit = gp.fit_gaussian_process(kernel, X, Y)
    truth = np.sin(fit.x)
    assert np.mean((fit.mean - truth) ** 2) < 0.02
    assert fit.x.size == gp.GRID_POINTS
    assert np.all(fit.upper >= fit.mean) and np.all(fit.lower <= fit.mean)


def test_every_kernel_follows_a_smooth_curve() -> None:
    check_all(_every_kernel_follows_a_smooth_curve, _CASES_EVERY_KERNEL_FOLLOWS_A_SMOOTH_CURVE)


def test_the_band_is_two_sigmas_wide_and_holds_the_truth_most_of_the_time() -> None:
    fit = gp.fit_gaussian_process(gp.KERNEL_RBF, X, Y)
    inside = (np.sin(fit.x) >= fit.lower) & (np.sin(fit.x) <= fit.upper)
    assert np.mean(inside) > 0.85


def test_the_band_is_wider_where_there_are_no_data() -> None:
    gap = (X < 3.0) | (X > 6.0)  # nothing between 3 and 6
    fit = gp.fit_gaussian_process(gp.KERNEL_RBF, X[gap], Y[gap])
    width = fit.upper - fit.lower
    in_gap = (fit.x > 3.6) & (fit.x < 5.4)
    at_data = (fit.x < 1.5) | (fit.x > 8.0)
    assert np.mean(width[in_gap]) > 2.0 * np.mean(width[at_data])


def test_the_fit_reports_its_optimised_kernel_and_a_likelihood() -> None:
    fit = gp.fit_gaussian_process(gp.KERNEL_RBF, X, Y)
    assert "RBF" in fit.kernel and "WhiteKernel" in fit.kernel
    assert np.isfinite(fit.log_marginal_likelihood)


def test_the_kernel_is_a_base_kernel_plus_white_noise_and_an_unknown_name_is_rbf() -> None:
    kernel = gp.build_kernel("nope", length_scale=2.0, noise_level=0.5)
    assert repr(kernel) == "RBF(length_scale=2) + WhiteKernel(noise_level=0.5)"
    matern = gp.build_kernel(gp.KERNEL_MATERN_52, length_scale=1.0, noise_level=1.0)
    assert "nu=2.5" in repr(matern)


def test_the_fit_is_repeatable() -> None:
    first = gp.fit_gaussian_process(gp.KERNEL_MATERN_52, X, Y)
    second = gp.fit_gaussian_process(gp.KERNEL_MATERN_52, X, Y)
    np.testing.assert_array_equal(first.mean, second.mean)
