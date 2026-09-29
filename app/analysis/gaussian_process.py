"""Gaussian process regression: the arithmetic behind the GP operation.

A GP is the one smoother in the application that quantifies its own
uncertainty: fitting it gives a mean curve and a posterior standard
deviation at every point, so the result is three curves - the mean and a
+/-2 sigma band. Plain arrays in, no Qt, nothing logged (todo R-01).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern, RationalQuadratic, WhiteKernel

from app.analysis.regression import GRID_POINTS

KERNEL_RBF = "rbf"
KERNEL_MATERN_32 = "matern_1_5"
KERNEL_MATERN_52 = "matern_2_5"
KERNEL_RATIONAL_QUADRATIC = "rational_quadratic"

#: How many standard deviations the band flanks the mean by.
BAND_SIGMAS = 2.0


@dataclass(frozen=True, slots=True)
class GaussianProcessFit:
    """The posterior mean, its band, and how well the kernel explains the data."""

    x: np.ndarray
    mean: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    #: The kernel after its hyperparameters were optimised, as sklearn prints it.
    kernel: str
    #: nan when it cannot be computed (a diagnostic, never fatal).
    log_marginal_likelihood: float


def build_kernel(name: str, *, length_scale: float, noise_level: float) -> Kernel:
    """Return a base kernel plus additive white noise, ready to fit.

    The length_scale/noise_level are a starting guess, not a frozen choice:
    GaussianProcessRegressor re-optimises both (and every kernel
    hyperparameter) from here by maximising the marginal likelihood. An
    unknown *name* gets the RBF kernel.
    """
    if name == KERNEL_MATERN_32:
        base = Matern(length_scale=length_scale, nu=1.5)
    elif name == KERNEL_MATERN_52:
        base = Matern(length_scale=length_scale, nu=2.5)
    elif name == KERNEL_RATIONAL_QUADRATIC:
        base = RationalQuadratic(length_scale=length_scale)
    else:
        base = RBF(length_scale=length_scale)
    return base + WhiteKernel(noise_level=noise_level)


def fit_gaussian_process(
    kernel_name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    length_scale: float = 1.0,
    noise_level: float = 1.0,
) -> GaussianProcessFit:
    """Fit a GP to ``(x, y)`` and predict it over :data:`GRID_POINTS` points."""
    model = GaussianProcessRegressor(
        kernel=build_kernel(kernel_name, length_scale=length_scale, noise_level=noise_level),
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0,
    )
    model.fit(x.reshape(-1, 1), y)

    x_grid = np.linspace(float(x.min()), float(x.max()), GRID_POINTS)
    # With return_std=True, predict returns exactly (mean, std).
    mean, std = cast(tuple[np.ndarray, np.ndarray], model.predict(x_grid.reshape(-1, 1), return_std=True))

    try:
        log_likelihood = float(cast(float, model.log_marginal_likelihood()))
    except Exception:  # noqa: BLE001 - diagnostic only, never fatal
        log_likelihood = float("nan")

    return GaussianProcessFit(
        x=x_grid,
        mean=mean,
        upper=mean + BAND_SIGMAS * std,
        lower=mean - BAND_SIGMAS * std,
        kernel=repr(model.kernel_),
        log_marginal_likelihood=log_likelihood,
    )
