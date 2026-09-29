"""Baseline correction: the arithmetic behind the Baseline operation.

Two estimators of the slow background under a peaky series - asymmetric
least squares (Eilers & Boelens 2005) and the rubber band (the lower
convex hull) - over plain arrays. No Qt, nothing logged (todo R-01); a bad
parameter or too short a series is a ``ValueError``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

METHOD_ASLS = "asls"
METHOD_RUBBER_BAND = "rubber_band"

#: Below this the second-difference penalty matrix has nothing to act on.
ASLS_MINIMUM_POINTS = 5


def asls_baseline(y: np.ndarray, lam: float, p: float, iterations: int = 10) -> np.ndarray:
    """Eilers & Boelens' asymmetric least squares baseline.

    Fits ``z`` to minimise ``sum(w * (y - z)^2) + lam * sum(diff(z, 2)^2)``,
    re-weighting after each solve so points above the current curve count
    for only ``p`` (a peak should not pull the baseline up towards it) and
    points below count for ``1 - p`` (the background should).
    """
    size = int(np.asarray(y).size)
    if size < ASLS_MINIMUM_POINTS:
        raise ValueError(
            f"AsLS needs at least {ASLS_MINIMUM_POINTS} points, got {size}"
        )
    if lam <= 0.0:
        raise ValueError("lambda must be positive")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be between 0 and 1")

    y = np.asarray(y, dtype=float)
    # The discrete second-difference operator: (D @ D.T) penalises curvature.
    # scipy does not annotate diags; Pylance guesses offsets is an int from
    # its default (0), but a list of offsets is what it takes.
    diagonals = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(size, size - 2))  # pyright: ignore[reportArgumentType]
    penalty = float(lam) * diagonals.dot(diagonals.transpose())

    weights = np.ones(size)
    fitted = y.copy()
    smoother = sparse.spdiags(weights, 0, size, size)
    for _iteration in range(max(1, int(iterations))):
        smoother.setdiag(weights)
        fitted = spsolve((smoother + penalty).tocsc(), weights * y)
        weights = p * (y > fitted) + (1.0 - p) * (y <= fitted)
    return np.asarray(fitted, dtype=float)


def rubber_band_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The lower convex hull of ``(x, y)``, linearly interpolated at every x.

    ``x`` must already be sorted (the caller guarantees this via
    ``prepare_input_xy``); the hull is built with the standard monotone-
    chain algorithm restricted to its lower half.
    """
    size = x.size
    hull: list[int] = []
    for index in range(size):
        while len(hull) >= 2:
            ox, oy = x[hull[-2]], y[hull[-2]]
            ax, ay = x[hull[-1]], y[hull[-1]]
            bx, by = x[index], y[index]
            # <= 0: the last hull point does not turn left of O->new - it is
            # above the segment, so it cannot be part of a *lower* hull.
            cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(index)
    return np.interp(x, x[hull], y[hull])


@dataclass(frozen=True, slots=True)
class BaselineCorrection:
    """The estimated background, the series with it taken off, and its area."""

    baseline: np.ndarray
    corrected: np.ndarray
    #: Area under the baseline (trapezoid rule over x).
    area: float


def correct_baseline(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = 1e5,
    p: float = 0.01,
    iterations: int = 10,
) -> BaselineCorrection:
    """Estimate the baseline of ``(x, y)`` with *method* and subtract it.

    *lam*, *p* and *iterations* are AsLS's; the rubber band takes none.
    """
    if method == METHOD_ASLS:
        baseline = asls_baseline(y, lam=lam, p=p, iterations=iterations)
    elif method == METHOD_RUBBER_BAND:
        baseline = rubber_band_baseline(x, y)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    return BaselineCorrection(
        baseline=baseline,
        corrected=y - baseline,
        area=float(np.trapezoid(baseline, x)),
    )
