"""Calculus: the arithmetic behind the Calculus series operation.

Derivatives of a sampled curve (Savitzky-Golay, finite difference, smoothing
spline), its running and definite integrals (trapezoid, Simpson) with an
optional baseline taken off first, and the gradient and the volume of a
gridded surface. Plain arrays in, no Qt, nothing logged (todo R-01): what the
caller should tell the user comes back as a note.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

DERIVATIVE_SAVGOL = "savgol"
DERIVATIVE_GRADIENT = "gradient"
DERIVATIVE_SPLINE = "spline"

BASELINE_NONE = "none"
BASELINE_MINIMUM = "minimum"
BASELINE_ENDPOINTS = "endpoints"

# A dated series' x arrives as seconds since the epoch - correct to
# differentiate or integrate against, but a derivative "per second" of daily
# data is ~86400x smaller than what a person looking at the chart expects,
# and a definite integral is inflated by the same factor. The coarsest unit
# that keeps the median sample spacing at 0.5 or more of it turns "per
# second" into "per day" for the common case without asking for the cadence.
TIME_UNITS: tuple[tuple[str, float], ...] = (
    ("year", 365.25 * 86400.0),
    ("week", 7.0 * 86400.0),
    ("day", 86400.0),
    ("hour", 3600.0),
    ("minute", 60.0),
    ("second", 1.0),
)


def pick_time_unit(median_spacing_seconds: float) -> tuple[str, float]:
    """Return (unit name, seconds per unit) for a median sample spacing."""
    if median_spacing_seconds > 0:
        for name, seconds in TIME_UNITS:
            if median_spacing_seconds >= seconds * 0.5:
                return name, seconds
    return "second", 1.0


def time_unit_of(x_seconds: np.ndarray) -> tuple[str, float]:
    """The unit to express a dated x in, from its median spacing."""
    steps = np.diff(x_seconds)
    return pick_time_unit(float(np.median(steps)) if steps.size else 0.0)


# ----------------------------------------------------------------------
# Derivatives
# ----------------------------------------------------------------------
def savgol_window(available: int, window: int, polyorder: int) -> tuple[int, int]:
    """Return a (window, polyorder) savgol_filter will actually accept.

    Both of its constraints are reported from inside SciPy in terms of array
    shapes rather than of the controls the user moved, so they are resolved
    here: the window cannot exceed the series, and the polynomial order must
    be below the window.
    """
    window = int(window)
    polyorder = int(polyorder)
    if window > available:
        window = available if available % 2 == 1 else available - 1
    window = max(3, window)
    if polyorder >= window:
        polyorder = window - 1
    return window, max(1, polyorder)


def uniform_spacing(x: np.ndarray) -> tuple[float, str]:
    """The sample spacing savgol should assume, and a note when there is none.

    savgol_filter takes a single delta, so it can only be right for evenly
    spaced data; the median step is the best single answer for data that is
    nearly even.
    """
    steps = np.diff(x)
    if steps.size == 0:
        return 1.0, ""
    spacing = float(np.median(steps))
    if spacing <= 0.0:
        return 1.0, "could not determine a sample spacing; assuming 1."
    return spacing, ""


@dataclass(frozen=True, slots=True)
class Derivative:
    values: np.ndarray
    #: How it was computed, for the report: "central difference", "window 11, order 3"...
    detail: str
    notes: tuple[str, ...] = ()


def differentiate(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 1,
    window: int = 11,
    polyorder: int = 3,
    smoothing: float = 0.0,
) -> Derivative:
    """The *order*-th derivative of y(x) at every x, by *method*."""
    if method == DERIVATIVE_GRADIENT:
        # np.gradient, not np.diff: it takes x explicitly, so it is correct on
        # unevenly sampled data, and it returns one value per input point.
        return Derivative(np.asarray(np.gradient(y, x, edge_order=2), dtype=float), "central difference")

    if method == DERIVATIVE_SPLINE:
        spline = UnivariateSpline(x, y, k=min(5, max(order + 1, 3)), s=float(smoothing))
        return Derivative(
            np.asarray(spline.derivative(n=order)(x), dtype=float), f"spline, s={smoothing}"
        )

    if method != DERIVATIVE_SAVGOL:
        raise ValueError(f"Unsupported derivative: {method}")
    window, polyorder = savgol_window(x.size, window, polyorder)
    spacing, note = uniform_spacing(x)
    # delta scales the result into units of y per unit of x. Without it savgol
    # returns a derivative per sample index, off by the sampling interval.
    values = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=order, delta=spacing)
    return Derivative(
        np.asarray(values, dtype=float), f"window {window}, order {polyorder}", (note,) if note else ()
    )


# ----------------------------------------------------------------------
# Integrals
# ----------------------------------------------------------------------
def subtract_baseline(x: np.ndarray, y: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    """Return the signal with its baseline removed, and what was done."""
    if mode == BASELINE_MINIMUM:
        floor = float(np.min(y))
        return y - floor, f"minimum ({floor:g}) subtracted"

    if mode == BASELINE_ENDPOINTS:
        if x.size < 2:
            return y, "none"
        # The straight line through the first and last points: the usual
        # approximation for a peak sitting on a sloping background.
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        line = y[0] + slope * (x - x[0])
        return y - line, "endpoint line subtracted"

    return y, "none"


def cumulative_integral(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The running integral, one value per point, starting at zero."""
    return np.asarray(cumulative_trapezoid(y, x, initial=0.0), dtype=float)


@dataclass(frozen=True, slots=True)
class Area:
    total: float
    #: "Simpson" or "trapezoidal".
    rule: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def definite_integral(x: np.ndarray, y: np.ndarray, *, simpson_rule: bool = False) -> Area:
    """The area under y(x), by Simpson's rule if asked and possible, else the trapezoid."""
    if simpson_rule and x.size % 2 == 0:
        # Simpson's rule pairs intervals, so it needs an odd number of points.
        # SciPy silently changes method on an even sample rather than saying
        # so, which would make the reported rule wrong.
        return Area(
            float(trapezoid(y, x)),
            "trapezoidal",
            (
                f"Simpson's rule needs an odd number of points; the series has {x.size}, "
                "so the trapezoidal rule was used instead.",
            ),
        )
    if simpson_rule:
        return Area(float(simpson(y, x=x)), "Simpson")
    return Area(float(trapezoid(y, x)), "trapezoidal")


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SurfaceGradient:
    dz_dx: np.ndarray
    dz_dy: np.ndarray
    magnitude: np.ndarray


def surface_gradient(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> SurfaceGradient:
    """The gradient of a gridded surface, on the same grid.

    ``np.gradient`` on ``Z[row, col]`` returns ``(dZ/d(axis0), dZ/d(axis1))``,
    and a grid built by ``np.meshgrid``'s default ('xy') indexing walks y
    along axis 0 and x along axis 1 - so the first array back is dz/dy. NaN
    cells propagate to their neighbours, as they should: there is no real
    slope next to a hole in the data.
    """
    x_axis = x_grid[0, :]
    y_axis = y_grid[:, 0]
    with np.errstate(invalid="ignore"):
        dz_dy, dz_dx = np.gradient(z_grid, y_axis, x_axis)
    return SurfaceGradient(dz_dx, dz_dy, np.sqrt(dz_dx**2 + dz_dy**2))


@dataclass(frozen=True, slots=True)
class Volume:
    total: float
    #: Cells that were not numbers and were counted as 0.
    missing_cells: int


def surface_volume(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> Volume:
    """The volume under a gridded surface, by Simpson's rule along x then along y.

    Cells that are not numbers (an interpolated grid outside the data's convex
    hull) count as zero rather than being cut out: Simpson's rule cannot
    integrate around a hole, and the caller reports how many there were.
    """
    x_axis = x_grid[0, :]
    y_axis = y_grid[:, 0]
    finite = np.isfinite(z_grid)
    missing = int(z_grid.size - np.count_nonzero(finite))
    inner = simpson(np.where(finite, z_grid, 0.0), x=x_axis, axis=1)
    return Volume(float(simpson(inner, x=y_axis)), missing)
