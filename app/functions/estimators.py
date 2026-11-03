"""The starting values of the library's functions, read off the data: one estimator per function that has one.

Moved unchanged from the classes the library was translated from (step 4 of dev/plans/sympy_library_plan.txt):
an entry's "guess" names its estimator here. A curve's takes (x, y), a surface's (x, y, z); each returns the
parameters in the entry's order, or None when the data say nothing. A function without one - a project's own
- starts from app.analysis.initial_guess.generic_guess.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from app.plugin_api.functions.base import (
    _baseline,
    _exponential,
    _measured_fwhm,
    _peak_shape,
    _periodic_guess,
    _polynomial_guess,
)


def _value(name: str, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The library function *name* at x - for an estimator that compares its candidates."""
    from app.functions import library

    return library.numeric(library.get(name))(x, p)


def constant(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    del x
    return [float(np.mean(y))] if y.size else None

def cosine(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    guess = _periodic_guess(x, y)
    if guess is None:
        return None
    amplitude, omega, offset = guess
    return [amplitude, omega, 0.0, offset]

def cubic(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 3)

def exponential_decay(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _exponential(x, y, False)

def exponential_growth(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _exponential(x, y, True)

def gaussian_peak(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    shape = _peak_shape(x, y)
    if shape is None:
        return None
    amplitude, centre, width, offset = shape
    return [amplitude, centre, max(width, 1e-9), offset]

def linear(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 1)

def logistic4(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    if x.size < 4:
        return None

    edge = max(1, x.size // 10)
    bottom = float(np.mean(y[:edge]))
    top = float(np.mean(y[-edge:]))
    span = top - bottom
    if span == 0.0:
        return None

    half = bottom + 0.5 * span
    midpoint = float(x[int(np.argmin(np.abs(y - half)))])

    # A transition occupying about a tenth of the plotted range; the sign
    # follows the direction of travel because k carries it.
    x_span = float(np.ptp(x)) or 1.0
    steepness = (10.0 / x_span) * (1.0 if span > 0 else -1.0)
    return [bottom, span, steepness, midpoint]

def lorentzian_peak(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    shape = _peak_shape(x, y)
    if shape is None:
        return None
    amplitude, centre, moment_width, offset = shape

    width = _measured_fwhm(x, y, centre, amplitude, offset)
    if width is None:
        # Falling back to the moment is wrong by a factor for a Lorentzian,
        # but a too-wide peak in the right place still converges; no guess at
        # all does not.
        width = 2.355 * moment_width
    return [amplitude, centre, max(width, 1e-9), offset]

def michaelis_menten(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    if x.size < 3:
        return None

    usable = (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(usable)) >= 2:
        try:
            slope, intercept = np.polyfit(x[usable], x[usable] / y[usable], 1)
        except NUMERICAL_FAILURES:
            slope = intercept = 0.0
        if np.isfinite(slope) and np.isfinite(intercept) and slope > 0.0:
            v_max = float(1.0 / slope)
            km = float(intercept * v_max)
            if km > 0.0 and np.isfinite(km):
                # Returned here rather than falling through: the
                # Lineweaver-Burk reading is the better of the two, and
                # the direct one below would overwrite it.
                return [v_max, km, 0.0]

    # The direct reading, for data the transform cannot take: every y at or
    # below zero, or a slope the wrong way.
    v_max = float(np.max(y))
    if v_max <= 0.0:
        return None
    km = float(x[int(np.argmin(np.abs(y - 0.5 * v_max)))])
    if km <= 0.0:
        km = float(np.median(x[x > 0.0])) if np.any(x > 0.0) else 1.0
    return [v_max, km, 0.0]

def polynomial10(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 10)

def polynomial7(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 7)

def polynomial8(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 8)

def polynomial9(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 9)

def power_law(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    """``A x^n + C``, via a straight line through log x against log y.

    Both offsets are tried and the better one kept.  A power law rises
    from its own offset, so the value at the low end of x is very nearly
    that offset - except when there is no offset at all, where the same
    reading subtracts a real part of the curve and the log-log line comes
    out through skewed data.  The two candidates cost one polyfit each and
    are told apart by the residual they leave, which is not something a
    rule of thumb can decide in advance.
    """
    candidates: list[list[float]] = []
    for offset in (_baseline(y), 0.0):
        above = y - offset
        usable = (x > 0.0) & (above > 0.0)
        if int(np.count_nonzero(usable)) < 2:
            continue
        try:
            exponent, intercept = np.polyfit(
                np.log(x[usable]), np.log(above[usable]), 1
            )
        except NUMERICAL_FAILURES:
            continue
        if np.isfinite(exponent) and np.isfinite(intercept):
            candidates.append([float(np.exp(intercept)), float(exponent), float(offset)])

    if not candidates:
        return None

    def misfit(guess: list[float]) -> float:
        values = _value("power_law", x, np.asarray(guess, dtype=float))
        residual = y - values
        return float(np.sum(residual * residual)) if np.all(np.isfinite(residual)) else np.inf

    return min(candidates, key=misfit)

def pseudo_voigt_peak(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    shape = _peak_shape(x, y)
    if shape is None:
        return None
    amplitude, centre, moment_width, offset = shape
    width = _measured_fwhm(x, y, centre, amplitude, offset) or (2.355 * moment_width)
    # eta 0.5: an even mix is the honest starting point, since nothing in the
    # data says how Gaussian or Lorentzian the peak is.
    return [amplitude, centre, max(width, 1e-9), 0.5, offset]

def quadratic(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 2)

def quartic(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 4)

def quintic(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 5)

def sextic(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    return _polynomial_guess(x, y, 6)

def sine(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    guess = _periodic_guess(x, y)
    if guess is None:
        return None
    amplitude, omega, offset = guess
    return [amplitude, omega, 0.0, offset]

def gaussian2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
    del x, y, z
    return None

def paraboloid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
    del x, y, z
    # Non-linear in x0/y0; left to the Monte Carlo search, same as most
    # of the 1D library's own shaped functions (Gaussian, logistic, ...).
    return None

def plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
    """Solve the linear least squares directly - a plane always can.

    Same spirit as the 1D library's ``_polynomial_guess`` (``np.polyfit``
    under the hood): a plane is linear in its parameters, so there is no
    need to guess when the exact least-squares answer is one
    ``np.linalg.lstsq`` call away.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if x.size < 3:
        return None
    design = np.column_stack([np.ones_like(x), x, y])
    try:
        coeffs, *_rest = np.linalg.lstsq(design, z, rcond=None)
    except NUMERICAL_FAILURES:
        return None
    if not np.all(np.isfinite(coeffs)):
        return None
    return [float(coeffs[0]), float(coeffs[1]), float(coeffs[2])]

def ripple2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
    del x, y, z
    # Frequency-in-two-directions is exactly the kind of many-valleyed
    # search the Monte Carlo starting-point search exists for.
    return None

def saddle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
    del x, y, z
    return None

def nist_eckerle4(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    del x, y
    return [1.5, 5.0, 450.0]

def nist_gauss3(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    del x, y
    return [96.0, 0.0096, 80.0, 110.0, 25.0, 74.0, 139.0, 25.0]

def nist_misra1a(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    del x, y
    return [250.0, 0.0005]

def nist_rat43(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    del x, y
    return [700.0, 5.0, 0.75, 1.3]

def nist_thurber(x: np.ndarray, y: np.ndarray) -> list[float] | None:
    del x, y
    return [1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05]

#: The estimators by the name an entry's "guess" holds.
ESTIMATORS: dict[str, Callable[..., list[float] | None]] = {
    "constant": constant,
    "cosine": cosine,
    "cubic": cubic,
    "exponential_decay": exponential_decay,
    "exponential_growth": exponential_growth,
    "gaussian_peak": gaussian_peak,
    "linear": linear,
    "logistic4": logistic4,
    "lorentzian_peak": lorentzian_peak,
    "michaelis_menten": michaelis_menten,
    "polynomial10": polynomial10,
    "polynomial7": polynomial7,
    "polynomial8": polynomial8,
    "polynomial9": polynomial9,
    "power_law": power_law,
    "pseudo_voigt_peak": pseudo_voigt_peak,
    "quadratic": quadratic,
    "quartic": quartic,
    "quintic": quintic,
    "sextic": sextic,
    "sine": sine,
    "gaussian2d": gaussian2d,
    "paraboloid": paraboloid,
    "plane": plane,
    "ripple2d": ripple2d,
    "saddle": saddle,
    "nist_eckerle4": nist_eckerle4,
    "nist_gauss3": nist_gauss3,
    "nist_misra1a": nist_misra1a,
    "nist_rat43": nist_rat43,
    "nist_thurber": nist_thurber,
}
