"""Interpolation: the arithmetic behind the Interpolation series operation.

A series is fitted or interpolated - polynomial and linear fits, curve fits of
five closed forms (exponential, logarithmic, power, Gaussian, sigmoid), NumPy's
linear interpolation and SciPy's PCHIP, Akima, cubic, B-spline and smoothing
splines - and evaluated at x values chosen by a spacing rule (evenly, on a log
or geometric scale, on integer steps, at Chebyshev nodes, at the original x or
at values typed in). Plain arrays in, no Qt, nothing logged (todo R-01): a
request the data cannot meet is a ``ValueError``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    UnivariateSpline,
    make_interp_spline,
)
from scipy.optimize import curve_fit

MODEL_POLYNOMIAL: Final[str] = "Polynomial"
MODEL_LINEAR: Final[str] = "Linear"
MODEL_EXPONENTIAL: Final[str] = "Exponential"
MODEL_LOGARITHMIC: Final[str] = "Logarithmic"
MODEL_POWER: Final[str] = "Power"
MODEL_GAUSSIAN: Final[str] = "Gaussian"
MODEL_SIGMOID: Final[str] = "Sigmoid"
MODEL_NUMPY_INTERP: Final[str] = "NumPy interp"
MODEL_SCIPY_PCHIP: Final[str] = "SciPy PCHIP"
MODEL_SCIPY_CUBIC: Final[str] = "SciPy CubicSpline"
MODEL_SCIPY_SPLINE: Final[str] = "SciPy spline family"
MODEL_SCIPY_AKIMA: Final[str] = "SciPy Akima"

#: The closed forms fitted with curve_fit, with their parameter names in order.
CURVE_FIT_MODELS: dict[str, tuple[str, ...]] = {
    MODEL_EXPONENTIAL: ("a", "b", "c"),
    MODEL_LOGARITHMIC: ("a", "b"),
    MODEL_POWER: ("a", "b", "c"),
    MODEL_GAUSSIAN: ("a", "mu", "sigma", "c"),
    MODEL_SIGMOID: ("a", "x0", "k", "c"),
}

SPACING_LINEAR = "linspace"
SPACING_LOG = "logspace"
SPACING_GEOMETRIC = "geomspace"
SPACING_INTEGER_STEP = "integer step"
SPACING_CHEBYSHEV = "chebyshev nodes"
SPACING_ORIGINAL = "original data X"
SPACING_CUSTOM = "custom X values"


@dataclass(frozen=True, slots=True)
class InterpolationSettings:
    """Everything a model reads besides the data; each model reads its own."""

    #: Polynomial degree (clamped to the points available).
    degree: int = 3
    #: Evaluate outside the data's x range rather than leaving it empty.
    extrapolate: bool = False
    #: CubicSpline's and make_interp_spline's boundary condition.
    cubic_bc: str = "not-a-knot"
    #: For the spline family: "CubicSpline", "PCHIP", "Akima1D", "B-spline" or
    #: anything else for UnivariateSpline.
    spline_type: str = "CubicSpline"
    spline_degree: int = 3
    #: UnivariateSpline's smoothing factor.
    smoothing: float = 0.0


@dataclass(frozen=True, slots=True)
class Interpolated:
    y: np.ndarray
    #: The fitted parameters, by name (empty for a pure interpolant).
    params: dict[str, float] = field(default_factory=dict)
    #: What was done, for the report: "NumPy polyfit degree=3"...
    message: str = ""


# ----------------------------------------------------------------------
# Where to evaluate
# ----------------------------------------------------------------------
def parse_values(text: str) -> np.ndarray:
    """Numbers typed as a list: separated by commas, semicolons, spaces or new lines."""
    raw = text.replace(";", ",").replace("\n", ",")
    tokens: list[str] = []
    for chunk in raw.split(","):
        tokens.extend(part for part in chunk.split(" ") if part.strip())
    return np.asarray([float(token) for token in tokens], dtype=float)


def evaluation_range(
    x_data: np.ndarray,
    *,
    explicit: tuple[float, float] | None = None,
    extend_percent: float = 0.0,
    extrapolate: bool = False,
) -> tuple[float, float]:
    """The x range to evaluate over: *explicit*, or the data's, widened when extrapolating."""
    if explicit is not None:
        start, stop = float(explicit[0]), float(explicit[1])
    else:
        x_min = float(np.min(x_data))
        x_max = float(np.max(x_data))
        span = x_max - x_min
        if span <= 0.0:
            raise ValueError("X data must contain more than one unique value.")
        pad = span * float(extend_percent) / 100.0 if extrapolate else 0.0
        start, stop = x_min - pad, x_max + pad
    if start == stop:
        raise ValueError("X range start and stop must differ.")
    return (stop, start) if start > stop else (start, stop)


def evaluation_x(
    x_data: np.ndarray,
    spacing: str,
    *,
    start: float,
    stop: float,
    count: int = 200,
    step: float = 1.0,
    custom: np.ndarray | None = None,
) -> np.ndarray:
    """The x values to evaluate at, by *spacing* (one of the SPACING_ names)."""
    if spacing == SPACING_CUSTOM:
        values = np.asarray(custom if custom is not None else [], dtype=float)
        if values.size == 0:
            raise ValueError("Enter at least one Eval X value.")
        return np.sort(values)
    if spacing == SPACING_ORIGINAL:
        return np.sort(np.asarray(x_data, dtype=float))
    if spacing == SPACING_LOG:
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("logspace requires a positive X range.")
        return np.logspace(np.log10(start), np.log10(stop), count)
    if spacing == SPACING_GEOMETRIC:
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("geomspace requires a positive X range.")
        return np.geomspace(start, stop, count)
    if spacing == SPACING_INTEGER_STEP:
        step = float(step)
        first = math.ceil(start / step) * step
        values = np.arange(first, stop + step * 0.5, step, dtype=float)
        if values.size == 0:
            raise ValueError("Step spacing produced no X values.")
        return values
    if spacing == SPACING_CHEBYSHEV:
        k = np.arange(count, dtype=float)
        nodes = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * count))
        return np.sort(0.5 * (start + stop) + 0.5 * (stop - start) * nodes)
    return np.linspace(start, stop, count)


# ----------------------------------------------------------------------
# The models
# ----------------------------------------------------------------------
def default_params(model: str) -> dict[str, float]:
    """The starting parameters shown for a curve-fit model before any data is seen."""
    if model == MODEL_EXPONENTIAL:
        return {"a": 1.0, "b": 0.1, "c": 0.0}
    if model == MODEL_LOGARITHMIC:
        return {"a": 1.0, "b": 0.0}
    if model == MODEL_POWER:
        return {"a": 1.0, "b": 1.0, "c": 0.0}
    if model == MODEL_GAUSSIAN:
        return {"a": 1.0, "mu": 0.0, "sigma": 1.0, "c": 0.0}
    if model == MODEL_SIGMOID:
        return {"a": 1.0, "x0": 0.0, "k": 1.0, "c": 0.0}
    return {}


def guess_params(model: str, x_data: np.ndarray, y_data: np.ndarray) -> dict[str, float]:
    """A starting guess for a curve-fit model, read off the data's ranges."""
    x_min = float(np.min(x_data))
    x_max = float(np.max(x_data))
    y_min = float(np.min(y_data))
    y_max = float(np.max(y_data))
    y_span = y_max - y_min if y_max != y_min else 1.0
    x_mid = float(np.median(x_data))
    if model == MODEL_EXPONENTIAL:
        return {"a": y_span, "b": 1.0 / max(abs(x_max - x_min), 1.0), "c": y_min}
    if model == MODEL_LOGARITHMIC:
        return {"a": y_span, "b": y_min}
    if model == MODEL_POWER:
        return {"a": 1.0, "b": 1.0, "c": y_min}
    if model == MODEL_GAUSSIAN:
        return {"a": y_span, "mu": x_mid, "sigma": max((x_max - x_min) / 6.0, 1e-9), "c": y_min}
    if model == MODEL_SIGMOID:
        return {"a": y_span, "x0": x_mid, "k": 1.0 / max(abs(x_max - x_min), 1.0), "c": y_min}
    return {}


def model_function(model: str) -> Callable[..., np.ndarray]:
    """The closed form of a curve-fit model, f(x, *params)."""
    if model == MODEL_EXPONENTIAL:
        return lambda x, a, b, c: a * np.exp(b * x) + c
    if model == MODEL_LOGARITHMIC:
        return lambda x, a, b: a * np.log(x) + b
    if model == MODEL_POWER:
        return lambda x, a, b, c: a * np.power(x, b) + c
    if model == MODEL_GAUSSIAN:
        return lambda x, a, mu, sigma, c: a * np.exp(-0.5 * np.square((x - mu) / sigma)) + c
    if model == MODEL_SIGMOID:
        return lambda x, a, x0, k, c: a / (1.0 + np.exp(-k * (x - x0))) + c
    raise ValueError(f"Unsupported model: {model}")


def _mask_outside(x_data: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, extrapolate: bool) -> np.ndarray:
    """Leave the values outside the data's x range empty, unless extrapolating."""
    if extrapolate:
        return y_eval
    outside = (x_eval < float(np.min(x_data))) | (x_eval > float(np.max(x_data)))
    masked = y_eval.astype(float, copy=True)
    masked[outside] = np.nan
    return masked


def _spline_family(
    x_data: np.ndarray, y_data: np.ndarray, x_eval: np.ndarray, settings: InterpolationSettings
) -> Interpolated:
    spline_type = settings.spline_type
    extrapolate = settings.extrapolate

    if spline_type == "CubicSpline":
        curve = CubicSpline(x_data, y_data, bc_type=cast(str, settings.cubic_bc), extrapolate=extrapolate)
        return Interpolated(cast(np.ndarray, curve(x_eval)), {}, "SciPy CubicSpline")
    if spline_type == "PCHIP":
        curve = PchipInterpolator(x_data, y_data, extrapolate=extrapolate)
        return Interpolated(cast(np.ndarray, curve(x_eval)), {}, "SciPy PCHIP")
    if spline_type == "Akima1D":
        values = cast(np.ndarray, Akima1DInterpolator(x_data, y_data, extrapolate=extrapolate)(x_eval))
        return Interpolated(_mask_outside(x_data, x_eval, values, extrapolate), {}, "SciPy Akima1D")

    k = min(int(settings.spline_degree), max(1, x_data.size - 1))
    if spline_type == "B-spline":
        curve = make_interp_spline(x_data, y_data, k=k, bc_type=cast(str, settings.cubic_bc))
        values = cast(np.ndarray, curve(x_eval))
        return Interpolated(_mask_outside(x_data, x_eval, values, extrapolate), {"k": float(k)}, "SciPy B-spline")
    curve = UnivariateSpline(x_data, y_data, s=float(settings.smoothing), k=k)
    values = cast(np.ndarray, curve(x_eval))
    return Interpolated(_mask_outside(x_data, x_eval, values, extrapolate), {"k": float(k)}, "SciPy UnivariateSpline")


def interpolate(
    model: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_eval: np.ndarray,
    settings: InterpolationSettings | None = None,
    start_params: Mapping[str, float] | None = None,
) -> Interpolated:
    """Fit or interpolate ``(x_data, y_data)`` with *model* and evaluate it at *x_eval*.

    *x_data* must be sorted and unique (the dialog's input check guarantees
    it). *start_params* override the guessed starting point of a curve fit.
    """
    settings = settings or InterpolationSettings()
    if model == MODEL_POLYNOMIAL:
        degree = min(int(settings.degree), max(1, x_data.size - 1))
        coeff = np.polyfit(x_data, y_data, degree)
        return Interpolated(
            np.polyval(coeff, x_eval),
            {f"c{i}": float(v) for i, v in enumerate(coeff)},
            f"NumPy polyfit degree={degree}",
        )
    if model == MODEL_LINEAR:
        coeff = np.polyfit(x_data, y_data, 1)
        return Interpolated(np.polyval(coeff, x_eval), {"m": float(coeff[0]), "b": float(coeff[1])}, "Linear fit")
    if model == MODEL_NUMPY_INTERP:
        return Interpolated(np.interp(x_eval, x_data, y_data), {}, "NumPy interpolation")
    if model == MODEL_SCIPY_PCHIP:
        curve = PchipInterpolator(x_data, y_data, extrapolate=settings.extrapolate)
        return Interpolated(cast(np.ndarray, curve(x_eval)), {}, "SciPy PCHIP")
    if model == MODEL_SCIPY_AKIMA:
        # extrapolate passed on: SciPy's Akima leaves the outside empty unless
        # told otherwise, so the Extrapolate box used to do nothing for it.
        values = cast(
            np.ndarray, Akima1DInterpolator(x_data, y_data, extrapolate=settings.extrapolate)(x_eval)
        )
        return Interpolated(_mask_outside(x_data, x_eval, values, settings.extrapolate), {}, "SciPy Akima1D")
    if model == MODEL_SCIPY_CUBIC:
        curve = CubicSpline(
            x_data, y_data, bc_type=cast(str, settings.cubic_bc), extrapolate=settings.extrapolate
        )
        return Interpolated(cast(np.ndarray, curve(x_eval)), {}, "SciPy CubicSpline")
    if model == MODEL_SCIPY_SPLINE:
        return _spline_family(x_data, y_data, x_eval, settings)

    function = model_function(model)
    names = list(CURVE_FIT_MODELS[model])
    guess = guess_params(model, x_data, y_data)
    guess.update(start_params or {})
    popt, _covariance = curve_fit(function, x_data, y_data, p0=[guess[name] for name in names], maxfev=50_000)
    return Interpolated(
        function(x_eval, *popt),
        {name: float(value) for name, value in zip(names, popt)},
        f"SciPy series_fit {model}",
    )


def goodness(
    model: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    params: Mapping[str, float],
    settings: InterpolationSettings | None = None,
) -> dict[str, float]:
    """RMSE and R² of the model at the data's own x; empty when it cannot be evaluated there.

    A curve fit is evaluated from its fitted parameters (passed as the start,
    so the fit stays where it is).
    """
    try:
        y_hat = interpolate(model, x_data, y_data, x_data, settings, dict(params)).y
    except NUMERICAL_FAILURES:  # a diagnostic, never fatal
        return {}
    residual = y_data - y_hat
    ss_res = float(np.nansum(np.square(residual)))
    ss_tot = float(np.nansum(np.square(y_data - np.nanmean(y_data))))
    return {
        "rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
    }
