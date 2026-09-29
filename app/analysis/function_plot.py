"""Evaluating a function over a range: the arithmetic behind the Function operation.

The x values to evaluate at (linear or logarithmic spacing) and the
evaluation itself of a curve ``y = f(x)`` or a surface ``z = f(x, y)`` with
a set of parameter values, counting the points where the function is not
defined - many are legitimately undefined over part of a range (a log below
zero, a pole in a rational), and the useful behaviour is to draw the part
that exists and say how much was dropped. The function is any callable
``model(x, parameters)``; where the library of functions lives is not this
module's business. No Qt, nothing logged (todo R-01).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SPACING_LINEAR = "linear"
SPACING_LOG = "log"

#: Fewest points a range is made of.
MIN_POINTS = 2

Model = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class Curve:
    x: np.ndarray
    y: np.ndarray
    #: How many values came out as nan or infinity.
    undefined: int


@dataclass(frozen=True, slots=True)
class Surface:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    undefined: int


def build_range(start: float, stop: float, points: int, spacing: str = SPACING_LINEAR) -> np.ndarray:
    """The x values to evaluate at: *points* of them from *start* to *stop*."""
    count = max(MIN_POINTS, int(points))
    if start == stop:
        raise ValueError("the range is empty - From and To are the same")

    if spacing == SPACING_LOG:
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("logarithmic spacing needs a range strictly above zero")
        return np.logspace(np.log10(start), np.log10(stop), count)

    return np.linspace(start, stop, count)


def _count_undefined(values: np.ndarray) -> int:
    finite = int(np.count_nonzero(np.isfinite(values)))
    if finite == 0:
        raise ValueError(
            "the function is undefined everywhere in this range - check "
            "the range and the parameter values"
        )
    return values.size - finite


def evaluate_curve(model: Model, x: np.ndarray, parameters: np.ndarray) -> Curve:
    """Evaluate ``model(x, parameters)``; refuses a result of the wrong shape or with no defined value."""
    y = np.asarray(model(x, parameters), dtype=float)
    if y.shape != x.shape:
        raise ValueError(f"the function returned {y.size} value(s) for {x.size} input(s)")
    return Curve(x, y, _count_undefined(y))


def evaluate_surface(model: Model, axis: np.ndarray, parameters: np.ndarray) -> Surface:
    """Evaluate ``model(xy, parameters)`` over the square grid *axis* x *axis*, flattened.

    The same range serves for y as for x - a square domain, which keeps the
    range controls the same for a curve and a surface.
    """
    x_grid, y_grid = np.meshgrid(axis, axis.copy())
    xy = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    z = np.asarray(model(xy, parameters), dtype=float)
    return Surface(x_grid.ravel(), y_grid.ravel(), z, _count_undefined(z))
