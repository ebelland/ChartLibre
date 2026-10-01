"""Roots: where a series crosses a level - the arithmetic behind the Roots operation.

Brackets first, then SciPy: consecutive samples whose difference from the
level changes sign bracket a crossing, and each bracket is refined over an
interpolant of the series (linear by default, cubic or PCHIP on request) by
Brent, bisection, TOMS 748 or the secant method - the last checked against
the bracket it came from. On a surface the roots are a whole level curve,
read back from matplotlib's contour extraction. Plain arrays in, no Qt,
nothing logged (todo R-01): a bracket the interpolant does not really cross
comes back as a note.

See app/series_operations/roots_dialog.py's docstring for why the bracket,
not the solver, decides what can be found.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, newton, toms748

SOLVER_BRENT = "Brent"
SOLVER_BISECT = "Bisection"
SOLVER_TOMS748 = "TOMS 748"
SOLVER_NEWTON = "Newton (secant)"

INTERP_LINEAR = "linear"
INTERP_CUBIC = "cubic"
INTERP_PCHIP = "pchip"


@dataclass(slots=True)
class Root:
    """One located crossing.

    For a 2D crossing (a series with a z role, see ``_solve_one_3d``), ``x``
    and ``y`` are a point's actual coordinates rather than x and a residual,
    ``z`` holds the level (constant across every point), and
    ``curve_index`` groups points into the separate polylines
    ``ax.contour`` returned - a saddle's z=0 level set, for instance, is two
    unconnected diagonal lines, not one.
    """

    x: float
    #: The interpolant's value there. Not exactly the level - it is the
    #: residual that says how well the solver converged, and a large one is
    #: the sign of a bracket the interpolant does not really cross.
    #: For a 2D crossing this is instead the point's own y coordinate.
    y: float
    #: True when the series is going up through the level at this x. A
    #: rising and a falling crossing are different events - a threshold
    #: being exceeded and a recovery - and the report says which. Meaningless
    #: for a 2D crossing (a level *curve* has no single "up"), always False
    #: there.
    rising: bool
    #: How the value was arrived at: the solver's name, or "sample" for a
    #: point that sat on the level to begin with. "contour" for a 2D one.
    method: str
    iterations: int = 0
    z: float | None = None
    curve_index: int = 0


def extract_level_curves(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the ``z_grid == level`` polylines as ``(xs, ys)`` arrays.

    contourpy directly - the generator ``ax.contour`` runs, with the same
    algorithm and corner masking as matplotlib's defaults, so the curves are
    the ones a contour plot draws. Not through a Figure: that touched
    pyplot's global state, which is not safe off the GUI thread, and this
    runs in the background (todo R-02).
    """
    import contourpy

    z_values = np.ma.masked_invalid(np.asarray(z_grid, dtype=float), copy=False)
    generator = contourpy.contour_generator(
        x_grid,
        y_grid,
        z_values,
        name="mpl2014",
        corner_mask=True,
        line_type=contourpy.LineType.SeparateCode,
    )
    lines = cast(tuple[list[np.ndarray], list[Any]], generator.lines(float(level)))
    polylines: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in lines[0]:
        segment = np.asarray(segment, dtype=float)
        if segment.shape[0] >= 1:
            polylines.append((segment[:, 0], segment[:, 1]))
    return polylines


def brackets(
    x_values: np.ndarray, offset: np.ndarray
) -> list[tuple[float, float, float | None]]:
    """Return the intervals that must contain a crossing.

    ``(left, right, exact)``: *exact* is set when a sample sits on the
    level, in which case there is nothing to solve. A run of consecutive
    samples exactly on the level would otherwise be reported as one root
    each, so only the first of such a run is taken - the series does not
    cross there, it rests there.
    """
    brackets: list[tuple[float, float, float | None]] = []
    previous_was_zero = False

    for index in range(offset.size):
        value = float(offset[index])
        if value == 0.0:
            if not previous_was_zero:
                brackets.append((0.0, 0.0, float(x_values[index])))
            previous_was_zero = True
            continue
        previous_was_zero = False

        if index + 1 >= offset.size:
            break
        following = float(offset[index + 1])
        if following != 0.0 and (value > 0.0) != (following > 0.0):
            brackets.append(
                (float(x_values[index]), float(x_values[index + 1]), None)
            )

    return brackets


def interpolant(
    x_values: np.ndarray, offset: np.ndarray, kind: str
) -> Any:
    """Return a callable for ``y(x) - level`` between the samples.

    Cubic and PCHIP need four and two points respectively and both need
    strictly increasing x, which ``prepare_input_xy`` has already
    guaranteed. A series too short for the chosen interpolant falls back
    to the straight line rather than failing: the assumption degrades,
    the answer still exists.
    """
    if kind == INTERP_CUBIC and x_values.size >= 4:
        return CubicSpline(x_values, offset)
    if kind == INTERP_PCHIP and x_values.size >= 2:
        return PchipInterpolator(x_values, offset)
    return lambda value: np.interp(value, x_values, offset)


def xtol_from_digits(digits: Any) -> float:
    """Return the x tolerance, from the number of digits asked for.

    Asked for as digits rather than as a number because that is how
    anyone thinks about it, and because a spin box showing
    "0.000000001000" is a control nobody can read or set.
    """
    try:
        count = int(digits)
    except (TypeError, ValueError):
        count = 9
    return 10.0 ** -min(max(count, 1), 15)


def refine(
    interpolant: Any,
    left: float,
    right: float,
    model: str,
    *,
    xtol: float,
    max_iter: int,
    notes: list[str] | None = None,
) -> tuple[float, int] | None:
    """Solve inside one bracket, or return None (and say why in *notes*).

    Every solver here is ``scipy.optimize``'s; what differs is what each
    is allowed to do with the bracket. A failure costs the one crossing
    rather than the whole series: an interpolant that wanders can leave a
    bracket its samples did straddle, and the other twenty crossings are
    still worth reporting.
    """
    def evaluate(value: float) -> float:
        return float(interpolant(value))

    try:
        if model == SOLVER_NEWTON:
            return _refine_newton(evaluate, left, right, xtol, max_iter)

        solver = {
            SOLVER_BRENT: brentq,
            SOLVER_TOMS748: toms748,
        }.get(model)
        if solver is None:
            return _refine_bisect(evaluate, left, right, xtol, max_iter)

        root, result = solver(
            evaluate,
            left,
            right,
            xtol=xtol,
            maxiter=max_iter,
            full_output=True,
        )
        if not result.converged:
            raise RuntimeError("the solver did not converge")
        return float(root), int(result.iterations)
    except NUMERICAL_FAILURES as exc:
        if notes is not None:
            notes.append(
                f"No root in [{left:g}, {right:g}]: {exc}. The samples change sign "
                "across it, so the interpolant does not - which is the interpolant "
                "disagreeing with the data rather than an error."
            )
        return None


def _refine_bisect(
    evaluate: Any,
    left: float,
    right: float,
    xtol: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Halve the bracket until it is narrower than the tolerance.

    SciPy's own ``bisect`` would do this; it is written out because it is
    four lines and because the iteration count it reports is then the
    real one rather than the solver's internal bookkeeping.
    """
    low, high = float(left), float(right)
    f_low = evaluate(low)
    iterations = 0
    while high - low > xtol and iterations < max_iter:
        middle = 0.5 * (low + high)
        f_middle = evaluate(middle)
        if f_middle == 0.0:
            return middle, iterations + 1
        if (f_low > 0.0) != (f_middle > 0.0):
            high = middle
        else:
            low, f_low = middle, f_middle
        iterations += 1
    return 0.5 * (low + high), iterations


def _refine_newton(
    evaluate: Any,
    left: float,
    right: float,
    xtol: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Secant from the middle of the bracket, then check it stayed in it.

    ``newton`` is not a bracketing method: with no derivative it runs the
    secant method, which converges faster than Brent on a smooth curve
    and is free to step anywhere. A result outside the bracket is a
    different crossing, or none - it is not the root of *this* interval,
    so it is refused rather than reported at the wrong x.
    """
    # full_output=True: newton returns (root, RootResults).
    root, result = cast(tuple[float, Any], newton(
        evaluate,
        0.5 * (left + right),
        tol=xtol,
        maxiter=max_iter,
        full_output=True,
        disp=False,
    ))
    if not result.converged:
        raise RuntimeError("the secant iteration did not converge")
    if not (min(left, right) <= float(root) <= max(left, right)):
        raise RuntimeError(
            "the secant iteration left the bracket it started in"
        )
    return float(root), int(result.iterations)


def is_rising(
    interpolant: Any, x_root: float, x_values: np.ndarray
) -> bool:
    """Say whether the series goes up through the level at *x_root*.

    Measured over a small step either side rather than from a derivative,
    so it means the same thing for every interpolant - ``np.interp`` has
    no derivative to ask for.
    """
    span = float(x_values[-1] - x_values[0])
    step = (span / max(x_values.size - 1, 1)) * 1.0e-3 if span > 0.0 else 1.0e-9
    try:
        before = float(interpolant(x_root - step))
        after = float(interpolant(x_root + step))
    except ValueError:  # outside the interpolant's domain
        return True
    return after >= before


@dataclass(frozen=True, slots=True)
class RootSearch:
    """The crossings found, whether the list was cut at the limit, and why any bracket gave nothing."""

    roots: list[Root]
    truncated: bool
    notes: tuple[str, ...] = ()
    #: Surfaces only: how many separate level curves there were.
    curves: int = 0


def find_roots(
    x: np.ndarray,
    y: np.ndarray,
    *,
    level: float = 0.0,
    solver: str = SOLVER_BRENT,
    interpolation: str = INTERP_LINEAR,
    tolerance_digits: int = 9,
    max_iter: int = 100,
    limit: int = 100,
) -> RootSearch:
    """Every x where ``y(x) = level``, in x order, at most *limit* of them.

    *x* must be sorted and unique (the dialog's input check guarantees it).
    """
    xtol = xtol_from_digits(tolerance_digits)
    offset = y - level
    curve = interpolant(x, offset, interpolation)
    notes: list[str] = []

    roots: list[Root] = []
    for left, right, exact in brackets(x, offset):
        if exact is not None:
            roots.append(Root(x=float(exact), y=level, rising=is_rising(curve, float(exact), x), method="sample"))
            continue
        found = refine(curve, left, right, solver, xtol=xtol, max_iter=max_iter, notes=notes)
        if found is None:
            continue
        x_root, iterations = found
        roots.append(
            Root(
                x=float(x_root),
                y=float(curve(x_root)) + level,
                rising=is_rising(curve, float(x_root), x),
                method=solver,
                iterations=int(iterations),
            )
        )

    roots.sort(key=lambda root: root.x)
    truncated = len(roots) > limit
    if truncated:
        # In x order, not by any measure of quality: a crossing is a
        # crossing, and the first hundred is the only defensible "first"
        # when they are all equally real.
        roots = roots[:limit]
    return RootSearch(roots, truncated, tuple(notes))


def find_level_curve(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    level: float = 0.0,
    limit: int = 100,
) -> RootSearch:
    """The points of the curve ``z = level`` on a gridded surface, at most *limit*."""
    polylines = extract_level_curves(x_grid, y_grid, z_grid, level)
    roots: list[Root] = []
    truncated = False
    for curve_index, (xs, ys) in enumerate(polylines):
        for x_value, y_value in zip(xs.tolist(), ys.tolist()):
            if len(roots) >= limit:
                truncated = True
                break
            roots.append(
                Root(
                    x=float(x_value), y=float(y_value), rising=False, method="contour",
                    z=level, curve_index=curve_index,
                )
            )
        if truncated:
            break
    return RootSearch(roots, truncated, (), len(polylines))
