"""Starting values for a fit of any SymPy function - a project's own, which has no estimator of its own.

Step 2 of dev/plans/sympy_library_plan.txt. Three tries, the first that works kept:

1. Linear in its parameters (a polynomial, a sum of known shapes with unknown weights: SymPy says so) - one
   least-squares solve, exact, no starting point needed.
2. Read off the data by the parameters' names: an amplitude (A, amp, height, Vmax...) the range of y, a centre
   (x0, mu, xc, midpoint...) the x of the peak or of the half-way point, a width (sigma, s, w, fwhm, gamma)
   from the half-height width, an offset (C, offset, baseline, bottom) the low end of y, a rate (k, rate) one
   over the range of x, a time constant (tau) a third of it, a slope and an intercept from the line through
   the ends. A name it does not know keeps its stored value.
3. From there, a short multi-start: the guess scaled up and down by factors of ten, a few dozen points, the
   best few polished by least squares - within a budget of about half a second.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp

#: Seconds the multi-start may take.
BUDGET = 0.5
_ROLES: tuple[tuple[str, str], ...] = (
    ("amplitude", r"^(a|amp|amplitude|height|h|vmax|area|span|peak|top|l|ymax)\d*$"),
    ("centre", r"^(x0|xc|mu|center|centre|location|midpoint|x_?c|t0|ec50|k_?half|km)\d*$"),
    ("width", r"^(sigma|s|w|width|fwhm|gamma|hwhm|sd)\d*$"),
    ("offset", r"^(c|offset|baseline|bottom|y0|b0|background|bg)\d*$"),
    ("rate", r"^(k|rate|lam|lambda_?|r|omega)\d*$"),
    ("time", r"^(tau|t|tc)\d*$"),
    ("slope", r"^(m|slope)\d*$"),
    ("intercept", r"^(b|intercept|q)\d*$"),
)


def role(name: str) -> str:
    """What a parameter's name says it is ("amplitude", "centre"...), or "" for a name it does not know."""
    lowered = name.lower()
    return next((kind for kind, pattern in _ROLES if re.match(pattern, lowered)), "")


def _features(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    order = np.argsort(x)
    x, y = x[order], y[order]
    ends = float(min(np.median(y[: max(2, y.size // 10)]), np.median(y[-max(2, y.size // 10):])))
    rise = y - ends
    peak = int(np.argmax(np.abs(rise)))
    amplitude = float(rise[peak])
    half = np.abs(rise) >= abs(amplitude) / 2.0
    width = float(x[half].max() - x[half].min()) if half.sum() >= 2 else float(np.ptp(x)) / 4.0
    span = float(np.ptp(x)) or 1.0
    # A step (a sigmoid, a growth) has its centre where it is half-way, not where it is highest.
    monotone = abs(np.corrcoef(x, y)[0, 1]) > 0.9 if y.size > 2 and np.std(y) > 0 else False
    middle = float(np.interp(0.5, (y - y.min()) / (np.ptp(y) or 1.0), x)) if monotone and y[-1] > y[0] \
        else float(x[np.argmin(np.abs(y - (y.min() + y.max()) / 2.0))]) if monotone else float(x[peak])
    slope = float((y[-1] - y[0]) / (x[-1] - x[0])) if x[-1] != x[0] else 0.0
    return {"amplitude": float(np.ptp(y)) if monotone else amplitude, "centre": middle,
            "width": max(width / 2.3548, span / 1000.0), "offset": float(y.min()) if monotone else ends,
            "rate": 4.0 / span, "time": span / 3.0, "slope": slope, "intercept": float(y[0] - slope * x[0])}


def from_names(params: Sequence[str], values: Sequence[float], x: np.ndarray, y: np.ndarray) -> list[float]:
    """Try 2: each parameter its role's reading of the data; its stored value when its name says nothing."""
    features = _features(np.asarray(x, float), np.asarray(y, float))
    guess = []
    for name, value in zip(params, values, strict=True):
        kind = role(name)
        reading = features.get(kind)
        if kind == "width" and name.lower() == "fwhm":
            reading = features["width"] * 2.3548
        guess.append(float(reading) if reading is not None and np.isfinite(reading) else float(value))
    return guess


def linear_solution(expression: sp.Expr, params: Sequence[str], x: np.ndarray, y: np.ndarray) -> list[float] | None:
    """Try 1: the exact least-squares values when *expression* is linear in its parameters; else None."""
    symbols = [sp.Symbol(name) for name in params]
    x_symbol = sp.Symbol("x")
    try:
        columns = [sp.diff(expression, symbol) for symbol in symbols]
    except (TypeError, ValueError):
        return None
    if any(column.free_symbols & set(symbols) for column in columns):
        return None  # a parameter inside another's term: not linear
    rest = expression.subs({symbol: 0 for symbol in symbols})
    try:
        design = np.column_stack([np.broadcast_to(np.asarray(sp.lambdify(x_symbol, column, "numpy")(x), float),
                                                  x.shape) for column in columns])
        base = np.broadcast_to(np.asarray(sp.lambdify(x_symbol, rest, "numpy")(x), float), x.shape)
        solution, *_rest = np.linalg.lstsq(design, y - base, rcond=None)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return [float(value) for value in solution] if np.all(np.isfinite(solution)) else None


def multi_start(model: Callable[[np.ndarray, np.ndarray], np.ndarray], start: Sequence[float], x: np.ndarray,
                y: np.ndarray, *, budget: float = BUDGET, seed: int = 0) -> list[float]:
    """Try 3: the start, and points around it scaled by factors of ten, the best few polished; within *budget*."""
    from scipy.optimize import least_squares

    deadline = time.monotonic() + budget
    start = np.asarray(start, float)
    rng = np.random.default_rng(seed)

    def misfit(p: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            residual = model(x, p) - y
        return float(np.sum(residual ** 2)) if np.all(np.isfinite(residual)) else np.inf

    scale = np.where(start != 0, np.abs(start), 1.0)
    candidates = [start] + [start * 10.0 ** rng.uniform(-1, 1, start.size) * rng.choice([1, -1], start.size, p=[0.85, 0.15])
                            + (start == 0) * rng.normal(0, 1, start.size) * scale for _ in range(40)]
    scored = []
    for candidate in candidates:
        scored.append((misfit(candidate), candidate))
        if time.monotonic() > deadline:
            break
    scored.sort(key=lambda item: item[0])
    best_cost, best = scored[0]
    for _cost, candidate in scored[:4]:
        if time.monotonic() > deadline:
            break
        try:
            with np.errstate(all="ignore"):
                fitted = least_squares(lambda p: np.nan_to_num(model(x, p) - y, nan=1e12, posinf=1e12, neginf=-1e12),
                                       candidate, max_nfev=200 * candidate.size)
        except (ValueError, ArithmeticError):
            continue
        cost = misfit(fitted.x)
        if cost < best_cost:
            best_cost, best = cost, fitted.x
    return [float(value) for value in best]


def generic_guess(code: str, params: Sequence[str], values: Sequence[float], x: np.ndarray, y: np.ndarray, *,
                  budget: float = BUDGET) -> list[float]:
    """Starting values for the expression *code* in *params* (stored *values*) from the data x, y."""
    from app.functions import library

    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < len(params) or x.size < 2:
        return [float(value) for value in values]
    expression = sp.sympify(str(code).replace("^", "**"), locals=library.sympy_locals(list(params)))
    exact = linear_solution(expression, params, x, y)
    if exact is not None:
        return exact
    entry = {"name": "guess", "kind": "expression", "code": str(code), "params": list(params), "ndim": 1}
    model = library.numeric(entry)
    return multi_start(model, from_names(params, values, x, y), x, y, budget=budget)
