"""Where a fit starts, and which algorithm finishes it.

Both were previously one hard-wired answer: the declared ``p0`` and
``least_squares`` with its defaults. Both failures they cause are quiet ones -
a converged fit in the wrong valley reports a plausible RMSE and completely
wrong parameters, and a fit dragged by three outliers reports success too.
So these tests assert the properties that distinguish a right answer from a
confident wrong one, not that the code runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.functions import functions as library
from app.functions.optimizers import OPTIMIZERS, TRUST_REGION, run_optimizer
from app.functions.starting_point import choose_starting_point

X = np.linspace(0.5, 10.0, 240)


# ======================================================================
# The functions' own estimators
# ======================================================================
@pytest.mark.parametrize(
    "function, truth, tolerance",
    [
        (library.linear, [3.0, 2.0], 1e-6),
        (library.quadratic, [1.0, -2.0, 0.5], 1e-6),
        (library.cubic, [0.0, 1.0, -0.3, 0.02], 1e-6),
        (library.gaussian_peak, [5.0, 4.0, 0.8, 1.0], 0.05),
        (library.michaelis_menten, [10.0, 2.0, 0.0], 0.05),
        (library.logistic4, [1.0, 8.0, 2.0, 5.0], 1.5),
        (library.power_law, [2.0, 1.5, 0.0], 0.2),
    ],
)
def test_an_estimator_lands_near_the_parameters_that_made_the_data(
    function, truth, tolerance
) -> None:
    """The point of an estimator: the optimiser starts in the right valley."""
    y = function.execute(X, np.asarray(truth, dtype=float))

    guess = function.initial_guess(X, y)

    assert guess is not None, "this function declares an estimator"
    assert len(guess) == len(truth)
    assert np.allclose(guess, truth, atol=tolerance, rtol=0.2)


# ======================================================================
# Choosing a starting point
# ======================================================================
def _model(function):
    return lambda x, p: function.execute(np.asarray(x, dtype=float), np.asarray(p, dtype=float))


def test_a_fixed_parameter_keeps_the_value_the_user_fixed() -> None:
    """The estimator does not get to overrule an answer the user gave."""
    truth = np.array([5.0, 4.0, 0.8, 1.0])
    y = library.gaussian_peak.execute(X, truth)

    start = choose_starting_point(
        _model(library.gaussian_peak), X, y,
        declared=[1.0, 1.0, 1.0, 99.0],
        lower=[-np.inf] * 4, upper=[np.inf] * 4,
        function_class=library.gaussian_peak,
        fixed=[False, False, False, True],
    )

    assert start.values[3] == 99.0
    assert start.values[1] == pytest.approx(4.0, abs=0.05)


# ======================================================================
# The random search
# ======================================================================


# ======================================================================
# The algorithms
# ======================================================================
DECAY_TRUTH = np.array([4.0, 0.7, 0.5])


def _decay_residual() -> tuple:
    y = library.exponential_decay.execute(X, DECAY_TRUTH)
    return y, (lambda p: y - library.exponential_decay.execute(X, p))


@pytest.mark.parametrize("optimizer", [o.key for o in OPTIMIZERS])
def test_every_algorithm_recovers_a_known_curve(optimizer: str) -> None:
    """Ten ways to the same answer; each one has to actually arrive."""
    _y, residual = _decay_residual()

    outcome = run_optimizer(
        optimizer, residual, [1.0, 1.0, 0.0], [-np.inf] * 3, [np.inf] * 3,
        max_nfev=600, seed=2, iterations=3000,
    )

    assert np.allclose(outcome.params, DECAY_TRUTH, atol=1e-4)


def test_an_unknown_algorithm_is_named_not_replaced() -> None:
    """A misspelt key used to run the trust region and report it as such."""
    _y, residual = _decay_residual()
    with pytest.raises(ValueError, match="levenberg-marquardt"):
        run_optimizer("levenberg-marquardt", residual, [1.0, 1.0, 0.0], [-np.inf] * 3, [np.inf] * 3)


def test_a_robust_loss_survives_outliers_that_drag_least_squares() -> None:
    """A squared residual weights a point ten times off a hundred times more
    than a point one off, so a handful of bad points move the whole line."""
    truth = np.array([2.0, 1.5])
    y = library.linear.execute(X, truth)
    spoiled = y.copy()
    spoiled[[10, 60, 130]] += 60.0
    residual = lambda p: spoiled - library.linear.execute(X, p)  # noqa: E731

    ordinary = run_optimizer(TRUST_REGION, residual, [0.0, 0.0],
                             [-np.inf] * 2, [np.inf] * 2, max_nfev=400)
    robust = run_optimizer(TRUST_REGION, residual, [0.0, 0.0],
                           [-np.inf] * 2, [np.inf] * 2, max_nfev=400,
                           loss="cauchy")

    ordinary_error = float(np.max(np.abs(ordinary.params - truth)))
    robust_error = float(np.max(np.abs(robust.params - truth)))
    assert robust_error < ordinary_error / 5.0


def test_bounds_are_honoured_where_they_are_supported() -> None:
    _y, residual = _decay_residual()

    outcome = run_optimizer(TRUST_REGION, residual, [1.0, 1.0, 0.0],
                            [-np.inf, -np.inf, 1.0], [np.inf, np.inf, np.inf],
                            max_nfev=400)

    assert outcome.params[2] >= 1.0 - 1e-9


# ======================================================================
# The library has to be loadable the way the scanner loads it
# ======================================================================


def test_every_discovered_function_evaluates() -> None:
    """A class that loads but cannot be called is a model that fails on Fit."""
    from app.scanners.functions_scanner import FunctionScanner

    scanner = FunctionScanner()
    broken: list[str] = []

    for payloads in scanner.catalog().values():
        for payload in payloads:
            model = scanner.make_model(payload)
            p0 = np.asarray(payload.get("p0") or [1.0], dtype=float)
            try:
                values = np.asarray(model(X, p0), dtype=float)
            except Exception as exc:  # noqa: BLE001
                broken.append(f"{payload['name']}: {type(exc).__name__}: {exc}")
                continue
            if values.shape != X.shape:
                broken.append(f"{payload['name']}: returned {values.shape}, not {X.shape}")

    assert broken == []
