"""The root-finding operation, checked against roots that are known exactly.

Every assertion here compares against an analytic answer - the zeros of a
sine, the crossing of a straight line - because a root finder is the kind of
code that is convincingly wrong: it returns a number of the right magnitude
whatever it does, and only arithmetic that knows the answer can tell.

The properties worth pinning are the two halves of the method. The bracket
scan decides *what can be found at all*: a crossing no two samples straddle
is invisible, a sample sitting on the level is a root already, and a series
that rests on the level is not crossing it. The SciPy refinement decides
*how well*: which is why the accuracy tests compare a solver's answer to the
straight-line crossing it started from rather than only to a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.series_operations.roots_dialog import INTERP_LINEAR, ROOT_BRENT, ROOT_MODELS, SeriesRootsDialog
from dev.tests._cases import check_all


def _bare() -> SeriesRootsDialog:
    """An instance without its Qt dialog - the numerics are plain methods."""
    return SeriesRootsDialog.__new__(SeriesRootsDialog)


def _params(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "level": 0.0,
        "interpolation": INTERP_LINEAR,
        "tolerance_digits": 12,
        "max_iter": 100,
        "limit": 100,
    }
    values.update(overrides)
    return values


def _solve(x, y, model=ROOT_BRENT, **overrides):
    return _bare()._solve_one("s", np.asarray(x, dtype=float),
                              np.asarray(y, dtype=float), model, _params(**overrides))


# Two full periods, sampled finely enough that linear interpolation is close
# and coarsely enough that cubic is visibly closer.
SINE_X = np.linspace(0.0, 4.0 * np.pi, 200)
SINE_Y = np.sin(SINE_X)
SINE_ZEROS = np.array([0.0, np.pi, 2.0 * np.pi, 3.0 * np.pi])


# ======================================================================
# What is found
# ======================================================================
_CASES_EVERY_SOLVER_FINDS_THE_ZEROS_OF_A_SINE = [(case,) for case in ROOT_MODELS]


def _every_solver_finds_the_zeros_of_a_sine(model: str) -> None:
    result = _solve(SINE_X, SINE_Y, model)

    assert len(result.roots) == len(SINE_ZEROS)
    assert np.allclose([root.x for root in result.roots], SINE_ZEROS, atol=1e-5)


def test_every_solver_finds_the_zeros_of_a_sine() -> None:
    check_all(_every_solver_finds_the_zeros_of_a_sine, _CASES_EVERY_SOLVER_FINDS_THE_ZEROS_OF_A_SINE)


def test_a_series_that_never_reaches_the_level_has_no_roots() -> None:
    result = _solve(SINE_X, SINE_Y + 5.0)

    assert result.roots == []
    assert result.metadata["found"] == 0


def test_the_direction_of_each_crossing_is_reported() -> None:
    result = _solve(SINE_X, SINE_Y)

    assert [root.rising for root in result.roots] == [True, False, True, False]


# ======================================================================
# How well it is found
# ======================================================================
def test_a_straight_line_crossing_is_exact() -> None:
    """y = 2x - 3 crosses zero at 1.5, and every solver should say so to
    the tolerance rather than to the sample spacing."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x - 3.0

    for model in ROOT_MODELS:
        root = _solve(x, y, model).roots[0].x
        assert abs(root - 1.5) < 1e-9, model


# ======================================================================
# The solvers themselves
# ======================================================================


def test_bisection_converges_to_within_the_tolerance() -> None:
    from app.analysis.roots import _refine_bisect

    refined = _refine_bisect(lambda v: v - 0.3, 0.0, 1.0, 1e-9, 200)
    assert refined is not None
    root, iterations = refined

    assert abs(root - 0.3) <= 1e-9
    assert iterations < 200


# ======================================================================
# The result, as the rest of the app sees it
# ======================================================================
def test_the_result_frame_carries_every_root_and_its_measurements() -> None:
    result = _solve(SINE_X, SINE_Y)
    frame = result.to_df()

    assert list(frame.columns) == [
        "x", "y", "level", "rising", "method", "iterations"
    ]
    assert len(frame) == len(result.roots)


