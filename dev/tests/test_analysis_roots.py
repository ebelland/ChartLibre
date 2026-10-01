"""Roots (app.analysis.roots): crossings of a level with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import roots as rt

X = np.linspace(0.0, 4.0 * np.pi, 200)
SINE = np.sin(X)
ZEROS = [0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi]


@pytest.mark.parametrize("solver", [rt.SOLVER_BRENT, rt.SOLVER_BISECT, rt.SOLVER_TOMS748, rt.SOLVER_NEWTON])
def test_every_solver_finds_the_zeros_of_a_sine(solver: str) -> None:
    search = rt.find_roots(X, SINE, solver=solver, interpolation=rt.INTERP_CUBIC, tolerance_digits=12)
    found = [root.x for root in search.roots]
    assert found == pytest.approx(ZEROS[: len(found)], abs=1e-5)
    assert len(found) >= 4
    assert not search.truncated


def test_crossings_say_whether_they_rise_or_fall() -> None:
    search = rt.find_roots(X, SINE, level=0.5)
    assert [root.rising for root in search.roots] == [True, False, True, False]
    # sin(x) = 0.5 at pi/6 and 5 pi/6, then again one period later.
    expected = [np.pi / 6, 5 * np.pi / 6, 2 * np.pi + np.pi / 6, 2 * np.pi + 5 * np.pi / 6]
    assert [root.x for root in search.roots] == pytest.approx(expected, abs=2e-3)  # linear interpolation


def test_cubic_interpolation_is_closer_than_linear_on_a_coarse_smooth_curve() -> None:
    x = np.linspace(0.0, 6.0, 9)
    y = np.cos(x)
    linear = rt.find_roots(x, y, interpolation=rt.INTERP_LINEAR).roots[0].x
    cubic = rt.find_roots(x, y, interpolation=rt.INTERP_CUBIC).roots[0].x
    assert abs(cubic - np.pi / 2) < abs(linear - np.pi / 2)


def test_a_sample_on_the_level_is_a_root_and_a_run_of_them_counts_once() -> None:
    x = np.arange(8.0)
    y = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.5, 2.0, 3.0])
    search = rt.find_roots(x, y)
    assert [(root.x, root.method) for root in search.roots][:1] == [(1.0, "sample")]
    assert [root.x for root in search.roots] == pytest.approx([1.0, 4 + 2 / 3])


def test_the_limit_keeps_the_first_crossings_in_x_order() -> None:
    search = rt.find_roots(X, SINE, level=0.1, limit=2)
    assert search.truncated and len(search.roots) == 2
    assert search.roots[0].x < search.roots[1].x


def test_no_crossing_gives_no_root() -> None:
    assert rt.find_roots(X, SINE + 5.0).roots == []


def test_the_tolerance_is_asked_for_in_digits() -> None:
    assert rt.xtol_from_digits(9) == pytest.approx(1e-9)
    assert rt.xtol_from_digits(99) == pytest.approx(1e-15)
    assert rt.xtol_from_digits("nonsense") == pytest.approx(1e-9)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy warns as the secant runs away - which is the point
def test_newton_refuses_a_root_outside_its_bracket() -> None:
    notes: list[str] = []
    # A function whose secant step from the bracket's middle jumps far out.
    found = rt.refine(lambda v: np.arctan(v - 10.0), 0.0, 1.0, rt.SOLVER_NEWTON, xtol=1e-9, max_iter=50, notes=notes)
    assert found is None and notes and "No root in [0, 1]" in notes[0]


def test_a_saddle_has_two_crossing_lines_at_level_zero() -> None:
    gx, gy = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))
    search = rt.find_level_curve(gx, gy, gx * gy, level=0.0)
    assert search.curves >= 2
    for root in search.roots:
        assert abs(root.x * root.y) < 1e-9 and root.z == 0.0 and root.method == "contour"


def test_a_circle_is_found_as_one_closed_curve_of_the_right_radius() -> None:
    gx, gy = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-2, 2, 81))
    search = rt.find_level_curve(gx, gy, gx**2 + gy**2, level=1.0, limit=10_000)
    assert search.curves == 1
    radii = [np.hypot(root.x, root.y) for root in search.roots]
    assert radii == pytest.approx([1.0] * len(radii), abs=5e-3)
