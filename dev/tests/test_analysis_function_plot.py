"""Evaluating a function over a range (app.analysis.function_plot)."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import function_plot as fp


def test_a_linear_range_spans_the_requested_endpoints() -> None:
    assert fp.build_range(0.0, 10.0, 5, fp.SPACING_LINEAR).tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_a_log_range_is_evenly_spaced_in_decades() -> None:
    assert fp.build_range(1.0, 1000.0, 4, fp.SPACING_LOG).tolist() == pytest.approx([1.0, 10.0, 100.0, 1000.0])


def test_a_range_can_run_downwards_and_has_at_least_two_points() -> None:
    assert fp.build_range(10.0, 0.0, 3).tolist() == [10.0, 5.0, 0.0]
    assert fp.build_range(0.0, 1.0, 0).tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    ("args", "match"),
    [((3.0, 3.0, 10, fp.SPACING_LINEAR), "range is empty"),
     ((0.0, 10.0, 10, fp.SPACING_LOG), "strictly above zero"),
     ((-1.0, 10.0, 10, fp.SPACING_LOG), "strictly above zero")],
)
def test_an_empty_range_or_a_log_range_through_zero_is_refused(args: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fp.build_range(*args)


def test_a_curve_is_the_function_at_each_x() -> None:
    x = fp.build_range(-2.0, 2.0, 9)
    curve = fp.evaluate_curve(lambda x, p: p[0] * x**2 + p[1], x, np.array([3.0, 1.0]))
    np.testing.assert_allclose(curve.y, 3.0 * x**2 + 1.0)
    assert curve.undefined == 0


def test_the_undefined_part_is_counted_not_refused() -> None:
    x = fp.build_range(-1.0, 1.0, 11)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = fp.evaluate_curve(lambda x, p: np.log(x), x, np.array([]))
    assert curve.undefined == 6  # -1 ... 0 inclusive: log is nan below zero and -inf at zero
    assert np.count_nonzero(np.isfinite(curve.y)) == 5  # the five points above zero survive


def test_a_function_undefined_everywhere_is_refused() -> None:
    x = fp.build_range(-5.0, -1.0, 5)
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match="undefined everywhere"):
            fp.evaluate_curve(lambda x, p: np.sqrt(x), x, np.array([]))


def test_a_result_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="returned 1 value"):
        fp.evaluate_curve(lambda x, p: np.array([1.0]), np.arange(5.0), np.array([]))


def test_a_surface_is_evaluated_over_the_square_of_the_range_and_flattened() -> None:
    axis = fp.build_range(0.0, 2.0, 3)
    surface = fp.evaluate_surface(lambda xy, p: xy[:, 0] * 10 + xy[:, 1], axis, np.array([]))
    assert surface.x.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert surface.y.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert surface.z.tolist() == [0, 10, 20, 1, 11, 21, 2, 12, 22]
    assert surface.undefined == 0


def test_a_surface_undefined_everywhere_is_refused() -> None:
    with pytest.raises(ValueError, match="undefined everywhere"):
        fp.evaluate_surface(lambda xy, p: np.full(xy.shape[0], np.nan), np.arange(3.0), np.array([]))
