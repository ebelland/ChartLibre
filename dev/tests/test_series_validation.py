"""What the pre-flight check catches, and what it must not reject.

The second half matters as much as the first: a validator that rejects data an
operation handles perfectly well is worse than none, because it stops real work
and teaches people to distrust it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.utils.series_validation import DUPLICATE_X, EMPTY, LENGTH_MISMATCH, UNSORTED_X, clean_xy, errors, validate_xy
from dev.tests._cases import check_all


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


# ----------------------------------------------------------------------
# Clean input is left alone
# ----------------------------------------------------------------------

def test_a_good_series_produces_no_issues() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = np.sin(x)
    assert validate_xy(x, y) == []


# ----------------------------------------------------------------------
# The three that give wrong answers rather than errors
# ----------------------------------------------------------------------

def test_unsorted_x_is_reported_when_the_operation_needs_order() -> None:
    x = np.array([0.0, 3.0, 1.0, 2.0])
    issues = validate_xy(x, x.copy(), require_sorted_x=True)
    assert UNSORTED_X in codes(issues)


def test_duplicate_x_is_an_error_for_an_interpolating_operation() -> None:
    x = np.array([0.0, 1.0, 1.0, 2.0])
    issues = validate_xy(x, np.array([0.0, 5.0, 9.0, 1.0]), require_unique_x=True)
    assert DUPLICATE_X in codes(issues)
    assert errors(issues), "a spline through two y at one x has no solution"


# ----------------------------------------------------------------------
# Shape and size
# ----------------------------------------------------------------------

_CASES_AN_EMPTY_SERIES_IS_AN_ERROR = [([], []), (None, None), ([], [1.0])]


def _an_empty_series_is_an_error(x, y) -> None:
    assert EMPTY in codes(validate_xy(x, y))


def test_an_empty_series_is_an_error() -> None:
    check_all(_an_empty_series_is_an_error, _CASES_AN_EMPTY_SERIES_IS_AN_ERROR)


def test_mismatched_lengths_are_an_error() -> None:
    issues = validate_xy(np.arange(5.0), np.arange(3.0))
    assert LENGTH_MISMATCH in codes(issues)


# ----------------------------------------------------------------------
# Spacing and variation
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Labelling
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Repair
# ----------------------------------------------------------------------

def test_clean_drops_non_finite_pairs_and_says_how_many() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, np.nan, 3.0, np.inf])
    x_out, y_out, report = clean_xy(x, y)
    assert x_out.tolist() == [0.0, 2.0]
    assert y_out.tolist() == [1.0, 3.0]
    assert report.dropped_non_finite == 2
    assert "dropped 2" in report.describe()


def test_clean_averages_duplicate_x() -> None:
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0, 30.0])
    x_out, y_out, report = clean_xy(x, y, merge_duplicate_x=True)
    assert x_out.tolist() == [0.0, 1.0, 2.0]
    assert y_out.tolist() == [0.0, 15.0, 30.0]
    assert report.merged_duplicates == 1


# ----------------------------------------------------------------------
# Severity depends on whether a repair is coming
# ----------------------------------------------------------------------


