"""Randomness and serial-independence diagnostics (app/analysis/randomness.py)."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import randomness as rd
from app.analysis import statistics as st


def test_an_alternating_sequence_has_too_many_runs_and_turning_points() -> None:
    x = np.tile([1.0, -1.0], 20) + np.linspace(0, 0.01, 40)
    runs = rd.runs_test(x, centre=0.0)
    assert runs.available and runs.statistic > 5 and runs.pvalue < 1e-6
    turning = rd.turning_point_test(x)
    assert turning.statistic > 4 and turning.pvalue < 1e-4


def test_a_trend_has_too_few_runs_and_a_small_durbin_watson() -> None:
    x = np.linspace(0, 10, 60)
    assert rd.runs_test(x).statistic < -5
    assert rd.durbin_watson(x).statistic < 0.1


def test_white_noise_is_not_rejected() -> None:
    x = np.random.default_rng(3).normal(size=400)
    assert rd.runs_test(x).pvalue > 0.01
    assert rd.turning_point_test(x).pvalue > 0.01
    assert 1.7 < rd.durbin_watson(x).statistic < 2.3
    assert all(r.pvalue > 0.01 for r in rd.ljung_box(x, lags=(1, 5, 10)))


def test_the_runs_statistic_matches_the_textbook_formula() -> None:
    # + - + - ... : 10 runs, 5 above, 5 below
    x = np.array([1, -1] * 5, dtype=float)
    result = rd.runs_test(x, centre=0.0)
    expected, variance = 1 + 2 * 5 * 5 / 10, 2 * 5 * 5 * (2 * 5 * 5 - 10) / (100 * 9)
    assert result.statistic == pytest.approx((10 - expected - 0.5) / variance**0.5)
    assert "runs = 10" in result.note


def test_limits_become_notes_not_exceptions() -> None:
    assert not rd.runs_test([1.0, 2.0]).available and rd.runs_test([1.0, 2.0]).note
    assert "tie" in rd.turning_point_test([1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0]).note
    assert not rd.durbin_watson([0.0, 0.0, 0.0]).available
    assert "constant" in rd.ljung_box([2.0] * 30)[0].note
    assert np.isnan(rd.durbin_watson([1.0, 2.0, 0.5, 3.0]).pvalue)


def test_the_statistics_report_gets_one_row_per_diagnostic() -> None:
    rows = st.randomness_tests(np.random.default_rng(1).normal(size=100), lags=(1, 5))
    assert len(rows) == 5  # runs, turning points, Durbin-Watson, two Ljung-Box
