"""Control charts (app.analysis.control_charts): limits worked by hand, flags where planted."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import control_charts as cc


def test_individuals_limits_come_from_the_moving_range_not_the_spread() -> None:
    # Moving ranges 2, 2, 2, 2: mean 2, sigma = 2 / d2(2) = 2 / 1.128.
    y = np.array([10.0, 12.0, 10.0, 12.0, 10.0])
    chart = cc.variables_chart(cc.CHART_INDIVIDUALS, np.arange(5.0), y)
    sigma = 2.0 / 1.128
    assert chart.center == pytest.approx(10.8)
    assert chart.sigma == pytest.approx(sigma)
    assert chart.upper == pytest.approx(10.8 + 3.0 * sigma)
    assert chart.lower == pytest.approx(10.8 - 3.0 * sigma)
    assert chart.violations == []


def test_a_shifted_process_is_flagged_even_though_its_overall_spread_is_wide() -> None:
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(10.0, 0.5, 30), rng.normal(14.0, 0.5, 30)])
    chart = cc.variables_chart(cc.CHART_INDIVIDUALS, np.arange(60.0), y, nelson=False)
    # Three sigma of all the data would hold every point; the within-process
    # sigma does not.
    assert 10.0 + 3.0 * float(np.std(y)) > y.max()
    assert any(v.index >= 30 for v in chart.violations)


def test_the_moving_range_chart_uses_d3_d4() -> None:
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    chart = cc.variables_chart(cc.CHART_MOVING_RANGE, np.arange(5.0), y)
    np.testing.assert_allclose(chart.y, [2.0, 1.0, 3.0, 1.0])
    assert chart.x.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert chart.center == pytest.approx(1.75)
    assert chart.upper == pytest.approx(3.267 * 1.75)
    assert chart.lower == 0.0


def test_xbar_r_plots_subgroup_means_against_the_standard_error() -> None:
    y = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 9.0])  # the 9th is dropped
    chart = cc.variables_chart(cc.CHART_XBAR_R, np.arange(9.0), y, subgroup=4)
    np.testing.assert_allclose(chart.y, [4.0, 5.0])
    assert chart.subgroup_size == 4
    assert chart.details["dropped"] == 1
    sigma = 6.0 / 2.059  # mean range 6, d2(4)
    assert chart.sigma == pytest.approx(sigma)
    assert chart.upper == pytest.approx(4.5 + 3.0 * sigma / 2.0)


def test_too_few_subgroups_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        cc.variables_chart(cc.CHART_XBAR_S, np.arange(6.0), np.ones(6), subgroup=5)


def test_spc_constants_fall_back_to_the_nearest_smaller_row() -> None:
    assert cc.spc_constants(5) == (cc.SPC_CONSTANTS[5], True)
    assert cc.spc_constants(17) == (cc.SPC_CONSTANTS[15], False)


def test_nelson_rule_2_flags_a_run_on_one_side() -> None:
    y = np.array([0.0, 1.0, -1.0, 1.0, -1.0] * 4 + [0.5] * 9)
    chart = cc.variables_chart(cc.CHART_INDIVIDUALS, np.arange(y.size, dtype=float), y)
    assert any(2 in v.rules for v in chart.violations)
    plain = cc.variables_chart(cc.CHART_INDIVIDUALS, np.arange(y.size, dtype=float), y, nelson=False)
    assert all(v.rules == (1,) for v in plain.violations)


def test_excluding_violations_tightens_the_limits() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, 50)
    y[[10, 30]] = [25.0, -25.0]
    x = np.arange(50.0)
    first = cc.variables_chart(cc.CHART_INDIVIDUALS, x, y, nelson=False)
    revised = cc.variables_chart(cc.CHART_INDIVIDUALS, x, y, nelson=False, exclude_violations=True)
    assert revised.sigma < first.sigma
    assert revised.details["excluded"] == len(first.violations)
    assert revised.notes == ()


def test_excluding_everything_keeps_every_point_and_says_so() -> None:
    # Two plateaus: every point sits far from the centre line between them.
    y = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0])
    chart = cc.variables_chart(cc.CHART_INDIVIDUALS, np.arange(8.0), y, exclude_violations=True)
    assert len(chart.violations) == 8
    assert chart.notes and "too few points" in chart.notes[0]
    assert "excluded" not in chart.details
    assert chart.center == pytest.approx(5.0)


def test_p_chart_limits_vary_with_the_sample_size() -> None:
    counts = np.array([5.0, 10.0, 5.0, 20.0])
    sizes = np.array([50.0, 100.0, 50.0, 200.0])
    chart = cc.attribute_chart(cc.CHART_P, np.arange(4.0), counts, sizes)
    pbar = 40.0 / 400.0
    assert chart.center == pytest.approx(pbar)
    np.testing.assert_allclose(chart.y, [0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(chart.upper_band, pbar + 3.0 * np.sqrt(pbar * (1 - pbar) / sizes))
    assert chart.details["limits"] == "vary with the sample size"
    assert "only 4 subgroups" in chart.details["note"]


def test_c_chart_is_poisson_and_clipped_at_zero() -> None:
    counts = np.array([1.0, 2.0, 0.0, 1.0])
    chart = cc.attribute_chart(cc.CHART_C, np.arange(4.0), counts, np.ones(4), recommended_subgroups=2)
    assert chart.center == pytest.approx(1.0)
    np.testing.assert_allclose(chart.upper_band, 4.0)
    np.testing.assert_allclose(chart.lower_band, 0.0)
    assert "note" not in chart.details


def test_an_attribute_spike_is_flagged_and_excluded() -> None:
    rng = np.random.default_rng(5)
    sizes = np.full(30, 100.0)
    counts = rng.binomial(100, 0.05, 30).astype(float)
    counts[12] = 40.0
    x = np.arange(30.0)
    first = cc.attribute_chart(cc.CHART_NP, x, counts, sizes, nelson=False)
    assert [v.index for v in first.violations] == [12]
    revised = cc.attribute_chart(cc.CHART_NP, x, counts, sizes, nelson=False, exclude_violations=True)
    assert revised.center < first.center
    assert revised.details["excluded"] == 1
    assert revised.upper_band.size == 30
