"""The numbers behind the diagnostic plots (todo R-06), against published values."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from app.analysis import diagnostics as dg

# The 6-MP leukaemia trial (Freireich et al. 1963): weeks in remission.
TREATED_T = np.array([6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35], float)
TREATED_E = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], float)
PLACEBO_T = np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23], float)
PLACEBO_E = np.ones(21)


def test_kaplan_meier_matches_the_textbook_and_statsmodels() -> None:
    curve = dg.kaplan_meier(TREATED_T, TREATED_E)
    reference = SurvfuncRight(TREATED_T, TREATED_E)
    np.testing.assert_allclose(curve.time[1:], reference.surv_times)
    np.testing.assert_allclose(curve.survival[1:], reference.surv_prob)
    np.testing.assert_allclose(curve.survival, [1, 0.857, 0.807, 0.753, 0.690, 0.627, 0.538, 0.448], atol=5e-4)
    assert curve.median == 23.0
    # Greenwood on the log(-log) scale: 0.857 (0.620, 0.952) at six weeks.
    assert curve.lower[1] == pytest.approx(0.620, abs=5e-4)
    assert curve.upper[1] == pytest.approx(0.952, abs=5e-4)
    assert list(curve.censored_time) == [6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35]
    assert curve.censored_survival[0] == pytest.approx(0.857, abs=5e-4)


def test_a_curve_that_never_halves_has_no_median_and_negative_times_are_refused() -> None:
    assert np.isnan(dg.kaplan_meier(np.array([1.0, 2, 3, 4]), np.array([0, 1, 0, 0])).median)
    with pytest.raises(ValueError):
        dg.kaplan_meier(np.array([-1.0, 2]), np.array([1, 1]))


def test_log_rank_matches_statsmodels() -> None:
    test = dg.log_rank([(TREATED_T, TREATED_E), (PLACEBO_T, PLACEBO_E)])
    statistic, pvalue = survdiff(
        np.concatenate([TREATED_T, PLACEBO_T]), np.concatenate([TREATED_E, PLACEBO_E]),
        np.repeat([0, 1], 21),
    )
    assert test.statistic == pytest.approx(statistic) == pytest.approx(16.79, abs=0.01)
    assert test.pvalue == pytest.approx(pvalue)
    assert test.dof == 1


def test_three_groups_log_rank_matches_statsmodels() -> None:
    rng = np.random.default_rng(3)
    groups = [(rng.exponential(scale, 30).round(1), rng.integers(0, 2, 30)) for scale in (5.0, 8.0, 12.0)]
    test = dg.log_rank(groups)
    statistic, pvalue = survdiff(
        np.concatenate([t for t, _e in groups]), np.concatenate([e for _t, e in groups]).astype(float),
        np.repeat([0, 1, 2], 30),
    )
    assert test.statistic == pytest.approx(statistic) and test.pvalue == pytest.approx(pvalue) and test.dof == 2


def test_plotting_positions_are_r_ppoints() -> None:
    np.testing.assert_allclose(dg.plotting_positions(5), (np.arange(1, 6) - 0.375) / 5.25)
    np.testing.assert_allclose(dg.plotting_positions(20), (np.arange(1, 21) - 0.5) / 20)


def test_normal_qq_plot_is_qqnorm_with_qqline() -> None:
    sample = np.random.default_rng(1).normal(5.0, 2.0, 50)
    plot = dg.qq_plot(sample)
    np.testing.assert_allclose(plot.x, stats.norm.ppf(dg.plotting_positions(50)))
    np.testing.assert_allclose(plot.y, np.sort(sample))
    q1, q3 = np.quantile(sample, [0.25, 0.75])
    z1, z3 = stats.norm.ppf([0.25, 0.75])
    assert plot.slope == pytest.approx((q3 - q1) / (z3 - z1))
    assert plot.intercept == pytest.approx(q1 - plot.slope * z1)
    # Every point of a sample the distribution really produced sits mostly inside the band.
    inside = (plot.y >= plot.lower) & (plot.y <= plot.upper)
    assert inside.mean() > 0.9


def test_qq_in_data_units_lies_on_the_identity_for_a_good_fit() -> None:
    sample = stats.gamma(2.5, scale=3.0).rvs(size=400, random_state=4)
    plot = dg.qq_plot(sample, "gamma", standardized=False, line="identity")
    assert (plot.slope, plot.intercept) == (1.0, 0.0)
    assert np.median(np.abs(plot.y - plot.x)) < 0.5
    with pytest.raises(ValueError):
        dg.qq_plot(np.ones(10))


def test_pp_plot_is_the_fitted_cdf_against_the_positions() -> None:
    sample = np.random.default_rng(2).normal(size=30)
    plot = dg.pp_plot(sample)
    mean, sd = np.mean(sample), np.std(sample, ddof=1)
    np.testing.assert_allclose(plot.x, stats.norm.cdf(np.sort(sample), mean, sd))
    np.testing.assert_allclose(plot.y, dg.plotting_positions(30))
    assert np.all(plot.lower < plot.upper)


def test_cell_means_keep_the_levels_in_reading_order() -> None:
    means = dg.cell_means(
        np.array(["low", "high", "low", "high", "low", "mid"], dtype=object),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]),
        np.array(["A", "A", "B", "B", "A", "B"], dtype=object),
    )
    assert means.levels == ["low", "high"] and means.traces == ["A", "B"]
    np.testing.assert_allclose(means.mean, [[3.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(means.count, [[2, 1], [1, 1]])
    assert means.half_width("se")[0, 0] == pytest.approx(np.std([1, 5], ddof=1) / np.sqrt(2))
    numeric = dg.cell_means(np.array([10, 2, 10], dtype=object), np.array([1.0, 2.0, 3.0]))
    assert numeric.levels == [2, 10]


def test_mosaic_tiles_are_the_joint_proportions_and_the_test_is_pearsons() -> None:
    first = np.array(["a"] * 30 + ["b"] * 70, dtype=object)
    second = np.array(["x"] * 20 + ["y"] * 10 + ["x"] * 20 + ["y"] * 50, dtype=object)
    result = dg.mosaic(first, second, gap=0.0)
    areas = {(tile.column, tile.row): tile.width * tile.height for tile in result.tiles}
    assert areas == pytest.approx({("a", "x"): 0.2, ("a", "y"): 0.1, ("b", "x"): 0.2, ("b", "y"): 0.5})
    chi2, pvalue, dof, _expected = stats.chi2_contingency([[20, 10], [20, 50]], correction=False)
    assert result.chi_square == pytest.approx(chi2) and result.pvalue == pytest.approx(pvalue) and result.dof == dof
    weighted = dg.mosaic(np.array(["a", "a", "b", "b"], dtype=object), np.array(["x", "y", "x", "y"], dtype=object),
                         np.array([20.0, 10, 20, 50]), gap=0.0)
    assert weighted.chi_square == pytest.approx(chi2)
