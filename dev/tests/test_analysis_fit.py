"""The fit engine against certified reference results.

NIST StRD Misra1a (https://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml):
y = b1 * (1 - exp(-b2 * x)), with certified estimates and standard errors -
the dataset commercial packages use to validate nonlinear regression.
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from app.analysis import fit

MISRA1A_X = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0])
MISRA1A_Y = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78])
CERTIFIED = np.array([2.3894212918e02, 5.5015643181e-04])
CERTIFIED_STD = np.array([2.7070075241e00, 7.2668688436e-06])
CERTIFIED_SS_RES = 1.2455138894e-01


def misra1a(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * (1.0 - np.exp(-p[1] * np.asarray(x, dtype=float)))


@pytest.mark.parametrize("start", [(500.0, 1e-4), (250.0, 5e-4)])
def test_misra1a_matches_the_certified_values(start) -> None:
    result = fit.fit_curve(misra1a, MISRA1A_X, MISRA1A_Y, np.array(start))
    assert result.success
    np.testing.assert_allclose(result.params, CERTIFIED, rtol=1e-6)
    np.testing.assert_allclose(result.std, CERTIFIED_STD, rtol=1e-4)
    assert result.metrics["ss_res"] == pytest.approx(CERTIFIED_SS_RES, rel=1e-6)
    assert result.dof == 12


def test_parameter_inference_uses_student_t() -> None:
    result = fit.fit_curve(misra1a, MISRA1A_X, MISRA1A_Y, np.array([250.0, 5e-4]))
    np.testing.assert_allclose(result.tvalues, CERTIFIED / CERTIFIED_STD, rtol=1e-4)
    assert np.all(result.pvalues < 1e-10)
    assert np.all(result.ci_low < result.params) and np.all(result.params < result.ci_high)


def test_bands_of_a_straight_line_agree_with_statsmodels() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 30)
    y = 1.5 + 0.8 * x + rng.normal(0, 0.6, x.size)

    def line(xv: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * np.asarray(xv, dtype=float)

    result = fit.fit_curve(line, x, y, np.array([0.0, 1.0]))
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(result.params, ols.params, rtol=1e-6)
    np.testing.assert_allclose(result.std, ols.bse, rtol=1e-4)
    np.testing.assert_allclose(result.pvalues, ols.pvalues, rtol=1e-3)

    frame = ols.get_prediction(sm.add_constant(x)).summary_frame(alpha=0.05)
    low, high = fit.confidence_band(line, x, result.params, result.cov, dof=result.dof)
    np.testing.assert_allclose(low, frame["mean_ci_lower"], rtol=1e-4)
    np.testing.assert_allclose(high, frame["mean_ci_upper"], rtol=1e-4)
    low, high = fit.confidence_band(
        line, x, result.params, result.cov,
        dof=result.dof, residual_variance=result.metrics["reduced_chi2"], prediction=True,
    )
    np.testing.assert_allclose(low, frame["obs_ci_lower"], rtol=1e-4)
    np.testing.assert_allclose(high, frame["obs_ci_upper"], rtol=1e-4)


def test_evaluating_without_optimising_keeps_the_parameters() -> None:
    p = np.array([240.0, 5.5e-4])
    result = fit.fit_curve(misra1a, MISRA1A_X, MISRA1A_Y, p, optimise=False)
    np.testing.assert_array_equal(result.params, p)
    assert np.all(np.isfinite(result.std))  # a hand-tuned curve still gets error bars


def test_a_fixed_parameter_has_no_error() -> None:
    result = fit.fit_curve(
        misra1a, MISRA1A_X, MISRA1A_Y, np.array([240.0, 5e-4]), fixed=np.array([True, False])
    )
    assert result.params[0] == 240.0
    assert np.isnan(result.std[0]) and np.isfinite(result.std[1])
    assert result.dof == 13


def test_multi_peak_model_recovers_two_gaussians() -> None:
    x = np.linspace(0, 10, 400)
    model = fit.multi_peak_model("Gaussian", 2)
    truth = np.array([3.0, 3.0, 0.5, 2.0, 6.5, 0.8, 0.2])
    y = model(x, truth)
    result = fit.fit_curve(model, x, y, np.array([2.5, 2.8, 0.6, 1.5, 6.3, 1.0, 0.0]))
    np.testing.assert_allclose(result.params, truth, rtol=1e-5, atol=1e-7)


def test_a_band_series_is_drawn_as_a_filled_region() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    from app.charts.base import SeriesData
    from app.charts.render_figure import _draw_band
    from app.data.series_frame import SeriesFrame

    ax = Figure().add_subplot()
    frame = SeriesFrame({"x": [3.0, 1.0, 2.0], "ci_low": [0.0, 0.0, 0.0], "ci_high": [1.0, 2.0, 3.0]})
    band = SeriesData(
        name="band", df=frame, style={"draw_as": "band", "label": "95% band"},
        roles={"x": "x", "y": "ci_low", "y2": "ci_high"},
    )
    _draw_band(ax, band)
    fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(fills) == 1 and fills[0].get_label() == "95% band"


@pytest.mark.parametrize("optimizer", ["trf", "nelder-mead", "monte-carlo", "differential-evolution"])
def test_a_stop_request_ends_the_fit(optimizer) -> None:
    """Every family: least squares, a simplex, and the global searches that
    catch every error a sample raises - the stop has to get through them."""
    calls = {"n": 0}

    def stop_after_five() -> bool:
        calls["n"] += 1
        return calls["n"] > 5

    with pytest.raises(fit.FitStopped):
        fit.fit_curve(
            misra1a, MISRA1A_X, MISRA1A_Y, np.array([500.0, 1e-4]),
            optimizer=optimizer, should_stop=stop_after_five,
        )
    assert calls["n"] == 6


def test_an_unknown_optimizer_is_an_error_even_for_a_linear_model() -> None:
    """A line is solved exactly without an optimiser; a misspelt one is still caught."""
    with pytest.raises(ValueError, match="Unknown optimizer"):
        fit.fit_curve(lambda x, p: p[0] * x, MISRA1A_X, MISRA1A_Y, np.array([1.0]), optimizer="levenberg-marquardt")
