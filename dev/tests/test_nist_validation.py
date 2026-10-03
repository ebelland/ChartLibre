"""ChartLibre against the NIST Statistical Reference Datasets (todo R-13).

The certified values come with the data in dev/tests/data/nist; agreement is
counted in correct significant digits (LRE, see _nist_strd.py), the measure
NIST and the reviews of commercial packages use. Each threshold below is a
floor with some margin under what ChartLibre measured when the test was
written (2026-10-01), so a regression in the arithmetic shows up here
first. ``python3 dev/validation/nist_report.py`` prints the full table.

What was measured, worst case per family:

* Nonlinear least squares, 27 problems x 2 NIST starting points, trust
  region (the default) and Levenberg-Marquardt at the Fit dialog's default
  budget of 800 evaluations: 6.5 digits or more in every parameter and 6.0
  or more in every standard error. The exceptions - one start that needs a
  larger budget, one that Levenberg-Marquardt does not converge from, one
  exact-fit problem - are stated where they are tested.
* Linear least squares - polynomials up to degree 10 (Filip, which many
  packages fail): 6.2 digits or more through the Interpolation polynomial,
  7 or more through the Fit engine.
* Summary statistics: mean 15 digits, standard deviation 8.3 or more.
* One-way ANOVA: 10 digits or more in F, except where the data themselves
  cannot be stored to more than about four (SmLs07-09).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest

from app.analysis.fit import fit_curve
from app.analysis.interpolation import MODEL_POLYNOMIAL, InterpolationSettings, interpolate
from app.analysis.statistics import describe, group_tests
from app.functions.functions import linear, quadratic
from app.functions.optimizers import LEVENBERG, TRUST_REGION
from dev.tests import _nist_strd as nist
from dev.tests._nist_models import LOG_RESPONSE, MODELS
from dev.tests._cases import check_all

# ----------------------------------------------------------------------
# Nonlinear regression: the Fit engine
# ----------------------------------------------------------------------
#: The Fit dialog's default evaluation budget.
DEFAULT_BUDGET = 800
#: MGH17 from its first starting point needs more than the default budget
#: (it reaches 8 digits with this one) - as NIST warns it might: the start
#: is far from the answer on purpose.
LARGER_BUDGET = {("MGH17", 1): 5000}
#: Lanczos1 is generated from the model with no noise, so its certified
#: residual sum of squares is 1.4e-25 - rounding error. The parameters are
#: right to 10.6 digits; the residuals, and the standard errors computed
#: from them, cannot be.
EXACT_FIT = frozenset({"Lanczos1"})


#: Levenberg-Marquardt ignores bounds and takes BoxBOD's first, far start
#: to a different valley; trust region from the same start, and both from
#: the second, find the certified answer.
DIVERGES = frozenset({("BoxBOD", 1, LEVENBERG)})


#: Every (problem, NIST start, optimiser) that converges - all but DIVERGES.
NONLINEAR_CASES: list[tuple[str, int, str]] = [
    (name, start, optimizer)
    for name in MODELS
    for start in (1, 2)
    for optimizer in (TRUST_REGION, LEVENBERG)
    if (name, start, optimizer) not in DIVERGES
]


def _the_fit_engine_reproduces_the_certified_nonlinear_results(name: str, start: int, optimizer: str) -> None:
    data = nist.nonlinear(name)
    target = np.log(data.y) if name in LOG_RESPONSE else data.y
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # The far starting points send the optimiser through overflow on
        # its way in; that is the point of them.
        warnings.simplefilter("ignore")
        fit = fit_curve(
            MODELS[name],
            data.x,
            target,
            data.start1 if start == 1 else data.start2,
            optimizer=optimizer,
            max_nfev=LARGER_BUDGET.get((name, start), DEFAULT_BUDGET),
        )

    digits = nist.NONLINEAR_DIGITS
    params = [nist.lre(value, certified, digits) for value, certified in zip(fit.params, data.params)]
    assert min(params) >= 6.0, f"{name} start {start}: parameters to {params}"
    if name in EXACT_FIT:
        return
    std = [nist.lre(value, certified, digits) for value, certified in zip(fit.std, data.std)]
    assert min(std) >= 6.0, f"{name} start {start}: standard errors to {std}"
    assert nist.lre(fit.metrics["ss_res"], data.rss, digits) >= 9.0


def test_the_fit_engine_reproduces_the_certified_nonlinear_results() -> None:
    check_all(_the_fit_engine_reproduces_the_certified_nonlinear_results, NONLINEAR_CASES)


def test_the_known_divergence_still_diverges() -> None:
    """BoxBOD from its far start under Levenberg-Marquardt: if it starts
    converging, DIVERGES should lose it and the main test gain it."""
    for case in DIVERGES:
        with pytest.raises(AssertionError):
            _the_fit_engine_reproduces_the_certified_nonlinear_results(*case)


# ----------------------------------------------------------------------
# Linear regression: polynomials, two ways
# ----------------------------------------------------------------------
_CASES_THE_INTERPOLATION_POLYNOMIAL_REPRODUCES_THE_CERTIFIED_COEFFICIENTS = [
        ("Norris", 12.0),
        ("Pontius", 12.0),
        ("Filip", 7.0),
        ("Wampler1", 8.5),
        ("Wampler2", 12.0),
        ("Wampler3", 8.5),
        ("Wampler4", 7.5),
        ("Wampler5", 6.0),
    ]


def _the_interpolation_polynomial_reproduces_the_certified_coefficients(name: str, floor: float) -> None:
    data = nist.linear(name)
    degree = data.params.size - 1
    order = np.argsort(data.x, kind="stable")
    x, y = data.x[order], data.y[order]
    fitted = interpolate(MODEL_POLYNOMIAL, x, y, x, InterpolationSettings(degree=degree))
    # c0 is the highest power, as np.polyfit returns them; NIST lists B0 first.
    coefficients = [fitted.params[f"c{i}"] for i in range(degree + 1)][::-1]

    digits = [nist.lre(value, certified) for value, certified in zip(coefficients, data.params)]
    assert min(digits) >= floor, f"{name}: {digits}"
    residual = y - fitted.y
    residual_sd = float(np.sqrt(residual @ residual / (y.size - degree - 1)))
    assert nist.lre(residual_sd, data.residual_sd) >= 8.0


def test_the_interpolation_polynomial_reproduces_the_certified_coefficients() -> None:
    check_all(_the_interpolation_polynomial_reproduces_the_certified_coefficients, _CASES_THE_INTERPOLATION_POLYNOMIAL_REPRODUCES_THE_CERTIFIED_COEFFICIENTS)


_CASES_THE_FIT_ENGINE_REPRODUCES_THE_CERTIFIED_LINEAR_FITS = [("Norris", linear), ("Pontius", quadratic)]


def _the_fit_engine_reproduces_the_certified_linear_fits(name: str, function: type) -> None:
    data = nist.linear(name)
    start = np.asarray(function.initial_guess(data.x, data.y), dtype=float)
    fit = fit_curve(function.execute, data.x, data.y, start)

    assert min(nist.lre(value, certified) for value, certified in zip(fit.params, data.params)) >= 7.5
    assert min(nist.lre(value, certified) for value, certified in zip(fit.std, data.std)) >= 6.5
    assert nist.lre(fit.metrics["r2"], data.r_squared) >= 12.0


def test_the_fit_engine_reproduces_the_certified_linear_fits() -> None:
    check_all(_the_fit_engine_reproduces_the_certified_linear_fits, _CASES_THE_FIT_ENGINE_REPRODUCES_THE_CERTIFIED_LINEAR_FITS)


_CASES_A_LINE_THROUGH_THE_ORIGIN = [(case,) for case in ["NoInt1", "NoInt2"]]


def _a_line_through_the_origin(name: str) -> None:
    """y = b1 x. R² is not compared: NIST certifies the uncentred one a model
    without intercept calls for, the Fit reports the usual centred one."""
    data = nist.linear(name)
    fit = fit_curve(lambda x, p: p[0] * x, data.x, data.y, np.array([1.0]))
    assert nist.lre(fit.params[0], data.params[0]) >= 10.0
    assert nist.lre(fit.std[0], data.std[0]) >= 9.0


def test_a_line_through_the_origin() -> None:
    check_all(_a_line_through_the_origin, _CASES_A_LINE_THROUGH_THE_ORIGIN)


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------
_CASES_THE_SUMMARY_STATISTICS_REPRODUCE_THE_CERTIFIED_MEAN_AND_DEVIATION = [(case,) for case in ["PiDigits", "Lottery", "Lew", "Mavro", "Michelso", "NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4"]]


def _the_summary_statistics_reproduce_the_certified_mean_and_deviation(name: str) -> None:
    data = nist.univariate(name)
    summary = describe(data.values)
    assert nist.lre(summary["mean"], data.mean) >= 14.0
    # NumAcc3/4: a spread of 0.1 on values of 1e6 and 1e7 - the hardest.
    assert nist.lre(summary["std"], data.std) >= 8.0


def test_the_summary_statistics_reproduce_the_certified_mean_and_deviation() -> None:
    check_all(_the_summary_statistics_reproduce_the_certified_mean_and_deviation, _CASES_THE_SUMMARY_STATISTICS_REPRODUCE_THE_CERTIFIED_MEAN_AND_DEVIATION)


# ----------------------------------------------------------------------
# One-way analysis of variance
# ----------------------------------------------------------------------
_CASES_THE_ANOVA_REPRODUCES_THE_CERTIFIED_F_STATISTIC = [
        ("SiRstv", 12.0),
        ("SmLs01", 14.0),
        ("SmLs02", 14.0),
        ("SmLs03", 13.0),
        ("AtmWtAg", 9.5),
        ("SmLs04", 9.5),
        ("SmLs05", 9.5),
        ("SmLs06", 9.5),
        # 1000000000000.4 is stored to the nearest double, 6e-5 away, and the
        # spread the test is about is 0.1: no double-precision program can
        # recover more than about four digits of F from these.
        ("SmLs07", 4.0),
        ("SmLs08", 4.0),
        ("SmLs09", 4.0),
    ]


def _the_anova_reproduces_the_certified_f_statistic(name: str, floor: float) -> None:
    data = nist.anova(name)
    anova = group_tests(data.groups)[0]
    assert anova["n"] == sum(values.size for values in data.groups.values())
    assert f"df = {data.between_df}, {data.within_df}" in anova["note"]
    assert nist.lre(float(anova["statistic"]), data.f_statistic) >= floor


def test_the_anova_reproduces_the_certified_f_statistic() -> None:
    check_all(_the_anova_reproduces_the_certified_f_statistic, _CASES_THE_ANOVA_REPRODUCES_THE_CERTIFIED_F_STATISTIC)


# ----------------------------------------------------------------------
# Polynomials through the Fit engine: solved exactly, from any start
# ----------------------------------------------------------------------
def _polynomial(degree: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def model(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(x, p[: degree + 1])

    return model


_CASES_THE_FIT_ENGINE_SOLVES_A_POLYNOMIAL_EXACTLY_FROM_ANY_START = [("Wampler1", 9.0), ("Wampler2", 13.0), ("Wampler3", 8.5), ("Wampler4", 9.0), ("Wampler5", 7.0), ("Filip", 7.0)]


def _the_fit_engine_solves_a_polynomial_exactly_from_any_start(name: str, floor: float) -> None:
    """Iterated, the optimiser took polyfit's 7.3 digits on Filip down to 5.4
    and its errors to none; a model linear in its parameters is now solved."""
    data = nist.linear(name)
    degree = data.params.size - 1
    fit = fit_curve(_polynomial(degree), data.x, data.y, np.zeros(degree + 1))
    assert "solved exactly" in fit.message
    assert min(nist.lre(value, certified) for value, certified in zip(fit.params, data.params)) >= floor
    assert min(nist.lre(value, certified) for value, certified in zip(fit.std, data.std)) >= 7.0


def test_the_fit_engine_solves_a_polynomial_exactly_from_any_start() -> None:
    check_all(_the_fit_engine_solves_a_polynomial_exactly_from_any_start, _CASES_THE_FIT_ENGINE_SOLVES_A_POLYNOMIAL_EXACTLY_FROM_ANY_START)


def test_a_nonlinear_model_is_not_taken_for_a_linear_one() -> None:
    data = nist.nonlinear("Misra1a")
    fit = fit_curve(MODELS["Misra1a"], data.x, data.y, data.start1)
    assert "solved exactly" not in fit.message


def test_a_linear_answer_outside_the_bounds_falls_back_to_the_optimiser() -> None:
    data = nist.linear("Norris")
    # Certified slope is 1.002; bounded below that, the exact answer is refused.
    fit = fit_curve(_polynomial(1), data.x, data.y, np.array([0.0, 0.5]), upper=np.array([np.inf, 0.9]))
    assert "solved exactly" not in fit.message
    assert fit.params[1] <= 0.9 + 1e-12
