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

import numpy as np
import pytest

from app.analysis.fit import fit_curve
from app.analysis.interpolation import MODEL_POLYNOMIAL, InterpolationSettings, interpolate
from app.analysis.statistics import describe, group_tests
from app.functions.functions import linear, quadratic
from app.functions.optimizers import LEVENBERG, TRUST_REGION
from dev.tests import _nist_strd as nist
from dev.tests._nist_models import LOG_RESPONSE, MODELS

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


def _nonlinear_cases() -> list[object]:
    return [
        pytest.param(
            name, start, optimizer,
            marks=[pytest.mark.xfail(strict=True, reason="does not converge from here")]
            if (name, start, optimizer) in DIVERGES else [],
        )
        for name in MODELS
        for start in (1, 2)
        for optimizer in (TRUST_REGION, LEVENBERG)
    ]


@pytest.mark.parametrize(("name", "start", "optimizer"), _nonlinear_cases())
def test_the_fit_engine_reproduces_the_certified_nonlinear_results(name: str, start: int, optimizer: str) -> None:
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


# ----------------------------------------------------------------------
# Linear regression: polynomials, two ways
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "floor"),
    [
        ("Norris", 12.0),
        ("Pontius", 12.0),
        ("Filip", 7.0),
        ("Wampler1", 8.5),
        ("Wampler2", 12.0),
        ("Wampler3", 8.5),
        ("Wampler4", 7.5),
        ("Wampler5", 6.0),
    ],
)
def test_the_interpolation_polynomial_reproduces_the_certified_coefficients(name: str, floor: float) -> None:
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


@pytest.mark.parametrize(("name", "function"), [("Norris", linear), ("Pontius", quadratic)])
def test_the_fit_engine_reproduces_the_certified_linear_fits(name: str, function: type) -> None:
    data = nist.linear(name)
    start = np.asarray(function.initial_guess(data.x, data.y), dtype=float)
    fit = fit_curve(function.execute, data.x, data.y, start)

    assert min(nist.lre(value, certified) for value, certified in zip(fit.params, data.params)) >= 7.5
    assert min(nist.lre(value, certified) for value, certified in zip(fit.std, data.std)) >= 6.5
    assert nist.lre(fit.metrics["r2"], data.r_squared) >= 12.0


@pytest.mark.parametrize("name", ["NoInt1", "NoInt2"])
def test_a_line_through_the_origin(name: str) -> None:
    """y = b1 x. R² is not compared: NIST certifies the uncentred one a model
    without intercept calls for, the Fit reports the usual centred one."""
    data = nist.linear(name)
    fit = fit_curve(lambda x, p: p[0] * x, data.x, data.y, np.array([1.0]))
    assert nist.lre(fit.params[0], data.params[0]) >= 10.0
    assert nist.lre(fit.std[0], data.std[0]) >= 9.0


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["PiDigits", "Lottery", "Lew", "Mavro", "Michelso", "NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4"]
)
def test_the_summary_statistics_reproduce_the_certified_mean_and_deviation(name: str) -> None:
    data = nist.univariate(name)
    summary = describe(data.values)
    assert nist.lre(summary["mean"], data.mean) >= 14.0
    # NumAcc3/4: a spread of 0.1 on values of 1e6 and 1e7 - the hardest.
    assert nist.lre(summary["std"], data.std) >= 8.0


# ----------------------------------------------------------------------
# One-way analysis of variance
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "floor"),
    [
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
    ],
)
def test_the_anova_reproduces_the_certified_f_statistic(name: str, floor: float) -> None:
    data = nist.anova(name)
    anova = group_tests(data.groups)[0]
    assert anova["n"] == sum(values.size for values in data.groups.values())
    assert f"df = {data.between_df}, {data.within_df}" in anova["note"]
    assert nist.lre(float(anova["statistic"]), data.f_statistic) >= floor
