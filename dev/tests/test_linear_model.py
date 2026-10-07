"""Fit Model's engine against experiments small enough to work out by hand."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.analysis import linear_model as lm
from app.analysis.linear_model import CONTINUOUS, NOMINAL, Factor, ModelSpec


def _two_by_two(noise: tuple[float, ...] = (0.1, -0.1, 0.2, -0.2, -0.1, 0.1, -0.2, 0.2)) -> pd.DataFrame:
    """A 2x2 factorial, twice: y = 10 + 2A + 1B + 0.5AB, plus a known noise."""
    a = [-1, 1, -1, 1] * 2
    b = [-1, -1, 1, 1] * 2
    y = [10 + 2 * i + 1 * j + 0.5 * i * j + e for i, j, e in zip(a, b, noise)]
    return pd.DataFrame({"A": a, "B": b, "Y": y})


def test_estimates_are_half_effects_and_match_least_squares() -> None:
    frame = _two_by_two()
    factors = [Factor("A", low=-1, high=1), Factor("B", low=-1, high=1)]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.factorial(factors, 2)))[0]
    expected, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(8), frame.A, frame.B, frame.A * frame.B]), frame.Y, rcond=None
    )
    np.testing.assert_allclose(fit.estimates["Estimate"], expected, atol=1e-12)
    assert list(fit.estimates["Term"]) == ["Intercept", "A", "B", "A*B"]
    total = ((frame.Y - frame.Y.mean()) ** 2).sum()
    assert fit.anova.loc[2, "Sum of Squares"] == pytest.approx(total)
    assert fit.summary["Observations"] == 8
    assert 0.99 < fit.summary["RSquare"] <= 1
    assert fit.lack_of_fit is None  # four settings, four parameters: nothing left to lack


def test_continuous_factors_are_coded_from_their_range() -> None:
    frame = _two_by_two()
    frame["T"] = frame.A.map({-1: 100.0, 1: 200.0})
    factors = [Factor("T", low=100, high=200), Factor("B", low=-1, high=1)]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.main_effects(factors)))[0]
    # +1 coded unit is +50 degrees: the estimate is still half the effect over the range.
    assert fit.estimates.set_index("Term").loc["T", "Estimate"] == pytest.approx(2.0, abs=0.1)
    assert fit.coding["T"] == (150.0, 50.0)


def test_lack_of_fit_needs_replicates_and_spare_terms() -> None:
    frame = _two_by_two()
    factors = [Factor("A", low=-1, high=1), Factor("B", low=-1, high=1)]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.main_effects(factors)))[0]
    lack = fit.lack_of_fit
    assert lack is not None
    assert list(lack["DF"]) == [1, 4, 5]  # the missing A*B is the lack of fit
    assert lack.loc[0, "Sum of Squares"] == pytest.approx(8 * 0.5**2, rel=1e-6)


def test_a_nominal_factor_is_one_test_of_its_levels() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"Cat": list("abc") * 4, "A": [-1, 1] * 6})
    frame["Y"] = frame.Cat.map({"a": 1.0, "b": 2.0, "c": 6.0}) + frame.A + rng.normal(0, 0.05, 12)
    factors = [Factor("Cat", NOMINAL), Factor("A", low=-1, high=1)]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.main_effects(factors)))[0]
    tests = fit.effect_tests.set_index("Term")
    assert tests.loc["Cat", "DF"] == 2 and tests.loc["A", "DF"] == 1
    assert tests.loc["Cat", "Prob > F"] < 0.001
    levels = [t for t in fit.estimates["Term"] if t.startswith("Cat[")]
    assert levels == ["Cat[a]", "Cat[b]"]  # sum-to-zero: the last level is minus their sum
    a, b = fit.estimates.set_index("Term").loc[levels, "Estimate"]
    assert a == pytest.approx(1 - 3, abs=0.1) and b == pytest.approx(2 - 3, abs=0.1)


def test_reduction_drops_noise_but_keeps_the_hierarchy() -> None:
    rng = np.random.default_rng(7)
    a = np.tile([-1, 1, -1, 1], 4)
    b = np.tile([-1, -1, 1, 1], 4)
    c = np.repeat([-1, 1], 8)
    # A does nothing alone, but A*B does a lot; C does nothing at all.
    y = 5 + 3 * b + 2 * a * b + rng.normal(0, 0.1, 16)
    frame = pd.DataFrame({"A": a, "B": b, "C": c, "Y": y})
    factors = [Factor(n, low=-1, high=1) for n in "ABC"]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.factorial(factors, 2), reduce=True))[0]
    assert ("A", "B") in fit.terms and ("A",) in fit.terms and ("B",) in fit.terms
    assert ("C",) in fit.removed and ("A", "C") in fit.removed and ("B", "C") in fit.removed


def test_a_saturated_model_says_so_instead_of_testing() -> None:
    frame = _two_by_two().iloc[:4]
    factors = [Factor("A", low=-1, high=1), Factor("B", low=-1, high=1)]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.factorial(factors, 2)))[0]
    assert fit.note
    assert fit.effect_tests["Prob > F"].isna().all()
    assert math.isnan(fit.summary["Root Mean Square Error"])


def test_response_surface_squares_only_continuous_factors() -> None:
    factors = [Factor("A"), Factor("B"), Factor("Cat", NOMINAL)]
    terms = lm.response_surface(factors)
    assert ("A", "A") in terms and ("B", "B") in terms and ("Cat", "Cat") not in terms
    assert lm.term_label(("A", "B")) == "A*B"


def test_several_responses_and_missing_values() -> None:
    frame = _two_by_two()
    frame["Z"] = frame.Y * 2
    frame.loc[0, "Z"] = np.nan
    factors = [Factor("A", low=-1, high=1), Factor("B", low=-1, high=1)]
    fits = lm.fit_models(frame, ModelSpec(factors, ["Y", "Z"], lm.main_effects(factors)))
    assert [f.response for f in fits] == ["Y", "Z"]
    assert fits[1].summary["Observations"] == 7 and 0 not in fits[1].rows
    assert len(fits[1].predicted) == 7 and fits[1].predicted.index.equals(fits[1].rows)
    assert CONTINUOUS == "continuous"
