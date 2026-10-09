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


def test_the_profile_follows_each_continuous_factor_with_the_others_held() -> None:
    frame = _two_by_two()
    frame["T"] = frame.A.map({-1: 100.0, 1: 200.0})
    factors = [Factor("T", low=100, high=200), Factor("B", low=-1, high=1)]
    fit = lm.fit_models(frame, ModelSpec(factors, ["Y"], lm.main_effects(factors)))[0]
    profile = fit.profile
    assert set(profile["Factor"]) == {"T", "B"}
    t = profile[profile.Factor == "T"]
    assert len(t) == lm.PROFILE_POINTS
    assert t["Value"].iloc[0] == 100.0 and t["Value"].iloc[-1] == 200.0
    # Along T, with B at its centre: intercept + 2 * coded T.
    intercept = fit.estimates.set_index("Term").loc["Intercept", "Estimate"]
    np.testing.assert_allclose(t["Predicted"], intercept + fit.estimates.set_index("Term").loc["T", "Estimate"] * t["Coded"])
    assert (t["Lower"] <= t["Predicted"]).all() and (t["Predicted"] <= t["Upper"]).all()


def _noisy(seed: int = 4, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"A": rng.uniform(-1, 1, n), "B": rng.uniform(-1, 1, n), "C": rng.uniform(-1, 1, n)})
    frame["eta"] = 0.3 + 1.5 * frame.A - 1.0 * frame.B
    return frame


def test_forward_stepwise_adds_only_what_matters() -> None:
    frame = _noisy()
    frame["Y"] = 2 + 3 * frame.A - 2 * frame.B + np.random.default_rng(1).normal(0, 0.3, len(frame))
    factors = [Factor(n, low=-1, high=1) for n in "ABC"]
    spec = ModelSpec(factors, ["Y"], lm.main_effects(factors), personality="stepwise", direction="forward")
    fit = lm.fit_models(frame, spec)[0]
    assert sorted(fit.terms) == [("A",), ("B",)] and fit.removed == [("C",)]


def test_a_normal_glm_is_least_squares_and_poisson_finds_its_rate() -> None:
    frame = _noisy()
    rng = np.random.default_rng(2)
    frame["Y"] = 1 + 2 * frame.A + rng.normal(0, 0.2, len(frame))
    frame["N"] = rng.poisson(np.exp(0.5 + 0.8 * frame.A))
    factors = [Factor("A", low=-1, high=1), Factor("B", low=-1, high=1)]
    terms = lm.main_effects(factors)
    ols = lm.fit_models(frame, ModelSpec(factors, ["Y"], terms))[0]
    glm = lm.fit_models(frame, ModelSpec(factors, ["Y"], terms, personality="glm"))[0]
    np.testing.assert_allclose(glm.estimates["Estimate"], ols.estimates["Estimate"], atol=1e-8)
    poisson = lm.fit_models(frame, ModelSpec(factors, ["N"], terms, personality="glm", family="poisson"))[0]
    assert poisson.estimates.set_index("Term").loc["A", "Estimate"] == pytest.approx(0.8, abs=0.2)
    tests = poisson.effect_tests.set_index("Term")
    assert tests.loc["A", "Prob > ChiSq"] < 0.001 < tests.loc["B", "Prob > ChiSq"]
    assert poisson.p_column == "Prob > ChiSq" and poisson.anova.loc[0, "Prob > ChiSq"] < 0.001
    assert poisson.anova.loc[0, "-LogLikelihood"] > 0


def test_nominal_logistic_binary_and_multinomial() -> None:
    frame = _noisy(n=300)
    rng = np.random.default_rng(3)
    p_good = 1 / (1 + np.exp(-frame.eta))
    frame["Ok"] = np.where(rng.uniform(size=len(frame)) < p_good, "good", "bad")
    factors = [Factor("A", low=-1, high=1), Factor("B", low=-1, high=1)]
    terms = lm.main_effects(factors)
    fit = lm.fit_models(frame, ModelSpec(factors, ["Ok"], terms, personality="nominal"))[0]
    assert fit.levels == ["bad", "good"]  # the predictions are P(bad)
    a = fit.estimates.set_index("Term").loc["A", "Estimate"]
    assert a == pytest.approx(-1.5, abs=0.6)  # P(bad) falls as A rises
    assert fit.effect_tests.set_index("Term").loc["A", "Prob > ChiSq"] < 0.001
    assert fit.predicted.between(0, 1).all() and not fit.profile.empty

    frame["Grade"] = pd.cut(frame.eta + rng.normal(0, 0.7, len(frame)), [-np.inf, -0.3, 0.9, np.inf],
                            labels=["c", "b", "a"]).astype(str)
    multi = lm.fit_models(frame, ModelSpec(factors, ["Grade"], terms, personality="nominal"))[0]
    assert multi.levels == ["a", "b", "c"]
    assert len(multi.estimates) == 3 * 2  # intercept, A, B for each level after the first
    assert multi.effect_tests.set_index("Term").loc["A", "DF"] == 2

    ordinal = lm.fit_models(frame, ModelSpec(factors, ["Grade"], terms, personality="ordinal"))[0]
    tests = ordinal.effect_tests.set_index("Term")
    assert tests.loc["A", "DF"] == 1 and tests.loc["A", "Prob > ChiSq"] < 0.001
    assert ordinal.anova.loc[0, "Prob > ChiSq"] < 0.001


def test_a_model_without_design_info_still_has_its_effect_tests_and_estimates() -> None:
    """Some models carry no ``data.design_info`` ("'PandasData' object has no
    attribute 'design_info'"): the term columns are rebuilt from the formula."""
    import statsmodels.formula.api as smf

    from app.analysis import linear_model as lm

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"a": rng.normal(size=40), "g": rng.choice(["p", "q", "r"], 40)})
    frame["y"] = 2 * frame["a"] + np.where(frame["g"] == "q", 1.0, 0.0) + rng.normal(0, 0.1, 40)
    terms = [("a",), ("g",)]

    def piece(term):
        return "a" if term == ("a",) else "C(g, Sum)"

    model = smf.ols("y ~ a + C(g, Sum)", data=frame).fit()
    expected = lm._effect_tests(model, terms, piece)
    del model.model.data.design_info
    tests = lm._effect_tests(model, terms, piece)
    pd.testing.assert_frame_equal(tests, expected)
    assert len(lm._estimates(model, terms, piece)) == 4
