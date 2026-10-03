"""Tests for fitting continuous distributions and ranking them.

The ranking is the product here, so the tests are about the ranking: that a
sample drawn from a known family puts that family at the top, that a candidate
which cannot describe the sample drops out instead of raising, and that the
histogram draws the same fit the table reports.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.utils.distribution_fit import best_fit, curve_points, fit_distributions, fit_one
from dev.tests._cases import check_all

RNG = np.random.default_rng(11)


# ----------------------------------------------------------------------
# Fitting one
# ----------------------------------------------------------------------
def test_a_known_sample_recovers_its_own_parameters() -> None:
    fit = fit_one(RNG.normal(10.0, 2.0, 4000), "norm")

    assert fit is not None
    location, scale = fit.params
    assert location == pytest.approx(10.0, abs=0.2)
    assert scale == pytest.approx(2.0, abs=0.2)


def test_a_degenerate_sample_is_dropped_not_raised() -> None:
    """A sample with no spread has no scale, so the fit is undefined."""
    assert fit_one(np.full(200, 4.0), "norm") is None


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------
_CASES_A_SAMPLE_RANKS_ITS_OWN_FAMILY_AT_THE_TOP = [
        ("norm", RNG.normal(10.0, 2.0, 3000)),
        ("lognorm", RNG.lognormal(1.0, 0.5, 3000)),
        ("expon", RNG.exponential(3.0, 3000)),
        ("uniform", RNG.uniform(0.0, 1.0, 3000)),
    ]


def _a_sample_ranks_its_own_family_at_the_top(family: str, sample) -> None:
    """The one claim the ranking has to earn, and only AIC earns it."""
    fits = fit_distributions(sample)

    assert fits, "nothing fitted at all"
    assert fits[0].name == family


def test_a_sample_ranks_its_own_family_at_the_top() -> None:
    check_all(_a_sample_ranks_its_own_family_at_the_top, _CASES_A_SAMPLE_RANKS_ITS_OWN_FAMILY_AT_THE_TOP)


# ----------------------------------------------------------------------
# The candidate list
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The curve the histogram draws
# ----------------------------------------------------------------------
def test_the_density_curve_integrates_to_about_one() -> None:
    """It is a probability density, which is why the histogram must be one too."""
    fit = best_fit(RNG.normal(10.0, 2.0, 3000))
    assert fit is not None

    x, pdf = curve_points(fit, 0.0, 20.0, points=2000)

    assert np.isfinite(pdf).all()
    assert float(np.trapezoid(pdf, x)) == pytest.approx(1.0, abs=0.02)


# ----------------------------------------------------------------------
# Resolving what a chart asked for
# ----------------------------------------------------------------------


