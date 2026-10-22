"""Fit Model against the NIST StRD Longley problem: 16 rows, six strongly collinear predictors.

The certified values are NIST's (the Statistical Reference Datasets, linear regression, higher
difficulty). The predictors are used in their own units - Factor low/high of -1 and +1 is the
identity coding - so the estimates are the certified coefficients themselves. Longley is the
classic test of a least-squares code's conditioning: a normal-equations solver loses most of its digits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import linear_model as lm

COLUMNS = ["deflator", "gnp", "unemployed", "armed", "population", "year"]
RESPONSE = [60323, 61122, 60171, 61187, 63221, 63639, 64989, 63761, 66019, 67857, 68169, 66513, 68655, 69564,
            69331, 70551]
PREDICTORS = [
    [83.0, 234289, 2356, 1590, 107608, 1947], [88.5, 259426, 2325, 1456, 108632, 1948],
    [88.2, 258054, 3682, 1616, 109773, 1949], [89.5, 284599, 3351, 1650, 110929, 1950],
    [96.2, 328975, 2099, 3099, 112075, 1951], [98.1, 346999, 1932, 3594, 113270, 1952],
    [99.0, 365385, 1870, 3547, 115094, 1953], [100.0, 363112, 3578, 3350, 116219, 1954],
    [101.2, 397469, 2904, 3048, 117388, 1955], [104.6, 419180, 2822, 2857, 118734, 1956],
    [108.4, 442769, 2936, 2798, 120445, 1957], [110.8, 444546, 4681, 2637, 121950, 1958],
    [112.6, 482704, 3813, 2552, 123366, 1959], [114.2, 502601, 3931, 2514, 125368, 1960],
    [115.7, 518173, 4806, 2572, 127852, 1961], [116.9, 554894, 4007, 2827, 130081, 1962],
]

#: NIST's certified values, in the order Intercept, then the six predictors.
CERTIFIED_ESTIMATES = [
    -3482258.63459582, 15.0618722713733, -0.358191792925910e-01, -2.02022980381683,
    -1.03322686717359, -0.511041056535807e-01, 1829.15146461355,
]
CERTIFIED_STD_ERRORS = [
    890420.383607373, 84.9149257747669, 0.334910077722432e-01, 0.488399681651699,
    0.214274163161675, 0.226073200069370, 455.478499142212,
]
CERTIFIED_RESIDUAL_SD = 304.854073561965
CERTIFIED_R_SQUARED = 0.995479004577296


@pytest.fixture(scope="module")
def longley() -> lm.ResponseFit:
    frame = pd.DataFrame(PREDICTORS, columns=COLUMNS)
    frame["employed"] = RESPONSE
    spec = lm.ModelSpec(
        factors=[lm.Factor(name, "continuous", -1.0, 1.0) for name in COLUMNS],
        responses=["employed"], terms=[(name,) for name in COLUMNS], alpha=0.05,
        personality="standard", family="normal", penalty="lasso", direction="backward",
    )
    return lm.fit_models(frame, spec)[0]


def test_the_estimates_are_the_certified_coefficients(longley: lm.ResponseFit) -> None:
    assert list(longley.estimates["Term"]) == ["Intercept", *COLUMNS]
    np.testing.assert_allclose(longley.estimates["Estimate"].to_numpy(), CERTIFIED_ESTIMATES, rtol=1e-8)


def test_the_standard_errors_are_the_certified_ones(longley: lm.ResponseFit) -> None:
    np.testing.assert_allclose(longley.estimates["Std Error"].to_numpy(), CERTIFIED_STD_ERRORS, rtol=1e-8)


def test_the_residual_standard_deviation_and_r_squared_are_certified(longley: lm.ResponseFit) -> None:
    assert longley.summary["Root Mean Square Error"] == pytest.approx(CERTIFIED_RESIDUAL_SD, rel=1e-9)
    assert longley.summary["RSquare"] == pytest.approx(CERTIFIED_R_SQUARED, rel=1e-11)
