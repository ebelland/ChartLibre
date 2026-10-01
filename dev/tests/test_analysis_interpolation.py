"""Interpolation (app.analysis.interpolation): fits and interpolants with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import interpolation as ip

X = np.linspace(1.0, 10.0, 25)


def test_a_polynomial_of_the_right_degree_is_recovered_exactly() -> None:
    y = 2.0 * X**2 - 3.0 * X + 1.0
    result = ip.interpolate(ip.MODEL_POLYNOMIAL, X, y, X, ip.InterpolationSettings(degree=2))
    np.testing.assert_allclose(list(result.params.values()), [2.0, -3.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(result.y, y, atol=1e-9)


def test_the_polynomial_degree_is_clamped_to_the_points() -> None:
    x = np.array([0.0, 1.0, 2.0])
    result = ip.interpolate(ip.MODEL_POLYNOMIAL, x, x**2, x, ip.InterpolationSettings(degree=9))
    assert result.message == "NumPy polyfit degree=2"


@pytest.mark.parametrize(
    ("model", "truth", "params"),
    [
        (ip.MODEL_EXPONENTIAL, lambda x: 2.0 * np.exp(0.3 * x) + 1.0, {"a": 2.0, "b": 0.3, "c": 1.0}),
        (ip.MODEL_LOGARITHMIC, lambda x: 4.0 * np.log(x) - 2.0, {"a": 4.0, "b": -2.0}),
        (ip.MODEL_POWER, lambda x: 1.5 * x**1.7 + 0.5, {"a": 1.5, "b": 1.7, "c": 0.5}),
        (ip.MODEL_GAUSSIAN, lambda x: 3.0 * np.exp(-0.5 * ((x - 5.0) / 1.2) ** 2) + 0.2, {"a": 3.0, "mu": 5.0, "sigma": 1.2, "c": 0.2}),
        (ip.MODEL_SIGMOID, lambda x: 4.0 / (1.0 + np.exp(-1.5 * (x - 6.0))) + 1.0, {"a": 4.0, "x0": 6.0, "k": 1.5, "c": 1.0}),
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the sigmoid's exp overflows on far guesses
def test_each_curve_fit_recovers_its_own_parameters(model, truth, params) -> None:
    result = ip.interpolate(model, X, truth(X), X)
    for name, value in params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-4, abs=1e-6)
    assert ip.goodness(model, X, truth(X), result.params)["r2"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "model", [ip.MODEL_NUMPY_INTERP, ip.MODEL_SCIPY_PCHIP, ip.MODEL_SCIPY_AKIMA, ip.MODEL_SCIPY_CUBIC]
)
def test_an_interpolant_passes_through_every_point(model: str) -> None:
    y = np.sin(X)
    np.testing.assert_allclose(ip.interpolate(model, X, y, X).y, y, atol=1e-12)


@pytest.mark.parametrize("spline_type", ["CubicSpline", "PCHIP", "Akima1D", "B-spline"])
def test_the_spline_family_passes_through_every_point(spline_type: str) -> None:
    y = np.cos(X)
    settings = ip.InterpolationSettings(spline_type=spline_type)
    np.testing.assert_allclose(ip.interpolate(ip.MODEL_SCIPY_SPLINE, X, y, X, settings).y, y, atol=1e-10)


@pytest.mark.parametrize("settings", [{}, {"spline_type": "Akima1D"}])
def test_outside_the_data_is_left_empty_unless_extrapolating(settings: dict) -> None:
    x_eval = np.array([0.0, 5.0, 11.0])
    y = X.copy()
    model = ip.MODEL_SCIPY_SPLINE if settings else ip.MODEL_SCIPY_AKIMA
    closed = ip.interpolate(model, X, y, x_eval, ip.InterpolationSettings(**settings))
    assert np.isnan(closed.y[0]) and closed.y[1] == pytest.approx(5.0) and np.isnan(closed.y[2])
    # Akima used to ignore Extrapolate: SciPy leaves the outside empty unless asked.
    opened = ip.interpolate(model, X, y, x_eval, ip.InterpolationSettings(extrapolate=True, **settings))
    assert opened.y.tolist() == pytest.approx([0.0, 5.0, 11.0])


def test_a_smoothing_spline_with_a_large_factor_does_not_pass_through_noise() -> None:
    noisy = X + np.random.default_rng(1).normal(0, 0.5, X.size)
    smooth = ip.interpolate(ip.MODEL_SCIPY_SPLINE, X, noisy, X, ip.InterpolationSettings(spline_type="Univariate", smoothing=50.0))
    assert np.max(np.abs(smooth.y - noisy)) > 0.1
    assert smooth.message == "SciPy UnivariateSpline"


# ----------------------------------------------------------------------
# Where to evaluate
# ----------------------------------------------------------------------
def test_the_spacings() -> None:
    lin = ip.evaluation_x(X, ip.SPACING_LINEAR, start=0.0, stop=1.0, count=5)
    assert lin.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    log = ip.evaluation_x(X, ip.SPACING_LOG, start=1.0, stop=100.0, count=3)
    assert log.tolist() == pytest.approx([1.0, 10.0, 100.0])
    steps = ip.evaluation_x(X, ip.SPACING_INTEGER_STEP, start=0.5, stop=6.2, step=2.0)
    assert steps.tolist() == [2.0, 4.0, 6.0]
    cheb = ip.evaluation_x(X, ip.SPACING_CHEBYSHEV, start=-1.0, stop=1.0, count=4)
    assert cheb.tolist() == pytest.approx(sorted(np.cos((2 * np.arange(4) + 1) * np.pi / 8)))
    assert ip.evaluation_x(X[::-1], ip.SPACING_ORIGINAL, start=0, stop=1).tolist() == X.tolist()
    assert ip.evaluation_x(X, ip.SPACING_CUSTOM, start=0, stop=1, custom=np.array([3.0, 1.0])).tolist() == [1.0, 3.0]


@pytest.mark.parametrize(
    ("spacing", "kwargs", "match"),
    [(ip.SPACING_LOG, {"start": -1.0, "stop": 2.0}, "positive"),
     (ip.SPACING_GEOMETRIC, {"start": 0.0, "stop": 2.0}, "positive"),
     (ip.SPACING_CUSTOM, {"start": 0.0, "stop": 1.0}, "at least one"),
     (ip.SPACING_INTEGER_STEP, {"start": 0.1, "stop": 0.2, "step": 5.0}, "no X values")],
)
def test_a_spacing_the_range_cannot_give_is_a_value_error(spacing, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ip.evaluation_x(X, spacing, **kwargs)


def test_the_range_is_the_data_widened_only_when_extrapolating() -> None:
    assert ip.evaluation_range(X, extend_percent=10.0) == (1.0, 10.0)
    assert ip.evaluation_range(X, extend_percent=10.0, extrapolate=True) == pytest.approx((0.1, 10.9))
    assert ip.evaluation_range(X, explicit=(5.0, 2.0)) == (2.0, 5.0)
    with pytest.raises(ValueError, match="must differ"):
        ip.evaluation_range(X, explicit=(3.0, 3.0))
    with pytest.raises(ValueError, match="more than one unique"):
        ip.evaluation_range(np.array([2.0, 2.0]))


def test_typed_values_accept_any_separator() -> None:
    assert ip.parse_values("1, 2;3 4\n5").tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_an_unknown_model_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        ip.interpolate("nope", X, X, X)
