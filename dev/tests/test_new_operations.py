"""Numerics for the four operations added after the review.

These were each verified against analytic results while being written; this is
that verification made permanent. The properties asserted are the ones that
would be silently wrong rather than loud - a derivative off by the sample
spacing, a control limit built from the wrong variance - because those are the
ones a smoke test would not catch.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pytest

from app.series_operations.calculus_dialog import BASELINE_NONE, DERIV_GRADIENT, DERIV_SAVGOL, DERIV_SPLINE, DEST_NEW_AXIS, INTEGRAL_DEFINITE, SeriesCalculusDialog
from app.series_operations.control_chart_dialog import CHART_INDIVIDUALS, CHART_XBAR_R, SPC_CONSTANTS, SeriesControlChartDialog
from app.series_operations.function_dialog import (
    SPACING_LINEAR,
    SPACING_LOG,
    SeriesFunctionDialog,
)
from app.series_operations.peaks_dialog import PEAKS_MAXIMA, SeriesPeaksDialog


T = TypeVar("T")


def _bare(cls: type[T]) -> T:
    """Build an instance without its Qt dialog.

    The numerics are plain methods on the class and need no window; going
    through __init__ would require a repository and a figure and would test
    the shell rather than the arithmetic.
    """
    instance = cls.__new__(cls)
    return instance


# ======================================================================
# Calculus: derivatives
# ======================================================================

X = np.linspace(0.0, 2.0 * np.pi, 201)
Y = np.sin(X)


@pytest.mark.parametrize(
    "model, params, tolerance",
    [
        (DERIV_GRADIENT, {"order": 1}, 1e-3),
        (DERIV_SAVGOL, {"order": 1, "window": 11, "polyorder": 3}, 1e-3),
        (DERIV_SPLINE, {"order": 1, "smoothing": 0}, 1e-5),
    ],
)
def test_the_derivative_of_sine_is_cosine(model, params, tolerance) -> None:
    result = _bare(SeriesCalculusDialog)._differentiate("s", X, Y, model, params)
    assert np.max(np.abs(result.y - np.cos(X))) < tolerance


# ======================================================================
# Calculus: integrals
# ======================================================================

def test_the_definite_integral_of_sine_over_half_a_period_is_two() -> None:
    x = np.linspace(0.0, np.pi, 201)
    result = _bare(SeriesCalculusDialog)._integrate(
        "s", x, np.sin(x), INTEGRAL_DEFINITE,
        {"baseline": BASELINE_NONE, "simpson": True},
    )
    assert result.total == pytest.approx(2.0, abs=1e-6)


# ======================================================================
# Peaks
# ======================================================================

PEAK_X = np.linspace(0.0, 100.0, 1001)


def _gaussian(centre: float, amplitude: float, width: float) -> np.ndarray:
    return amplitude * np.exp(-((PEAK_X - centre) ** 2) / (2.0 * width**2))


THREE_PEAKS = _gaussian(20.0, 1.0, 2.0) + _gaussian(50.0, 0.5, 4.0) + _gaussian(80.0, 2.0, 1.5)

DEFAULT_PEAK_PARAMS = {
    "filter_by": "prominence",
    "threshold": 0.05,
    "distance": 1,
    "min_width": 0.0,
    "limit": 50,
}


def test_three_known_peaks_are_found_at_their_known_positions() -> None:
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, THREE_PEAKS, PEAKS_MAXIMA, DEFAULT_PEAK_PARAMS
    )
    positions = sorted(peak.x for peak in result.peaks)
    assert len(positions) == 3
    for found, expected in zip(positions, (20.0, 50.0, 80.0)):
        assert found == pytest.approx(expected, abs=0.2)


def test_a_flat_series_has_no_peaks() -> None:
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, np.ones_like(PEAK_X), PEAKS_MAXIMA, DEFAULT_PEAK_PARAMS
    )
    assert result.peaks == []


# ======================================================================
# Control charts
# ======================================================================

CONTROL_PARAMS = {
    "subgroup": 5,
    "sigma_limit": 3.0,
    "nelson": False,
    "exclude_violations": False,
}


def test_xbar_limits_match_the_textbook_a2_formula() -> None:
    """Agreement is to about 1e-3, which is the rounding in the published d2
    and A2 rather than a disagreement about the method."""
    rng = np.random.default_rng(5)
    values = rng.normal(50.0, 2.0, 100)
    x = np.arange(100, dtype=float)

    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, values, CHART_XBAR_R, CONTROL_PARAMS
    )

    grouped = values.reshape(20, 5)
    centre = grouped.mean(axis=1).mean()
    mean_range = (grouped.max(axis=1) - grouped.min(axis=1)).mean()
    a2 = SPC_CONSTANTS[5][3]

    assert result.upper == pytest.approx(centre + a2 * mean_range, abs=1e-2)
    assert result.lower == pytest.approx(centre - a2 * mean_range, abs=1e-2)


def test_the_run_rules_catch_a_shift_that_stays_inside_the_limits() -> None:
    """What a limits-only chart misses, and the reason the run rules exist."""
    rng = np.random.default_rng(11)
    x = np.arange(60, dtype=float)
    values = rng.normal(0.0, 1.0, 60)
    values[40:] += 1.2

    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, values, CHART_INDIVIDUALS, {**CONTROL_PARAMS, "nelson": True}
    )

    assert ((values <= result.upper) & (values >= result.lower)).all()
    assert result.violations, "no point breaches the limits, so only a run rule can see it"
    assert 1 not in {rule for v in result.violations for rule in v.rules}


# ======================================================================
# Function plotting
# ======================================================================

def test_a_linear_range_spans_the_requested_endpoints() -> None:
    values = SeriesFunctionDialog._build_range(
        {"start": 0.0, "stop": 10.0, "points": 5, "spacing": SPACING_LINEAR}
    )
    assert values.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_a_log_range_is_evenly_spaced_in_decades() -> None:
    values = SeriesFunctionDialog._build_range(
        {"start": 1.0, "stop": 1000.0, "points": 4, "spacing": SPACING_LOG}
    )
    assert values.tolist() == pytest.approx([1.0, 10.0, 100.0, 1000.0])


def test_a_discovered_function_evaluates_over_a_range() -> None:
    """End to end: the scanner's callable applied to a built range."""
    from app.scanners.functions_scanner import FunctionScanner

    scanner = FunctionScanner()
    payload = next(
        payload
        for functions in scanner.catalog().values()
        for payload in functions
        if payload.get("discovery_entry", {}).get("name") == "linear"
    )
    model = scanner.make_model(dict(payload))
    x = SeriesFunctionDialog._build_range(
        {"start": 1.0, "stop": 10.0, "points": 7, "spacing": SPACING_LINEAR}
    )
    y = np.asarray(model(x, np.asarray([0.0, 1.0])), dtype=float)

    assert y == pytest.approx(x), "intercept 0, slope 1 is the identity"


# ======================================================================
# Calculus: where the result is drawn
# ======================================================================

class _AxisSpy:
    """Stands in for the repository and for the base's creation helpers.

    The routing is decided entirely from the parameter value and the result
    metadata, so it can be exercised without a figure or a database.
    """

    def __init__(self) -> None:
        self.axes: list[dict] = []
        self.figures: list[dict] = []
        self.deleted_axes: list[int] = []
        self.deleted_figures: list[int] = []
        self.labelled: dict | None = None

    def create_result_axis(self, **kwargs) -> int:
        self.axes.append(kwargs)
        return 99

    def create_result_figure(self, **kwargs) -> tuple[int, int]:
        self.figures.append(kwargs)
        return 42, 99

    def update_axis_descriptor(self, **kwargs) -> None:
        self.labelled = kwargs

    def delete_axis(self, axis_id: int) -> None:
        self.deleted_axes.append(axis_id)

    def delete_figure(self, figure_id: int) -> None:
        self.deleted_figures.append(figure_id)


def _calculus_with_axis(destination: str) -> tuple[SeriesCalculusDialog, _AxisSpy]:
    spy = _AxisSpy()
    dialog = _bare(SeriesCalculusDialog)
    dialog._result_axis_id = None
    dialog._result_figure_id = None
    dialog._applied = False
    dialog._repo = spy
    dialog.create_result_axis = spy.create_result_axis
    dialog.create_result_figure = spy.create_result_figure
    dialog.parameter_values = lambda: {"destination": destination}
    dialog._model = lambda: DERIV_SAVGOL
    return dialog, spy


def _derivative_result(order: int = 1, model: str = DERIV_SAVGOL):
    from app.series_operations.calculus_dialog import CalculusResult

    return CalculusResult(
        source_name="s",
        result_name="s - d",
        model=model,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        metadata={"order": order},
    )


def test_calculus_draws_on_a_new_axis_by_default() -> None:
    """A derivative divides by dx, so it rarely shares a scale with its
    source; overlaid, one of the two is a flat line on the zero gridline."""
    dialog, spy = _calculus_with_axis(DEST_NEW_AXIS)
    target = dialog.resolve_target_axis_id(7, [_derivative_result()])

    assert target == 99
    assert target != 7
    assert len(spy.axes) == 1


def test_closing_without_applying_removes_the_axis_it_created() -> None:
    """Creating an axis commits, so the preview savepoint does not cover it."""
    dialog, spy = _calculus_with_axis(DEST_NEW_AXIS)
    dialog.resolve_target_axis_id(7, [_derivative_result()])
    dialog.discard_operation_artifacts()

    assert spy.deleted_axes == [99]


# ======================================================================
# Control charts: the lines, not just the points
# ======================================================================

def _chart_result(nelson: bool = False):
    rng = np.random.default_rng(4)
    x = np.arange(60, dtype=float)
    values = rng.normal(100.0, 1.0, 60)
    values[30] = 130.0
    return _bare(SeriesControlChartDialog)._build_chart(
        "s", x, values, CHART_INDIVIDUALS, {**CONTROL_PARAMS, "nelson": nelson}
    )


DRAW_ALL = {
    "draw_center": True,
    "draw_limits": True,
    "draw_zones": True,
    "draw_violations": True,
}


def _specs(result, **overrides):
    dialog = _bare(SeriesControlChartDialog)
    dialog.parameter_values = lambda: {**DRAW_ALL, **overrides}
    return dialog.result_series_specs(1, "_T", result)


def _labels(specs) -> list[str]:
    return [spec.name.split(" - ")[-1] for spec in specs]


def test_a_control_chart_draws_its_limits_by_default() -> None:
    """Without them it is a run chart: the same numbers, a different question."""
    labels = _labels(_specs(_chart_result(), draw_zones=False))
    assert "UCL" in labels
    assert "LCL" in labels
    assert "CL" in labels


