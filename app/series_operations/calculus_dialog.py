"""Numerical derivative and integral of a chart series.

Both halves of calculus in one dialog, because they are the same conversation
about the same series and neither is much use without the other: you
differentiate to find where a signal turns, and integrate to find how much of
it there was.

Two things are deliberately built in rather than left as steps the user is
expected to remember first.

**Smoothing, inside the derivative.**  Differentiation amplifies noise - that
is not a caveat, it is the whole difficulty of doing it numerically.  A
finite difference divides by a small dx, so noise of size e on neighbouring
points becomes noise of size 2e/dx on the result; halve the sampling interval
and the noise doubles.  Offering a raw ``np.gradient`` and trusting people to
smooth first produces a spiky mess that looks like data.  Savitzky-Golay is
therefore the default: it fits a low-order polynomial over a window and
differentiates *that*, so smoothing and differentiating happen in one step
rather than as two the user can get the order of wrong.

**Baseline subtraction, inside the integral.**  The area under a peak means
nothing if the signal does not return to zero: a constant offset contributes
offset x width to the result, which for a broad peak on a raised baseline is
most of the answer.  So the integral offers to remove a baseline first.

Peak finding lives next door in ``series_peaks_dialog``; find a peak there,
read its bounds, integrate between them here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QWidget
from app.analysis import calculus as calc
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, ChoiceParam, IntParam
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.utils import report_html
from app.utils.i18n import _

# --- Models -----------------------------------------------------------

DERIV_SAVGOL = "Derivative (Savitzky-Golay)"
DERIV_GRADIENT = "Derivative (finite difference)"
DERIV_SPLINE = "Derivative (spline)"
INTEGRAL_CUMULATIVE = "Integral (cumulative)"
INTEGRAL_DEFINITE = "Integral (total area)"

#: A series with a z role gets two more entries in the same combo, rather
#: than a second model list gated on the selection: a data-driven list is
#: what makes a *third* one (curl/divergence, once vector fields - x, y, u,
#: v - are in scope) a matter of adding another string here later, not a new
#: branch of dialog. Picking one of these on a series with no z role simply
#: fails per-row with a clear error (see compute_results), the same way
#: every other row-level mismatch here already does.
DERIV_GRADIENT_SURFACE = "Derivative (surface gradient)"
INTEGRAL_VOLUME_SURFACE = "Integral (surface volume)"


@dataclass(frozen=True, slots=True, kw_only=True)
class CalculusModel(OperationModel):
    #: "derivative" or "integral".
    kind: str
    #: Works on a surface (a series with a z role) rather than on y(x).
    surface: bool = False


#: The calculations offered, in combo order.
CALCULUS_MODELS: dict[str, CalculusModel] = {
    DERIV_SAVGOL: CalculusModel(
        doc_title="Savitzky-Golay filter",
        doc_url="https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter",
        kind="derivative",
    ),
    DERIV_GRADIENT: CalculusModel(
        doc_title="Finite difference",
        doc_url="https://en.wikipedia.org/wiki/Finite_difference",
        kind="derivative",
    ),
    DERIV_SPLINE: CalculusModel(
        doc_title="Smoothing spline",
        doc_url="https://en.wikipedia.org/wiki/Smoothing_spline",
        kind="derivative",
    ),
    DERIV_GRADIENT_SURFACE: CalculusModel(
        doc_title="Gradient",
        doc_url="https://en.wikipedia.org/wiki/Gradient",
        kind="derivative",
        surface=True,
    ),
    INTEGRAL_CUMULATIVE: CalculusModel(
        doc_title="Trapezoidal rule",
        doc_url="https://en.wikipedia.org/wiki/Trapezoidal_rule",
        kind="integral",
    ),
    INTEGRAL_DEFINITE: CalculusModel(
        doc_title="Simpson's rule",
        doc_url="https://en.wikipedia.org/wiki/Simpson%27s_rule",
        kind="integral",
    ),
    INTEGRAL_VOLUME_SURFACE: CalculusModel(
        doc_title="Simpson's rule",
        doc_url="https://en.wikipedia.org/wiki/Simpson%27s_rule",
        kind="integral",
        surface=True,
    ),
}

#: The one-dimensional integrals: the models the baseline choice applies to.
INTEGRALS = tuple(
    name for name, model in CALCULUS_MODELS.items() if model.kind == "integral" and not model.surface
)


# Re-exported from the base so this module's callers and tests can name them
# without reaching through the class.
DEST_SAME_AXIS = SeriesOperationDialogBase.DEST_SAME_AXIS
DEST_NEW_AXIS = SeriesOperationDialogBase.DEST_NEW_AXIS
DEST_NEW_FIGURE = SeriesOperationDialogBase.DEST_NEW_FIGURE

# The baseline choices are the engine's; re-exported for callers and tests.
BASELINE_NONE = calc.BASELINE_NONE
BASELINE_MINIMUM = calc.BASELINE_MINIMUM
BASELINE_ENDPOINTS = calc.BASELINE_ENDPOINTS

#: The engine's name for each one-dimensional derivative.
_DERIVATIVE_METHOD: dict[str, str] = {
    DERIV_SAVGOL: calc.DERIVATIVE_SAVGOL,
    DERIV_GRADIENT: calc.DERIVATIVE_GRADIENT,
    DERIV_SPLINE: calc.DERIVATIVE_SPLINE,
}

@dataclass(slots=True)
class CalculusResult(TableResult):
    """Derivative or integral of one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    #: Set only for the definite integral, which produces one number rather
    #: than a curve. Kept separate from ``y`` so the report can show it without
    #: the chart having to special-case a one-point series.
    total: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Set only for the surface gradient: ``x``/``y`` are then the flattened
    #: grid's own coordinates (not a curve's x and a computed y), and ``z``
    #: is the gradient magnitude at each of them. ``dz_dx``/``dz_dy`` carry
    #: the two components separately, for a report that wants more than the
    #: magnitude the chart draws.
    z: np.ndarray | None = None
    dz_dx: np.ndarray | None = None
    dz_dy: np.ndarray | None = None

    def to_df(self) -> pd.DataFrame:
        if self.z is not None:
            data: dict[str, np.ndarray] = {"x": self.x, "y": self.y, "z": self.z}
            if self.dz_dx is not None:
                data["dz_dx"] = self.dz_dx
            if self.dz_dy is not None:
                data["dz_dy"] = self.dz_dy
            return pd.DataFrame(data)
        return pd.DataFrame({"x": self.x, "y": self.y})


class SeriesCalculusDialog(SeriesOperationDialogBase):
    """Differentiate or integrate a chart series."""

    MODELS = CALCULUS_MODELS
    MODEL_TOOLTIP = "Choose the calculation."

    Name: str = "Calculus"
    Description = "Differentiate or integrate"

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    # Every model here treats the series as y = f(x), so the points have to be
    # in x order and single-valued. Both are repaired and reported.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # A derivative needs a point either side of the one it is estimating.
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        IntParam(
            "window",
            "Smoothing window:",
            tooltip=(
                "Points per polynomial fit. Larger smooths more and rounds "
                "sharp features; must exceed the polynomial order."
            ),
            default_value=11,
            minimum=3,
            maximum=9999,
            odd_only=True,
            visible_for={"model": (DERIV_SAVGOL,)},
        ),
        IntParam(
            "polyorder",
            "Polynomial order:",
            tooltip="Order of the polynomial fitted over each window.",
            default_value=3,
            minimum=1,
            maximum=9,
            visible_for={"model": (DERIV_SAVGOL,)},
        ),
        IntParam(
            "order",
            "Derivative order:",
            tooltip="1 for the slope, 2 for the curvature.",
            default_value=1,
            minimum=1,
            maximum=2,
            visible_for={"model": (DERIV_SAVGOL, DERIV_SPLINE)},
        ),
        IntParam(
            "smoothing",
            "Spline smoothing:",
            tooltip=(
                "0 interpolates every point exactly; larger values allow the "
                "spline to depart from noisy data."
            ),
            default_value=0,
            minimum=0,
            maximum=1_000_000,
            visible_for={"model": (DERIV_SPLINE,)},
        ),
        ChoiceParam(
            "baseline",
            "Baseline:",
            tooltip=(
                "Removed before integrating. Without this, a raised baseline "
                "contributes its offset times the width to the area."
            ),
            choices=(
                ("None", BASELINE_NONE),
                ("Subtract minimum", BASELINE_MINIMUM),
                ("Straight line between endpoints", BASELINE_ENDPOINTS),
            ),
            visible_for={"model": INTEGRALS},
        ),
        SeriesOperationDialogBase.destination_param(
            tooltip=(
                "A derivative or an integral rarely shares a scale with the "
                "series it came from - d/dx of a slow drift is near zero "
                "beside data in the thousands - so a new axis is the default. "
                "Same axis overlays them when the ranges are comparable; a new "
                "figure keeps the result out of this chart entirely."
            ),
            default=SeriesOperationDialogBase.DEST_NEW_AXIS,
        ),
        BoolParam(
            "simpson",
            "Use Simpson's rule:",
            tooltip=(
                "More accurate on smooth data than the trapezoidal rule. "
                "Assumes the curve is well approximated by parabolas."
            ),
            default_value=False,
            visible_for={"model": (INTEGRAL_DEFINITE,)},
        ),
    )

    Icon = """
    <path d="M7 19c0-6 2-14 5-14"/>
    <path d="M5 12h9"/>
    <path d="M14 19h5"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesCalculusDialog requires a repository instance.")

        self._last_results: list[CalculusResult] = []
        self._parameter_form: QFormLayout | None = None
        # Created lazily on the first Preview/Apply that asks for it, and
        # reused after, so adjusting the window repeatedly does not leave a
        # trail of empty axes behind.
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Calculus",
            parent=parent,
            width=760,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def series_settings(self) -> tuple[str, dict[str, Any]]:
        return self.current_model(), self.parameter_values()

    def read_series(self, row: Any, name: str) -> Any:
        if CALCULUS_MODELS[self.current_model()].surface:
            return self.series_grid_xyz(row, name)
        return self.series_xy(row, name)

    def compute_series(
        self, name: str, data: Any, settings: tuple[str, dict[str, Any]]
    ) -> CalculusResult:
        model, params = settings
        if CALCULUS_MODELS[model].surface:
            return self._compute_one_3d(data, name, model)
        x_values, y_values = data
        return self._compute_one(name, x_values, y_values, model, params)

    def _compute_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> CalculusResult:
        if CALCULUS_MODELS[model].kind == "derivative":
            return self._differentiate(name, x_values, y_values, model, params)
        return self._integrate(name, x_values, y_values, model, params)

    # --- Surfaces (a series with a z role): gradient and volume ---------

    def _compute_one_3d(
        self, grid: tuple[np.ndarray, np.ndarray, np.ndarray, bool], name: str, model: str
    ) -> CalculusResult:
        x_grid, y_grid, z_grid, interpolated = grid
        if model == DERIV_GRADIENT_SURFACE:
            return self._gradient_surface(name, x_grid, y_grid, z_grid, interpolated)
        return self._volume_surface(name, x_grid, y_grid, z_grid, interpolated)

    def _gradient_surface(
        self,
        name: str,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        interpolated: bool,
    ) -> CalculusResult:
        """Return the surface's gradient magnitude on the same grid (app.analysis.calculus)."""
        gradient = calc.surface_gradient(x_grid, y_grid, z_grid)
        return CalculusResult(
            source_name=name,
            result_name=f"{name} - |grad z|",
            model=DERIV_GRADIENT_SURFACE,
            x=x_grid.ravel(),
            y=y_grid.ravel(),
            z=gradient.magnitude.ravel(),
            dz_dx=gradient.dz_dx.ravel(),
            dz_dy=gradient.dz_dy.ravel(),
            metadata={
                "detail": "np.gradient over the x/y grid",
                "interpolated": bool(interpolated),
            },
        )

    def _volume_surface(
        self,
        name: str,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        interpolated: bool,
    ) -> CalculusResult:
        """Return the volume under the surface, by double Simpson integration."""
        volume = calc.surface_volume(x_grid, y_grid, z_grid)
        detail = "double Simpson's rule (x then y)"
        if volume.missing_cells:
            detail += (
                f"; {volume.missing_cells} of {z_grid.size} interpolated cell(s) outside "
                "the data's convex hull were treated as 0"
            )
        x_axis = x_grid[0, :]
        return CalculusResult(
            source_name=name,
            result_name=f"{name} - volume under surface",
            model=INTEGRAL_VOLUME_SURFACE,
            # A single number is still reported as a two-point flat line so it
            # can be stored and drawn like every other result.
            x=np.array([float(np.min(x_axis)), float(np.max(x_axis))]),
            y=np.array([volume.total, volume.total]),
            total=volume.total,
            metadata={"detail": detail, "interpolated": bool(interpolated)},
        )

    # --- Derivatives ---------------------------------------------------

    def _differentiate(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> CalculusResult:
        order = int(params.get("order", 1))
        unit_label, unit_seconds = self._temporal_unit(name, x_values)
        # x_calc is only for the derivative's own math - the result keeps the
        # original x_values (raw seconds for a dated series) so it still lands
        # on the axis's real date scale (restore_temporal_x).
        x_calc = x_values / unit_seconds if unit_label else x_values

        derivative = calc.differentiate(
            _DERIVATIVE_METHOD.get(model, calc.DERIVATIVE_SAVGOL),
            x_calc,
            y_values,
            order=order,
            window=int(params.get("window", 11)),
            polyorder=int(params.get("polyorder", 3)),
            smoothing=params.get("smoothing", 0),
        )
        for note in derivative.notes:
            applogger.warning(f"{name}: {note}")

        power = "²" if order == 2 else ""
        base_name = f"{name} - d{power}y/dx{power}"
        result_name = f"{base_name} (x in {unit_label}s)" if unit_label else base_name
        return CalculusResult(
            source_name=name,
            result_name=result_name,
            model=model,
            x=x_values,
            y=derivative.values,
            metadata={"order": order, "detail": derivative.detail, "per": unit_label},
        )

    def _temporal_unit(self, name: str, x_values: np.ndarray) -> tuple[str, float]:
        """Return (unit name, seconds per unit) for *name*'s x, if temporal.

        ("", 1.0) for a plain numeric x - the empty label is also the signal
        callers use to skip rescaling entirely.
        """
        # getattr, not a direct read: tests run the numeric methods on a bare
        # cls.__new__(cls) instance, where _temporal_x_sources does not exist.
        if not getattr(self, "_temporal_x_sources", {}).get(name):
            return "", 1.0
        return calc.time_unit_of(x_values)

    # --- Integrals -----------------------------------------------------

    def _integrate(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> CalculusResult:
        corrected, baseline_detail = calc.subtract_baseline(
            x_values, y_values, str(params.get("baseline", BASELINE_NONE))
        )
        unit_label, unit_seconds = self._temporal_unit(name, x_values)
        x_calc = x_values / unit_seconds if unit_label else x_values
        unit_suffix = f" (x in {unit_label}s)" if unit_label else ""

        if model == INTEGRAL_CUMULATIVE:
            running = calc.cumulative_integral(x_calc, corrected)
            return CalculusResult(
                source_name=name,
                result_name=f"{name} - ∫y dx{unit_suffix}",
                model=model,
                x=x_values,
                y=running,
                total=float(running[-1]) if running.size else 0.0,
                metadata={"baseline": baseline_detail, "per": unit_label},
            )

        area = calc.definite_integral(x_calc, corrected, simpson_rule=bool(params.get("simpson", False)))
        for note in area.notes:
            applogger.warning(f"{name}: {note}")
        return CalculusResult(
            source_name=name,
            result_name=f"{name} - area{unit_suffix}",
            model=model,
            # A single number still has to be a series to be stored and drawn,
            # so it is reported across the range it was computed over.
            x=np.array([float(x_values[0]), float(x_values[-1])]),
            y=np.array([area.total, area.total]),
            total=area.total,
            metadata={"baseline": baseline_detail, "rule": area.rule, "per": unit_label},
        )

    # ------------------------------------------------------------------
    # Where the results are drawn
    # ------------------------------------------------------------------

    def resolve_target_axis_id(
        self,
        selected_axis_id: int,
        results: Sequence[Any],
    ) -> int:
        """Return the axis to draw on: a new one unless asked otherwise.

        A derivative and its source almost never share a scale. Differentiating
        divides by dx, so a slow drift over thousands of seconds has a
        derivative near zero; drawn on the source's axis the result is a flat
        line on the zero gridline while the source fills the plot. An integral
        goes the other way and grows without bound. Either way both curves
        become unreadable, which is why this defaults to a new axis rather than
        following the usual "write back where the input came from" rule.

        A new axis on the same figure, not a new tab: the result is a second
        view of this chart's data and belongs beside it.
        """
        is_surface_gradient = bool(results and results[0].model == DERIV_GRADIENT_SURFACE)
        if is_surface_gradient:
            # Contour Plot, not Surface: the magnitude is a scalar field
            # over (x, y), which is exactly what a contour map shows, and
            # (unlike Surface/Scatter3D) it needs no 3D projection.
            chart_type = "Contour Plot"
            options: dict[str, Any] = {"grid": True}
        else:
            chart_type = "Scatter Plot"
            options = {"grid": True, "linestyle": "-", "marker": ""}

        axis_id = self.resolve_destination_axis(
            selected_axis_id,
            chart_type=chart_type,
            title=self.current_model(),
            figure_name=self.result_figure_name(results, self.current_model()),
            options=options,
        )
        self._label_result_axis(results)
        return axis_id

    def _label_result_axis(self, results: Sequence[Any]) -> None:
        """Name the axis after what was actually computed.

        From the results rather than from the model name, because the same
        model produces different quantities: a first derivative and a second
        are both "Savitzky-Golay".
        """
        if self._result_axis_id is None or not results:
            return

        first = results[0]
        order = int(first.metadata.get("order", 1) or 1)
        spec = CALCULUS_MODELS.get(first.model)
        if spec is not None and spec.kind == "derivative" and not spec.surface:
            y_label = "d\u00b2y/dx\u00b2" if order == 2 else "dy/dx"
        elif first.model == DERIV_GRADIENT_SURFACE:
            y_label = "|\u2207z|"
        elif first.model == INTEGRAL_VOLUME_SURFACE:
            y_label = "volume"
        else:
            y_label = "\u222by dx"
        unit_label = str(first.metadata.get("per") or "")
        if unit_label:
            y_label = f"{y_label} (x in {unit_label}s)"

        self.label_result_axis(title=str(first.model or ""), y_label=y_label)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: CalculusResult,
    ) -> ResultSeriesSpec:
        del axis_id
        if result.z is not None:
            return ResultSeriesSpec(
                name=result.result_name,
                sql_query=f'SELECT x, y, z FROM "{table_name}"',
                roles={"x": "x", "y": "y", "z": "z"},
                style={
                    "generated_calculus": True,
                    "calculus_dialog": "series_calculus",
                    "source_name": result.source_name,
                    "model": result.model,
                },
            )
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_calculus": True,
                "calculus_dialog": "series_calculus",
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    RESULT_TABLE_PREFIX = 'Calculus'
    RESULT_TABLE_VARIANT = 'model'

    @property
    def operation_label(self) -> str:
        return "Calculus"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[CalculusResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        rows = [
            (
                result.source_name,
                result.model,
                report_html.format_number(result.y.size, digits=0),
                report_html.format_number(result.total)
                if result.total is not None
                else "&mdash;",
                ", ".join(
                    f"{key}: {value}" for key, value in result.metadata.items() if value
                ),
            )
            for result in results
        ]

        return report_html.document(
            _("Calculus"),
            self.current_model(),
            report_html.section(
                _("Results"),
                report_html.table(
                    (_("Series"), _("Model"), _("Points"), _("Total"), _("Detail")),
                    rows,
                    align=("left", "left", "right", "right", "left"),
                    empty_message=_("No results for this selection."),
                ),
            ),
        )
