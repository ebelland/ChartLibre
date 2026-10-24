"""Compact chart-series series  interpolation dialog.

Features:
- Select one chart axis and one or more series.
- Fit/interpolate selected series with NumPy/SciPy models.
- Dynamic settings visibility for compact PySide6 strict UI.
- Evaluate generated Y values at multiple X spacing modes/custom X values.
- Apply generated series to the selected axis without closing the dialog.
- Re-apply removes previously generated series by this dialog and replaces them.
- Save generated datapoints as a normal SQLite table.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal

from app.data.data_source import parse_roles, resolve_role_column
from app.plugin_api.data_operations.results import TableResult
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam, TextParam
from app.plugin_api.data_operations.dialog_base import (
    CallJob,
    OperationModel,
    ResultSeriesSpec,
    DataOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.i18n import _

from app.analysis.interpolation import (
    MODEL_EXPONENTIAL,
    MODEL_GAUSSIAN,
    MODEL_LINEAR,
    MODEL_LOGARITHMIC,
    MODEL_NUMPY_INTERP,
    MODEL_POLYNOMIAL,
    MODEL_POWER,
    MODEL_SCIPY_AKIMA,
    MODEL_SCIPY_CUBIC,
    MODEL_SCIPY_PCHIP,
    MODEL_SCIPY_SPLINE,
    MODEL_SIGMOID,
    SPACING_CHEBYSHEV,
    SPACING_CUSTOM,
    SPACING_GEOMETRIC,
    SPACING_INTEGER_STEP,
    SPACING_LINEAR,
    SPACING_LOG,
    SPACING_ORIGINAL,
    InterpolationSettings,
    default_params,
    evaluation_range,
    evaluation_x,
    goodness,
    interpolate,
    parse_values,
)


# The model names are the engine's (app.analysis.interpolation).


@dataclass(frozen=True, slots=True, kw_only=True)
class InterpolationModel(OperationModel):
    #: Fitted by scipy.optimize.curve_fit from starting parameters, which the
    #: user may guess or type in.
    curve_fit: bool = False


#: The models offered, in combo order.
MODEL_NAMES: dict[str, InterpolationModel] = {
    MODEL_POLYNOMIAL: InterpolationModel(
        doc_title="NumPy polyfit",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html",
    ),
    MODEL_LINEAR: InterpolationModel(
        doc_title="NumPy polyfit",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html",
    ),
    MODEL_EXPONENTIAL: InterpolationModel(
        doc_title="SciPy curve_fit",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
        curve_fit=True,
    ),
    MODEL_LOGARITHMIC: InterpolationModel(
        doc_title="SciPy curve_fit",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
        curve_fit=True,
    ),
    MODEL_POWER: InterpolationModel(
        doc_title="SciPy curve_fit",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
        curve_fit=True,
    ),
    MODEL_GAUSSIAN: InterpolationModel(
        doc_title="SciPy curve_fit",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
        curve_fit=True,
    ),
    MODEL_SIGMOID: InterpolationModel(
        doc_title="SciPy curve_fit",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
        curve_fit=True,
    ),
    MODEL_NUMPY_INTERP: InterpolationModel(
        doc_title="NumPy interp",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.interp.html",
        group="interpolation",
    ),
    MODEL_SCIPY_PCHIP: InterpolationModel(
        doc_title="SciPy PchipInterpolator",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html",
        group="interpolation",
    ),
    MODEL_SCIPY_AKIMA: InterpolationModel(
        doc_title="SciPy Akima1DInterpolator",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html",
        group="interpolation",
    ),
    MODEL_SCIPY_CUBIC: InterpolationModel(
        doc_title="SciPy CubicSpline",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html",
        group="interpolation",
    ),
    MODEL_SCIPY_SPLINE: InterpolationModel(
        doc_title="SciPy interpolation",
        doc_url="https://docs.scipy.org/doc/scipy/tutorial/interpolate.html",
        group="interpolation",
    ),
}

_SPACINGS = (SPACING_LINEAR, SPACING_LOG, SPACING_GEOMETRIC, SPACING_ORIGINAL, SPACING_INTEGER_STEP,
             SPACING_CHEBYSHEV, SPACING_CUSTOM)
#: The spacings that build x over a range (the data's, extended, or one typed).
_RANGED = {"spacing": tuple(spacing for spacing in _SPACINGS if spacing not in (SPACING_CUSTOM, SPACING_ORIGINAL))}
_CURVE_FIT = {"model": tuple(name for name, spec in MODEL_NAMES.items() if spec.curve_fit)}
_SPLINE = {"model": (MODEL_SCIPY_SPLINE,)}

#: The settings, each shown for the models and the spacings that use it.
INTERPOLATION_PARAMS = (
    IntParam("degree", "Degree:", default_value=2, minimum=1, maximum=12, visible_for={"model": (MODEL_POLYNOMIAL,)},
             tooltip="Polynomial degree for NumPy polyfit."),
    IntParam("points", "Points:", default_value=600, minimum=10, maximum=200_000,
             visible_for={"spacing": tuple(spacing for spacing in _RANGED["spacing"] if spacing != SPACING_INTEGER_STEP)},
             tooltip="Number of generated points for continuous spacing modes."),
    ChoiceParam("spacing", "X spacing:", choices=_SPACINGS, tooltip="Choose how generated X values are built."),
    TextParam("x_range", "X range:", placeholder="auto, or start, stop", visible_for=_RANGED,
              tooltip="Optional range as 'start, stop'. Empty = data range plus Extend."),
    FloatParam("step", "Step:", default_value=1.0, minimum=1e-12, maximum=1e12, decimals=6, step=0.001,
               visible_for={"spacing": (SPACING_INTEGER_STEP,)}, tooltip="Step size used by integer/fixed-step spacing."),
    TextParam("custom_x", "Eval X:", placeholder="1, 2.5, 10 or one per line", visible_for={"spacing": (SPACING_CUSTOM,)},
              tooltip="Explicit X values where Y is evaluated."),
    FloatParam("extend", "Extend:", default_value=0.0, minimum=0.0, maximum=500.0, decimals=1, step=1.0, suffix=" %",
               visible_for=_RANGED, tooltip="Extend automatic data range by this percent on both sides."),
    BoolParam("extrapolate", "Allow extrapolation", default_value=True,
              visible_if=lambda values: values.get("spacing") in _RANGED["spacing"] or values.get("model") in (
                  MODEL_SCIPY_PCHIP, MODEL_SCIPY_AKIMA, MODEL_SCIPY_CUBIC, MODEL_SCIPY_SPLINE),
              tooltip="Allow evaluation outside source data X range."),
    ChoiceParam("spline_type", "Spline:", choices=("UnivariateSpline", "CubicSpline", "PCHIP", "Akima1D", "B-spline"),
                visible_for=_SPLINE, tooltip="Spline algorithm for SciPy spline family."),
    IntParam("spline_degree", "Spline k:", default_value=3, minimum=1, maximum=5,
             visible_for={**_SPLINE, "spline_type": ("UnivariateSpline", "B-spline")},
             tooltip="Spline degree k for UnivariateSpline/B-spline."),
    ChoiceParam("cubic_bc", "Boundary:", choices=("not-a-knot", "natural", "clamped", "periodic"),
                visible_if=lambda values: values.get("model") == MODEL_SCIPY_CUBIC or (
                    values.get("model") == MODEL_SCIPY_SPLINE and values.get("spline_type") in ("CubicSpline", "B-spline")),
                tooltip="Spline boundary condition."),
    FloatParam("smoothing", "Smooth s:", default_value=0.0, minimum=0.0, maximum=1_000_000.0, decimals=3, step=0.1,
               visible_for={**_SPLINE, "spline_type": ("UnivariateSpline",)}, tooltip="UnivariateSpline smoothing factor s."),
    BoolParam("guess", "Guess starting parameters", default_value=True, visible_for=_CURVE_FIT,
              tooltip="Guess starting parameters for nonlinear models."),
    TextParam("start_params", "Start params:", placeholder='Example: {"a": 1.0, "b": 0.1}', visible_for=_CURVE_FIT,
              tooltip="JSON starting parameters for nonlinear series_fit models."),
)

_TABLE_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class _EvaluationGrid:
    """Where to evaluate the interpolant, read off the controls once per run."""

    spacing: str
    custom: np.ndarray | None
    explicit: tuple[float, float] | None
    extend_percent: float
    extrapolate: bool
    count: int
    step: float

    def x_for(self, x_data: np.ndarray) -> np.ndarray:
        start, stop = evaluation_range(
            x_data,
            explicit=self.explicit,
            extend_percent=self.extend_percent,
            extrapolate=self.extrapolate,
        )
        return evaluation_x(
            x_data,
            self.spacing,
            start=start,
            stop=stop,
            count=self.count,
            step=self.step,
            custom=self.custom,
        )


@dataclass(slots=True)
class SeriesChoice:
    """Selectable chart series descriptor."""

    series_id: int
    series_index: int
    name: str
    sql_query: str
    roles: dict[str, Any]

    def label(self) -> str:
        name = self.name.strip() or f"Series {self.series_index}"
        return f"{self.series_index}: {name}"


@dataclass(slots=True)
class FitResult(TableResult):
    """interpolation output for one source series."""

    source: SeriesChoice
    model: str
    table_name: str
    output_name: str
    x_eval: np.ndarray
    y_eval: np.ndarray
    params: dict[str, float]
    metrics: dict[str, float]
    message: str


    @property
    def series(self) -> tuple[str, ...]:
        return (self.source.name,)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_series_id": self.source.series_id,
                "source_series_name": self.source.name,
                "model": self.model,
                "x": self.x_eval,
                "y": self.y_eval,
            }
        )


class SeriesInterpolateDialog(DataOperationDialogBase):
    """Compact series fitting/interpolation dialog for one chart panel."""

    #: Where the result goes: the Charts tab's "Draw on".
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS

    MODELS = MODEL_NAMES
    PARAMS = INTERPOLATION_PARAMS
    MODEL_TOOLTIP = "Choose the fitting/interpolation model."
    Name: str  = "Interpolation"
    Description = "Fill missing values"

    # Every model here interpolates y as a function of x, so x has to be
    # ordered and single-valued: a spline through two different y at one x has
    # no solution, and SciPy reports that as a singular matrix rather than as
    # a problem with the data.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True

    #: Computed on a worker thread: see DataOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    Icon = """
    <path d="M4 18.5h16"/>
    <path d="M4.5 18V5"/>
    <path d="M6.5 15.5l4.2-4.2 3.2 2.4 4.2-6.2"/>
    <circle cx="6.5" cy="15.5" r="1"/>
    <circle cx="10.7" cy="11.3" r="1"/>
    <circle cx="13.9" cy="13.7" r="1"/>
    <circle cx="18.1" cy="7.5" r="1"/>
    """
    Keywords: str = (
        "interpolazione missing values valori mancanti fill riempi spline linear lineare resample "
        "ricampiona"
    )
    Goals: list[str] = ["prepare", "signal"]
    Category: str = "Curves"
    Order: int = 74
    applied = Signal()

    WINDOW_SIZE = (720, 640)

    def __init__(self, **kwargs: Any) -> None:
        #: The model and "Guess" the starting values were last filled in for.
        self._guessed_for: tuple[str, bool] | None = None
        super().__init__(**kwargs)
        self._doc_link.setToolTip(_("Open NumPy/SciPy documentation for the model."))
        self._guess_start_params()

    # ------------------------------------------------------------------
    # Descriptor loading
    # ------------------------------------------------------------------


    def _series_choice_from_row(self, row: Any) -> SeriesChoice:
        series_index = int(row["series_index"])
        return SeriesChoice(
            series_id=int(row["id"]),
            series_index=series_index,
            name=str(row["name"] or f"Series {series_index}"),
            sql_query=str(row["sql_query"]),
            roles=parse_roles(row["roles"]),
        )

    # ------------------------------------------------------------------
    # Dynamic settings
    # ------------------------------------------------------------------

    def _model_name(self) -> str:
        return str(self.model_combo.currentText())

    def _refresh_visibility(self, *_ignored: Any) -> None:
        super()._refresh_visibility()
        self._guess_start_params()

    def parameters_changed(self) -> None:
        self._guess_start_params()
        super().parameters_changed()

    def _guess_start_params(self) -> None:
        """With "Guess" ticked, the model's own starting values in the box - once per model and tick, so what
        is typed there afterwards stays. They are curve_fit's p0: only the fitted models take them."""
        form = getattr(self, "_parameter_form_spec", None)
        if form is None:
            return
        model = self._model_name()
        guess = bool(form.values()["guess"])
        if (model, guess) == self._guessed_for:
            return
        self._guessed_for = (model, guess)
        if guess and MODEL_NAMES[model].curve_fit:
            form.set_values({"start_params": json.dumps(self._default_params_for_model(model))})

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _start_params(self) -> dict[str, float]:
        if not MODEL_NAMES[self._model_name()].curve_fit:
            return {}
        text = str(self.parameter_values()["start_params"]).strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid parameter JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Start parameters must be a JSON object.")
        return {str(k): float(v) for k, v in value.items()}

    @staticmethod
    def _default_params_for_model(model: str) -> dict[str, float]:
        return default_params(model)

    def prepare_job(self, **options: Any) -> CallJob:
        """Read the series and every control here; the interpolation runs in the job."""
        del options
        model = self._model_name()
        settings = self._settings()
        start_params = self._start_params()
        grid = self._evaluation_grid()
        inputs = [
            (series, *self._series_data(series))
            for series in (self._series_choice_from_row(row) for row in self.selected_series())
        ]
        # After the series: reading them may draw a table's columns, which
        # is the axis the result belongs on.
        axis_id = self.series_selector.selected_axis_id()

        def compute() -> list[FitResult]:
            return [
                self._interpolate_one(series, x_data, y_data, model, settings, start_params, grid, axis_id)
                for series, x_data, y_data in inputs
            ]

        return CallJob(compute)

    def _series_data(self, series: SeriesChoice) -> tuple[np.ndarray, np.ndarray]:
        df = self._repo.query_df(series.sql_query)
        x_col, y_col = self._xy_columns(df, series.roles)

        # prepare_input_xy replaces the old _clean_xy/_sort_unique_xy pair. It
        # does the same three things - drop non-finite, sort, average duplicate
        # x - but says so. The old pair repaired silently, so a series with two
        # readings at one x was interpolated through neither and nothing in the
        # interface ever mentioned it.
        return self.prepare_input_xy(
            self.numeric_x(df[x_col], series.name),
            self.numeric_y(df[y_col]),
            label=series.name,
        )

    @staticmethod
    def _interpolate_one(
        series: SeriesChoice,
        x_data: np.ndarray,
        y_data: np.ndarray,
        model: str,
        settings: InterpolationSettings,
        start_params: dict[str, float],
        grid: _EvaluationGrid,
        axis_id: int | None,
    ) -> FitResult:
        x_eval = grid.x_for(x_data)
        result = interpolate(model, x_data, y_data, x_eval, settings, start_params)
        metrics = goodness(model, x_data, y_data, result.params, settings)

        safe_model_name = _TABLE_SAFE_RE.sub(
            "_",
            model.strip().lower(),
        ).strip("_") or "series"

        table_name = generated_table_name(
            f"Interpolation_axis{axis_id}"
            f"_series{series.series_id}_{safe_model_name}",
            fallback="Interpolation_Result",
        )

        return FitResult(
            source=series,
            model=model,
            table_name=table_name,
            output_name=f"Interpolate: {series.name} [{model}]",
            x_eval=x_eval,
            y_eval=result.y,
            params=result.params,
            metrics=metrics,
            message=result.message,
        )

    @staticmethod
    def _xy_columns(df: pd.DataFrame, roles: Mapping[str, Any]) -> tuple[str, str]:
        if df.empty:
            applogger.error("Series query returned no rows.")

        columns = [str(col) for col in df.columns]
        x_name = resolve_role_column(columns, roles, "x") or ""
        y_name = resolve_role_column(columns, roles, "y") or ""

        if x_name in columns and y_name in columns:
            return x_name, y_name

        numeric = [
            str(col)
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(numeric) < 2:
            applogger.error("Series query must expose at least two numeric columns.")

        return numeric[0], numeric[1]

    def _evaluation_grid(self) -> _EvaluationGrid:
        p = self.parameter_values()
        spacing = str(p["spacing"])
        custom = parse_values(p["custom_x"]) if spacing == SPACING_CUSTOM else None
        text = str(p["x_range"]).strip()
        explicit = None
        if text:
            values = parse_values(text)
            if values.size < 2:
                raise ValueError("X range must contain start and stop.")
            explicit = (float(values[0]), float(values[1]))
        return _EvaluationGrid(
            spacing=spacing, custom=custom, explicit=explicit, extend_percent=float(p["extend"]),
            extrapolate=bool(p["extrapolate"]), count=int(p["points"]), step=float(p["step"]),
        )

    def _settings(self) -> InterpolationSettings:
        """The model settings, read off the form."""
        p = self.parameter_values()
        return InterpolationSettings(
            degree=int(p["degree"]), extrapolate=bool(p["extrapolate"]), cubic_bc=str(p["cubic_bc"]),
            spline_type=str(p["spline_type"]), spline_degree=int(p["spline_degree"]), smoothing=float(p["smoothing"]),
        )

    # ------------------------------------------------------------------
    # DataOperationDialogBase hooks
    # ------------------------------------------------------------------


    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated": True, "dialog": "series_interpolation"}

    def result_table_name(self, axis_id: int, result: FitResult) -> str:
        return result.table_name

    def result_series_spec(self, axis_id: int, table_name: str, result: FitResult) -> ResultSeriesSpec:
        return ResultSeriesSpec(
            name=result.output_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "linestyle": "--",
                "linewidth": 2.0,
                "marker": "",
                "source_series_id": result.source.series_id,
                "model": result.model,
                "generated": True,
                "dialog": "series_interpolation",
            },
        )
