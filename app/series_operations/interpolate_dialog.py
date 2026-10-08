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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from app.data.data_source import parse_roles, resolve_role_column
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    CallJob,
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
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
    SPACING_CUSTOM,
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


class SeriesInterpolateDialog(SeriesOperationDialogBase):
    """Compact series fitting/interpolation dialog for one chart panel."""

    MODELS = MODEL_NAMES
    MODEL_TOOLTIP = "Choose the fitting/interpolation model."
    Name: str  = "Interpolation"
    Description = "Fill missing values"

    # Every model here interpolates y as a function of x, so x has to be
    # ordered and single-valued: a spline through two different y at one x has
    # no solution, and SciPy reports that as a singular matrix rather than as
    # a problem with the data.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
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
    applied = Signal()

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QMainWindow,
    ) -> None:
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Interpolation",
            parent=parent,
            width=720,
            height=640
        )
        self._refresh_model_defaults()
        self.model_combo.setVisible(True)

    def init_operation_widgets(self) -> None:
        """Create interpolation controls before base builder hooks run."""
        self._last_results: list[FitResult] = []
        self._settings_form: QFormLayout | None = None

        self._degree_spin = QSpinBox(self)
        self._points_spin = QSpinBox(self)
        self._spacing_combo = QComboBox(self)
        self._range_edit = QLineEdit(self)
        self._integer_step_spin = QDoubleSpinBox(self)
        self._custom_x_edit = QLineEdit(self)
        self._extend_spin = QDoubleSpinBox(self)
        self._extrap_check = QCheckBox(_("Allow extrapolation"), self)
        self._spline_type_combo = QComboBox(self)
        self._spline_degree_spin = QSpinBox(self)
        self._cubic_bc_combo = QComboBox(self)
        self._smoothing_spin = QDoubleSpinBox(self)
        self._guess_check = QCheckBox(_("Guess starting parameters"), self)
        self._params_label = QLabel(_("Start params:"), self)
        self._params_edit = QPlainTextEdit(self)
        #self.series_selector.reload(select_all_series=False)


    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_parameter_selector(self) -> QWidget:
        settings = QWidget(self)
        form = QFormLayout(settings)
        self._settings_form = form
        form.setContentsMargins(4, 4, 4, 4)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._degree_spin.setRange(1, 12)
        self._degree_spin.setValue(2)
        form.addRow(_("Degree:"), self._degree_spin)

        self._points_spin.setRange(10, 200_000)
        self._points_spin.setValue(600)
        form.addRow(_("Points:"), self._points_spin)

        self._spacing_combo.addItems([
            "linspace",
            "logspace",
            "geomspace",
            "original data X",
            "integer step",
            "chebyshev nodes",
            "custom X values",
        ])
        form.addRow(_("X spacing:"), self._spacing_combo)

        self._range_edit.setPlaceholderText(_("auto, or start, stop"))
        form.addRow(_("X range:"), self._range_edit)

        self._integer_step_spin.setRange(1e-12, 1e12)
        self._integer_step_spin.setDecimals(6)
        self._integer_step_spin.setValue(1.0)
        form.addRow(_("Step:"), self._integer_step_spin)

        self._custom_x_edit.setPlaceholderText(_("1, 2.5, 10 or one per line"))
        form.addRow(_("Eval X:"), self._custom_x_edit)

        self._extend_spin.setRange(0.0, 500.0)
        self._extend_spin.setDecimals(1)
        self._extend_spin.setSuffix(" %")
        form.addRow(_("Extend:"), self._extend_spin)

        self._extrap_check.setChecked(True)
        form.addRow("", self._extrap_check)

        self._spline_type_combo.addItems([
            "UnivariateSpline",
            "CubicSpline",
            "PCHIP",
            "Akima1D",
            "B-spline",
        ])
        form.addRow(_("Spline:"), self._spline_type_combo)

        self._spline_degree_spin.setRange(1, 5)
        self._spline_degree_spin.setValue(3)
        form.addRow(_("Spline k:"), self._spline_degree_spin)

        self._cubic_bc_combo.addItems(["not-a-knot", "natural", "clamped", "periodic"])
        form.addRow(_("Boundary:"), self._cubic_bc_combo)

        self._smoothing_spin.setRange(0.0, 1_000_000.0)
        self._smoothing_spin.setDecimals(3)
        form.addRow(_("Smooth s:"), self._smoothing_spin)

        self._guess_check.setChecked(True)
        form.addRow("", self._guess_check)

        self._params_edit.setMaximumHeight(64)
        self._params_edit.setPlaceholderText(_('Example: {"a": 1.0, "b": 0.1}'))
        form.addRow(self._params_label, self._params_edit)

        for widget in (
            self._range_edit,
            self._custom_x_edit,
            self._spacing_combo,
            self._spline_type_combo,
            self._cubic_bc_combo,
            self._params_edit,
        ):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        scroll = QScrollArea(self)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        self._set_setting_tooltips()
        return scroll

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._degree_spin.valueChanged.connect(self._refresh_model_defaults)
        self._points_spin.valueChanged.connect(self.mark_results_stale)
        self._spacing_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._range_edit.textChanged.connect(self.mark_results_stale)
        self._integer_step_spin.valueChanged.connect(self.mark_results_stale)
        self._custom_x_edit.textChanged.connect(self.mark_results_stale)
        self._extend_spin.valueChanged.connect(self.mark_results_stale)
        self._extrap_check.stateChanged.connect(self.mark_results_stale)
        self._spline_type_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._spline_degree_spin.valueChanged.connect(self.mark_results_stale)
        self._cubic_bc_combo.currentIndexChanged.connect(self.mark_results_stale)
        self._smoothing_spin.valueChanged.connect(self.mark_results_stale)
        self._guess_check.stateChanged.connect(self._refresh_model_defaults)
        self._params_edit.textChanged.connect(self.mark_results_stale)

    def _set_setting_tooltips(self) -> None:
        self._doc_link.setToolTip(_("Open NumPy/SciPy documentation for the model."))
        self._degree_spin.setToolTip(_("Polynomial degree for NumPy polyfit."))
        self._points_spin.setToolTip(_("Number of generated points for continuous spacing modes."))
        self._spacing_combo.setToolTip(_("Choose how generated X values are built."))
        self._range_edit.setToolTip(_("Optional range as 'start, stop'. Empty = data range plus Extend."))
        self._integer_step_spin.setToolTip(_("Step size used by integer/fixed-step spacing."))
        self._custom_x_edit.setToolTip(_("Explicit X values where Y is evaluated."))
        self._extend_spin.setToolTip(_("Extend automatic data range by this percent on both sides."))
        self._extrap_check.setToolTip(_("Allow evaluation outside source data X range."))
        self._spline_type_combo.setToolTip(_("Spline algorithm for SciPy spline family."))
        self._spline_degree_spin.setToolTip(_("Spline degree k for UnivariateSpline/B-spline."))
        self._cubic_bc_combo.setToolTip(_("Spline boundary condition."))
        self._smoothing_spin.setToolTip(_("UnivariateSpline smoothing factor s."))
        self._guess_check.setToolTip(_("Guess starting parameters for nonlinear models."))
        self._params_edit.setToolTip(_("JSON starting parameters for nonlinear series_fit models."))

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        field.setVisible(visible)
        form = self._settings_form
        if form is None:
            return
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

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

    def _refresh_model_defaults(self) -> None:
        model = self._model_name()
        spacing = self._spacing_combo.currentText()
        spline_type = self._spline_type_combo.currentText()

        # Starting parameters are curve_fit's p0: the fitted models take them,
        # interpolation has none to take.
        uses_start_params = MODEL_NAMES[model].curve_fit
        uses_poly_degree = model == MODEL_POLYNOMIAL
        uses_spline_menu = model == MODEL_SCIPY_SPLINE
        uses_spline_k = uses_spline_menu and spline_type in {
            "UnivariateSpline",
            "B-spline",
        }
        uses_smoothing = uses_spline_menu and spline_type == "UnivariateSpline"
        uses_boundary = model == MODEL_SCIPY_CUBIC or (
            uses_spline_menu and spline_type in {"CubicSpline", "B-spline"}
        )

        uses_custom_x = spacing == "custom X values"
        uses_original_x = spacing == "original data X"
        uses_step = spacing == "integer step"
        uses_points = spacing not in {
            "custom X values",
            "original data X",
            "integer step",
        }
        uses_range = spacing not in {"custom X values", "original data X"}
        uses_extrap = uses_range or model in {
            MODEL_SCIPY_PCHIP,
            MODEL_SCIPY_AKIMA,
            MODEL_SCIPY_CUBIC,
            MODEL_SCIPY_SPLINE,
        }

        self._set_form_row_visible(self._degree_spin, uses_poly_degree)
        self._set_form_row_visible(self._points_spin, uses_points)
        self._set_form_row_visible(self._range_edit, uses_range)
        self._set_form_row_visible(self._integer_step_spin, uses_step)
        self._set_form_row_visible(self._custom_x_edit, uses_custom_x)
        self._set_form_row_visible(self._extend_spin, uses_range)
        self._set_form_row_visible(self._extrap_check, uses_extrap)
        self._set_form_row_visible(self._spline_type_combo, uses_spline_menu)
        self._set_form_row_visible(self._spline_degree_spin, uses_spline_k)
        self._set_form_row_visible(self._cubic_bc_combo, uses_boundary)
        self._set_form_row_visible(self._smoothing_spin, uses_smoothing)
        self._set_form_row_visible(self._guess_check, uses_start_params)
        self._set_form_row_visible(self._params_edit, uses_start_params)
        self._params_label.setVisible(uses_start_params)


        if uses_start_params and self._guess_check.isChecked():
            self._params_edit.blockSignals(True)
            self._params_edit.setPlainText(
                json.dumps(self._default_params_for_model(model), indent=2)
            )
            self._params_edit.blockSignals(False)

        # Suppress unused variable warning while keeping logic readable.
        _unused = uses_original_x
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _start_params(self) -> dict[str, float]:
        if not MODEL_NAMES[self._model_name()].curve_fit:
            return {}
        text = self._params_edit.toPlainText().strip()
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
        axis_id = self.series_selector.selected_axis_id()
        inputs = [
            (series, *self._series_data(series))
            for series in (self._series_choice_from_row(row) for row in self.selected_series())
        ]

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
        spacing = self._spacing_combo.currentText()
        custom = parse_values(self._custom_x_edit.text()) if spacing == SPACING_CUSTOM else None
        text = self._range_edit.text().strip()
        explicit = None
        if text:
            values = parse_values(text)
            if values.size < 2:
                raise ValueError("X range must contain start and stop.")
            explicit = (float(values[0]), float(values[1]))
        return _EvaluationGrid(
            spacing=spacing,
            custom=custom,
            explicit=explicit,
            extend_percent=float(self._extend_spin.value()),
            extrapolate=self._extrap_check.isChecked(),
            count=int(self._points_spin.value()),
            step=float(self._integer_step_spin.value()),
        )

    def _settings(self) -> InterpolationSettings:
        """The model settings, read off the controls."""
        return InterpolationSettings(
            degree=int(self._degree_spin.value()),
            extrapolate=self._extrap_check.isChecked(),
            cubic_bc=self._cubic_bc_combo.currentText(),
            spline_type=self._spline_type_combo.currentText(),
            spline_degree=int(self._spline_degree_spin.value()),
            smoothing=float(self._smoothing_spin.value()),
        )

    @staticmethod
    def format_results(results: Sequence[FitResult]) -> str:
        lines: list[str] = []
        for result in results:
            lines.append(result.source.name)
            if result.metrics:
                r2 = result.metrics.get("r2")
                rmse = result.metrics.get("rmse")
                if r2 is not None and np.isfinite(r2):
                    lines.append(f"R² = {r2:.5g}")
                if rmse is not None and np.isfinite(rmse):
                    lines.append(f"RMSE = {rmse:.5g}")
            else:
                lines.append(_("Generated {count} points").format(count=len(result.x_eval)))
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # SeriesOperationDialogBase hooks
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
