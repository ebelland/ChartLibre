"""Chart-series smoothing dialog.

This module is intended to be a drop-in companion to ``dialog_series_interpolate``.
It keeps the same high-level construction pattern::

    dialog = SeriesSmoothingDialog(repo, figure_id, parent=self)

The dialog itself shows the same selection concept as the interpolation dialog:
axis list on the left, series list below it, and method/settings on the right.
The selected ``figure_id`` is the data context; the selected axis only scopes the
series shown in the dialog and is returned in metadata/callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QWidget,
)
from app.analysis.smoothing import (
    SMOOTH2D_GAUSSIAN,
    SMOOTH2D_MEDIAN,
    SMOOTH2D_RBF,
    SMOOTH2D_RECT_SPLINE,
    SMOOTH2D_SPLINE,
    SMOOTH2D_TV,
    SMOOTH3D_FFT,
    SMOOTH3D_GAUSSIAN,
    SMOOTH3D_MEDIAN,
    SMOOTH3D_RBF,
    SMOOTH3D_TV,
    SMOOTH_BUTTERWORTH,
    SMOOTH_FFT,
    SMOOTH_GAUSSIAN,
    SMOOTH_HP,
    SMOOTH_KALMAN,
    SMOOTH_LOWESS,
    SMOOTH_MEDIAN,
    SMOOTH_MOVING_AVERAGE,
    SMOOTH_SAVGOL,
    SMOOTH_SPLINE,
    SMOOTH_TV,
    SMOOTH_WAVELET,
    SMOOTH_WHITTAKER,
    SMOOTH_WIENER,
    is_available,
    smooth_series,
)
from app.data.data_source import parse_roles, row_value

from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.data_operations.results import TableResult
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam
from app.plugin_api.data_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesJob,
    DataOperationDialogBase,
    SeriesOutcome,
)
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.utils.i18n import _
from app.utils.coercion import to_numbers


# ---------------------------------------------------------------------------
# Method constants and documentation links
# ---------------------------------------------------------------------------

DIM_1D = "1D series"
DIM_2D = "2D surface / XYZ"
DIM_3D = "3D volume / XYZW"

#: The engine speaks in dimensions, the dialog in the names above.
_DIMENSION_NUMBER: dict[str, int] = {DIM_1D: 1, DIM_2D: 2, DIM_3D: 3}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SeriesChoice:
    """Selectable series descriptor loaded from the repository."""

    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    values: np.ndarray | None = None
    source: Any | None = None


@dataclass(slots=True)
class SmoothResult(TableResult):
    """Result returned by the smoothing dialog."""

    source_name: str
    result_name: str
    method: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    values: np.ndarray | None
    metadata: dict[str, Any]

    def to_df(self) -> pd.DataFrame:
        return SeriesSmoothingDialog.results_to_dataframe([self])


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSpec(OperationModel):
    """One smoothing model: the data it works on and the parameters it shows."""

    #: DIM_1D, DIM_2D or DIM_3D: the Data type the model is listed under.
    dimension: str
    #: The parameter rows the model shows.
    fields: frozenset[str]


#: Every smoothing model, in the Method combo's order within each data type.
_ALL_MODELS: dict[str, ModelSpec] = {
    SMOOTH_MOVING_AVERAGE: ModelSpec(
        dimension=DIM_1D,
        doc_title="NumPy convolve",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.convolve.html",
        fields=frozenset(("window", "centered")),
    ),
    SMOOTH_SAVGOL: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy savgol_filter",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html",
        fields=frozenset(("window", "polyorder", "deriv", "delta", "savgol_mode")),
    ),
    SMOOTH_GAUSSIAN: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy gaussian_filter1d",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html",
        fields=frozenset(("sigma", "truncate", "mode")),
    ),
    SMOOTH_MEDIAN: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy medfilt",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html",
        fields=frozenset(("kernel",)),
    ),
    SMOOTH_WIENER: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy wiener",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html",
        fields=frozenset(("window", "noise")),
    ),
    SMOOTH_SPLINE: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy UnivariateSpline",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html",
        fields=frozenset(("spline_s", "spline_k")),
    ),
    SMOOTH_LOWESS: ModelSpec(
        dimension=DIM_1D,
        doc_title="Statsmodels LOWESS",
        doc_url="https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html",
        fields=frozenset(("lowess_frac", "lowess_it")),
    ),
    SMOOTH_KALMAN: ModelSpec(
        dimension=DIM_1D,
        doc_title="Kalman smoothing",
        doc_url="https://en.wikipedia.org/wiki/Kalman_filter",
        fields=frozenset(("kalman_process_variance", "kalman_measurement_variance", "kalman_initial_covariance")),
    ),
    SMOOTH_FFT: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy FFT",
        doc_url="https://docs.scipy.org/doc/scipy/reference/fft.html",
        fields=frozenset(("fft_cutoff_ratio",)),
    ),
    SMOOTH_BUTTERWORTH: ModelSpec(
        dimension=DIM_1D,
        doc_title="SciPy butter",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html",
        fields=frozenset(("butter_fs", "butter_cutoff", "butter_high_cutoff", "butter_order", "butter_type")),
    ),
    SMOOTH_WAVELET: ModelSpec(
        dimension=DIM_1D,
        doc_title="PyWavelets",
        doc_url="https://pywavelets.readthedocs.io/",
        fields=frozenset(("wavelet", "wavelet_level", "wavelet_threshold_factor", "wavelet_threshold_mode")),
    ),
    SMOOTH_WHITTAKER: ModelSpec(
        dimension=DIM_1D,
        doc_title="Whittaker smoothing",
        doc_url="https://pybaselines.readthedocs.io/",
        fields=frozenset(("whittaker_lambda", "whittaker_order")),
    ),
    SMOOTH_HP: ModelSpec(
        dimension=DIM_1D,
        doc_title="Statsmodels hpfilter",
        doc_url="https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html",
        fields=frozenset(("hp_lambda",)),
    ),
    SMOOTH_TV: ModelSpec(
        dimension=DIM_1D,
        doc_title="Total variation denoising (Chambolle)",
        doc_url="https://en.wikipedia.org/wiki/Total_variation_denoising",
        fields=frozenset(("tv_weight",)),
    ),
    SMOOTH2D_GAUSSIAN: ModelSpec(
        dimension=DIM_2D,
        doc_title="SciPy gaussian_filter",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html",
        fields=frozenset(("sigma_x", "sigma_y", "truncate", "mode")),
    ),
    SMOOTH2D_MEDIAN: ModelSpec(
        dimension=DIM_2D,
        doc_title="SciPy median_filter",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html",
        fields=frozenset(("kernel_x", "kernel_y", "mode")),
    ),
    SMOOTH2D_SPLINE: ModelSpec(
        dimension=DIM_2D,
        doc_title="SciPy SmoothBivariateSpline",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.html",
        fields=frozenset(("spline_s", "kx", "ky")),
    ),
    SMOOTH2D_RECT_SPLINE: ModelSpec(
        dimension=DIM_2D,
        doc_title="SciPy RectBivariateSpline",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html",
        fields=frozenset(("spline_s", "kx", "ky")),
    ),
    SMOOTH2D_RBF: ModelSpec(
        dimension=DIM_2D,
        doc_title="SciPy RBFInterpolator",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html",
        fields=frozenset(("rbf_kernel", "rbf_smoothing", "rbf_epsilon", "rbf_neighbors")),
    ),
    SMOOTH2D_TV: ModelSpec(
        dimension=DIM_2D,
        doc_title="Total variation denoising (Chambolle)",
        doc_url="https://en.wikipedia.org/wiki/Total_variation_denoising",
        fields=frozenset(("tv_weight",)),
    ),
    SMOOTH3D_GAUSSIAN: ModelSpec(
        dimension=DIM_3D,
        doc_title="SciPy gaussian_filter",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html",
        fields=frozenset(("sigma_x", "sigma_y", "sigma_z", "truncate", "mode")),
    ),
    SMOOTH3D_MEDIAN: ModelSpec(
        dimension=DIM_3D,
        doc_title="SciPy median_filter",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html",
        fields=frozenset(("kernel_x", "kernel_y", "kernel_z", "mode")),
    ),
    SMOOTH3D_FFT: ModelSpec(
        dimension=DIM_3D,
        doc_title="SciPy FFT",
        doc_url="https://docs.scipy.org/doc/scipy/reference/fft.html",
        fields=frozenset(("fft_cutoff_ratio",)),
    ),
    SMOOTH3D_RBF: ModelSpec(
        dimension=DIM_3D,
        doc_title="SciPy RBFInterpolator",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html",
        fields=frozenset(("rbf_kernel", "rbf_smoothing", "rbf_epsilon", "rbf_neighbors")),
    ),
    SMOOTH3D_TV: ModelSpec(
        dimension=DIM_3D,
        doc_title="Total variation denoising (Chambolle)",
        doc_url="https://en.wikipedia.org/wiki/Total_variation_denoising",
        fields=frozenset(("tv_weight",)),
    ),
}

#: The models offered here: every one whose optional package is installed.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    name: spec for name, spec in _ALL_MODELS.items() if is_available(name)
}


def models_for_dimension(dimension: str) -> list[str]:
    """Return model names shown in the method combo for one dimension."""
    return [name for name, spec in MODEL_REGISTRY.items() if spec.dimension == dimension]


def model_spec(method: str) ->ModelSpec |None:
    """Return the registry spec or raise a clean error for stale UI values."""
    try:
        return MODEL_REGISTRY[method]
    except KeyError:
        applogger.error(f"Unsupported smoothing method: {method}")
        return None

def _shown_for(field: str) -> dict[str, tuple[str, ...]]:
    """The methods that use *field* (ModelSpec.fields): its row is shown for them only."""
    return {"method": tuple(name for name, spec in _ALL_MODELS.items() if field in spec.fields)}


def _int(name: str, label: str, default: int, minimum: int, maximum: int, tooltip: str,
         special_text: str = "") -> IntParam:
    return IntParam(name, label, default_value=default, minimum=minimum, maximum=maximum,
                    special_text=special_text, tooltip=tooltip, visible_for=_shown_for(name))


def _float(name: str, label: str, default: float, minimum: float, maximum: float, decimals: int, tooltip: str,
           special_text: str = "") -> FloatParam:
    return FloatParam(name, label, default_value=default, minimum=minimum, maximum=maximum, decimals=decimals,
                      step=10 ** -min(decimals, 3), special_text=special_text, tooltip=tooltip,
                      visible_for=_shown_for(name))


def _choice(name: str, label: str, choices: tuple[str, ...], tooltip: str) -> ChoiceParam:
    return ChoiceParam(name, label, choices=choices, tooltip=tooltip, visible_for=_shown_for(name))


#: Every method's settings; each row shows for the methods that list it (ModelSpec.fields).
SMOOTHING_PARAMS = (
    _int("window", "Window size:", 7, 3, 9999, "Window size, in points, used by the smoothing filter."),
    BoolParam("centered", "Centered", default_value=True, visible_for=_shown_for("centered"),
              tooltip="Center the window on each point instead of trailing it."),
    _int("polyorder", "Polynomial order:", 2, 0, 10, "Polynomial order fit inside each Savitzky-Golay window."),
    _int("deriv", "Derivative order:", 0, 0, 5, "Order of the derivative to compute (0 = smoothed value only)."),
    _float("delta", "Delta:", 1.0, 1e-12, 1e12, 6, "Sample spacing used when computing derivatives."),
    _float("sigma", "Sigma:", 2.0, 0.001, 1e6, 4, "Standard deviation of the Gaussian kernel."),
    _float("sigma_x", "Sigma X:", 1.5, 0.001, 1e6, 4, "Standard deviation of the Gaussian kernel along X."),
    _float("sigma_y", "Sigma Y:", 1.5, 0.001, 1e6, 4, "Standard deviation of the Gaussian kernel along Y."),
    _float("sigma_z", "Sigma Z:", 1.5, 0.001, 1e6, 4, "Standard deviation of the Gaussian kernel along Z."),
    _float("truncate", "Gaussian truncate:", 4.0, 0.1, 50.0, 2,
           "Truncate the Gaussian kernel at this many standard deviations."),
    _int("kernel", "Kernel size:", 5, 3, 9999, "Size, in points, of the moving kernel window."),
    _int("kernel_x", "Kernel X:", 3, 3, 9999, "Kernel size along X."),
    _int("kernel_y", "Kernel Y:", 3, 3, 9999, "Kernel size along Y."),
    _int("kernel_z", "Kernel Z:", 3, 3, 9999, "Kernel size along Z."),
    _float("noise", "Noise estimate:", 0.0, 0.0, 1e12, 6,
           "Noise level estimate; leave at 'auto' to estimate from the data.", special_text="auto"),
    _float("spline_s", "Smoothing factor:", 10.0, 0.0, 1e18, 4, "Smoothing factor: higher values produce a smoother spline."),
    _int("spline_k", "Spline order:", 3, 1, 5, "Degree of the smoothing spline."),
    _int("kx", "Spline order X:", 3, 1, 5, "Spline degree along X."),
    _int("ky", "Spline order Y:", 3, 1, 5, "Spline degree along Y."),
    _float("lowess_frac", "LOWESS fraction:", 0.25, 0.01, 1.0, 3,
           "Fraction of points used for each local LOWESS regression."),
    _int("lowess_it", "LOWESS iterations:", 3, 0, 20, "Number of robustifying iterations for LOWESS."),
    _float("fft_cutoff_ratio", "FFT cutoff ratio:", 0.20, 0.001, 0.999, 3,
           "Fraction of frequency components kept by the FFT low-pass filter."),
    _float("butter_fs", "Sampling frequency:", 1.0, 1e-12, 1e12, 6, "Sampling frequency of the input data."),
    _float("butter_cutoff", "Cutoff frequency:", 0.20, 1e-12, 1e12, 6, "Cutoff frequency of the Butterworth filter."),
    _float("butter_high_cutoff", "High cutoff:", 0.40, 1e-12, 1e12, 6,
           "Upper cutoff frequency for bandpass/bandstop filters."),
    _int("butter_order", "Filter order:", 4, 1, 20, "Order of the Butterworth filter."),
    _choice("butter_type", "Filter type:", ("lowpass", "highpass", "bandpass", "bandstop"), "Butterworth filter type."),
    _choice("mode", "Boundary mode:", ("nearest", "reflect", "mirror", "constant", "wrap"),
            "Boundary handling mode applied at the edges of the data."),
    _choice("savgol_mode", "Edge mode:", ("interp", "mirror", "nearest", "constant", "wrap"),
            "Edge handling mode for the Savitzky-Golay filter."),
    _choice("wavelet", "Wavelet:", ("db2", "db4", "sym4", "coif1", "haar"), "Wavelet family used for wavelet denoising."),
    _int("wavelet_level", "Wavelet level:", 0, 0, 20, "Decomposition level; 0 lets the algorithm choose automatically."),
    _float("wavelet_threshold_factor", "Threshold factor:", 1.0, 0.01, 100.0, 3,
           "Multiplier applied to the estimated noise threshold."),
    _choice("wavelet_threshold_mode", "Threshold mode:", ("soft", "hard"),
            "Soft or hard thresholding of wavelet coefficients."),
    _float("whittaker_lambda", "Whittaker lambda:", 1000.0, 0.0, 1e12, 3,
           "Smoothness penalty for the Whittaker smoother (higher = smoother)."),
    _int("whittaker_order", "Difference order:", 2, 1, 5, "Order of the finite-difference penalty."),
    _float("hp_lambda", "HP lambda:", 1600.0, 0.0, 1e12, 3, "Smoothness penalty for the Hodrick-Prescott filter."),
    _float("tv_weight", "TV weight:", 0.15, 0.0, 1e6, 4, "Regularization weight for total-variation denoising."),
    _float("kalman_process_variance", "Process variance:", 1e-4, 1e-12, 1e6, 8,
           "Kalman filter: expected variance of the underlying process (model uncertainty)."),
    _float("kalman_measurement_variance", "Measurement variance:", 1e-2, 1e-12, 1e6, 8,
           "Kalman filter: expected variance of the measurement noise."),
    _float("kalman_initial_covariance", "Initial covariance:", 1.0, 1e-12, 1e6, 8,
           "Kalman filter: initial state covariance."),
    _choice("rbf_kernel", "RBF kernel:", ("thin_plate_spline", "linear", "cubic", "quintic", "multiquadric",
                                          "inverse_multiquadric", "inverse_quadratic", "gaussian"),
            "Radial basis function kernel used for interpolation-based smoothing."),
    _float("rbf_smoothing", "RBF smoothing:", 0.1, 0.0, 1e9, 4,
           "Smoothing factor for the RBF interpolant (0 = exact interpolation)."),
    _float("rbf_epsilon", "RBF epsilon:", 0.0, 0.0, 1e9, 4,
           "Shape parameter for the RBF kernel; leave at 'auto' to estimate it.", special_text="auto"),
    _int("rbf_neighbors", "RBF neighbors:", 0, 0, 1_000_000,
         "Number of nearest neighbors used per point; 'all' uses the full dataset.", special_text="all"),
)


# ---------------------------------------------------------------------------
# PySide6 dialog
# ---------------------------------------------------------------------------

class SeriesSmoothingDialog(DataOperationDialogBase):
    """Smoothing dialog that follows the shared operation-dialog pattern."""

    #: Where the result goes: the Charts tab's "Draw on".
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS
    Name: str = "Smoothing"
    Description = "Reduce noise"

    # Savitzky-Golay, the rolling filters and every window method treat the
    # series as an ordered sequence rather than as f(x), so points arriving
    # in query order are smoothed across a fold in the data - which returns
    # numbers that look like a result. Duplicate x is harmless here.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_MINIMUM_POINTS = 3

    #: Computed on a worker thread: see DataOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    Icon = """
    <path d="M4 13c1.7-4 3.4 4 5.1 0s3.4-4 5.1 0 3.4 4 5.8-1"/>
    <path d="M4 17c3.3-2.3 6.7-2.3 10 0 2 1.3 4 1.3 6 0"/>
    """
    Keywords: str = (
        "smussamento lisciatura noise rumore moving average media mobile savitzky-golay lowess loess "
        "filtro smooth"
    )
    Goals: list[str] = ["signal"]
    Category: str = "Signal Processing"
    Order: int = 80

    PARAMS = SMOOTHING_PARAMS

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        applied_callback: Callable[[], None] | None = None,
        table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesSmoothingDialog requires a repository instance.")

        self._repo: Any = repo
        self._figure_id = int(figure_id)
        self._applied_callback = applied_callback
        self._initial_table = table
        self._last_results: list[SmoothResult] = []

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            parent=parent,
            width=720,
            height=640,
        )
        self.model_combo.setVisible(False)
        self._refresh_methods()
        self._refresh_visibility()
        self.mark_results_stale()

    def build_model_selector(self) -> QWidget:
        self.dimension_combo = QComboBox(self)
        self.dimension_combo.addItems([DIM_1D, DIM_2D, DIM_3D])
        self.dimension_combo.setToolTip(_("Choose whether the source series is 1D, 2D, or 3D data."))

        self.method_combo = QComboBox(self)
        self.method_combo.setToolTip(_("Choose the smoothing/filtering method."))

        panel = self.model_form_card(
            "smoothingModelCard",
            [(_("Data type:"), self.dimension_combo), (_("Model:"), self.method_combo)],
        )
        self._doc_link.setToolTip(_("Open documentation for the selected method."))
        return panel

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(lambda *_args: self.mark_results_stale())
        self.series_selector.axis_changed.connect(lambda *_args: self.mark_results_stale())
        self.dimension_combo.currentIndexChanged.connect(self._refresh_methods)
        self.dimension_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.method_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.method_combo.currentIndexChanged.connect(self.mark_results_stale)

    def _series_choice_from_row(self, row: Any) -> SeriesChoice|None:
        name = row_value(row, "name", "series_name", "label", "title", default="Series")
        sql_query = row_value(row, "sql_query", "query", "sql", default="")
        if not sql_query:
            applogger.error("Selected series has no SQL query.")
            return None

        frame = self._repo.query_df(str(sql_query))
        if frame.empty:
            applogger.error("Selected series query returned no rows.")
            return None

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = str(roles.get("x") or "")
        y_col = str(roles.get("y") or "")
        z_col = str(roles.get("z") or "")
        values_col = str(roles.get("values") or roles.get("value") or "")

        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0] if columns else "x"
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else columns[1] if len(columns) > 1 else "y"
        if z_col and z_col not in columns:
            z_col = ""
        if values_col and values_col not in columns:
            values_col = ""

        x_values = self.numeric_x(frame[x_col], name)
        y_values = self.numeric_y(frame[y_col])
        z_values = None
        values = None
        if z_col:
            z_values = to_numbers(frame[z_col]).to_numpy(dtype=float)
        if values_col:
            values = to_numbers(frame[values_col]).to_numpy(dtype=float)

        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        z_values = (
            np.asarray(z_values, dtype=float).reshape(-1)
            if z_values is not None
            else None
        )
        values = (
            np.asarray(values, dtype=float).reshape(-1)
            if values is not None
            else None
        )

        # to_numeric(errors="coerce") turns any unparseable cell into NaN, so a
        # text column picked up as x arrives here as an all-NaN array rather
        # than as an error. That is exactly what the check is for.
        if z_values is None and values is None:
            x_values, y_values = self.prepare_input_xy(
                x_values, y_values, label=str(name)
            )
        else:
            # 3D: x, y, z and values are parallel, so reordering the first two
            # would silently pair each x with another point's z. Report the
            # problems, repair nothing.
            self.validate_input_xy(x_values, y_values, label=str(name))

        return SeriesChoice(
            name=str(name),
            x=x_values,
            y=y_values,
            z=z_values,
            values=values,
            source=row,
        )

    def _refresh_methods(self) -> None:
        """Rebuild method list from the registry for the selected dimension."""
        current_method = self.method_combo.currentText()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItems(models_for_dimension(self.dimension_combo.currentText()))

        old_index = self.method_combo.findText(current_method)
        self.method_combo.setCurrentIndex(old_index if old_index >= 0 else 0)
        self.method_combo.blockSignals(False)
        self._refresh_visibility()

    def parameter_context(self) -> Mapping[str, Any]:
        """The method chosen: each setting shows for the methods that use it (SMOOTHING_PARAMS)."""
        return {"method": self.method_combo.currentText()}

    def _refresh_visibility(self, *_ignored: Any) -> None:
        """The selected method's settings, and its documentation."""
        super()._refresh_visibility()
        self._update_description_link()

    def _update_description_link(self) -> None:
        """Update the inline documentation hyperlink for the current method."""
        method = self.method_combo.currentText()
        if not method:
            self._doc_link.clear()
            return

        spec = model_spec(method)
        if spec is not None:
            self.set_doc_link(spec.doc_title, spec.doc_url)

    def _params(self) -> dict[str, Any]:
        """The settings, as the smoothing functions take them: 0 is "auto" for the noise and the RBF epsilon."""
        params = self.parameter_values()
        params.pop("destination", None)  # the Charts tab's: where, not how
        for name in ("noise", "rbf_epsilon"):
            if params.get(name) == 0.0:
                params[name] = None
        params["replace_preview"] = True
        return params

    def prepare_job(self, **options: Any) -> SeriesJob | None:
        """Read the selected series and the controls; the smoothing itself runs in the job."""
        del options
        axis_name = self.current_axis_name()
        settings = (self.dimension_combo.currentText(), self.method_combo.currentText(), self._params())
        selected_rows = self.selected_series()
        if not selected_rows:
            return None

        inputs: list[tuple[str, Any]] = []
        errors: list[str] = []
        for row in selected_rows:
            try:
                series = self._series_choice_from_row(row)
                if series is None:
                    continue
                metadata = {
                    **dict(settings[2]),
                    "figure_id": self._figure_id,
                    "axis_name": axis_name,
                    "source_series_id": self._source_series_id(series),
                }
                inputs.append((self._series_display_name(row), (series, metadata)))
            except Exception as exc:  # noqa: BLE001 - collected per series, reported with the others
                errors.append(f"{self._series_display_name(row)}: {exc}")
        if not inputs and not errors:
            return None
        return SeriesJob(inputs, settings, self.compute_series, errors)

    def compute_series(
        self, name: str, data: tuple[SeriesChoice, dict[str, Any]], settings: tuple[str, str, dict[str, Any]]
    ) -> SmoothResult | None:
        del name
        series, metadata = data
        dimension, method, params = settings
        return self._smooth_one_series(series, dimension, method, params, metadata)

    def finish_job(self, job: Any, outcome: Any) -> list[SmoothResult]:
        """The smoothed series; a failed one is listed in a message, all failed is an error."""
        if not isinstance(outcome, SeriesOutcome):
            return list(super().finish_job(job, outcome))
        results = [result for _name, result in outcome.outcomes if result]
        errors = outcome.errors
        if errors and not results:
            applogger.error("\n".join(errors))
            return []
        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )
        return results

    def _smooth_one_series(
        self,
        series: SeriesChoice,
        dimension: str,
        method: str,
        params: Mapping[str, Any],
        metadata: dict[str, Any],
    ) -> SmoothResult|None:
        """Smooth one materialized series with the engine (app.analysis.smoothing)."""
        spec = model_spec(method)
        if spec is None:
            return None
        if spec.dimension != dimension:
            raise ValueError(
                f"Method '{method}' is registered for {spec.dimension}, "
                f"not {dimension}."
            )

        smoothed = smooth_series(
            _DIMENSION_NUMBER[dimension], method, params,
            x=series.x, y=series.y, z=series.z, values=series.values,
        )
        return SmoothResult(
            source_name=series.name,
            result_name=f"{series.name} - {method}",
            method=method,
            x=smoothed.x,
            y=smoothed.y,
            z=smoothed.z,
            values=smoothed.values,
            metadata=metadata,
        )

    def result_series_spec(self, axis_id: int, table_name: str, result: SmoothResult) -> ResultSeriesSpec:
        del axis_id
        if result.values is not None and result.z is not None:
            sql_query = f'SELECT x, y, z, value FROM "{table_name}" ORDER BY z, y, x'
            roles = {"x": "x", "y": "y", "z": "z", "values": "value"}
        elif result.z is not None:
            sql_query = f'SELECT x, y, z FROM "{table_name}" ORDER BY x, y'
            roles = {"x": "x", "y": "y", "z": "z"}
        else:
            sql_query = f'SELECT x, y FROM "{table_name}" ORDER BY x'
            roles = {"x": "x", "y": "y"}

        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=sql_query,
            roles=roles,
            style={
                **self.generated_style_filter,
                "source_name": result.source_name,
                "source_series_id": result.metadata.get("source_series_id"),
                "method": result.method,
                "linestyle": "-",
                "linewidth": 2.0,
                "marker": "",
            },
        )

    RESULT_TABLE_PREFIX = 'Smoothing'
    RESULT_TABLE_VARIANT = 'method'

    @staticmethod
    def _source_series_id(series: SeriesChoice) -> int | None:
        source = series.source
        if source is None:
            return None
        keys = source.keys() if hasattr(source, "keys") else []
        if "id" in keys:
            return int(source["id"])
        if isinstance(source, Mapping) and "id" in source:
            return int(source["id"])
        value = getattr(source, "id", None)
        return int(value) if value is not None else None

    def operation_succeeded(self, *, commit: bool) -> None:
        """Tell the owner window once the smoothing is applied."""
        if commit and self._applied_callback is not None:
            self._applied_callback()

    @staticmethod
    def results_to_dataframe(results: Sequence[SmoothResult]) -> pd.DataFrame:
        """Flatten smoothing results into a DataFrame for export/storage."""
        rows: list[dict[str, Any]] = []

        for result in results:
            if result.values is not None and result.z is not None:
                SeriesSmoothingDialog._append_3d_rows(rows, result)
            elif result.z is not None:
                SeriesSmoothingDialog._append_2d_rows(rows, result)
            else:
                for x_value, y_value in zip(result.x, result.y, strict=False):
                    rows.append(
                        {
                            "result_name": result.result_name,
                            "source_name": result.source_name,
                            "method": result.method,
                            "x": float(x_value),
                            "y": float(y_value),
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def _append_2d_rows(rows: list[dict[str, Any]], result: SmoothResult) -> None:
        z_values = np.asarray(result.z, dtype=float)
        x_values = np.asarray(result.x, dtype=float).reshape(-1)
        y_values = np.asarray(result.y, dtype=float).reshape(-1)

        if z_values.ndim == 2:
            for y_index, y_value in enumerate(y_values):
                for x_index, x_value in enumerate(x_values):
                    rows.append(
                        {
                            "result_name": result.result_name,
                            "source_name": result.source_name,
                            "method": result.method,
                            "x": float(x_value),
                            "y": float(y_value),
                            "z": float(z_values[y_index, x_index]),
                        }
                    )
            return

        for x_value, y_value, z_value in zip(
            x_values,
            y_values,
            z_values.reshape(-1),
            strict=False,
        ):
            rows.append(
                {
                    "result_name": result.result_name,
                    "source_name": result.source_name,
                    "method": result.method,
                    "x": float(x_value),
                    "y": float(y_value),
                    "z": float(z_value),
                }
            )

    @staticmethod
    def _append_3d_rows(rows: list[dict[str, Any]], result: SmoothResult) -> None:
        values = np.asarray(result.values, dtype=float)
        x_values = np.asarray(result.x, dtype=float).reshape(-1)
        y_values = np.asarray(result.y, dtype=float).reshape(-1)
        z_values = np.asarray(result.z, dtype=float).reshape(-1)

        if values.ndim == 3:
            for z_index, z_value in enumerate(z_values):
                for y_index, y_value in enumerate(y_values):
                    for x_index, x_value in enumerate(x_values):
                        rows.append(
                            {
                                "result_name": result.result_name,
                                "source_name": result.source_name,
                                "method": result.method,
                                "x": float(x_value),
                                "y": float(y_value),
                                "z": float(z_value),
                                "value": float(values[z_index, y_index, x_index]),
                            }
                        )
            return

        for x_value, y_value, z_value, scalar_value in zip(
            x_values,
            y_values,
            z_values,
            values.reshape(-1),
            strict=False,
        ):
            rows.append(
                {
                    "result_name": result.result_name,
                    "source_name": result.source_name,
                    "method": result.method,
                    "x": float(x_value),
                    "y": float(y_value),
                    "z": float(z_value),
                    "value": float(scalar_value),
                }
            )