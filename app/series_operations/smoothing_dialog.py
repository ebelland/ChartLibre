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

import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
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
from app.widgets.axis_series_selector import AxisSeriesSelector

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesJob,
    SeriesOperationDialogBase,
    SeriesOutcome,
)
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.styles.style import (
    CardFrame,
    stdSizeAndlayout,
)
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
        doc_title="scikit-image total variation",
        doc_url="https://scikit-image.org/docs/stable/api/skimage.restoration.html",
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
        doc_title="scikit-image total variation",
        doc_url="https://scikit-image.org/docs/stable/api/skimage.restoration.html",
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
        doc_title="scikit-image total variation",
        doc_url="https://scikit-image.org/docs/stable/api/skimage.restoration.html",
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

# ---------------------------------------------------------------------------
# PySide6 dialog
# ---------------------------------------------------------------------------

class SeriesSmoothingDialog(SeriesOperationDialogBase):
    """Smoothing dialog that follows the shared operation-dialog pattern."""
    Name: str = "Smoothing"
    Description = "Reduce noise"

    # Savitzky-Golay, the rolling filters and every window method treat the
    # series as an ordered sequence rather than as f(x), so points arriving
    # in query order are smoothed across a fold in the data - which returns
    # numbers that look like a result. Duplicate x is harmless here.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_MINIMUM_POINTS = 3

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    Icon = """
    <path d="M4 13c1.7-4 3.4 4 5.1 0s3.4-4 5.1 0 3.4 4 5.8-1"/>
    <path d="M4 17c3.3-2.3 6.7-2.3 10 0 2 1.3 4 1.3 6 0"/>
    """
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
        self._field_rows: dict[str, tuple[QWidget, QWidget]] = {}

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Smoothing",
            parent=parent,
            width=720,
            height=640,
        )
        self.model_combo.setVisible(False)
        self._refresh_methods()
        self._refresh_visibility()
        self.mark_results_stale()

    def create_axis_series_selector(self) -> AxisSeriesSelector:
        return AxisSeriesSelector(self._repo, self._figure_id, self)

    def init_operation_widgets(self) -> None:
        self._create_controls()

    def build_model_selector(self) -> QWidget:
        panel = CardFrame(self, "smoothingModelCard")
        layout = panel.layout()

        self.dimension_combo = QComboBox(self)
        self.dimension_combo.addItems([DIM_1D, DIM_2D, DIM_3D])
        self.dimension_combo.setToolTip(_("Choose whether the source series is 1D, 2D, or 3D data."))

        self.method_combo = QComboBox(self)
        self.method_combo.setToolTip(_("Choose the smoothing/filtering method."))

        form_widget = QWidget(panel)
        form = QFormLayout(form_widget)
        stdSizeAndlayout(form)
        form.addRow(_("Data type:"), self.dimension_combo)
        form.addRow(_("Model:"), self.method_combo)

        self._doc_link = QLabel(self)
        self._doc_link.setOpenExternalLinks(True)
        self._doc_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._doc_link.setToolTip(_("Open documentation for the selected method."))
        form.addRow(_("Docs:"), self._doc_link)
        layout.addWidget(form_widget)

        return panel

    def build_parameter_selector(self) -> QWidget:
        settings_widget = CardFrame(self, "smoothingParamsCard")
        form_widget = QWidget(settings_widget)
        self.form = QFormLayout(form_widget)
        stdSizeAndlayout(self.form)
        self._add_parameter_rows()
        settings_widget.layout().addWidget(form_widget)

        scroll = QScrollArea(self)
        stdSizeAndlayout(scroll)
        scroll.setWidget(settings_widget)
        return scroll

    def build_results_pane(self) -> QWidget:
        return super().build_results_pane()

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(lambda *_args: self.mark_results_stale())
        self.series_selector.axis_changed.connect(lambda *_args: self.mark_results_stale())
        self.dimension_combo.currentIndexChanged.connect(self._refresh_methods)
        self.dimension_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.method_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.method_combo.currentIndexChanged.connect(self.mark_results_stale)
        self._doc_link.linkActivated.connect(self._open_description)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setDecimals(decimals)
        widget.setSingleStep(10 ** -min(decimals, 3))
        return widget

    def _create_controls(self) -> None:
        """Create all parameter widgets once; visibility is managed dynamically."""
        self.window_spin = self._spin(3, 9999, 7)
        self.centered_check = QCheckBox()
        self.centered_check.setChecked(True)
        self.polyorder_spin = self._spin(0, 10, 2)
        self.deriv_spin = self._spin(0, 5, 0)
        self.delta_spin = self._double_spin(1e-12, 1e12, 1.0, 6)

        self.sigma_spin = self._double_spin(0.001, 1e6, 2.0, 4)
        self.sigma_x_spin = self._double_spin(0.001, 1e6, 1.5, 4)
        self.sigma_y_spin = self._double_spin(0.001, 1e6, 1.5, 4)
        self.sigma_z_spin = self._double_spin(0.001, 1e6, 1.5, 4)
        self.truncate_spin = self._double_spin(0.1, 50.0, 4.0, 2)

        self.kernel_spin = self._spin(3, 9999, 5)
        self.kernel_x_spin = self._spin(3, 9999, 3)
        self.kernel_y_spin = self._spin(3, 9999, 3)
        self.kernel_z_spin = self._spin(3, 9999, 3)

        self.noise_spin = self._double_spin(0.0, 1e12, 0.0, 6)
        self.noise_spin.setSpecialValueText(_("auto"))

        self.spline_s_spin = self._double_spin(0.0, 1e18, 10.0, 4)
        self.spline_k_spin = self._spin(1, 5, 3)
        self.kx_spin = self._spin(1, 5, 3)
        self.ky_spin = self._spin(1, 5, 3)

        self.lowess_frac_spin = self._double_spin(0.01, 1.0, 0.25, 3)
        self.lowess_it_spin = self._spin(0, 20, 3)
        self.fft_cutoff_spin = self._double_spin(0.001, 0.999, 0.20, 3)

        self.butter_fs_spin = self._double_spin(1e-12, 1e12, 1.0, 6)
        self.butter_cutoff_spin = self._double_spin(1e-12, 1e12, 0.20, 6)
        self.butter_high_cutoff_spin = self._double_spin(1e-12, 1e12, 0.40, 6)
        self.butter_order_spin = self._spin(1, 20, 4)
        self.butter_type_combo = QComboBox()
        self.butter_type_combo.addItems(["lowpass", "highpass", "bandpass", "bandstop"])

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["nearest", "reflect", "mirror", "constant", "wrap"])
        self.savgol_mode_combo = QComboBox()
        self.savgol_mode_combo.addItems(["interp", "mirror", "nearest", "constant", "wrap"])

        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["db2", "db4", "sym4", "coif1", "haar"])
        self.wavelet_level_spin = self._spin(0, 20, 0)
        self.wavelet_threshold_factor_spin = self._double_spin(0.01, 100.0, 1.0, 3)
        self.wavelet_threshold_mode_combo = QComboBox()
        self.wavelet_threshold_mode_combo.addItems(["soft", "hard"])

        self.whittaker_lambda_spin = self._double_spin(0.0, 1e12, 1000.0, 3)
        self.whittaker_order_spin = self._spin(1, 5, 2)
        self.hp_lambda_spin = self._double_spin(0.0, 1e12, 1600.0, 3)
        self.tv_weight_spin = self._double_spin(0.0, 1e6, 0.15, 4)

        self.kalman_process_variance_spin = self._double_spin(1e-12, 1e6, 1e-4, 8)
        self.kalman_measurement_variance_spin = self._double_spin(1e-12, 1e6, 1e-2, 8)
        self.kalman_initial_covariance_spin = self._double_spin(1e-12, 1e6, 1.0, 8)

        self.rbf_kernel_combo = QComboBox()
        self.rbf_kernel_combo.addItems(
            [
                "thin_plate_spline",
                "linear",
                "cubic",
                "quintic",
                "multiquadric",
                "inverse_multiquadric",
                "inverse_quadratic",
                "gaussian",
            ]
        )
        self.rbf_smoothing_spin = self._double_spin(0.0, 1e9, 0.1, 4)
        self.rbf_epsilon_spin = self._double_spin(0.0, 1e9, 0.0, 4)
        self.rbf_epsilon_spin.setSpecialValueText(_("auto"))
        self.rbf_neighbors_spin = self._spin(0, 1_000_000, 0)
        self.rbf_neighbors_spin.setSpecialValueText(_("all"))

        self.preview_check = QCheckBox(_("Replace previous smoothing preview in chart"))
        self.preview_check.setChecked(True)
        self.preview_check.setToolTip(
            _("When enabled, applying this dialog again replaces the previously generated preview series.")
        )

    _PARAMETER_TOOLTIPS: dict[str, str] = {
        "window": "Window size, in points, used by the smoothing filter.",
        "centered": "Center the window on each point instead of trailing it.",
        "polyorder": "Polynomial order fit inside each Savitzky-Golay window.",
        "deriv": "Order of the derivative to compute (0 = smoothed value only).",
        "delta": "Sample spacing used when computing derivatives.",
        "sigma": "Standard deviation of the Gaussian kernel.",
        "sigma_x": "Standard deviation of the Gaussian kernel along X.",
        "sigma_y": "Standard deviation of the Gaussian kernel along Y.",
        "sigma_z": "Standard deviation of the Gaussian kernel along Z.",
        "truncate": "Truncate the Gaussian kernel at this many standard deviations.",
        "kernel": "Size, in points, of the moving kernel window.",
        "kernel_x": "Kernel size along X.",
        "kernel_y": "Kernel size along Y.",
        "kernel_z": "Kernel size along Z.",
        "noise": "Noise level estimate; leave at 'auto' to estimate from the data.",
        "spline_s": "Smoothing factor: higher values produce a smoother spline.",
        "spline_k": "Degree of the smoothing spline.",
        "kx": "Spline degree along X.",
        "ky": "Spline degree along Y.",
        "lowess_frac": "Fraction of points used for each local LOWESS regression.",
        "lowess_it": "Number of robustifying iterations for LOWESS.",
        "fft_cutoff_ratio": "Fraction of frequency components kept by the FFT low-pass filter.",
        "butter_fs": "Sampling frequency of the input data.",
        "butter_cutoff": "Cutoff frequency of the Butterworth filter.",
        "butter_high_cutoff": "Upper cutoff frequency for bandpass/bandstop filters.",
        "butter_order": "Order of the Butterworth filter.",
        "butter_type": "Butterworth filter type.",
        "mode": "Boundary handling mode applied at the edges of the data.",
        "savgol_mode": "Edge handling mode for the Savitzky-Golay filter.",
        "wavelet": "Wavelet family used for wavelet denoising.",
        "wavelet_level": "Decomposition level; 0 lets the algorithm choose automatically.",
        "wavelet_threshold_factor": "Multiplier applied to the estimated noise threshold.",
        "wavelet_threshold_mode": "Soft or hard thresholding of wavelet coefficients.",
        "whittaker_lambda": "Smoothness penalty for the Whittaker smoother (higher = smoother).",
        "whittaker_order": "Order of the finite-difference penalty.",
        "hp_lambda": "Smoothness penalty for the Hodrick-Prescott filter.",
        "tv_weight": "Regularization weight for total-variation denoising.",
        "kalman_process_variance": "Expected variance of the underlying process (model uncertainty).",
        "kalman_measurement_variance": "Expected variance of the measurement noise.",
        "kalman_initial_covariance": "Initial state covariance for the Kalman filter.",
        "rbf_kernel": "Radial basis function kernel used for interpolation-based smoothing.",
        "rbf_smoothing": "Smoothing factor for the RBF interpolant (0 = exact interpolation).",
        "rbf_epsilon": "Shape parameter for the RBF kernel; leave at 'auto' to estimate it.",
        "rbf_neighbors": "Number of nearest neighbors used per point; 'all' uses the full dataset.",
    }

    def _add_parameter_rows(self) -> None:
        """Add all parameter rows to the form; hidden rows stay in layout."""
        rows: tuple[tuple[str, str, QWidget], ...] = (
            ("window", "Window size:", self.window_spin),
            ("centered", "Centered:", self.centered_check),
            ("polyorder", "Polynomial order:", self.polyorder_spin),
            ("deriv", "Derivative order:", self.deriv_spin),
            ("delta", "Delta:", self.delta_spin),
            ("sigma", "Sigma:", self.sigma_spin),
            ("sigma_x", "Sigma X:", self.sigma_x_spin),
            ("sigma_y", "Sigma Y:", self.sigma_y_spin),
            ("sigma_z", "Sigma Z:", self.sigma_z_spin),
            ("truncate", "Gaussian truncate:", self.truncate_spin),
            ("kernel", "Kernel size:", self.kernel_spin),
            ("kernel_x", "Kernel X:", self.kernel_x_spin),
            ("kernel_y", "Kernel Y:", self.kernel_y_spin),
            ("kernel_z", "Kernel Z:", self.kernel_z_spin),
            ("noise", "Noise estimate:", self.noise_spin),
            ("spline_s", "Smoothing factor:", self.spline_s_spin),
            ("spline_k", "Spline order:", self.spline_k_spin),
            ("kx", "Spline order X:", self.kx_spin),
            ("ky", "Spline order Y:", self.ky_spin),
            ("lowess_frac", "LOWESS fraction:", self.lowess_frac_spin),
            ("lowess_it", "LOWESS iterations:", self.lowess_it_spin),
            ("fft_cutoff_ratio", "FFT cutoff ratio:", self.fft_cutoff_spin),
            ("butter_fs", "Sampling frequency:", self.butter_fs_spin),
            ("butter_cutoff", "Cutoff frequency:", self.butter_cutoff_spin),
            ("butter_high_cutoff", "High cutoff:", self.butter_high_cutoff_spin),
            ("butter_order", "Filter order:", self.butter_order_spin),
            ("butter_type", "Filter type:", self.butter_type_combo),
            ("mode", "Boundary mode:", self.mode_combo),
            ("savgol_mode", "Edge mode:", self.savgol_mode_combo),
            ("wavelet", "Wavelet:", self.wavelet_combo),
            ("wavelet_level", "Wavelet level:", self.wavelet_level_spin),
            (
                "wavelet_threshold_factor",
                "Threshold factor:",
                self.wavelet_threshold_factor_spin,
            ),
            (
                "wavelet_threshold_mode",
                "Threshold mode:",
                self.wavelet_threshold_mode_combo,
            ),
            ("whittaker_lambda", "Whittaker lambda:", self.whittaker_lambda_spin),
            ("whittaker_order", "Difference order:", self.whittaker_order_spin),
            ("hp_lambda", "HP lambda:", self.hp_lambda_spin),
            ("tv_weight", "TV weight:", self.tv_weight_spin),
            (
                "kalman_process_variance",
                "Kalman process variance:",
                self.kalman_process_variance_spin,
            ),
            (
                "kalman_measurement_variance",
                "Kalman measurement variance:",
                self.kalman_measurement_variance_spin,
            ),
            (
                "kalman_initial_covariance",
                "Kalman initial covariance:",
                self.kalman_initial_covariance_spin,
            ),
            ("rbf_kernel", "RBF kernel:", self.rbf_kernel_combo),
            ("rbf_smoothing", "RBF smoothing:", self.rbf_smoothing_spin),
            ("rbf_epsilon", "RBF epsilon:", self.rbf_epsilon_spin),
            ("rbf_neighbors", "RBF neighbors:", self.rbf_neighbors_spin),
        )

        for key, label, widget in rows:
            self._add_row(key, label, widget)

    def _add_row(self, key: str, label: str, widget: QWidget) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        stdSizeAndlayout(row_layout)
        row_layout.addWidget(widget)
        row_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label_widget = QLabel(label)
        tooltip = self._PARAMETER_TOOLTIPS.get(key)
        if tooltip:
            widget.setToolTip(tooltip)
            label_widget.setToolTip(tooltip)
        self.form.addRow(label_widget, row_widget)
        self._field_rows[key] = (label_widget, row_widget)

    def _current_axis_name(self) -> str:
        return self.series_selector.selected_axis_name()



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

    def _on_axis_changed(self, axis_name: str) -> None:
        del axis_name
        self.mark_results_stale()

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

    def _refresh_visibility(self) -> None:
        """Show only the settings required by the selected registry model."""
        method = self.method_combo.currentText()
        m=model_spec(method)
        if m is None:
            return
        visible = m.fields if method else frozenset()

        for key, widgets in self._field_rows.items():
            is_visible = key in visible
            label_widget, row_widget = widgets
            label_widget.setVisible(is_visible)
            row_widget.setVisible(is_visible)

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

    def _open_description(self, _link: str = "") -> None:
        """Open the documentation URL from the selected registry entry."""
        method = self.method_combo.currentText()
        if not method:
            return

        spec = model_spec(method)
        if spec is not None:
            try:
                webbrowser.open(spec.doc_url)
            except Exception:
                show_message(
                    self,
                    "series.open_docs_failed",
                    title=spec.doc_title,
                    url=spec.doc_url,
                )

    def _params(self) -> dict[str, Any]:
        """Collect current UI settings into a plain dict for metadata/reuse."""
        return {
            "window": self.window_spin.value(),
            "centered": self.centered_check.isChecked(),
            "polyorder": self.polyorder_spin.value(),
            "deriv": self.deriv_spin.value(),
            "delta": self.delta_spin.value(),
            "sigma": self.sigma_spin.value(),
            "sigma_x": self.sigma_x_spin.value(),
            "sigma_y": self.sigma_y_spin.value(),
            "sigma_z": self.sigma_z_spin.value(),
            "truncate": self.truncate_spin.value(),
            "kernel": self.kernel_spin.value(),
            "kernel_x": self.kernel_x_spin.value(),
            "kernel_y": self.kernel_y_spin.value(),
            "kernel_z": self.kernel_z_spin.value(),
            "noise": None if self.noise_spin.value() == 0.0 else self.noise_spin.value(),
            "spline_s": self.spline_s_spin.value(),
            "spline_k": self.spline_k_spin.value(),
            "kx": self.kx_spin.value(),
            "ky": self.ky_spin.value(),
            "lowess_frac": self.lowess_frac_spin.value(),
            "lowess_it": self.lowess_it_spin.value(),
            "fft_cutoff_ratio": self.fft_cutoff_spin.value(),
            "butter_fs": self.butter_fs_spin.value(),
            "butter_cutoff": self.butter_cutoff_spin.value(),
            "butter_high_cutoff": self.butter_high_cutoff_spin.value(),
            "butter_order": self.butter_order_spin.value(),
            "butter_type": self.butter_type_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "savgol_mode": self.savgol_mode_combo.currentText(),
            "wavelet": self.wavelet_combo.currentText(),
            "wavelet_level": self.wavelet_level_spin.value(),
            "wavelet_threshold_factor": self.wavelet_threshold_factor_spin.value(),
            "wavelet_threshold_mode": self.wavelet_threshold_mode_combo.currentText(),
            "whittaker_lambda": self.whittaker_lambda_spin.value(),
            "whittaker_order": self.whittaker_order_spin.value(),
            "hp_lambda": self.hp_lambda_spin.value(),
            "tv_weight": self.tv_weight_spin.value(),
            "kalman_process_variance": self.kalman_process_variance_spin.value(),
            "kalman_measurement_variance": self.kalman_measurement_variance_spin.value(),
            "kalman_initial_covariance": self.kalman_initial_covariance_spin.value(),
            "rbf_kernel": self.rbf_kernel_combo.currentText(),
            "rbf_smoothing": self.rbf_smoothing_spin.value(),
            "rbf_epsilon": (
                None if self.rbf_epsilon_spin.value() == 0.0
                else self.rbf_epsilon_spin.value()
            ),
            "rbf_neighbors": self.rbf_neighbors_spin.value(),
            "replace_preview": self.preview_check.isChecked(),
        }

    def prepare_job(self, **options: Any) -> SeriesJob | None:
        """Read the selected series and the controls; the smoothing itself runs in the job."""
        del options
        axis_name = self._current_axis_name()
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
            except Exception as exc:
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
                "generated_smoothing": True,
                "smoothing_dialog": "series_smoothing",
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

    def format_results(self, results: Sequence[SmoothResult]) -> str:
        if not results:
            return ""
        return f"Preview for {len(results)} smoothed series"

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