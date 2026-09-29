"""Dialog for spectral and correlation analysis of chart series.

Estimators, and why each is here:

* **PSD (Welch)** - how power is distributed over frequency for one signal.
* **CSD (Welch)** - the same for a *pair*, showing where two signals share
  power and with what phase.
* **Coherence** - the normalised version of the CSD: how linearly related two
  signals are per frequency, on a 0-1 scale that is comparable across data.
* **Magnitude / phase / angle spectrum** - the raw one-sided FFT, for when the
  signal is deterministic rather than a noise process and Welch's averaging
  would smear the very peaks being measured.
* **Autocorrelation / cross-correlation** - the time-domain view: at what lag
  does a signal repeat, or does one signal lead another.

Two-input estimators (CSD, coherence, cross-correlation) pair the **first
selected series** with each of the others. That rule is arbitrary but it has to
be *some* rule, and "first is the reference" is the one a user can predict.

Sampling frequency is derived from the x role when it is uniformly spaced,
because a frequency axis in samples-per-x is meaningless otherwise; the value
can be overridden. Results are written to normal tables and attached to the
axis as ordinary series, so they persist in the .dhub and re-render like any
other data.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)
from app.analysis.spectral import (
    CORRELATION_NORMALISATIONS,
    DETREND_MODES,
    METHOD_ACORR,
    METHOD_ANGLE,
    METHOD_COHERENCE,
    METHOD_CSD,
    METHOD_LAPLACE,
    METHOD_MAGNITUDE,
    METHOD_PHASE,
    METHOD_PSD,
    METHOD_WAVELET,
    METHOD_XCORR,
    MIN_PAIR_SAMPLES,
    WINDOWS,
    SpectralParams,
    Spectrum,
    estimate,
    estimate_pair,
    sampling_frequency,
)
from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
import html

from app.utils.messages import show_message
from app.utils import report_html
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.utils.i18n import _
from app.utils.coercion import to_numbers

# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------
# The method names, and the windows and modes, are the engine's: app.analysis.spectral.


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectralMethod(OperationModel):
    #: Reads a pair of signals rather than one.
    paired: bool = False
    #: Its output lives in the frequency domain.
    frequency: bool = False
    #: Uses Welch segmentation (nperseg / overlap / window).
    welch: bool = False


#: The models offered, in combo order.
SPECTRAL_METHODS: dict[str, SpectralMethod] = {
    METHOD_PSD: SpectralMethod(
        doc_title="scipy.signal.welch",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html",
        frequency=True,
        welch=True,
    ),
    METHOD_CSD: SpectralMethod(
        doc_title="scipy.signal.csd",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html",
        paired=True,
        frequency=True,
        welch=True,
    ),
    METHOD_COHERENCE: SpectralMethod(
        doc_title="scipy.signal.coherence",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html",
        paired=True,
        frequency=True,
        welch=True,
    ),
    METHOD_MAGNITUDE: SpectralMethod(
        doc_title="Magnitude spectrum",
        doc_url="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.magnitude_spectrum.html",
        frequency=True,
    ),
    METHOD_PHASE: SpectralMethod(
        doc_title="Phase spectrum",
        doc_url="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.phase_spectrum.html",
        frequency=True,
    ),
    METHOD_ANGLE: SpectralMethod(
        doc_title="Angle spectrum",
        doc_url="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.angle_spectrum.html",
        frequency=True,
    ),
    METHOD_LAPLACE: SpectralMethod(
        doc_title="Laplace transform",
        doc_url="https://en.wikipedia.org/wiki/Laplace_transform",
        frequency=True,
    ),
    METHOD_WAVELET: SpectralMethod(
        doc_title="Continuous wavelet transform",
        doc_url="https://en.wikipedia.org/wiki/Continuous_wavelet_transform",
        frequency=True,
    ),
    METHOD_ACORR: SpectralMethod(
        doc_title="Autocorrelation",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.correlate.html",
    ),
    METHOD_XCORR: SpectralMethod(
        doc_title="Cross-correlation",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.correlate.html",
        paired=True,
    ),
}

@dataclass(slots=True)
class SpectralResult(TableResult):
    """One spectral or correlation estimate, ready to save and plot."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        """Return the two-column frame written to the result table."""
        return pd.DataFrame({self.x_label: self.x, self.y_label: self.y})


class SeriesSpectralDialog(SeriesOperationDialogBase):
    """Compute spectra and correlations and attach them to the chart."""

    MODELS = SPECTRAL_METHODS
    MODEL_TOOLTIP = "Choose the spectral or correlation estimator."
    MODEL_LABEL = "Method:"

    # format_results builds a table; without this the pane would show the
    # markup as literal text.
    Name: str = "Spectral Analysis"
    Description = "Analyse frequencies"

    # An FFT maps sample index to frequency, so it assumes a constant step.
    # Uneven x does not fail - it returns a spectrum whose frequency axis is
    # meaningless, which is the worst of both worlds, hence the warning.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIFORM_X = True
    # Eight, matching _series_signal: a spectrum from fewer points has so few
    # frequency bins that the result says nothing.
    INPUT_MINIMUM_POINTS = 8

    Icon = """
    <path d="M4 18.5h16"/>
    <path d="M4.5 18V5"/>
    <path d="M7 16v-3"/>
    <path d="M10 16V8"/>
    <path d="M13 16v-6"/>
    <path d="M16 16V6"/>
    <path d="M19 16v-4"/>
    """
    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error(
                "SeriesSpectralDialog requires a repository instance.",
                show_dialog=True,
                raise_error=True,
            )

        self._last_results: list[SpectralResult] = []
        self._parameter_form: QFormLayout | None = None

        # The axis this dialog adds to the current figure for its results.
        # Created on the first Preview, kept across further previews, deleted
        # on Close unless Apply has confirmed it.
        self._result_axis_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Spectral Analysis",
            parent=parent,
            width=900,
            height=680,
        )
        self.setModal(True)
        self.series_selector.set_series_filter(self._has_query)
        self.series_selector.reload(select_all_series=False)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        """Create the controls before the base class builds the panels."""
        self._fs_auto_check = QCheckBox(_("Derive from the x role"), self)
        self._fs_spin = QDoubleSpinBox(self)
        self._nperseg_spin = QSpinBox(self)
        self._overlap_spin = QDoubleSpinBox(self)
        self._window_combo = QComboBox(self)
        self._detrend_combo = QComboBox(self)
        self._scaling_combo = QComboBox(self)
        self._onesided_check = QCheckBox(_("One-sided spectrum"), self)
        self._db_check = QCheckBox(_("Convert to decibels"), self)
        self._maxlags_spin = QSpinBox(self)
        self._sigma_spin = QDoubleSpinBox(self)
        self._wavelet_w0_spin = QDoubleSpinBox(self)
        self._wavelet_scales_spin = QSpinBox(self)
        self._corr_norm_combo = QComboBox(self)
        self._parameter_form = None

    def build_parameter_selector(self) -> QWidget:
        """Parameters, shown and hidden per method by _refresh_visibility."""
        widget = QWidget(self)
        self._parameter_form = QFormLayout(widget)
        self._parameter_form.setContentsMargins(0, 0, 0, 0)
        self._parameter_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._fs_auto_check.setChecked(True)
        self._fs_auto_check.setToolTip(
            _("Take the sampling frequency from the spacing of the x role.")
        )
        self._parameter_form.addRow(_("Sampling rate:"), self._fs_auto_check)

        self._fs_spin.setRange(1e-9, 1e12)
        self._fs_spin.setDecimals(6)
        self._fs_spin.setValue(1.0)
        self._fs_spin.setToolTip(_("Samples per unit of x, used when not derived."))
        self._parameter_form.addRow(_("fs:"), self._fs_spin)

        self._nperseg_spin.setRange(8, 1_048_576)
        self._nperseg_spin.setValue(256)
        self._nperseg_spin.setToolTip(
            _("Samples per Welch segment. Longer segments resolve finer frequency "
            "detail; shorter ones average away more noise.")
        )
        self._parameter_form.addRow(_("Segment length:"), self._nperseg_spin)

        self._overlap_spin.setRange(0.0, 0.95)
        self._overlap_spin.setDecimals(2)
        self._overlap_spin.setSingleStep(0.05)
        self._overlap_spin.setValue(0.5)
        self._overlap_spin.setToolTip(_("Fraction of each segment shared with the next."))
        self._parameter_form.addRow(_("Segment overlap:"), self._overlap_spin)

        self._window_combo.addItems(WINDOWS)
        self._window_combo.setToolTip(_("Taper applied to each segment before the FFT."))
        self._parameter_form.addRow(_("Window:"), self._window_combo)

        self._sigma_spin.setRange(0.0, 1e6)
        self._sigma_spin.setDecimals(6)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setValue(0.0)
        self._sigma_spin.setToolTip(
            _("Damping sigma of the Laplace variable s = sigma + i*omega. "
            "Zero reduces the transform to the Fourier transform; increasing it "
            "weights early samples more and suppresses late ones, which is what "
            "makes a decaying or unstable signal integrable.")
        )
        self._parameter_form.addRow(_("Damping (sigma):"), self._sigma_spin)

        self._wavelet_w0_spin.setRange(3.0, 30.0)
        self._wavelet_w0_spin.setDecimals(1)
        self._wavelet_w0_spin.setSingleStep(0.5)
        self._wavelet_w0_spin.setValue(6.0)
        self._wavelet_w0_spin.setToolTip(
            _("Morlet central frequency w0. Higher values resolve frequency more "
            "finely at the cost of time resolution; 6 is the usual compromise "
            "and the value at which the wavelet is near-admissible.")
        )
        self._parameter_form.addRow(_("Morlet w0:"), self._wavelet_w0_spin)

        self._wavelet_scales_spin.setRange(8, 512)
        self._wavelet_scales_spin.setValue(64)
        self._wavelet_scales_spin.setToolTip(
            _("How many scales to evaluate between the longest period the record "
            "supports and the Nyquist frequency, spaced logarithmically.")
        )
        self._parameter_form.addRow(_("Scales:"), self._wavelet_scales_spin)

        self._detrend_combo.addItems(DETREND_MODES)
        self._detrend_combo.setToolTip(
            _("Remove a constant or linear trend from each segment first. A trend "
            "leaks into the lowest frequency bins and hides everything near it.")
        )
        self._parameter_form.addRow(_("Detrend:"), self._detrend_combo)

        self._scaling_combo.addItems(("density", "spectrum"))
        self._scaling_combo.setToolTip(
            _("density: power per unit frequency. spectrum: power per segment.")
        )
        self._parameter_form.addRow(_("Scaling:"), self._scaling_combo)

        self._onesided_check.setChecked(True)
        self._onesided_check.setToolTip(
            _("Fold the negative frequencies onto the positive ones. Correct for "
            "real-valued signals.")
        )
        self._parameter_form.addRow("", self._onesided_check)

        self._db_check.setToolTip(_("Express the result as 10*log10 of the value."))
        self._parameter_form.addRow("", self._db_check)

        self._maxlags_spin.setRange(0, 1_000_000)
        self._maxlags_spin.setValue(0)
        self._maxlags_spin.setSpecialValueText(_("all lags"))
        self._maxlags_spin.setToolTip(_("Largest lag to keep. 0 keeps every lag."))
        self._parameter_form.addRow(_("Max lags:"), self._maxlags_spin)

        self._corr_norm_combo.addItems(CORRELATION_NORMALISATIONS)
        self._corr_norm_combo.setToolTip(
            _("unbiased divides by the overlap at each lag; biased divides by N; "
            "none returns the raw sum of products.")
        )
        self._parameter_form.addRow(_("Normalisation:"), self._corr_norm_combo)

        return widget

    def connect_operation_signals(self) -> None:
        """Mark the preview stale whenever a parameter changes."""
        super().connect_operation_signals()
        self._fs_auto_check.toggled.connect(self._refresh_visibility)
        self._fs_auto_check.toggled.connect(self.mark_results_stale)

        for widget in (
            self._fs_spin,
            self._overlap_spin,
        ):
            widget.valueChanged.connect(self.mark_results_stale)
        for widget in (self._nperseg_spin, self._maxlags_spin):
            widget.valueChanged.connect(self.mark_results_stale)
        for widget in (
            self._window_combo,
            self._detrend_combo,
            self._scaling_combo,
            self._corr_norm_combo,
        ):
            widget.currentIndexChanged.connect(self.mark_results_stale)
        for widget in (self._onesided_check, self._db_check):
            widget.toggled.connect(self.mark_results_stale)
        for widget in (
            self._sigma_spin,
            self._wavelet_w0_spin,
            self._wavelet_scales_spin,
        ):
            widget.valueChanged.connect(self.mark_results_stale)

    def _refresh_visibility(self) -> None:
        """Show only the parameters the selected method actually uses."""
        if self._parameter_form is None:
            return

        method = self.model_combo.currentText()
        is_welch = SPECTRAL_METHODS[method].welch
        is_frequency = SPECTRAL_METHODS[method].frequency
        is_correlation = not is_frequency

        self.set_row_visible(self._fs_auto_check, is_frequency)
        self.set_row_visible(self._fs_spin, is_frequency and not self._fs_auto_check.isChecked())
        self.set_row_visible(self._nperseg_spin, is_welch)
        self.set_row_visible(self._overlap_spin, is_welch)
        self.set_row_visible(self._window_combo, is_welch)
        self.set_row_visible(self._detrend_combo, is_welch)
        self.set_row_visible(self._scaling_combo, method in {METHOD_PSD, METHOD_CSD})
        self.set_row_visible(self._onesided_check, is_welch)
        self.set_row_visible(
            self._db_check,
            method in {
                METHOD_PSD,
                METHOD_CSD,
                METHOD_MAGNITUDE,
                METHOD_LAPLACE,
                METHOD_WAVELET,
            },
        )
        self.set_row_visible(self._maxlags_spin, is_correlation)
        self.set_row_visible(self._corr_norm_combo, is_correlation)
        self.set_row_visible(self._sigma_spin, method == METHOD_LAPLACE)
        self.set_row_visible(self._wavelet_w0_spin, method == METHOD_WAVELET)
        self.set_row_visible(self._wavelet_scales_spin, method == METHOD_WAVELET)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    @staticmethod
    def _has_query(row: Any) -> bool:
        """Only expose series backed by an SQL query."""
        return bool(row["sql_query"] != "")

    def _series_signal(self, row: Any) -> tuple[str, np.ndarray, np.ndarray]:
        """Return (name, x, y) for one selected series, finite values only."""
        roles = parse_roles(row_value(row, "roles"))
        frame = self._repo.query_df(str(row["sql_query"]))
        name = str(row["name"])

        y_column = str(roles.get("y", "y") or "y")
        if y_column not in frame.columns:
            y_column = "y"
        if y_column not in frame.columns:
            raise ValueError(f"series '{name}' has no y role")

        y_values = to_numbers(frame[y_column]).to_numpy(dtype=float)

        x_column = str(roles.get("x", "x") or "x")
        if x_column in frame.columns:
            # Seconds for a dated series, which is what a spectrum of one
            # needs: the sampling interval it reads off this is then in
            # seconds and its frequencies in Hz.
            x_values = self.numeric_x(frame[x_column], name)
        else:
            x_values = np.arange(y_values.size, dtype=float)

        # Drops non-finite points and sorts, as this did by hand, and also
        # reports the uneven spacing that makes a spectrum's frequency axis
        # meaningless - the failure this dialog is most exposed to.
        x_sorted, y_sorted = self.prepare_input_xy(x_values, y_values, label=name)
        return name, x_sorted, y_sorted

    def _sampling_frequency(self, x_values: np.ndarray, name: str) -> float:
        """Return fs in samples per unit of x: the typed one, or read off the x role."""
        if not self._fs_auto_check.isChecked():
            return float(self._fs_spin.value())

        fs, note = sampling_frequency(x_values)
        if note:
            applogger.warning(
                "Series '%s': %s.", name, note, show_dialog=False, raise_error=False,
            )
        return fs

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _spectral_params(self, fs: float) -> SpectralParams:
        """The estimate's settings, read off the controls."""
        return SpectralParams(
            fs=fs,
            window=self._window_combo.currentText(),
            nperseg=int(self._nperseg_spin.value()),
            overlap=float(self._overlap_spin.value()),
            detrend=self._detrend_combo.currentText(),
            one_sided=bool(self._onesided_check.isChecked()),
            scaling=self._scaling_combo.currentText(),
            decibels=bool(self._db_check.isChecked()),
            sigma=float(self._sigma_spin.value()),
            wavelet_w0=float(self._wavelet_w0_spin.value()),
            wavelet_scales=int(self._wavelet_scales_spin.value()),
            correlation_norm=self._corr_norm_combo.currentText(),
            max_lags=int(self._maxlags_spin.value()),
        )

    def compute_results(self) -> list[SpectralResult]:
        """Compute one result per selected series, or per pair when paired."""
        selected = self.selected_series()
        if not selected:
            return []

        method = self.model_combo.currentText()
        paired = SPECTRAL_METHODS[method].paired

        if paired and len(selected) < 2:
            show_message(
                self,
                "series.needs_two_series",
                title=self.operation_label,
                method=method,
            )
            return []

        signals: list[tuple[str, np.ndarray, np.ndarray]] = []
        errors: list[str] = []
        for row in selected:
            try:
                signals.append(self._series_signal(row))
            except Exception as exc:
                errors.append(str(exc))

        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )
        if not signals:
            return []

        results: list[SpectralResult] = []
        if paired:
            reference = signals[0]
            for other in signals[1:]:
                result = self._compute_pair(method, reference, other)
                if result is not None:
                    results.append(result)
        else:
            for entry in signals:
                result = self._compute_single(method, entry)
                if result is not None:
                    results.append(result)

        return results

    def _compute_single(
        self,
        method: str,
        entry: tuple[str, np.ndarray, np.ndarray],
    ) -> SpectralResult | None:
        """Compute a one-input estimate."""
        name, x_values, y_values = entry
        fs = self._sampling_frequency(x_values, name)

        try:
            spectrum = estimate(method, y_values, self._spectral_params(fs))
        except Exception as exc:
            applogger.exception("Spectral estimate failed for '%s'", name)
            show_message(
                self,
                "series.estimate_failed",
                title=self.operation_label,
                series=name,
                error=exc,
            )
            return None
        return self._result(method, name, spectrum)

    def _compute_pair(
        self,
        method: str,
        reference: tuple[str, np.ndarray, np.ndarray],
        other: tuple[str, np.ndarray, np.ndarray],
    ) -> SpectralResult | None:
        """Compute a two-input estimate against the reference series."""
        reference_name, reference_x, reference_y = reference
        other_name, _other_x, other_y = other

        length = min(reference_y.size, other_y.size)
        if length < MIN_PAIR_SAMPLES:
            applogger.warning(
                "Pair '%s' / '%s' skipped: fewer than %d shared samples.",
                reference_name,
                other_name,
                MIN_PAIR_SAMPLES,
            )
            return None

        fs = self._sampling_frequency(reference_x[:length], reference_name)
        pair_label = f"{reference_name} x {other_name}"

        try:
            spectrum = estimate_pair(
                method, reference_y, other_y, self._spectral_params(fs)
            )
        except Exception as exc:
            applogger.exception("Spectral estimate failed for '%s'", pair_label)
            show_message(
                self,
                "series.estimate_failed",
                title=self.operation_label,
                series=pair_label,
                error=exc,
            )
            return None
        return self._result(method, pair_label, spectrum)

    @staticmethod
    def _result(method: str, source_name: str, spectrum: Spectrum) -> SpectralResult:
        """Wrap an engine estimate in a result, named the way it always was.

        A frequency-domain estimate is "<source> - <method>"; a correlation
        is "<source> - Autocorrelation" / "- Cross-correlation".
        """
        suffix = {METHOD_ACORR: "Autocorrelation", METHOD_XCORR: "Cross-correlation"}.get(method, method)
        return SpectralResult(
            source_name=source_name,
            result_name=f"{source_name} - {suffix}",
            model=method,
            x=spectrum.x,
            y=spectrum.y,
            x_label=spectrum.x_label,
            y_label=spectrum.y_label,
            metadata=dict(spectrum.details),
        )

    # ------------------------------------------------------------------
    # Base-class hooks
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # The results get a chart of their own
    # ------------------------------------------------------------------
    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """Return an axis of this dialog's own, creating it once.

        A spectrum does not belong on the axis its input came from: the x axis
        stops being time and becomes frequency (or lag), so the two cannot
        share a scale, a label or a grid.

        A new axis on the *same figure*, not a new figure: the spectrum is a
        second view of the data in this chart, and a separate tab would put it
        where nobody compares it with the original.

        The axis is created on the first Preview and reused afterwards, so
        adjusting nperseg repeatedly does not leave a trail of empty axes.  If
        the dialog is closed without Apply, ``discard_operation_artifacts``
        removes it again.
        """
        del selected_axis_id

        if self._result_axis_id is None:
            self._result_axis_id = self.create_result_axis(
                chart_type="Scatter Plot",
                title=self.model_combo.currentText(),
                options={"grid": True, "linestyle": "-", "marker": ""},
            )

        self._label_result_axis(results)
        return self._result_axis_id

    def _label_result_axis(self, results: Sequence[Any]) -> None:
        """Name the axes after what the current estimate actually produced.

        The labels come from the results rather than from the method, because
        the same method yields different units depending on the options: a PSD
        is power or dB, a correlation is a lag axis rather than a frequency
        one.
        """
        if self._result_axis_id is None or not results:
            return

        first = results[0]
        x_label = str(getattr(first, "x_label", "") or "")
        y_label = str(getattr(first, "y_label", "") or "")
        if x_label == "frequency":
            x_label = "frequency [1/x]"

        try:
            self._repo.update_axis_descriptor(
                axis_id=self._result_axis_id,
                title=str(getattr(first, "model", "") or ""),
                x_label=x_label,
                y_label=y_label,
            )
        except Exception:
            applogger.exception("Failed to label the spectral result axis")

    def discard_operation_artifacts(self) -> None:
        """Delete the axis this dialog created, when Apply never happened.

        Closing without applying must leave the chart exactly as it was, and
        the axis is not covered by the preview savepoint: creating it commits.
        """
        if self._applied or self._result_axis_id is None:
            return

        axis_id = self._result_axis_id
        self._result_axis_id = None
        try:
            self._repo.delete_axis(axis_id)
            applogger.info("Discarded the unapplied spectral axis %s.", axis_id)
        except Exception:
            applogger.exception("Failed to discard spectral axis %s", axis_id)

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: SpectralResult,
    ) -> ResultSeriesSpec:
        """Attach the result as an ordinary x/y line series."""
        del axis_id
        quoted = f'"{table_name}"'
        style = dict(self.generated_style_filter)
        style.update(
            {
                "linestyle": "-",
                "marker": "",
                "label": result.result_name,
                "spectral_model": result.model,
            }
        )
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=(
                f'SELECT "{result.x_label}" AS x, "{result.y_label}" AS y FROM {quoted}'
            ),
            roles={"x": "x", "y": "y"},
            style=style,
        )

    #: format_results below returns a report built with report_html, and
    #: saying so is what stops the base escaping it. Without this the chart's
    #: notes pane showed the markup as text - "<html><body style=..." - because
    #: results_report_html ran it through plain_to_html.
    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[SpectralResult]) -> str:
        """Return an HTML summary of the computed estimates."""
        if not results:
            return report_html.note("Select one or more source series.")

        rows = []
        for result in results:
            finite = result.y[np.isfinite(result.y)]
            peak = ""
            if finite.size and result.x.size == result.y.size:
                peak_index = int(np.nanargmax(np.abs(result.y)))
                peak = (
                    f"{report_html.format_number(result.x[peak_index])}"
                    f" &rarr; {report_html.format_number(result.y[peak_index])}"
                )
            rows.append(
                (
                    html.escape(result.source_name),
                    html.escape(result.model),
                    str(result.x.size),
                    peak,
                )
            )

        first = results[0]
        return report_html.document(
            "Spectral analysis",
            first.model,
            report_html.section(
                _("Estimates"),
                report_html.table(
                    [
                        "Source",
                        "Method",
                        "Points",
                        f"Peak ({first.x_label} &rarr; {first.y_label})",
                    ],
                    rows,
                    align=["left", "left", "right", "right"],
                ),
            ),
        )

