"""Frequency-selective filtering and analytic-signal work (scipy.signal).

The gap between the two scipy.signal-based operations that already exist:
**Spectral Analysis** estimates a spectrum but never changes the series
itself, and **Smoothing** only ever lowpasses (one Butterworth model among
several denoising ones). This dialog is the frequency-domain complement:

* **IIR filter** - ``iirfilter`` -> ``sosfiltfilt``, zero-phase. Butterworth,
  Chebyshev I/II, Bessel or Elliptic; lowpass, highpass, bandpass or
  bandstop.
* **FIR filter** - ``firwin`` -> ``filtfilt``, zero-phase, linear-phase
  design (no ringing on a step the way an IIR filter can have).
* **Analytic signal** - ``hilbert``: amplitude envelope, instantaneous
  phase, or instantaneous frequency. AM demodulation, phase-based work.
* **Detrend** - ``scipy.signal.detrend``: remove a linear or constant trend
  before any of the above, or on its own.

fs (the sampling frequency) is derived from the median spacing of the
selected series' x role, exactly as the Spectral Analysis dialog does it -
including the same warning when the spacing is not uniform enough to trust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)
from app.analysis.filtering import (
    AUTO_CUTOFF1,
    AUTO_CUTOFF2,
    FIR_WINDOWS,
    IIR_FAMILIES,
    IirTerms,
    analytic_signal,
    apply_detrend,
    apply_fir_filter,
    apply_iir_filter,
    resolve_cutoffs,
    two_cutoffs,
)
from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.results import XYResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.utils.i18n import _

_SCIPY_DOCS = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.{}.html"


@dataclass(frozen=True, slots=True, kw_only=True)
class FilterModel(OperationModel):
    """What one of the four operations is, and what it asks the user for."""

    #: The scipy.signal function behind it, which is also its docs page.
    function: str
    #: Short name for the result series and table: "IIR", "FIR", ...
    tag: str
    #: Reads a sampling rate fs, rather than working sample by sample.
    needs_fs: bool = False
    #: Has a response shape: lowpass, highpass, bandpass or bandstop.
    has_response: bool = False

    @property
    def doc(self) -> tuple[str, str]:
        """(title, url) for the Docs link."""
        return f"scipy.signal.{self.function}", _SCIPY_DOCS.format(self.function)


FILTER_IIR = "IIR filter"
FILTER_FIR = "FIR filter"
FILTER_ANALYTIC = "Analytic signal (Hilbert)"
FILTER_DETREND = "Detrend"

#: The operations, in the order the Operation combo lists them.
FILTERS: dict[str, FilterModel] = {
    FILTER_IIR: FilterModel(function="iirfilter", tag="IIR", needs_fs=True, has_response=True),
    FILTER_FIR: FilterModel(function="firwin", tag="FIR", needs_fs=True, has_response=True),
    FILTER_ANALYTIC: FilterModel(function="hilbert", tag="Analytic", needs_fs=True),
    FILTER_DETREND: FilterModel(function="detrend", tag="Detrend"),
}

#: The labels; the key is iirfilter's ``btype`` (see app.analysis.filtering).
RESPONSE_LABELS: dict[str, str] = {
    "lowpass": _("Lowpass"),
    "highpass": _("Highpass"),
    "bandpass": _("Bandpass"),
    "bandstop": _("Bandstop"),
}

#: The labels; the key is iirfilter's ``ftype``.
IIR_FAMILY_LABELS: dict[str, str] = {
    "butter": _("Butterworth"),
    "cheby1": _("Chebyshev I"),
    "cheby2": _("Chebyshev II"),
    "bessel": _("Bessel"),
    "ellip": _("Elliptic"),
}

#: What the analytic signal gives back; the key is analytic_signal's ``output``.
ANALYTIC_LABELS: dict[str, str] = {
    "envelope": _("Amplitude envelope"),
    "phase": _("Instantaneous phase"),
    "frequency": _("Instantaneous frequency"),
}

#: The key is scipy.signal.detrend's ``type``.
DETREND_LABELS: dict[str, str] = {
    "linear": _("Linear"),
    "constant": _("Constant"),
}


@dataclass(slots=True)
class FilterResult(XYResult):
    """One filtered/derived series for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    

# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
class SeriesFilterDialog(SeriesOperationDialogBase):
    """Filter, detrend or demodulate a series (scipy.signal)."""

    Name: str = "Filtering"
    Description = "Filter, detrend or demodulate a series (scipy.signal)"

    MODELS = FILTERS
    MODEL_LABEL = "Operation:"
    MODEL_TOOLTIP = "Choose the operation."

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 8

    RESULTS_ARE_HTML = False

    Icon = """
    <path d="M3 12h4l2-7 4 14 2-7h6"/>
    """

    def __init__(
        self, *, repo: SqliteRepo, figure_id: int, parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesFilterDialog requires a repository instance.")

        self._last_results: list[FilterResult] = []
        super().__init__(
            repo=repo, figure_id=figure_id, title="Filtering", parent=parent,
            width=820, height=660,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        self.create_sampling_rate_widgets()
        self._family_combo = QComboBox(self)
        self._response_combo = QComboBox(self)
        self._order_spin = QSpinBox(self)
        self._numtaps_spin = QSpinBox(self)
        self._window_combo = QComboBox(self)
        self._cutoff1_spin = QDoubleSpinBox(self)
        self._cutoff2_spin = QDoubleSpinBox(self)
        self._ripple_spin = QDoubleSpinBox(self)
        self._atten_spin = QDoubleSpinBox(self)
        self._analytic_combo = QComboBox(self)
        self._detrend_combo = QComboBox(self)

    def build_parameter_selector(self) -> QWidget:
        widget = QWidget(self)
        self._parameter_form = QFormLayout(widget)
        self._parameter_form.setContentsMargins(0, 0, 0, 0)
        self._parameter_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.add_sampling_rate_rows(self._parameter_form)

        for value, label in IIR_FAMILY_LABELS.items():
            self._family_combo.addItem(label, value)
        self._family_combo.setToolTip(_("The filter family. Butterworth is maximally flat; "
                                         "Chebyshev/Elliptic trade ripple for a sharper roll-off."))
        self._parameter_form.addRow(_("Family:"), self._family_combo)

        for value, label in RESPONSE_LABELS.items():
            self._response_combo.addItem(label, value)
        self._response_combo.setToolTip(_("Which band the filter keeps."))
        self._parameter_form.addRow(_("Response:"), self._response_combo)

        self._order_spin.setRange(1, 20)
        self._order_spin.setValue(4)
        self._order_spin.setToolTip(_("Filter order. Higher is a sharper roll-off "
                                       "and more phase distortion before sosfiltfilt cancels it."))
        self._parameter_form.addRow(_("Order:"), self._order_spin)

        self._numtaps_spin.setRange(3, 20001)
        self._numtaps_spin.setValue(101)
        self._numtaps_spin.setToolTip(_("Number of FIR coefficients. More taps sharpen the "
                                         "transition band at the cost of needing more samples."))
        self._parameter_form.addRow(_("Taps:"), self._numtaps_spin)

        self._window_combo.addItems(FIR_WINDOWS)
        self._window_combo.setToolTip(_("Taper applied to the FIR coefficients."))
        self._parameter_form.addRow(_("Window:"), self._window_combo)

        # 0 is "Auto": a fraction of fs, worked out when the filter runs.
        # A fixed default (it was 1) is in the units of x, and for a series
        # dated in seconds fs is about 1e-5 - the default sat far above the
        # Nyquist frequency and Preview failed until the user found the box.
        for spin, tip, auto in (
            (self._cutoff1_spin, _("Cutoff frequency (same units as fs)."), AUTO_CUTOFF1),
            (self._cutoff2_spin, _("Second cutoff, for a bandpass/bandstop response."), AUTO_CUTOFF2),
        ):
            spin.setRange(0.0, 1e12)
            spin.setDecimals(6)
            spin.setSpecialValueText(_("Auto"))
            spin.setValue(0.0)
            spin.setToolTip(
                tip + " " + _("Auto is {fraction:g} of the sampling frequency.").format(fraction=auto)
            )
        self._parameter_form.addRow(_("Cutoff:"), self._cutoff1_spin)
        self._parameter_form.addRow(_("Second cutoff:"), self._cutoff2_spin)

        self._ripple_spin.setRange(0.001, 20.0)
        self._ripple_spin.setDecimals(3)
        self._ripple_spin.setValue(1.0)
        self._ripple_spin.setToolTip(_("Passband ripple, in dB (Chebyshev I / Elliptic)."))
        self._parameter_form.addRow(_("Ripple (dB):"), self._ripple_spin)

        self._atten_spin.setRange(1.0, 200.0)
        self._atten_spin.setDecimals(1)
        self._atten_spin.setValue(40.0)
        self._atten_spin.setToolTip(_("Minimum stopband attenuation, in dB (Chebyshev II / Elliptic)."))
        self._parameter_form.addRow(_("Attenuation (dB):"), self._atten_spin)

        for value, label in ANALYTIC_LABELS.items():
            self._analytic_combo.addItem(label, value)
        self._analytic_combo.setToolTip(_("What to derive from the analytic signal."))
        self._parameter_form.addRow(_("Output:"), self._analytic_combo)

        for value, label in DETREND_LABELS.items():
            self._detrend_combo.addItem(label, value)
        self._detrend_combo.setToolTip(_("Linear removes a best-fit line; "
                                          "constant removes only the mean."))
        self._parameter_form.addRow(_("Trend:"), self._detrend_combo)

        return widget

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self._family_combo.currentIndexChanged.connect(self._refresh_visibility)
        self._response_combo.currentIndexChanged.connect(self._refresh_visibility)

        for combo in (
            self.model_combo, self._family_combo, self._response_combo,
            self._window_combo, self._analytic_combo, self._detrend_combo,
        ):
            combo.currentIndexChanged.connect(self.mark_results_stale)
        for check in (self._fs_auto_check,):
            check.toggled.connect(self.mark_results_stale)
        for spin in (
            self._fs_spin, self._order_spin, self._numtaps_spin,
            self._cutoff1_spin, self._cutoff2_spin, self._ripple_spin, self._atten_spin,
        ):
            spin.valueChanged.connect(self.mark_results_stale)


    def _refresh_visibility(self) -> None:
        model = self.current_model()
        spec = FILTERS[model]
        is_iir = model == FILTER_IIR
        is_fir = model == FILTER_FIR
        needs_response = spec.has_response
        needs_second_cutoff = needs_response and two_cutoffs(str(self._response_combo.currentData()))
        family = IIR_FAMILIES.get(str(self._family_combo.currentData()), IirTerms())

        self.set_row_visible(self._fs_auto_check, spec.needs_fs)
        self.set_row_visible(self._fs_spin, spec.needs_fs and not self._fs_auto_check.isChecked())
        self.set_row_visible(self._family_combo, is_iir)
        self.set_row_visible(self._response_combo, needs_response)
        self.set_row_visible(self._order_spin, is_iir)
        self.set_row_visible(self._numtaps_spin, is_fir)
        self.set_row_visible(self._window_combo, is_fir)
        self.set_row_visible(self._cutoff1_spin, needs_response)
        self.set_row_visible(self._cutoff2_spin, needs_second_cutoff)
        self.set_row_visible(self._ripple_spin, is_iir and family.ripple)
        self.set_row_visible(self._atten_spin, is_iir and family.attenuation)
        self.set_row_visible(self._analytic_combo, model == FILTER_ANALYTIC)
        self.set_row_visible(self._detrend_combo, model == FILTER_DETREND)


    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _cutoffs(self, response: str, fs: float) -> tuple[float, float | None]:
        """The cutoff(s) to filter with: the typed ones, or Auto's fractions of fs."""
        return resolve_cutoffs(
            self._cutoff1_spin.value(), self._cutoff2_spin.value(), response, fs
        )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _apply_model(self, model: str, y: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        if model == FILTER_IIR:
            family = str(self._family_combo.currentData())
            response = str(self._response_combo.currentData())
            cutoff, cutoff2 = self._cutoffs(response, fs)
            y_out = apply_iir_filter(
                y, fs, family=family, response=response,
                order=int(self._order_spin.value()),
                cutoff=cutoff, cutoff2=cutoff2,
                ripple=float(self._ripple_spin.value()) if IIR_FAMILIES[family].ripple else None,
                atten=float(self._atten_spin.value()) if IIR_FAMILIES[family].attenuation else None,
            )
            meta = {
                "family": self._family_combo.currentText(),
                "response": self._response_combo.currentText(),
                "order": int(self._order_spin.value()),
                "cutoff": cutoff if cutoff2 is None else f"{cutoff:g} - {cutoff2:g}",
                "fs": fs,
            }
            return y_out, meta

        if model == FILTER_FIR:
            response = str(self._response_combo.currentData())
            cutoff, cutoff2 = self._cutoffs(response, fs)
            y_out = apply_fir_filter(
                y, fs, numtaps=int(self._numtaps_spin.value()),
                window=self._window_combo.currentText(), response=response,
                cutoff=cutoff, cutoff2=cutoff2,
            )
            meta = {
                "response": self._response_combo.currentText(),
                "taps": int(self._numtaps_spin.value()),
                "window": self._window_combo.currentText(),
                "cutoff": cutoff if cutoff2 is None else f"{cutoff:g} - {cutoff2:g}",
                "fs": fs,
            }
            return y_out, meta

        if model == FILTER_ANALYTIC:
            output = str(self._analytic_combo.currentData())
            y_out = analytic_signal(y, fs, output)
            return y_out, {"output": self._analytic_combo.currentText(), "fs": fs}

        # FILTER_DETREND
        kind = str(self._detrend_combo.currentData())
        y_out = apply_detrend(y, kind)
        return y_out, {"trend": self._detrend_combo.currentText()}

    def compute_results(self) -> list[FilterResult]:
        model = self.current_model()
        results: list[FilterResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                fs = self.sampling_rate(x_values, name) if FILTERS[model].needs_fs else 1.0
                y_out, meta = self._apply_model(model, y_values, fs)
                if y_out.size != x_values.size:
                    raise ValueError(
                        f"the result has {y_out.size} points but the series has "
                        f"{x_values.size}"
                    )
                results.append(
                    FilterResult(
                        source_name=name,
                        result_name=f"{name} - {FILTERS[model].tag}",
                        model=model,
                        x=x_values,
                        y=y_out,
                        metadata=meta,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message)
        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def result_series_spec(
        self, axis_id: int, table_name: str, result: FilterResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_filter": True,
                "filter_dialog": "series_filter",
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    def result_table_name(self, axis_id: int, result: FilterResult) -> str:
        return generated_table_name(
            f"Filter_axis{axis_id}_{result.source_name}_{FILTERS[result.model].tag}",
            fallback="Filter_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Filtering"

    def format_results(self, results: Sequence[FilterResult]) -> str:
        if not results:
            return _("No results.")
        lines = []
        for result in results:
            details = ", ".join(f"{key}={value}" for key, value in result.metadata.items())
            lines.append(f"{result.source_name}: {result.model} ({details})")
        return "\n".join(lines)
