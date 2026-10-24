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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from app.analysis.filtering import (
    AUTO_CUTOFF1,
    AUTO_CUTOFF2,
    FIR_WINDOWS,
    IIR_FAMILIES,
    analytic_signal,
    apply_detrend,
    apply_fir_filter,
    apply_iir_filter,
    resolve_cutoffs,
    two_cutoffs,
)
from app.plugin_api.data_operations.results import XYResult
from app.plugin_api.data_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.plugin_api.data_operations.dialog_base import (
    OperationModel,
    DataOperationDialogBase,
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


def _choices(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((label, value) for value, label in labels.items())


_IIR = {"model": (FILTER_IIR,)}
_FIR = {"model": (FILTER_FIR,)}
_WITH_RESPONSE = {"model": tuple(name for name, spec in FILTERS.items() if spec.has_response)}

#: The settings, each shown for the operations (and the families, the responses) it applies to.
FILTER_PARAMS = (
    *DataOperationDialogBase.sampling_rate_params(
        {"model": tuple(name for name, spec in FILTERS.items() if spec.needs_fs)}),
    ChoiceParam("family", "Family:", choices=_choices(IIR_FAMILY_LABELS), visible_for=_IIR,
                tooltip="The filter family. Butterworth is maximally flat; "
                        "Chebyshev/Elliptic trade ripple for a sharper roll-off."),
    ChoiceParam("response", "Response:", choices=_choices(RESPONSE_LABELS), visible_for=_WITH_RESPONSE,
                tooltip="Which band the filter keeps."),
    IntParam("order", "Order:", default_value=4, minimum=1, maximum=20, visible_for=_IIR,
             tooltip="Filter order. Higher is a sharper roll-off and more phase distortion before sosfiltfilt "
                     "cancels it."),
    IntParam("numtaps", "Taps:", default_value=101, minimum=3, maximum=20001, visible_for=_FIR,
             tooltip="Number of FIR coefficients. More taps sharpen the transition band at the cost of needing "
                     "more samples."),
    ChoiceParam("window", "Window:", choices=FIR_WINDOWS, visible_for=_FIR,
                tooltip="Taper applied to the FIR coefficients."),
    # 0 is "Auto": a fraction of fs, worked out when the filter runs. A fixed default (it was 1) is in the
    # units of x, and for a series dated in seconds fs is about 1e-5 - the default sat far above the
    # Nyquist frequency and Preview failed until the user found the box.
    FloatParam("cutoff1", "Cutoff:", default_value=0.0, minimum=0.0, maximum=1e12, decimals=6, step=0.001,
               special_text="Auto", visible_for=_WITH_RESPONSE,
               tooltip=f"Cutoff frequency (same units as fs). Auto is {AUTO_CUTOFF1:g} of the sampling frequency."),
    FloatParam("cutoff2", "Second cutoff:", default_value=0.0, minimum=0.0, maximum=1e12, decimals=6, step=0.001,
               special_text="Auto",
               visible_for={**_WITH_RESPONSE, "response": tuple(key for key in RESPONSE_LABELS if two_cutoffs(key))},
               tooltip=f"Second cutoff, for a bandpass/bandstop response. Auto is {AUTO_CUTOFF2:g} of the "
                       "sampling frequency."),
    FloatParam("ripple", "Ripple (dB):", default_value=1.0, minimum=0.001, maximum=20.0, decimals=3, step=0.1,
               visible_for={**_IIR, "family": tuple(key for key, terms in IIR_FAMILIES.items() if terms.ripple)},
               tooltip="Passband ripple, in dB (Chebyshev I / Elliptic)."),
    FloatParam("atten", "Attenuation (dB):", default_value=40.0, minimum=1.0, maximum=200.0, decimals=1, step=1.0,
               visible_for={**_IIR, "family": tuple(key for key, terms in IIR_FAMILIES.items() if terms.attenuation)},
               tooltip="Minimum stopband attenuation, in dB (Chebyshev II / Elliptic)."),
    ChoiceParam("analytic", "Output:", choices=_choices(ANALYTIC_LABELS), visible_for={"model": (FILTER_ANALYTIC,)},
                tooltip="What to derive from the analytic signal."),
    ChoiceParam("detrend", "Trend:", choices=_choices(DETREND_LABELS), visible_for={"model": (FILTER_DETREND,)},
                tooltip="Linear removes a best-fit line; constant removes only the mean."),
)


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
class SeriesFilterDialog(DataOperationDialogBase):
    """Filter, detrend or demodulate a series (scipy.signal)."""

    #: Where the result goes: the Charts tab's "Draw on".
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS

    Name: str = "Filtering"
    Description = "Filter, detrend or demodulate a series (scipy.signal)"

    MODELS = FILTERS
    MODEL_LABEL = "Operation:"
    MODEL_TOOLTIP = "Choose the operation."
    PARAMS = FILTER_PARAMS

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 8

    #: Computed on a worker thread: see DataOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    Icon = """
    <path d="M3 12h4l2-7 4 14 2-7h6"/>
    """
    Keywords: str = (
        "filtro filtraggio filter low-pass passa basso high-pass passa alto band-pass butterworth "
        "detrend trend demodulate scipy signal"
    )
    Goals: list[str] = ["signal"]
    Category: str = "Signal Processing"
    Order: int = 81

    WINDOW_SIZE = (820, 660)
    SELECTS_ALL_SERIES = True

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def series_settings(self) -> tuple[str, dict[str, Any]]:
        """The model and every setting, read here so the run needs no widget; the choices' words for the report."""
        params = self.parameter_values()
        for key, labels in (("family", IIR_FAMILY_LABELS), ("response", RESPONSE_LABELS),
                            ("analytic", ANALYTIC_LABELS), ("detrend", DETREND_LABELS)):
            params[f"{key}_text"] = labels.get(str(params[key]), str(params[key]))
        return self.current_model(), params

    def read_series(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray, float]:
        """x, y and the sampling rate (1 for a model that does not use one)."""
        x_values, y_values = self.series_xy(row, name)
        fs = self.sampling_rate(x_values, name) if FILTERS[self.current_model()].needs_fs else 1.0
        return x_values, y_values, fs

    def compute_series(
        self, name: str, data: tuple[np.ndarray, np.ndarray, float], settings: tuple[str, dict[str, Any]]
    ) -> FilterResult:
        model, params = settings
        x_values, y_values, fs = data
        y_out, meta = _apply_model(model, y_values, fs, params)
        if y_out.size != x_values.size:
            raise ValueError(f"the result has {y_out.size} points but the series has {x_values.size}")
        return FilterResult(
            source_name=name,
            result_name=f"{name} - {FILTERS[model].tag}",
            model=model,
            x=x_values,
            y=y_out,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def result_table_name(self, axis_id: int, result: FilterResult) -> str:
        return generated_table_name(
            f"Filter_axis{axis_id}_{result.source_name}_{FILTERS[result.model].tag}",
            fallback="Filter_Result",
        )

def _apply_model(model: str, y: np.ndarray, fs: float, p: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter *y* with *model* and the settings *p* (see SeriesFilterDialog.series_settings)."""
    if model in (FILTER_IIR, FILTER_FIR):
        response = p["response"]
        cutoff, cutoff2 = resolve_cutoffs(p["cutoff1"], p["cutoff2"], response, fs)
        shown_cutoff = cutoff if cutoff2 is None else f"{cutoff:g} - {cutoff2:g}"
    if model == FILTER_IIR:
        family = p["family"]
        y_out = apply_iir_filter(
            y, fs, family=family, response=response, order=p["order"],
            cutoff=cutoff, cutoff2=cutoff2,
            ripple=p["ripple"] if IIR_FAMILIES[family].ripple else None,
            atten=p["atten"] if IIR_FAMILIES[family].attenuation else None,
        )
        return y_out, {"family": p["family_text"], "response": p["response_text"], "order": p["order"],
                       "cutoff": shown_cutoff, "fs": fs}
    if model == FILTER_FIR:
        y_out = apply_fir_filter(
            y, fs, numtaps=p["numtaps"], window=p["window"], response=response,
            cutoff=cutoff, cutoff2=cutoff2,
        )
        return y_out, {"response": p["response_text"], "taps": p["numtaps"], "window": p["window"],
                       "cutoff": shown_cutoff, "fs": fs}
    if model == FILTER_ANALYTIC:
        return analytic_signal(y, fs, p["analytic"]), {"output": p["analytic_text"], "fs": fs}
    # FILTER_DETREND
    return apply_detrend(y, p["detrend"]), {"trend": p["detrend_text"]}
