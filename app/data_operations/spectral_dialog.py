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
    QFormLayout,
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
    METHOD_STFT,
    METHOD_WAVELET,
    METHOD_XCORR,
    MIN_PAIR_SAMPLES,
    WINDOWS,
    SpectralParams,
    Spectrum,
    estimate,
    estimate_pair,
)
from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
import html

from app.utils.messages import show_message
from app.utils import report_html
from app.plugin_api.data_operations.results import TableResult
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam
from app.plugin_api.data_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    DataOperationDialogBase,
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
    METHOD_STFT: SpectralMethod(
        doc_title="scipy.signal.spectrogram",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html",
        frequency=True,
        welch=True,
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


def _methods(keep: Any) -> dict[str, tuple[str, ...]]:
    """The methods *keep(name, method)* holds for: a row's visible_for."""
    return {"model": tuple(name for name, method in SPECTRAL_METHODS.items() if keep(name, method))}


_WELCH = _methods(lambda _name, method: method.welch)
_CORRELATION = _methods(lambda _name, method: not method.frequency)

#: The settings, each shown for the methods that use it.
SPECTRAL_PARAMS = (
    *DataOperationDialogBase.sampling_rate_params(_methods(lambda _name, method: method.frequency)),
    IntParam("nperseg", "Segment length:", default_value=256, minimum=8, maximum=1_048_576, visible_for=_WELCH,
             tooltip="Samples per Welch segment. Longer segments resolve finer frequency detail; shorter ones "
                     "average away more noise."),
    FloatParam("overlap", "Segment overlap:", default_value=0.5, minimum=0.0, maximum=0.95, decimals=2, step=0.05,
               visible_for=_WELCH, tooltip="Fraction of each segment shared with the next."),
    ChoiceParam("window", "Window:", choices=WINDOWS, visible_for=_WELCH,
                tooltip="Taper applied to each segment before the FFT."),
    FloatParam("sigma", "Damping (sigma):", default_value=0.0, minimum=0.0, maximum=1e6, decimals=6, step=0.1,
               visible_for={"model": (METHOD_LAPLACE,)},
               tooltip="Damping sigma of the Laplace variable s = sigma + i*omega. Zero reduces the transform to "
                       "the Fourier transform; increasing it weights early samples more and suppresses late ones, "
                       "which is what makes a decaying or unstable signal integrable."),
    FloatParam("wavelet_w0", "Morlet w0:", default_value=6.0, minimum=3.0, maximum=30.0, decimals=1, step=0.5,
               visible_for={"model": (METHOD_WAVELET,)},
               tooltip="Morlet central frequency w0. Higher values resolve frequency more finely at the cost of "
                       "time resolution; 6 is the usual compromise and the value at which the wavelet is "
                       "near-admissible."),
    IntParam("wavelet_scales", "Scales:", default_value=64, minimum=8, maximum=512,
             visible_for={"model": (METHOD_WAVELET,)},
             tooltip="How many scales to evaluate between the longest period the record supports and the Nyquist "
                     "frequency, spaced logarithmically."),
    ChoiceParam("detrend", "Detrend:", choices=DETREND_MODES, visible_for=_WELCH,
                tooltip="Remove a constant or linear trend from each segment first. A trend leaks into the lowest "
                        "frequency bins and hides everything near it."),
    ChoiceParam("scaling", "Scaling:", choices=("density", "spectrum"), visible_for={"model": (METHOD_PSD, METHOD_CSD)},
                tooltip="density: power per unit frequency. spectrum: power per segment."),
    # One-sided always for the STFT: a dominant frequency is a positive one.
    BoolParam("one_sided", "One-sided spectrum", default_value=True,
              visible_for=_methods(lambda name, method: method.welch and name != METHOD_STFT),
              tooltip="Fold the negative frequencies onto the positive ones. Correct for real-valued signals."),
    BoolParam("decibels", "Convert to decibels",
              visible_for={"model": (METHOD_PSD, METHOD_CSD, METHOD_MAGNITUDE, METHOD_LAPLACE, METHOD_WAVELET)},
              tooltip="Express the result as 10*log10 of the value."),
    IntParam("max_lags", "Max lags:", default_value=0, minimum=0, maximum=1_000_000, special_text="all lags",
             visible_for=_CORRELATION, tooltip="Largest lag to keep. 0 keeps every lag."),
    ChoiceParam("correlation_norm", "Normalisation:", choices=CORRELATION_NORMALISATIONS, visible_for=_CORRELATION,
                tooltip="unbiased divides by the overlap at each lag; biased divides by N; none returns the raw sum "
                        "of products."),
)


class SeriesSpectralDialog(DataOperationDialogBase):
    """Compute spectra and correlations and attach them to the chart."""

    MODELS = SPECTRAL_METHODS
    PARAMS = SPECTRAL_PARAMS
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
    Keywords: str = (
        "analisi spettrale fft fourier frequency frequenza spectrum spettro periodogram periodogramma "
        "welch power density densita spettrale"
    )
    Goals: list[str] = ["signal"]
    Category: str = "Signal Processing"
    Order: int = 82

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

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _spectral_params(self, fs: float) -> SpectralParams:
        """The estimate's settings, read off the form."""
        p = self.parameter_values()
        return SpectralParams(
            fs=fs, window=p["window"], nperseg=p["nperseg"], overlap=p["overlap"], detrend=p["detrend"],
            one_sided=p["one_sided"], scaling=p["scaling"], decibels=p["decibels"], sigma=p["sigma"],
            wavelet_w0=p["wavelet_w0"], wavelet_scales=p["wavelet_scales"],
            correlation_norm=p["correlation_norm"], max_lags=p["max_lags"],
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
            except Exception as exc:  # noqa: BLE001 - collected per series, reported with the others
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
        fs = self.sampling_rate(x_values, name)

        try:
            spectrum = estimate(method, y_values, self._spectral_params(fs))
        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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

        fs = self.sampling_rate(reference_x[:length], reference_name)
        pair_label = f"{reference_name} x {other_name}"

        try:
            spectrum = estimate_pair(
                method, reference_y, other_y, self._spectral_params(fs)
            )
        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        if y_label == "frequency":  # the dominant frequency over time
            y_label = "frequency [1/x]"

        self.label_result_axis(title=str(getattr(first, "model", "") or ""), x_label=x_label, y_label=y_label)

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
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
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

    def format_results(self, results: Sequence[SpectralResult]) -> str:
        """Return an HTML summary of the computed estimates."""
        if not results:
            return report_html.note(_("Select one or more source series."))

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
        return self.results_document(
            first.model,
            report_html.section(
                _("Estimates"),
                report_html.table(
                    [
                        _("Source"),
                        _("Method"),
                        _("Points"),
                        _("Peak ({x} &rarr; {y})").format(x=first.x_label, y=first.y_label),
                    ],
                    rows,
                    align=["left", "left", "right", "right"],
                ),
            ),
        )

