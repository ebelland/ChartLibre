"""Spectral analysis: the arithmetic behind the Spectral Analysis operation.

Ten estimates of one signal or a pair of them - Welch power and cross
spectral density, coherence, the plain FFT's magnitude and phase, the
Laplace transform along a vertical line of the s-plane, a Morlet wavelet
power spectrum, and auto- and cross-correlation. Plain arrays and a
:class:`SpectralParams` in, a :class:`Spectrum` out: no Qt, no repository,
so the same code runs behind the dialog, in a test or on a worker thread
(todo R-01).

A problem with the data or the method is a ``ValueError``. Nothing is
logged here; the caller decides how to tell the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

METHOD_PSD = "Power spectral density (Welch)"
METHOD_CSD = "Cross spectral density (Welch)"
METHOD_COHERENCE = "Coherence"
METHOD_MAGNITUDE = "Magnitude spectrum"
METHOD_PHASE = "Phase spectrum (unwrapped)"
METHOD_ANGLE = "Angle spectrum (wrapped)"
METHOD_ACORR = "Autocorrelation"
METHOD_XCORR = "Cross-correlation"
METHOD_LAPLACE = "Laplace transform (damped FFT)"
METHOD_WAVELET = "Wavelet power spectrum (Morlet)"

WINDOWS: tuple[str, ...] = (
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "flattop",
    "boxcar",
)

DETREND_MODES: tuple[str, ...] = ("constant", "linear", "none")

CORRELATION_NORMALISATIONS: tuple[str, ...] = ("unbiased", "biased", "none")

#: Fewest samples two signals must share for a paired estimate.
MIN_PAIR_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class SpectralParams:
    """Everything an estimate reads besides the signal itself."""

    #: Samples per unit of x.
    fs: float = 1.0
    window: str = "hann"
    #: Welch segment length; clamped to the data (and to at least 8).
    nperseg: int = 256
    #: Fraction of a segment shared with the next, 0..1.
    overlap: float = 0.5
    detrend: str = "constant"
    one_sided: bool = True
    scaling: str = "density"
    #: Return 10*log10 of a power/magnitude estimate.
    decibels: bool = False
    #: Damping of the Laplace transform; 0 is the Fourier magnitude.
    sigma: float = 0.0
    wavelet_w0: float = 6.0
    wavelet_scales: int = 64
    correlation_norm: str = "unbiased"
    #: Keep lags up to this many samples either way; 0 keeps them all.
    max_lags: int = 0


@dataclass(frozen=True, slots=True)
class Spectrum:
    """An estimate: values against frequency (or lag), with the unit named."""

    x: np.ndarray
    y: np.ndarray
    #: "frequency" or "lag".
    x_label: str
    #: The unit: "power", "dB", "magnitude", "radians", "coherence", "correlation".
    y_label: str
    #: What the estimate was made with, beyond fs (the number of points...).
    details: dict[str, Any]


def sampling_frequency(x_values: np.ndarray) -> tuple[float, str]:
    """Return (fs, note) - samples per unit of x read off the median spacing.

    *note* is empty when there is nothing to say, else why fs is doubtful:
    a spectrum of unevenly sampled data is not defined, so a non-uniform x
    is reported rather than silently averaged - the numbers would look fine
    and mean nothing.
    """
    if x_values.size < 2:
        return 1.0, ""

    spacing = np.diff(x_values)
    median_spacing = float(np.median(spacing))
    if median_spacing <= 0.0:
        return 1.0, "the x role is not increasing; assuming fs = 1"

    deviation = float(np.max(np.abs(spacing - median_spacing)) / median_spacing)
    note = ""
    if deviation > 0.01:
        note = (
            f"the series is not uniformly sampled (spacing varies by "
            f"{deviation * 100.0:.1f} %); the frequency axis is approximate"
        )
    return 1.0 / median_spacing, note


def welch_kwargs(params: SpectralParams, sample_count: int) -> dict[str, Any]:
    """Shared Welch parameters, with nperseg clamped to the data length."""
    nperseg = min(int(params.nperseg), sample_count)
    nperseg = max(8, nperseg)
    return {
        "fs": params.fs,
        "window": params.window,
        "nperseg": nperseg,
        "noverlap": int(nperseg * float(params.overlap)),
        "detrend": False if params.detrend == "none" else params.detrend,
        "return_onesided": bool(params.one_sided),
    }


def to_decibels(values: np.ndarray) -> np.ndarray:
    """Return 10*log10(values), with zeros floored to the smallest positive.

    Why floor rather than drop: a zero bin is a real measurement and
    removing it would shift every later point on the frequency axis.
    """
    positive = values[values > 0.0]
    floor = float(positive.min()) if positive.size else 1e-20
    return 10.0 * np.log10(np.maximum(values, floor))


def one_sided_fft(y_values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, complex spectrum) for the positive half."""
    spectrum = np.fft.rfft(y_values - float(np.mean(y_values)))
    frequencies = np.fft.rfftfreq(y_values.size, d=1.0 / fs)
    return frequencies, spectrum


def laplace_spectrum(
    y_values: np.ndarray, fs: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, |F(sigma + i*omega)|) along a vertical line in s.

    The one-sided Laplace transform of a sampled signal is

        F(s) = dt * sum_k y_k * exp(-s * t_k),   s = sigma + i*omega

    and splitting the exponential turns that into the Fourier transform of
    a damped signal::

        F(sigma + i*omega) = dt * FFT[ y_k * exp(-sigma * t_k) ]

    so no new machinery is needed - one multiply before the FFT already in
    use.  What it buys is the part of the s-plane the Fourier transform
    cannot reach: a growing or non-decaying signal has no Fourier transform,
    but with sigma large enough the damped signal does, which is the whole
    reason the Laplace transform exists.

    sigma = 0 is exactly the Fourier magnitude spectrum, which makes the
    control easy to understand: turn it up and watch the transform become
    defined.
    """
    raw = np.asarray(y_values, dtype=float)
    times = np.arange(raw.size, dtype=float) / float(fs)

    # exp(-sigma * t) underflows to zero over a long record; that is
    # arithmetically right - those samples contribute nothing - but it is
    # worth not letting it produce a NaN through 0 * inf.
    damping = np.exp(-float(sigma) * times)
    damped = np.nan_to_num(raw * damping, nan=0.0, posinf=0.0, neginf=0.0)

    # The mean is removed *after* damping, not before.  Removing it first
    # subtracts the mean of the undamped signal, which for a growing signal
    # is enormous and leaves a ramp that swamps every real peak - a sine
    # multiplied by exp(1.5t), damped by exp(-1.5t), came back peaked at
    # DC instead of at its own frequency.  Damping first and centring the
    # result leaves exactly the sine, which is the point of choosing that
    # sigma.
    damped = damped - float(np.mean(damped))

    spectrum = np.fft.rfft(damped) / float(fs)
    frequencies = np.fft.rfftfreq(damped.size, d=1.0 / fs)
    return frequencies, np.abs(spectrum)


def wavelet_power(
    y_values: np.ndarray, fs: float, w0: float, n_scales: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, time-averaged power) of a Morlet CWT.

    A scalogram is two-dimensional and the operation produces (x, y)
    series, so what is returned is the *global* wavelet spectrum: the power
    at each scale averaged over time.  That is the standard summary of a
    scalogram and is directly comparable with a Fourier power spectrum -
    with the difference that it is computed from a basis localised in time,
    so a frequency present in only part of the record still shows up
    without the leakage a single long FFT would spread around it.

    Computed by convolution in the Fourier domain, which is both the fast
    way and the only way that stays exact for a wavelet defined
    analytically in frequency::

        psi_hat(s*omega) = pi^-1/4 * H(omega) * exp(-(s*omega - w0)^2 / 2)

    ``H`` being the Heaviside step: the Morlet wavelet is analytic, so it
    has no negative-frequency content.  No PyWavelets dependency for one
    wavelet whose transform is four lines of numpy.
    """
    centred = np.asarray(y_values, dtype=float) - float(np.mean(y_values))
    n = centred.size

    # From the longest period the record can support to the Nyquist limit.
    # Anything outside that is not measurable from this data.
    lowest = max(float(fs) / float(n), 1e-12)
    highest = float(fs) / 2.0
    frequencies = np.logspace(np.log10(lowest), np.log10(highest), int(n_scales))

    # Morlet: the scale that responds to frequency f is w0 / (2*pi*f).
    scales = float(w0) / (2.0 * np.pi * frequencies)

    transformed = np.fft.fft(centred)
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / float(fs))

    power = np.empty(frequencies.size, dtype=float)
    for index, scale in enumerate(scales):
        scaled = scale * omega
        wavelet = (np.pi ** -0.25) * np.exp(-0.5 * (scaled - float(w0)) ** 2)
        wavelet[omega <= 0.0] = 0.0  # analytic: no negative frequencies
        # sqrt(scale) keeps power comparable across scales; without it the
        # spectrum slopes purely because wide wavelets integrate more.
        coefficients = np.fft.ifft(transformed * wavelet) * np.sqrt(scale)
        power[index] = float(np.mean(np.abs(coefficients) ** 2))

    return frequencies, power


def correlate(
    first: np.ndarray,
    second: np.ndarray,
    *,
    normalisation: str = "unbiased",
    max_lags: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lags, correlation) for two mean-removed signals."""
    length = min(first.size, second.size)
    a = first[:length] - float(np.mean(first[:length]))
    b = second[:length] - float(np.mean(second[:length]))

    raw = np.correlate(a, b, mode="full")
    lags = np.arange(-length + 1, length)

    if normalisation == "biased":
        raw = raw / float(length)
    elif normalisation == "unbiased":
        # Each lag averages a different number of overlapping samples;
        # dividing by that count removes the artificial taper towards the
        # extreme lags.
        raw = raw / (length - np.abs(lags))

    if max_lags > 0:
        keep = np.abs(lags) <= max_lags
        lags = lags[keep]
        raw = raw[keep]

    return lags.astype(float), raw


def _frequency_spectrum(frequencies: np.ndarray, values: np.ndarray, unit: str) -> Spectrum:
    return Spectrum(
        x=np.asarray(frequencies, dtype=float),
        y=np.asarray(values, dtype=float),
        x_label="frequency",
        y_label=unit,
        details={"points": int(np.size(frequencies))},
    )


def estimate(method: str, y_values: np.ndarray, params: SpectralParams) -> Spectrum:
    """Estimate the spectrum (or autocorrelation) of one signal."""
    if method == METHOD_PSD:
        frequencies, power = scipy_signal.welch(
            y_values, scaling=params.scaling, **welch_kwargs(params, y_values.size)
        )
        if params.decibels:
            return _frequency_spectrum(frequencies, to_decibels(power), "dB")
        return _frequency_spectrum(frequencies, power, "power")

    if method in {METHOD_MAGNITUDE, METHOD_PHASE, METHOD_ANGLE}:
        frequencies, spectrum = one_sided_fft(y_values, params.fs)
        if method == METHOD_PHASE:
            return _frequency_spectrum(frequencies, np.unwrap(np.angle(spectrum)), "radians")
        if method == METHOD_ANGLE:
            return _frequency_spectrum(frequencies, np.angle(spectrum), "radians")
        magnitude = np.abs(spectrum)
        if params.decibels:
            return _frequency_spectrum(frequencies, to_decibels(magnitude), "dB")
        return _frequency_spectrum(frequencies, magnitude, "magnitude")

    if method == METHOD_LAPLACE:
        frequencies, magnitude = laplace_spectrum(y_values, params.fs, params.sigma)
        if params.decibels:
            return _frequency_spectrum(frequencies, to_decibels(magnitude), "dB")
        return _frequency_spectrum(frequencies, magnitude, "magnitude")

    if method == METHOD_WAVELET:
        frequencies, power = wavelet_power(
            y_values, params.fs, params.wavelet_w0, int(params.wavelet_scales)
        )
        if params.decibels:
            return _frequency_spectrum(frequencies, to_decibels(power), "dB")
        return _frequency_spectrum(frequencies, power, "power")

    if method == METHOD_ACORR:
        lags, correlation = correlate(
            y_values, y_values,
            normalisation=params.correlation_norm, max_lags=params.max_lags,
        )
        return Spectrum(
            x=lags, y=correlation, x_label="lag", y_label="correlation",
            details={"fs": params.fs, "points": int(y_values.size)},
        )

    raise ValueError(f"Unhandled spectral method: {method!r}")


def estimate_pair(
    method: str, first: np.ndarray, second: np.ndarray, params: SpectralParams
) -> Spectrum:
    """Estimate a two-signal quantity: cross spectrum, coherence, cross-correlation.

    The signals are cut to the shorter one's length; fewer than
    :data:`MIN_PAIR_SAMPLES` shared samples is a ``ValueError``.
    """
    length = min(first.size, second.size)
    if length < MIN_PAIR_SAMPLES:
        raise ValueError(f"fewer than {MIN_PAIR_SAMPLES} shared samples")
    first = first[:length]
    second = second[:length]

    if method == METHOD_CSD:
        frequencies, cross = scipy_signal.csd(
            first, second, scaling=params.scaling, **welch_kwargs(params, length)
        )
        values = np.abs(cross)
        if params.decibels:
            return _frequency_spectrum(frequencies, to_decibels(values), "dB")
        return _frequency_spectrum(frequencies, values, "power")

    if method == METHOD_COHERENCE:
        kwargs = welch_kwargs(params, length)
        # coherence has no return_onesided or scaling parameter: it is a
        # ratio, so both would cancel.
        kwargs.pop("return_onesided", None)
        frequencies, coherence = scipy_signal.coherence(first, second, **kwargs)
        return _frequency_spectrum(frequencies, coherence, "coherence")

    if method == METHOD_XCORR:
        lags, correlation = correlate(
            first, second,
            normalisation=params.correlation_norm, max_lags=params.max_lags,
        )
        return Spectrum(
            x=lags, y=correlation, x_label="lag", y_label="correlation",
            details={"fs": params.fs, "points": int(length)},
        )

    raise ValueError(f"Unhandled paired method: {method!r}")
