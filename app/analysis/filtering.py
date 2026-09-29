"""Filtering: the arithmetic behind the Filtering series operation.

Zero-phase IIR and FIR filters, the analytic signal (envelope, phase,
instantaneous frequency) and detrending, over plain arrays - no Qt, no
repository, nothing logged (todo R-01). A problem with the request (a cutoff
above the Nyquist frequency, a missing second cutoff) is a ``ValueError``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.signal import filtfilt, firwin, hilbert, iirfilter, sosfiltfilt


class IirTerms(NamedTuple):
    """What one IIR family asks for besides an order and a cutoff."""

    #: Takes a passband ripple, iirfilter's ``rp``.
    ripple: bool = False
    #: Takes a stopband attenuation, iirfilter's ``rs``.
    attenuation: bool = False


#: The key is iirfilter's ``ftype``.
IIR_FAMILIES: dict[str, IirTerms] = {
    "butter": IirTerms(),
    "cheby1": IirTerms(ripple=True),
    "cheby2": IirTerms(attenuation=True),
    "bessel": IirTerms(),
    "ellip": IirTerms(ripple=True, attenuation=True),
}

#: How many cutoffs each response takes. The key goes straight to iirfilter's
#: ``btype`` and firwin's ``pass_zero``, which both accept these four strings.
RESPONSE_CUTOFFS: dict[str, int] = {
    "lowpass": 1,
    "highpass": 1,
    "bandpass": 2,
    "bandstop": 2,
}

FIR_WINDOWS = ("hamming", "hann", "blackman", "bartlett", "boxcar")

#: What the analytic signal can give back; the key is analytic_signal's ``output``.
ANALYTIC_OUTPUTS: tuple[str, ...] = ("envelope", "phase", "frequency")

#: The key is scipy.signal.detrend's ``type``.
DETREND_TYPES: tuple[str, ...] = ("linear", "constant")

#: What "Auto" means for the cutoffs, as fractions of the sampling
#: frequency: a tenth of it, and - for a band - a quarter (Nyquist is a half).
AUTO_CUTOFF1: float = 0.1
AUTO_CUTOFF2: float = 0.25


def two_cutoffs(response: str) -> bool:
    """True for a band (bandpass, bandstop): it takes a second cutoff."""
    return RESPONSE_CUTOFFS.get(response) == 2


def resolve_cutoffs(
    cutoff: float, cutoff2: float, response: str, fs: float
) -> tuple[float, float | None]:
    """The cutoff(s) to filter with: the typed ones, or Auto's fractions of fs.

    0 means Auto. The second cutoff is None for a response that has one.
    """
    first = float(cutoff) or AUTO_CUTOFF1 * fs
    second = (float(cutoff2) or AUTO_CUTOFF2 * fs) if two_cutoffs(response) else None
    return first, second


def check_cutoff(cutoff: float, fs: float) -> None:
    nyquist = fs / 2.0
    if not (0.0 < cutoff < nyquist):
        raise ValueError(
            f"a cutoff of {cutoff:g} must be between 0 and the Nyquist "
            f"frequency ({nyquist:g}, half of fs={fs:g})"
        )


def apply_iir_filter(
    y: np.ndarray, fs: float, *, family: str, response: str, order: int,
    cutoff: float, cutoff2: float | None = None,
    ripple: float | None = None, atten: float | None = None,
) -> np.ndarray:
    """Zero-phase IIR filter: ``iirfilter`` designs it, ``sosfiltfilt`` runs it."""
    check_cutoff(cutoff, fs)
    wn: float | list[float] = cutoff
    if two_cutoffs(response):
        if cutoff2 is None:
            raise ValueError(f"{response} needs a second cutoff")
        check_cutoff(cutoff2, fs)
        if cutoff2 <= cutoff:
            raise ValueError("the second cutoff must be higher than the first")
        wn = [cutoff, cutoff2]

    spec = IIR_FAMILIES[family]
    rp = (float(ripple) if ripple is not None else 1.0) if spec.ripple else None
    rs = (float(atten) if atten is not None else 40.0) if spec.attenuation else None

    sos = iirfilter(order, wn, rp=rp, rs=rs, btype=response, ftype=family, output="sos", fs=fs)
    return np.asarray(sosfiltfilt(sos, y), dtype=float)


def apply_fir_filter(
    y: np.ndarray, fs: float, *, numtaps: int, window: str, response: str,
    cutoff: float, cutoff2: float | None = None,
) -> np.ndarray:
    """Zero-phase FIR filter: ``firwin`` designs it, ``filtfilt`` runs it."""
    check_cutoff(cutoff, fs)
    cutoff_arg: float | list[float] = cutoff
    if two_cutoffs(response):
        if cutoff2 is None:
            raise ValueError(f"{response} needs a second cutoff")
        check_cutoff(cutoff2, fs)
        if cutoff2 <= cutoff:
            raise ValueError("the second cutoff must be higher than the first")
        cutoff_arg = [cutoff, cutoff2]

    # firwin requires an odd tap count for a filter that must pass Nyquist
    # (highpass, bandstop) - an even one has a zero response there instead.
    taps = int(numtaps)
    if response in ("highpass", "bandstop") and taps % 2 == 0:
        taps += 1

    pass_zero = response in ("lowpass", "bandstop")
    coefficients = firwin(taps, cutoff_arg, window=window, pass_zero=pass_zero, fs=fs)
    return np.asarray(filtfilt(coefficients, [1.0], y), dtype=float)


def analytic_signal(y: np.ndarray, fs: float, output: str) -> np.ndarray:
    """Envelope, unwrapped phase, or instantaneous frequency of ``y``.

    Frequency is one sample shorter than the input (it comes from a
    difference of the phase) and the last value is repeated so the result
    still lines up with the source's x column.
    """
    analytic = hilbert(y)
    if output == "envelope":
        return np.abs(analytic)

    phase = np.unwrap(np.angle(np.asarray(analytic, dtype=np.complex128)))
    if output == "phase":
        return phase

    frequency = np.diff(phase) / (2.0 * np.pi) * fs
    return np.concatenate([frequency, frequency[-1:]]) if frequency.size else frequency


def apply_detrend(y: np.ndarray, kind: str) -> np.ndarray:
    """Remove a linear or constant trend (against sample index, not x)."""
    if kind not in ("linear", "constant"):
        raise ValueError(f"unsupported detrend type: {kind}")
    return np.asarray(scipy_detrend(y, type=kind), dtype=float)
