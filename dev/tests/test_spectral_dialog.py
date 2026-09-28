"""Tests for the spectral estimators behind the analysis dialog.

Every test uses a signal whose answer is known in closed form - a sine at a
chosen frequency, a pair with a chosen lag - so a wrong window, a wrong
sampling rate or a swapped argument shows up as a wrong number rather than as
a plausible-looking curve.

The estimators are exercised through the dialog class without constructing the
Qt dialog: the numeric methods only read widget values, which the stand-in
supplies.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.series_operations.spectral_dialog import METHOD_COHERENCE, METHOD_MAGNITUDE, METHOD_PSD, METHOD_XCORR, SeriesSpectralDialog

FS = 200.0
DURATION = 8.0
TONE_HZ = 25.0


class _Value:
    """Stand-in for a Qt widget that only has to report one value."""

    def __init__(self, value: object) -> None:
        self._value = value

    def value(self):
        return self._value

    def isChecked(self) -> bool:
        return bool(self._value)

    def currentText(self) -> str:
        return str(self._value)


def _dialog(**overrides) -> SimpleNamespace:
    """Build the attribute set the numeric methods read."""
    defaults = {
        "_fs_auto_check": True,
        "_fs_spin": 1.0,
        "_nperseg_spin": 256,
        "_overlap_spin": 0.5,
        "_window_combo": "hann",
        "_detrend_combo": "constant",
        "_scaling_combo": "density",
        "_onesided_check": True,
        "_db_check": False,
        "_maxlags_spin": 0,
        "_corr_norm_combo": "unbiased",
    }
    defaults.update(overrides)
    namespace = SimpleNamespace(
        **{name: _Value(value) for name, value in defaults.items()}
    )

    # Bind the methods under test to the stand-in.
    for method in (
        "_sampling_frequency",
        "_welch_kwargs",
        "_to_decibels",
        "_one_sided_fft",
        "_correlate",
        "_compute_single",
        "_compute_pair",
    ):
        setattr(
            namespace,
            method,
            getattr(SeriesSpectralDialog, method).__get__(namespace, SeriesSpectralDialog),
        )
    namespace._frequency_result = SeriesSpectralDialog._frequency_result
    return namespace


def _tone(frequency: float = TONE_HZ, phase: float = 0.0, noise: float = 0.0):
    """Return (x, y) for a sine at a known frequency."""
    x = np.arange(0.0, DURATION, 1.0 / FS)
    y = np.sin(2.0 * np.pi * frequency * x + phase)
    if noise:
        y = y + np.random.default_rng(0).normal(0.0, noise, x.size)
    return x, y


# ----------------------------------------------------------------------
# Sampling frequency
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Frequency-domain estimators
# ----------------------------------------------------------------------
def test_psd_peaks_at_the_tone_frequency() -> None:
    dialog = _dialog()
    result = dialog._compute_single(METHOD_PSD, ("tone", *_tone()))

    assert result is not None
    peak = result.x[int(np.argmax(result.y))]
    assert peak == pytest.approx(TONE_HZ, abs=1.0)
    assert result.x_label == "frequency"


def test_magnitude_spectrum_peaks_at_the_tone_frequency() -> None:
    result = _dialog()._compute_single(METHOD_MAGNITUDE, ("tone", *_tone()))

    assert result is not None
    assert result.x[int(np.argmax(result.y))] == pytest.approx(TONE_HZ, abs=0.5)


# ----------------------------------------------------------------------
# Paired estimators
# ----------------------------------------------------------------------


def test_coherence_is_high_for_a_shared_tone_and_bounded() -> None:
    x, y = _tone(noise=0.05)
    _, y2 = _tone(phase=0.3, noise=0.05)
    result = _dialog()._compute_pair(METHOD_COHERENCE, ("a", x, y), ("b", x, y2))

    assert result is not None
    assert np.all(result.y >= -1e-9) and np.all(result.y <= 1.0 + 1e-9)
    peak_index = int(np.argmin(np.abs(result.x - TONE_HZ)))
    assert result.y[peak_index] > 0.8


def test_cross_correlation_finds_a_known_lag() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 2048)
    lag = 17
    shifted = np.roll(base, lag)

    x = np.arange(base.size, dtype=float)
    result = _dialog(_maxlags_spin=100, _corr_norm_combo="biased")._compute_pair(
        METHOD_XCORR, ("a", x, shifted), ("b", x, base)
    )

    assert result is not None
    assert result.x[int(np.argmax(result.y))] == pytest.approx(lag, abs=1.0)


# ----------------------------------------------------------------------
# Result plumbing
# ----------------------------------------------------------------------


