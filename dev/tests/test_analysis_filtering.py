"""Filtering (app.analysis.filtering): IIR/FIR filters, analytic signal, detrend.

Every numeric test uses a signal whose answer is known: a filter isolating
one of two mixed tones is checked against the tone itself, an AM envelope
against its own modulation, an instantaneous frequency against the tone
that produced it, so a swapped argument or a wrong Nyquist scaling shows up
as a wrong number rather than a plausible-looking curve. No dialog, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import filtering as flt
from app.analysis.filtering import analytic_signal, apply_detrend, apply_fir_filter, apply_iir_filter
from app.analysis.sampling import sampling_frequency

FS = 200.0
T = np.arange(0.0, 10.0, 1.0 / FS)
LOW_TONE = np.sin(2 * np.pi * 2.0 * T)
HIGH_TONE = np.sin(2 * np.pi * 40.0 * T)
MIXED = LOW_TONE + HIGH_TONE
EDGE = slice(150, -150)  # away from the filters' settling transients


# ----------------------------------------------------------------------
# IIR
# ----------------------------------------------------------------------
def test_iir_lowpass_recovers_the_low_tone_from_a_mix() -> None:
    filtered = apply_iir_filter(
        MIXED, FS, family="butter", response="lowpass", order=4, cutoff=10.0
    )
    assert filtered[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.05)


# ----------------------------------------------------------------------
# FIR
# ----------------------------------------------------------------------
def test_fir_lowpass_recovers_the_low_tone_from_a_mix() -> None:
    filtered = apply_fir_filter(
        MIXED, FS, numtaps=201, window="hamming", response="lowpass", cutoff=10.0
    )
    assert filtered[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.05)


# ----------------------------------------------------------------------
# Analytic signal
# ----------------------------------------------------------------------
def test_the_envelope_of_an_am_signal_tracks_its_modulation() -> None:
    modulation = 1.0 + 0.5 * np.sin(2.0 * np.pi * 1.0 * T)
    carrier = np.sin(2.0 * np.pi * 20.0 * T)
    envelope = analytic_signal(modulation * carrier, FS, "envelope")
    assert np.corrcoef(envelope[EDGE], modulation[EDGE])[0, 1] > 0.99


# ----------------------------------------------------------------------
# Detrend
# ----------------------------------------------------------------------
def test_linear_detrend_removes_a_ramp() -> None:
    ramp = np.linspace(0.0, 5.0, T.size)
    residual = apply_detrend(ramp, "linear")
    assert np.std(residual) < 1e-6


# ----------------------------------------------------------------------
# More IIR / FIR
# ----------------------------------------------------------------------
@pytest.mark.parametrize("family", list(flt.IIR_FAMILIES))
def test_every_iir_family_isolates_the_low_tone(family: str) -> None:
    filtered = apply_iir_filter(MIXED, FS, family=family, response="lowpass", order=5, cutoff=10.0,
                                ripple=0.5, atten=60.0)
    assert filtered[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.1)


def test_a_highpass_keeps_the_fast_tone_and_a_bandpass_only_its_band() -> None:
    high = apply_iir_filter(MIXED, FS, family="butter", response="highpass", order=4, cutoff=15.0)
    assert high[EDGE] == pytest.approx(HIGH_TONE[EDGE], abs=0.05)

    band = apply_iir_filter(MIXED, FS, family="butter", response="bandpass", order=4, cutoff=30.0, cutoff2=50.0)
    assert band[EDGE] == pytest.approx(HIGH_TONE[EDGE], abs=0.1)

    stop = apply_iir_filter(MIXED, FS, family="butter", response="bandstop", order=4, cutoff=30.0, cutoff2=50.0)
    assert stop[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.1)


def test_a_fir_highpass_with_an_even_tap_count_still_passes_nyquist() -> None:
    """firwin refuses an even count for a filter that must pass Nyquist; one is added."""
    out = apply_fir_filter(MIXED, FS, numtaps=100, window="hamming", response="highpass", cutoff=15.0)
    assert out[EDGE] == pytest.approx(HIGH_TONE[EDGE], abs=0.1)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, FS / 2.0, FS])
def test_a_cutoff_outside_zero_and_nyquist_is_a_value_error(cutoff: float) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        apply_iir_filter(MIXED, FS, family="butter", response="lowpass", order=2, cutoff=cutoff)
    with pytest.raises(ValueError, match="Nyquist"):
        apply_fir_filter(MIXED, FS, numtaps=51, window="hamming", response="lowpass", cutoff=cutoff)


def test_a_band_needs_a_second_cutoff_above_the_first() -> None:
    with pytest.raises(ValueError, match="second cutoff"):
        apply_iir_filter(MIXED, FS, family="butter", response="bandpass", order=2, cutoff=10.0)
    with pytest.raises(ValueError, match="higher than the first"):
        apply_iir_filter(MIXED, FS, family="butter", response="bandpass", order=2, cutoff=20.0, cutoff2=10.0)


# ----------------------------------------------------------------------
# Analytic signal
# ----------------------------------------------------------------------
def test_the_instantaneous_frequency_of_a_tone_is_the_tone() -> None:
    frequency = analytic_signal(np.sin(2 * np.pi * 20.0 * T), FS, "frequency")
    assert frequency.size == T.size  # the last value is repeated to keep the length
    assert frequency[EDGE] == pytest.approx(20.0, abs=0.2)


def test_the_unwrapped_phase_grows_by_two_pi_per_cycle() -> None:
    phase = analytic_signal(np.sin(2 * np.pi * 5.0 * T), FS, "phase")
    cycles = (phase[EDGE][-1] - phase[EDGE][0]) / (2 * np.pi)
    assert cycles == pytest.approx(5.0 * (T[EDGE][-1] - T[EDGE][0]), rel=0.01)


def test_the_constant_detrend_removes_the_mean_and_an_unknown_kind_is_refused() -> None:
    assert np.mean(apply_detrend(np.array([3.0, 4.0, 5.0, 6.0]), "constant")) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="unsupported detrend type"):
        apply_detrend(np.arange(5.0), "quadratic")


# ----------------------------------------------------------------------
# Cutoffs and sampling
# ----------------------------------------------------------------------
def test_auto_cutoffs_are_fractions_of_fs_and_typed_ones_are_kept() -> None:
    assert flt.resolve_cutoffs(0.0, 0.0, "lowpass", 100.0) == (10.0, None)
    assert flt.resolve_cutoffs(0.0, 0.0, "bandpass", 100.0) == (10.0, 25.0)
    assert flt.resolve_cutoffs(7.0, 0.0, "bandstop", 100.0) == (7.0, 25.0)
    assert flt.resolve_cutoffs(7.0, 30.0, "highpass", 100.0) == (7.0, None)  # no second cutoff to keep


def test_only_the_two_bands_take_a_second_cutoff() -> None:
    assert [name for name in flt.RESPONSE_CUTOFFS if flt.two_cutoffs(name)] == ["bandpass", "bandstop"]


def test_the_sampling_frequency_is_shared_with_spectral_analysis() -> None:
    from app.analysis import spectral

    assert spectral.sampling_frequency is sampling_frequency
    assert sampling_frequency(np.arange(0.0, 1.0, 0.01))[0] == pytest.approx(100.0)
