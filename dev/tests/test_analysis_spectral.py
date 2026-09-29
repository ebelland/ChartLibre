"""The spectral estimators (app.analysis.spectral): answers known in closed form.

Every test uses a signal whose answer is known - a sine at a chosen
frequency, a pair with a chosen lag, a damped exponential - so a wrong
window, a wrong sampling rate or a swapped argument shows up as a wrong
number rather than as a plausible-looking curve. No dialog, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import spectral as sp

FS = 200.0
DURATION = 8.0
TONE_HZ = 25.0
PARAMS = sp.SpectralParams(fs=FS)


def _tone(frequency: float = TONE_HZ, phase: float = 0.0, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(0.0, DURATION, 1.0 / FS)
    y = np.sin(2.0 * np.pi * frequency * x + phase)
    if noise:
        y = y + np.random.default_rng(0).normal(0.0, noise, x.size)
    return x, y


def _peak(spectrum: sp.Spectrum) -> float:
    return float(spectrum.x[int(np.argmax(spectrum.y))])


# ----------------------------------------------------------------------
# Sampling frequency
# ----------------------------------------------------------------------
def test_fs_is_read_off_the_median_spacing() -> None:
    fs, note = sp.sampling_frequency(np.arange(0.0, 10.0, 0.005))
    assert fs == pytest.approx(200.0)
    assert note == ""


def test_uneven_sampling_is_reported() -> None:
    x = np.cumsum(np.r_[0.0, np.full(100, 0.01), 0.05, np.full(100, 0.01)])
    fs, note = sp.sampling_frequency(x)
    assert fs == pytest.approx(100.0)
    assert "not uniformly sampled" in note


def test_a_non_increasing_x_assumes_fs_one() -> None:
    fs, note = sp.sampling_frequency(np.full(10, 3.0))
    assert fs == 1.0
    assert "not increasing" in note


# ----------------------------------------------------------------------
# One signal
# ----------------------------------------------------------------------
def test_psd_peaks_at_the_tone_frequency() -> None:
    result = sp.estimate(sp.METHOD_PSD, _tone()[1], PARAMS)
    assert _peak(result) == pytest.approx(TONE_HZ, abs=1.0)
    assert (result.x_label, result.y_label) == ("frequency", "power")


def test_magnitude_spectrum_peaks_at_the_tone_frequency_and_has_the_right_height() -> None:
    _x, y = _tone()
    result = sp.estimate(sp.METHOD_MAGNITUDE, y, PARAMS)
    assert _peak(result) == pytest.approx(TONE_HZ, abs=0.5)
    # A unit sine over N samples has an FFT magnitude of N/2 at its bin.
    assert float(result.y.max()) == pytest.approx(y.size / 2.0, rel=0.02)


def test_decibels_are_ten_log_ten_of_the_power() -> None:
    _x, y = _tone(noise=0.1)
    power = sp.estimate(sp.METHOD_PSD, y, PARAMS)
    in_db = sp.estimate(sp.METHOD_PSD, y, sp.SpectralParams(fs=FS, decibels=True))
    np.testing.assert_allclose(in_db.y, 10.0 * np.log10(np.maximum(power.y, power.y[power.y > 0].min())))
    assert in_db.y_label == "dB"


def test_a_zero_bin_is_floored_not_dropped() -> None:
    out = sp.to_decibels(np.array([0.0, 1.0, 10.0]))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [0.0, 0.0, 10.0])


def test_the_phase_of_a_cosine_at_its_bin_is_about_zero_and_a_sine_minus_ninety_degrees() -> None:
    n = 800
    t = np.arange(n) / FS
    cosine = np.cos(2 * np.pi * 25.0 * t)  # 25 Hz is exactly a bin: 200/800 = 0.25 Hz spacing
    sine = np.sin(2 * np.pi * 25.0 * t)
    for signal, expected in ((cosine, 0.0), (sine, -np.pi / 2)):
        result = sp.estimate(sp.METHOD_ANGLE, signal, PARAMS)
        at_tone = int(np.argmin(np.abs(result.x - 25.0)))
        assert result.y[at_tone] == pytest.approx(expected, abs=0.05)


def test_laplace_with_no_damping_is_the_fourier_magnitude_over_fs() -> None:
    _x, y = _tone(noise=0.05)
    fourier = sp.estimate(sp.METHOD_MAGNITUDE, y, PARAMS)
    laplace = sp.estimate(sp.METHOD_LAPLACE, y, sp.SpectralParams(fs=FS, sigma=0.0))
    np.testing.assert_allclose(laplace.y, fourier.y / FS)


def test_laplace_finds_the_tone_of_a_growing_signal() -> None:
    x = np.arange(0.0, 3.0, 1.0 / FS)
    growing = np.exp(1.5 * x) * np.sin(2 * np.pi * TONE_HZ * x)
    damped = sp.estimate(sp.METHOD_LAPLACE, growing, sp.SpectralParams(fs=FS, sigma=1.5))
    assert _peak(damped) == pytest.approx(TONE_HZ, abs=1.0)


def test_the_wavelet_spectrum_peaks_near_the_tone() -> None:
    result = sp.estimate(
        sp.METHOD_WAVELET, _tone()[1], sp.SpectralParams(fs=FS, wavelet_w0=6.0, wavelet_scales=96)
    )
    assert _peak(result) == pytest.approx(TONE_HZ, rel=0.15)
    assert result.x[0] < result.x[-1] <= FS / 2.0 + 1e-9


def test_the_autocorrelation_of_noise_is_a_spike_at_lag_zero() -> None:
    noise = np.random.default_rng(5).normal(size=1024)
    result = sp.estimate(
        sp.METHOD_ACORR, noise, sp.SpectralParams(fs=FS, correlation_norm="biased", max_lags=50)
    )
    assert result.x[int(np.argmax(result.y))] == 0.0
    assert float(result.y.max()) == pytest.approx(float(np.var(noise)), rel=1e-6)
    assert result.x.min() == -50 and result.x.max() == 50
    assert result.x_label == "lag"


def test_an_unknown_method_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unhandled spectral method"):
        sp.estimate("No Such Method", _tone()[1], PARAMS)


# ----------------------------------------------------------------------
# Two signals
# ----------------------------------------------------------------------
def test_coherence_is_high_for_a_shared_tone_and_bounded() -> None:
    _x, y = _tone(noise=0.05)
    _x2, y2 = _tone(phase=0.3, noise=0.05)
    result = sp.estimate_pair(sp.METHOD_COHERENCE, y, y2, PARAMS)
    assert np.all(result.y >= -1e-9) and np.all(result.y <= 1.0 + 1e-9)
    assert result.y[int(np.argmin(np.abs(result.x - TONE_HZ)))] > 0.8


def test_cross_spectrum_peaks_at_the_shared_tone() -> None:
    _x, y = _tone(noise=0.05)
    _x2, y2 = _tone(phase=0.5, noise=0.05)
    assert _peak(sp.estimate_pair(sp.METHOD_CSD, y, y2, PARAMS)) == pytest.approx(TONE_HZ, abs=1.0)


def test_cross_correlation_finds_a_known_lag() -> None:
    base = np.random.default_rng(3).normal(0.0, 1.0, 2048)
    lag = 17
    result = sp.estimate_pair(
        sp.METHOD_XCORR, np.roll(base, lag), base,
        sp.SpectralParams(fs=FS, max_lags=100, correlation_norm="biased"),
    )
    assert result.x[int(np.argmax(result.y))] == pytest.approx(lag, abs=1.0)


def test_a_pair_is_cut_to_the_shorter_signal_and_needs_eight_shared_samples() -> None:
    _x, y = _tone()
    short = sp.estimate_pair(sp.METHOD_XCORR, y, y[:100], sp.SpectralParams(fs=FS, max_lags=10))
    assert short.details["points"] == 100
    with pytest.raises(ValueError, match="fewer than 8 shared samples"):
        sp.estimate_pair(sp.METHOD_XCORR, y, y[:5], PARAMS)


# ----------------------------------------------------------------------
# Welch settings
# ----------------------------------------------------------------------
def test_nperseg_is_clamped_to_the_data_and_to_eight() -> None:
    assert sp.welch_kwargs(sp.SpectralParams(nperseg=256), 100)["nperseg"] == 100
    assert sp.welch_kwargs(sp.SpectralParams(nperseg=2), 100)["nperseg"] == 8
    kwargs = sp.welch_kwargs(sp.SpectralParams(nperseg=64, overlap=0.25, detrend="none"), 1000)
    assert kwargs["noverlap"] == 16
    assert kwargs["detrend"] is False
