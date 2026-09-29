"""Filtering: IIR/FIR filters, analytic-signal derivatives, detrend.

Every numeric test uses a signal whose answer is known: a filter isolating
one of two mixed tones is checked against the tone itself, an AM envelope
against its own modulation, an instantaneous frequency against the tone
that produced it, so a swapped argument or a wrong Nyquist scaling shows up
as a wrong number rather than a plausible-looking curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils.dialog_state import clear_state
from app.series_operations.filter_dialog import FILTER_IIR, SeriesFilterDialog, analytic_signal, apply_detrend, apply_fir_filter, apply_iir_filter

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
# The dialog, end to end
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """Dialog state (model/response/cutoffs) survives in the real config.json
    across runs - reset it so these tests never inherit another run's combo
    selection, and never leave one behind for the next."""
    clear_state("SeriesFilterDialog")
    yield
    clear_state("SeriesFilterDialog")


@pytest.fixture
def figure_with_mixed_signal(repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"t": T, "signal": MIXED}),
        table_name="sig", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="mix", x_label="t", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="mix",
        sql_query="SELECT t, signal FROM sig",
        roles={"x": "t", "y": "signal"}, style={},
    )
    return figure_id, axis_id


def test_the_dialog_applies_an_iir_lowpass(
    qapp, repo: SqliteRepo, figure_with_mixed_signal
) -> None:
    figure_id, axis_id = figure_with_mixed_signal
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_IIR)
        dialog._cutoff1_spin.setValue(10.0)

        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].y[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.05)

        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    assert any("IIR" in name for name in names)




# ----------------------------------------------------------------------
# The default cutoff follows fs (todo P0-1)
# ----------------------------------------------------------------------
@pytest.fixture
def dated_series(repo: SqliteRepo):
    """A daily series dated in text, the way SQLite hands dates back: fs
    comes out near 1e-5 per second, so the old fixed default cutoff of 1
    was far above Nyquist and Preview failed."""
    days = pd.date_range("2024-01-01", periods=400, freq="D")
    values = np.sin(2 * np.pi * np.arange(400) / 100.0) + 0.5 * np.sin(2 * np.pi * np.arange(400) / 3.0)
    repo.import_dataframe(
        pd.DataFrame({"day": days.strftime("%Y-%m-%d"), "value": values}),
        table_name="daily", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="D", nrows=1, ncols=1))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="daily", x_label="day", y_label="value", options={},
    ))
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="daily",
        sql_query="SELECT day, value FROM daily",
        roles={"x": "day", "y": "value"}, style={},
    )
    return figure_id, values


def test_the_untouched_cutoff_works_on_a_series_dated_in_seconds(
    qapp, repo: SqliteRepo, dated_series
) -> None:
    figure_id, values = dated_series
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_IIR)
        assert dialog._cutoff1_spin.value() == 0.0  # Auto - nobody typed anything

        result = dialog.compute_results()[0]
        fs = float(result.metadata["fs"])
        assert fs == pytest.approx(1.0 / 86400.0, rel=1e-3)
        assert float(result.metadata["cutoff"]) == pytest.approx(0.1 * fs)
        # A tenth of fs keeps the 100-day wave and drops the 3-day one.
        slow = np.sin(2 * np.pi * np.arange(400) / 100.0)
        assert result.y[100:-100] == pytest.approx(slow[100:-100], abs=0.15)
    finally:
        dialog.close()


def test_a_band_is_automatic_too_and_a_typed_cutoff_is_respected(
    qapp, repo: SqliteRepo, figure_with_mixed_signal
) -> None:
    figure_id, _axis_id = figure_with_mixed_signal
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_IIR)
        dialog._response_combo.setCurrentIndex(dialog._response_combo.findData("bandpass"))
        band = dialog.compute_results()[0]
        assert band.metadata["cutoff"] == f"{0.1 * FS:g} - {0.25 * FS:g}"

        dialog._response_combo.setCurrentIndex(dialog._response_combo.findData("lowpass"))
        dialog._cutoff1_spin.setValue(10.0)
        assert dialog.compute_results()[0].metadata["cutoff"] == 10.0
    finally:
        dialog.close()
