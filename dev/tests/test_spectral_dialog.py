"""The Spectral dialog on top of its engine: controls in, named results out.

The arithmetic is tested in test_analysis_spectral.py.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from app.analysis.spectral import METHOD_ACORR, METHOD_PSD, METHOD_XCORR
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.spectral_dialog import SeriesSpectralDialog
from app.utils.dialog_state import clear_state

FS = 200.0
TONE_HZ = 25.0


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state() -> Iterator[None]:
    clear_state("SeriesSpectralDialog")
    yield
    clear_state("SeriesSpectralDialog")


@pytest.fixture
def two_series(repo: SqliteRepo) -> int:
    t = np.arange(0.0, 8.0, 1.0 / FS)
    repo.import_dataframe(
        pd.DataFrame({
            "t": t,
            "a": np.sin(2 * np.pi * TONE_HZ * t),
            "b": np.sin(2 * np.pi * TONE_HZ * t + 0.4),
        }),
        table_name="sig", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="t", x_label="t", y_label="y", options={},
    ))
    for index, column in enumerate(("a", "b")):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=column,
            sql_query=f'SELECT t, {column} FROM sig', roles={"x": "t", "y": column}, style={},
        )
    return figure_id


def _dialog(repo: SqliteRepo, figure_id: int, method: str) -> SeriesSpectralDialog:
    dialog = SeriesSpectralDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    dialog.model_combo.setCurrentText(method)
    return dialog


def test_psd_of_each_series_peaks_at_the_tone_and_is_named_after_it(qapp, repo: SqliteRepo, two_series: int) -> None:
    dialog = _dialog(repo, two_series, METHOD_PSD)
    try:
        results = dialog.compute_results()
    finally:
        dialog.close()
    assert [result.result_name for result in results] == [f"a - {METHOD_PSD}", f"b - {METHOD_PSD}"]
    for result in results:
        assert result.x[int(np.argmax(result.y))] == pytest.approx(TONE_HZ, abs=1.0)
        assert (result.x_label, result.y_label) == ("frequency", "power")
        assert result.metadata == {"points": result.x.size}


def test_a_pair_is_estimated_against_the_first_series(qapp, repo: SqliteRepo, two_series: int) -> None:
    dialog = _dialog(repo, two_series, METHOD_XCORR)
    try:
        results = dialog.compute_results()
    finally:
        dialog.close()
    assert len(results) == 1
    assert results[0].result_name == "a x b - Cross-correlation"
    assert results[0].source_name == "a x b"
    assert results[0].x_label == "lag"
    assert results[0].metadata["fs"] == pytest.approx(FS)


def test_the_autocorrelation_result_is_named_as_before(qapp, repo: SqliteRepo, two_series: int) -> None:
    dialog = _dialog(repo, two_series, METHOD_ACORR)
    try:
        results = dialog.compute_results()
    finally:
        dialog.close()
    assert [result.result_name for result in results] == ["a - Autocorrelation", "b - Autocorrelation"]
