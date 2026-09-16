"""Heavy-compute spinboxes debounce their preview refresh.

Cluster (KMeans/hierarchical/DBSCAN), Spectral (Welch/FFT/wavelet) and
Baseline (AsLS) each connected their spinboxes' valueChanged straight to
refresh_results(), which recomputes from scratch synchronously on the UI
thread. Holding a spinbox's arrow down, or dragging it, fired that full
computation once per intermediate value. They now route through
SeriesOperationDialogBase._queue_refresh_results, which restarts a shared
QTimer instead of calling refresh_results() immediately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.baseline_dialog import SeriesBaselineDialog
from app.series_operations.cluster_dialog import SeriesClusterDialog
from app.series_operations.spectral_dialog import SeriesSpectralDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    for name in ("SeriesBaselineDialog", "SeriesClusterDialog", "SeriesSpectralDialog"):
        clear_state(name)
    yield
    for name in ("SeriesBaselineDialog", "SeriesClusterDialog", "SeriesSpectralDialog"):
        clear_state(name)


@pytest.fixture
def figure(repo: SqliteRepo) -> tuple[int, int]:
    x = np.linspace(0.0, 10.0, 200)
    y = np.sin(x) + 0.01 * x
    repo.import_dataframe(pd.DataFrame({"x": x, "y": y}), table_name="src", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM src", roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


def test_baseline_lambda_spin_debounces(qapp, repo: SqliteRepo, figure: tuple[int, int]) -> None:
    figure_id, _axis_id = figure
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        assert not dialog._refresh_timer.isActive()
        dialog._lambda_spin.setValue(dialog._lambda_spin.value() + 1.0)
        assert dialog._refresh_timer.isActive()
    finally:
        dialog.close()


def test_spectral_nperseg_spin_debounces(qapp, repo: SqliteRepo, figure: tuple[int, int]) -> None:
    figure_id, _axis_id = figure
    dialog = SeriesSpectralDialog(repo=repo, figure_id=figure_id)
    try:
        assert not dialog._refresh_timer.isActive()
        dialog._nperseg_spin.setValue(dialog._nperseg_spin.value() + 8)
        assert dialog._refresh_timer.isActive()
    finally:
        dialog.close()


def test_cluster_count_spin_debounces(qapp, repo: SqliteRepo, figure: tuple[int, int]) -> None:
    figure_id, _axis_id = figure
    dialog = SeriesClusterDialog(repo=repo, figure_id=figure_id)
    try:
        assert not dialog._refresh_timer.isActive()
        dialog.cluster_count_spin.setValue(dialog.cluster_count_spin.value() + 1)
        assert dialog._refresh_timer.isActive()
    finally:
        dialog.close()


def test_the_debounced_timer_still_runs_refresh_results(
    qapp, repo: SqliteRepo, figure: tuple[int, int]
) -> None:
    """Queuing is not the whole story - the timer firing must still reach
    refresh_results(), same as any other Qt timeout."""
    figure_id, _axis_id = figure
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        dialog._lambda_spin.setValue(dialog._lambda_spin.value() + 1.0)
        assert dialog._refresh_timer.isActive()

        dialog._refresh_timer.stop()
        dialog._refresh_timer.timeout.emit()

        assert dialog._results_label.toHtml().strip()
    finally:
        dialog.close()
