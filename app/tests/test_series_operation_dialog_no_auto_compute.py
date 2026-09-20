"""Series-operation dialogs compute only on Preview/OK, never on a control edit.

Cluster (KMeans/hierarchical/DBSCAN), Spectral (Welch/FFT/wavelet) and Baseline
(AsLS) used to connect their spinboxes' valueChanged straight to
refresh_results() (heavy ones through a debounced QTimer), which recomputed
from scratch off a plain control edit - a spinbox arrow held down, or dragged,
fired a full computation once per intermediate value, and every dialog kept a
"live" preview running even before the user asked for one.

Every control - light or heavy - now goes through
SeriesOperationDialogBase.mark_results_stale(), which only invalidates any
previously computed/previewed result and shows PENDING_RESULTS_MESSAGE.
Nothing is computed and nothing is drawn on the chart until the user actually
presses Preview (or OK), which is what these tests check.
"""
from __future__ import annotations

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


def _pending_message_shown(dialog) -> bool:
    return dialog.PENDING_RESULTS_MESSAGE in dialog._results_label.text()


def test_baseline_lambda_spin_only_marks_results_stale(
    qapp, repo: SqliteRepo, figure: tuple[int, int]
) -> None:
    figure_id, _axis_id = figure
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        assert dialog._last_results == []
        assert _pending_message_shown(dialog)

        dialog._lambda_spin.setValue(dialog._lambda_spin.value() + 1.0)

        # Editing the control invalidated the (empty) preview - it did not
        # compute anything.
        assert dialog._last_results == []
        assert _pending_message_shown(dialog)
    finally:
        dialog.close()


def test_spectral_nperseg_spin_only_marks_results_stale(
    qapp, repo: SqliteRepo, figure: tuple[int, int]
) -> None:
    figure_id, _axis_id = figure
    dialog = SeriesSpectralDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.series_selector.select_all_series()
        assert dialog._last_results == []

        dialog._nperseg_spin.setValue(dialog._nperseg_spin.value() + 8)

        assert dialog._last_results == []
    finally:
        dialog.close()


def test_cluster_count_spin_only_marks_results_stale(
    qapp, repo: SqliteRepo, figure: tuple[int, int]
) -> None:
    figure_id, _axis_id = figure
    dialog = SeriesClusterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.series_selector.select_all_series()
        assert dialog._last_results == []

        dialog.cluster_count_spin.setValue(dialog.cluster_count_spin.value() + 1)

        assert dialog._last_results == []
    finally:
        dialog.close()


def test_preview_is_the_only_thing_that_computes_a_baseline_result(
    qapp, repo: SqliteRepo, figure: tuple[int, int]
) -> None:
    """Preview (what the user actually presses) still runs compute_results()."""
    figure_id, _axis_id = figure
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        dialog._lambda_spin.setValue(dialog._lambda_spin.value() + 1.0)
        assert dialog._last_results == []

        assert dialog.preview() is True

        assert dialog._last_results
        assert not _pending_message_shown(dialog)
    finally:
        dialog.close()
