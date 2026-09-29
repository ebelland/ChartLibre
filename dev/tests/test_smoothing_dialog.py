"""The Smoothing dialog on top of its engine: parameters in, results out.

The arithmetic is tested in test_analysis_smoothing.py; this checks the
dialog still hands the engine what it needs and tells the user when the
engine says no.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

import app.series_operations.smoothing_dialog as dialog_module
from app.analysis.smoothing import SMOOTH_MOVING_AVERAGE
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.smoothing_dialog import SeriesSmoothingDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state() -> Iterator[None]:
    clear_state("SeriesSmoothingDialog")
    yield
    clear_state("SeriesSmoothingDialog")


def _figure(repo: SqliteRepo, x: list[float], y: list[float]) -> int:
    repo.import_dataframe(pd.DataFrame({"x": x, "y": y}), table_name="src", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="t", x_label="x", y_label="y", options={},
    ))
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM src", roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id


def test_a_moving_average_over_a_ramp_comes_back_as_the_ramp(qapp, repo: SqliteRepo) -> None:
    x = [float(i) for i in range(40, 0, -1)]  # stored in descending order on purpose
    figure_id = _figure(repo, x, [2.0 * v for v in x])
    dialog = SeriesSmoothingDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.series_selector.reload(select_all_series=True)
        dialog.method_combo.setCurrentText(SMOOTH_MOVING_AVERAGE)
        dialog.window_spin.setValue(5)
        result = dialog.compute_results()[0]
    finally:
        dialog.close()

    np.testing.assert_array_equal(result.x, np.arange(1.0, 41.0))  # the engine sorts by x
    np.testing.assert_allclose(result.y[3:-3], 2.0 * result.x[3:-3])
    assert result.method == SMOOTH_MOVING_AVERAGE
    assert result.metadata["window"] == 5


def test_the_engines_refusal_reaches_the_user(
    qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    told: list[str] = []
    monkeypatch.setattr(dialog_module.applogger, "error", lambda message, *a, **k: told.append(str(message)))
    figure_id = _figure(repo, [1.0, 2.0], [1.0, 2.0])  # two points: too few
    dialog = SeriesSmoothingDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.series_selector.reload(select_all_series=True)
        dialog.method_combo.setCurrentText(SMOOTH_MOVING_AVERAGE)
        assert dialog.compute_results() == []
    finally:
        dialog.close()

    # The shared input check may have spoken first; the engine's own reason
    # must be among what the user is told, with the series it is about.
    assert any("s: At least 3 finite points are required." in message for message in told)
