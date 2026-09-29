"""The Baseline dialog end to end (the estimators are in test_analysis_baseline.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.baseline_dialog import BASELINE_ASLS, SeriesBaselineDialog
from app.utils.dialog_state import clear_state

X = np.linspace(0.0, 100.0, 400)
TRUE_BASELINE = 5.0 + 0.02 * X + 3.0 * np.sin(X / 30.0)
PEAKS = 10.0 * np.exp(-((X - 30.0) ** 2) / 4.0) + 15.0 * np.exp(-((X - 70.0) ** 2) / 8.0)
NO_PEAK = PEAKS < 0.05


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    clear_state("SeriesBaselineDialog")
    yield
    clear_state("SeriesBaselineDialog")


# ----------------------------------------------------------------------
# The dialog, end to end
# ----------------------------------------------------------------------
@pytest.fixture
def figure_with_spectrum(repo: SqliteRepo):
    y = TRUE_BASELINE + PEAKS
    repo.import_dataframe(
        pd.DataFrame({"x": X, "y": y}), table_name="spec", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="spectrum", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="spec",
        sql_query="SELECT x, y FROM spec",
        roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


def test_the_dialog_applies_an_asls_correction(
    qapp, repo: SqliteRepo, figure_with_spectrum
) -> None:
    figure_id, axis_id = figure_with_spectrum
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(BASELINE_ASLS)
        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].corrected[NO_PEAK] == pytest.approx(
            np.zeros(int(NO_PEAK.sum())), abs=0.5
        )

        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    assert any("corrected" in name for name in names)
    assert any("baseline" in name for name in names)


