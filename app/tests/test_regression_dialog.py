"""Robust/ML regression dialog (todo.txt P2-15).

A dedicated dialog rather than another Fit model: Fit's shape is "an
algebraic expression with named parameters," and these five models have
no such expression - see regression_dialog.py's own module docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.regression_dialog import REGRESSION_MODELS, SeriesRegressionDialog

N = 60


def _dialog(repo: SqliteRepo, figure_id: int) -> SeriesRegressionDialog:
    dialog = SeriesRegressionDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


@pytest.fixture
def linear_figure(repo: SqliteRepo):
    """A noisy line with one gross outlier - robust models should shrug it off."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, N)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, size=N)
    y[5] += 30.0
    repo.import_dataframe(
        pd.DataFrame({"x": x, "y": y}), table_name="w", normalize_columns=False
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="w", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


@pytest.mark.parametrize("model", REGRESSION_MODELS)
def test_every_model_produces_a_sane_curve(qapp, repo: SqliteRepo, linear_figure, model) -> None:
    figure_id, _axis_id = linear_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    result = dialog.compute_results()[0]

    assert result.y.size == result.x.size
    assert np.all(np.isfinite(result.y))


