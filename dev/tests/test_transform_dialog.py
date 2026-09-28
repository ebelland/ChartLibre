"""Transform dialog: rescale or reshape a series' Y distribution (P2-15).

Unlike Calculus or Regression, x is carried through unchanged here - only Y
is reshaped, point for point, same length. See transform_dialog.py's own
module docstring for why that makes this a distinct operation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.transform_dialog import TRANSFORM_MODELS, SeriesTransformDialog

N = 60


def _dialog(repo: SqliteRepo, figure_id: int) -> SeriesTransformDialog:
    dialog = SeriesTransformDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


def _make_figure(repo: SqliteRepo, x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
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


@pytest.fixture
def positive_figure(repo: SqliteRepo):
    """All-positive, right-skewed - a natural Box-Cox candidate."""
    rng = np.random.default_rng(0)
    x = np.arange(N, dtype=float)
    y = rng.lognormal(mean=1.0, sigma=0.6, size=N)
    return _make_figure(repo, x, y)


@pytest.mark.parametrize("model", TRANSFORM_MODELS)
def test_every_model_transforms_in_place(qapp, repo: SqliteRepo, positive_figure, model) -> None:
    figure_id, _axis_id = positive_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    result = dialog.compute_results()[0]

    assert result.y.size == N
    assert np.array_equal(result.x, np.arange(N, dtype=float))
    assert np.all(np.isfinite(result.y))


