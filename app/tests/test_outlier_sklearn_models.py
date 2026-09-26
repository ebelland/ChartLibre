"""Shape-aware anomaly detection in the Outliers dialog (todo.txt P2-15).

The four original methods (Z-score, IQR, MAD, rolling median) all test y
alone. Isolation Forest, Local Outlier Factor, One-Class SVM and Elliptic
Envelope instead fit the joint (x, y) point cloud, which is what this file
actually exercises: a tight cluster of 29 points plus one point far away in
both x and y, unambiguous to every one of the four regardless of their
individual quirks (LOF's edge effects, SVM's kernel bandwidth, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.outlier_dialog import (
    OUTLIER_ELLIPTIC_ENVELOPE,
    OUTLIER_ISOLATION_FOREST,
    OUTLIER_LOCAL_OUTLIER_FACTOR,
    OUTLIER_ONE_CLASS_SVM,
    SeriesOutlierDialog,
)

OUTLIER_X = 30.0
OUTLIER_Y = 30.0


def _dialog(repo: SqliteRepo, figure_id: int) -> SeriesOutlierDialog:
    dialog = SeriesOutlierDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


def _make_figure(repo: SqliteRepo, table_name: str, x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    repo.import_dataframe(
        pd.DataFrame({"n": x, "v": y}), table_name=table_name, normalize_columns=False
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="s", x_label="n", y_label="v", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query=f"SELECT n AS x, v AS y FROM {table_name}",
        roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


@pytest.fixture
def shaped_figure(repo: SqliteRepo):
    """A tight 29-point cluster, one point far away in both x and y."""
    cluster_x = 10.0 + 0.1 * (np.arange(29) % 7)
    cluster_y = 10.0 + 0.1 * (np.arange(29) // 7)
    x = np.concatenate([cluster_x, [OUTLIER_X]])
    y = np.concatenate([cluster_y, [OUTLIER_Y]])
    return _make_figure(repo, "pts", x, y)


@pytest.mark.parametrize(
    "model",
    [
        OUTLIER_ISOLATION_FOREST,
        OUTLIER_LOCAL_OUTLIER_FACTOR,
        OUTLIER_ONE_CLASS_SVM,
        OUTLIER_ELLIPTIC_ENVELOPE,
    ],
)
def test_flags_the_shape_outlier(qapp, repo: SqliteRepo, shaped_figure, model: str) -> None:
    figure_id, _axis_id = shaped_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    results = dialog.compute_results()

    assert len(results) == 1
    result = results[0]
    assert result.outlier_count >= 1
    outlier_points = set(zip(result.outlier_x.tolist(), result.outlier_y.tolist()))
    assert (OUTLIER_X, OUTLIER_Y) in outlier_points






