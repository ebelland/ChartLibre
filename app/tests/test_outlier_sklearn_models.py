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


@pytest.mark.parametrize(
    "model",
    [
        OUTLIER_ISOLATION_FOREST,
        OUTLIER_LOCAL_OUTLIER_FACTOR,
        OUTLIER_ONE_CLASS_SVM,
        OUTLIER_ELLIPTIC_ENVELOPE,
    ],
)
def test_applying_marks_the_right_row_hidden(
    qapp, repo: SqliteRepo, shaped_figure, model: str
) -> None:
    figure_id, _axis_id = shaped_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    assert dialog.preview() is True

    assert repo.count_hidden_rows("pts") >= 1
    remaining = repo.query_df('SELECT n, v FROM pts WHERE COALESCE("Hide", 0) = 0')
    remaining_points = set(zip(remaining["n"].tolist(), remaining["v"].tolist()))
    assert (OUTLIER_X, OUTLIER_Y) not in remaining_points


def test_local_outlier_factor_survives_a_series_shorter_than_its_neighbourhood(
    qapp, repo: SqliteRepo
) -> None:
    """Default n_neighbors is 20; a 5-point series must not crash on it."""
    figure_id, _axis_id = _make_figure(
        repo, "short",
        np.arange(5.0),
        np.array([0.0, 1.0, 2.0, 3.0, 40.0]),
    )
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(OUTLIER_LOCAL_OUTLIER_FACTOR)

    results = dialog.compute_results()  # must not raise

    assert len(results) == 1


def test_elliptic_envelope_reports_a_clear_error_on_degenerate_input(
    qapp, repo: SqliteRepo
) -> None:
    """Every point identical: zero variance in x AND y singularises the
    covariance matrix outright (unlike zero variance in only one of the
    two, which this scikit-learn version tolerates with a warning)."""
    figure_id, _axis_id = _make_figure(
        repo, "flat", np.zeros(5), np.zeros(5)
    )
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(OUTLIER_ELLIPTIC_ENVELOPE)

    # compute_results() swallows a single series' failure into a shown
    # message rather than raising - assert the underlying method directly
    # to check the message itself is informative.
    with pytest.raises(ValueError, match="Elliptic Envelope"):
        dialog._detect_outliers(
            dialog.selected_series()[0],
            OUTLIER_ELLIPTIC_ENVELOPE,
            dialog._params(),
        )
