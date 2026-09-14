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
from app.series_operations.regression_dialog import (
    REGRESSION_GRADIENT_BOOSTING,
    REGRESSION_HUBER,
    REGRESSION_ISOTONIC,
    REGRESSION_MODELS,
    REGRESSION_RANDOM_FOREST,
    REGRESSION_RANSAC,
    SeriesRegressionDialog,
)
from app.utils.coercion import parse_datetimes

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


@pytest.fixture
def dated_figure(repo: SqliteRepo):
    days = pd.date_range("2024-01-01", periods=N, freq="D")
    x = np.arange(N, dtype=float)
    y = 2.0 * x + 1.0
    repo.import_dataframe(
        pd.DataFrame({"t": days.strftime("%Y-%m-%d %H:%M:%S"), "y": y}),
        table_name="ts", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Time Series",
            title="ts", x_label="t", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT t AS x, y AS y FROM ts", roles={"x": "x", "y": "y"}, style={},
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


@pytest.mark.parametrize("model", REGRESSION_MODELS)
def test_a_dated_series_is_written_back_as_dates(
    qapp, repo: SqliteRepo, dated_figure, model
) -> None:
    figure_id, axis_id = dated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)
    results = dialog.compute_results()

    dialog.apply_results_to_axis(axis_id, results)

    table = dialog.result_table_name(axis_id, results[0])
    stored = repo.query_df(f'SELECT x FROM "{table}"')
    parsed = parse_datetimes(stored["x"])
    assert parsed.notna().all()
    assert parsed.iloc[0].year == 2024


def test_ransac_reports_an_inlier_fraction(qapp, repo: SqliteRepo, linear_figure) -> None:
    figure_id, _axis_id = linear_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(REGRESSION_RANSAC)

    result = dialog.compute_results()[0]

    assert "inliers" in result.metadata


def test_huber_is_not_dragged_off_by_the_outlier(qapp, repo: SqliteRepo, linear_figure) -> None:
    """The true line is y = 2x + 1; a mean/OLS fit dragged toward the
    outlier at x=5 would overshoot the slope well past 2."""
    figure_id, _axis_id = linear_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(REGRESSION_HUBER)

    result = dialog.compute_results()[0]

    slope = (result.y[-1] - result.y[0]) / (result.x[-1] - result.x[0])
    assert 1.5 < slope < 2.5


@pytest.mark.parametrize("increasing_choice", ["auto", "true"])
def test_isotonic_never_decreases(qapp, repo: SqliteRepo, linear_figure, increasing_choice) -> None:
    figure_id, _axis_id = linear_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(REGRESSION_ISOTONIC)
    dialog.set_parameter_values({"increasing": increasing_choice})

    result = dialog.compute_results()[0]

    assert np.all(np.diff(result.y) >= -1e-9)


@pytest.mark.parametrize("model", [REGRESSION_RANDOM_FOREST, REGRESSION_GRADIENT_BOOSTING])
def test_tree_ensembles_report_r2(qapp, repo: SqliteRepo, linear_figure, model) -> None:
    figure_id, _axis_id = linear_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    result = dialog.compute_results()[0]

    assert 0.0 <= result.metadata["r2"] <= 1.0


def test_draws_on_the_same_axis_by_default(qapp, repo: SqliteRepo, linear_figure) -> None:
    figure_id, axis_id = linear_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(REGRESSION_RANSAC)
    results = dialog.compute_results()

    resolved = dialog.resolve_target_axis_id(axis_id, results)

    assert resolved == axis_id
