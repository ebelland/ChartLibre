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
from app.series_operations.transform_dialog import (
    TRANSFORM_MODELS,
    TRANSFORM_POWER,
    TRANSFORM_QUANTILE,
    POWER_BOX_COX,
    SeriesTransformDialog,
)
from app.utils.coercion import parse_datetimes

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


@pytest.fixture
def signed_figure(repo: SqliteRepo):
    """Contains a non-positive value - Box-Cox must refuse this one."""
    rng = np.random.default_rng(1)
    x = np.arange(N, dtype=float)
    y = rng.normal(0.0, 2.0, size=N)
    y[0] = -1.0
    return _make_figure(repo, x, y)


@pytest.fixture
def dated_figure(repo: SqliteRepo):
    days = pd.date_range("2024-01-01", periods=N, freq="D")
    y = np.abs(np.random.default_rng(0).normal(5.0, 2.0, size=N)) + 0.1
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


@pytest.mark.parametrize("model", TRANSFORM_MODELS)
def test_every_model_transforms_in_place(qapp, repo: SqliteRepo, positive_figure, model) -> None:
    figure_id, _axis_id = positive_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    result = dialog.compute_results()[0]

    assert result.y.size == N
    assert np.array_equal(result.x, np.arange(N, dtype=float))
    assert np.all(np.isfinite(result.y))


@pytest.mark.parametrize("model", TRANSFORM_MODELS)
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


def test_box_cox_succeeds_on_all_positive_data(qapp, repo: SqliteRepo, positive_figure) -> None:
    figure_id, _axis_id = positive_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(TRANSFORM_POWER)
    dialog.set_parameter_values({"method": POWER_BOX_COX})

    result = dialog.compute_results()[0]

    assert np.all(np.isfinite(result.y))
    assert result.metadata["method"] == POWER_BOX_COX


def test_box_cox_refuses_non_positive_data(qapp, repo: SqliteRepo, signed_figure) -> None:
    figure_id, _axis_id = signed_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(TRANSFORM_POWER)
    dialog.set_parameter_values({"method": POWER_BOX_COX})

    with pytest.raises(ValueError, match="[Pp]ositive"):
        dialog._transform_one(
            "sig",
            np.arange(5.0),
            np.array([-1.0, 1.0, 2.0, 3.0, 4.0]),
            TRANSFORM_POWER,
            {"method": POWER_BOX_COX},
        )


def test_yeo_johnson_accepts_signed_data(qapp, repo: SqliteRepo, signed_figure) -> None:
    figure_id, _axis_id = signed_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(TRANSFORM_POWER)

    result = dialog.compute_results()[0]

    assert np.all(np.isfinite(result.y))


def test_quantile_transform_clips_n_quantiles_to_the_sample_count(
    qapp, repo: SqliteRepo, positive_figure
) -> None:
    """The series here has N=60 points, well under the default 1000
    n_quantiles - scikit-learn raises if n_quantiles exceeds the sample
    count, so this must not raise."""
    figure_id, _axis_id = positive_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(TRANSFORM_QUANTILE)

    result = dialog.compute_results()[0]

    assert result.metadata["n_quantiles"] <= N
    assert np.all(np.isfinite(result.y))


def test_draws_on_a_new_axis_by_default(qapp, repo: SqliteRepo, positive_figure) -> None:
    figure_id, axis_id = positive_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(TRANSFORM_POWER)
    results = dialog.compute_results()

    resolved = dialog.resolve_target_axis_id(axis_id, results)

    assert resolved != axis_id
