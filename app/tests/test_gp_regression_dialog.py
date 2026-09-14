"""Gaussian process regression (todo.txt P2-15).

The one regression-shaped operation in this app whose whole point is that it
returns its own uncertainty: one call produces a mean curve plus a +/-2 sigma
band, as three series from one saved table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.gp_regression_dialog import (
    GP_KERNELS,
    SeriesGPRegressionDialog,
)
from app.utils.coercion import parse_datetimes


def _dialog(cls, repo: SqliteRepo, figure_id: int):
    dialog = cls(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


@pytest.fixture
def smooth_figure(repo: SqliteRepo):
    x = np.linspace(0.0, 10.0, 40)
    y = np.sin(x) + np.random.RandomState(0).normal(scale=0.05, size=x.size)
    repo.import_dataframe(
        pd.DataFrame({"x": x, "y": y}), table_name="w", normalize_columns=False
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


@pytest.fixture
def dated_figure(repo: SqliteRepo):
    days = pd.date_range("2024-01-01", periods=40, freq="D")
    y = np.sin(np.arange(40) / 5.0)
    repo.import_dataframe(
        pd.DataFrame({"t": days.strftime("%Y-%m-%d %H:%M:%S"), "v": y}),
        table_name="ts", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Time Series",
            title="ts", x_label="t", y_label="v", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT t AS x, v AS y FROM ts", roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


@pytest.mark.parametrize("kernel", GP_KERNELS)
def test_each_kernel_produces_a_band_around_the_mean(
    qapp, repo: SqliteRepo, smooth_figure, kernel: str
) -> None:
    figure_id, _axis_id = smooth_figure
    dialog = _dialog(SeriesGPRegressionDialog, repo, figure_id)
    dialog.model_combo.setCurrentText(kernel)

    result = dialog.compute_results()[0]

    assert len(result.x) == len(result.mean) == len(result.upper) == len(result.lower)
    assert np.all(result.upper >= result.mean)
    assert np.all(result.mean >= result.lower)


def test_apply_writes_the_mean_and_both_band_series(
    qapp, repo: SqliteRepo, smooth_figure
) -> None:
    figure_id, axis_id = smooth_figure
    dialog = _dialog(SeriesGPRegressionDialog, repo, figure_id)

    assert dialog.apply()

    new_axis_id = dialog._result_axis_id
    assert new_axis_id is not None
    assert new_axis_id != axis_id
    names = sorted(str(row["name"]) for row in repo.get_series(new_axis_id))
    assert names == ["sig - GP +2σ", "sig - GP -2σ", "sig - GP mean"]


def test_a_dated_series_is_written_back_as_dates(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    figure_id, _axis_id = dated_figure
    dialog = _dialog(SeriesGPRegressionDialog, repo, figure_id)

    assert dialog.apply()

    table = dialog.result_table_name(dialog._result_axis_id, dialog._last_results[0])
    stored = repo.query_df(f'SELECT x FROM "{table}"')
    parsed = parse_datetimes(stored["x"])
    assert parsed.notna().all()
    assert parsed.iloc[0].year == 2024
