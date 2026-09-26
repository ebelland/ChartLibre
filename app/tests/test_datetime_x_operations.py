"""Series operations on a dated x axis.

``pd.to_numeric`` turns a timestamp column into all-NaN, so an operation
reading its x that way reports "0 usable points" about a series of a
hundred perfectly good ones. ``app/utils/coercion`` was written for
precisely this - its own docstring says so - and only the outlier dialog
was calling it. Everything else broke on every dated series, which is a
lot of series: a time axis is what half of this application draws.

Two halves are tested here, because fixing only the first produces a
result that is right and unreadable:

*Reading* - the operation computes, instead of refusing a full table.
*Writing back* - the result's x returns as timestamps, so it lands on the
axis it came from rather than at x = 1 700 000 000 next to a chart drawn
in dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils.coercion import coerce_axis, parse_datetimes

DAYS = 60


@pytest.fixture
def dated_figure(repo: SqliteRepo):
    """One figure, one axis, one series whose x is an ISO date column."""
    days = pd.date_range("2024-01-01", periods=DAYS, freq="D")
    repo.import_dataframe(
        pd.DataFrame(
            {
                "t": days.strftime("%Y-%m-%d %H:%M:%S"),
                "v": np.sin(np.arange(DAYS) / 5.0),
            }
        ),
        table_name="ts",
        normalize_columns=False,
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
        sql_query="SELECT t AS x, v AS y FROM ts",
        roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


# ----------------------------------------------------------------------
# The parsing underneath, which had its own bug
# ----------------------------------------------------------------------
def test_a_column_of_mixed_precision_timestamps_parses_completely() -> None:
    """pandas infers one format from the first entry and coerces the rest to
    NaT. Computed timestamps come back from SQLite with fractional seconds
    on some rows and not others, so this is not a hypothetical - it is what
    every operation result looked like."""
    values = pd.Series(
        ["2024-01-01 00:00:00", "2024-01-16 16:58:38.426808", "2024-02-01 09:59:19.5"]
    )

    parsed = parse_datetimes(values)

    assert parsed.notna().all()
    assert parsed.iloc[1].microsecond == 426808


def test_a_numeric_axis_is_never_read_as_nanoseconds() -> None:
    """The reason coerce_axis tries numbers first: an x running 0..4000 read
    as timestamps is four microseconds of 1 January 1970."""
    coerced, is_temporal = coerce_axis(pd.Series([0.0, 1000.0, 4000.0]))

    assert is_temporal is False
    assert list(coerced) == [0.0, 1000.0, 4000.0]


# ----------------------------------------------------------------------
# Reading: the operations compute at all
# ----------------------------------------------------------------------
def _dialog(cls, repo: SqliteRepo, figure_id: int):
    dialog = cls(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


# ----------------------------------------------------------------------
# Calculus on dates: seconds-since-epoch is correct but unreadable
# (todo.txt P-01) - a derivative/integral against raw seconds is ~86400x
# smaller/larger than what daily data means intuitively.
# ----------------------------------------------------------------------
def test_a_derivative_of_a_dated_series_is_scaled_per_day_not_per_second(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    from app.series_operations.calculus_dialog import (
        DERIV_GRADIENT,
        SeriesCalculusDialog,
    )

    figure_id, _axis_id = dated_figure
    dialog = _dialog(SeriesCalculusDialog, repo, figure_id)
    dialog.model_combo.setCurrentText(DERIV_GRADIENT)
    result = dialog.compute_results()[0]

    # sin(n/5)'s slope is O(0.1-0.2) per day; per second it would be
    # ~86400x smaller, around 1e-6 - comfortably below this floor.
    assert np.abs(result.y).max() > 0.01
    assert "day" in result.result_name
    assert result.metadata.get("per") == "day"


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("control_chart_dialog", "SeriesControlChartDialog"),
        ("outlier_dialog", "SeriesOutlierDialog"),
        ("smoothing_dialog", "SeriesSmoothingDialog"),
        ("spectral_dialog", "SeriesSpectralDialog"),
        ("statistics_dialog", "SeriesStatisticsDialog"),
        ("cluster_dialog", "SeriesClusterDialog"),
    ],
)
def test_every_other_operation_produces_a_result_too(
    qapp, repo: SqliteRepo, dated_figure, module_name: str, class_name: str
) -> None:
    """Each of these read x with pd.to_numeric and refused the series."""
    import importlib

    figure_id, _axis_id = dated_figure
    cls = getattr(
        importlib.import_module(f"app.series_operations.{module_name}"), class_name
    )

    results = _dialog(cls, repo, figure_id).compute_results()

    assert results, f"{class_name} produced nothing from a dated series"


# ----------------------------------------------------------------------
# Writing back: the result lands on the axis it came from
# ----------------------------------------------------------------------
def test_a_result_computed_from_dates_is_written_back_as_dates(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    from app.series_operations.roots_dialog import SeriesRootsDialog

    figure_id, axis_id = dated_figure
    dialog = _dialog(SeriesRootsDialog, repo, figure_id)
    results = dialog.compute_results()

    dialog.apply_results_to_axis(axis_id, results)

    stored = repo.query_df(
        f'SELECT x FROM "{dialog.result_table_name(axis_id, results[0])}"'
    )
    parsed = parse_datetimes(stored["x"])
    assert parsed.notna().all(), "the x column did not come back as timestamps"
    assert parsed.iloc[0].year == 2024


