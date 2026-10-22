"""Control charts on a date column stored as text: every layout reads in the right order, and readings
sharing a date are all kept."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.utils.coercion import parse_datetimes, to_numeric_axis


def test_a_column_with_a_day_above_twelve_is_read_day_first() -> None:
    days = pd.date_range("2024-01-01", periods=40, freq="D")
    parsed = parse_datetimes(pd.Series(days.strftime("%d/%m/%Y")))
    assert parsed.tolist() == days.tolist()
    assert (parsed.diff().dropna() == pd.Timedelta(days=1)).all()


def test_a_month_first_column_is_still_month_first() -> None:
    days = pd.date_range("2024-01-01", periods=40, freq="D")
    assert parse_datetimes(pd.Series(days.strftime("%m/%d/%Y"))).tolist() == days.tolist()


def test_dots_and_dashes_and_two_digit_years_follow_the_same_rule() -> None:
    assert parse_datetimes(pd.Series(["13.02.2024", "14.02.2024", "01.03.2024"])).dt.day.tolist() == [13, 14, 1]
    assert parse_datetimes(pd.Series(["25-12-2024", "01-01-2025"])).dt.month.tolist() == [12, 1]


def test_an_all_ambiguous_column_is_left_as_pandas_reads_it() -> None:
    assert parse_datetimes(pd.Series(["01/02/2024", "03/04/2024"])).dt.month.tolist() == [1, 3]


def test_iso_and_named_month_dates_are_unaffected() -> None:
    assert parse_datetimes(pd.Series(["2024-02-13", "2024-03-01"])).dt.day.tolist() == [13, 1]
    assert parse_datetimes(pd.Series(["Feb 13 2024", "Mar 1 2024"])).dt.month.tolist() == [2, 3]
    assert np.all(np.diff(to_numeric_axis(pd.Series(["13/02/2024", "14/02/2024", "15/02/2024"]))) == 86400)


@pytest.fixture
def readings() -> np.ndarray:
    return np.round(np.random.default_rng(1).normal(10, 0.5, 40), 3)


def _chart(repo, dates: list[str], values: np.ndarray, shuffle: bool = False) -> tuple[int, str]:
    import re

    from dev.demo.demo_operations import OperationDemo, _create_source_figure, _run_operation

    frame = pd.DataFrame({"d": dates, "v": values})
    if shuffle:
        frame = frame.sample(frac=1, random_state=2).reset_index(drop=True)
    repo.import_dataframe(frame, table_name="t", normalize_columns=False)
    demo = OperationDemo("cc", "cc", "t", "SELECT d AS x, v AS y FROM t", {"x": "x", "y": "y"}, "d", "v",
                         "control_chart_dialog:SeriesControlChartDialog")
    html = _run_operation(repo, _create_source_figure(repo, demo), demo)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    points = int(re.search(r"Points plotted (\d+)", text).group(1))  # type: ignore[union-attr]
    return points, re.search(r"Upper control limit ([\d.]+)", text).group(1)  # type: ignore[union-attr]


def test_the_chart_keeps_every_reading_when_dates_repeat(qapp, repo, readings) -> None:
    dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=20, freq="D"), 2)).dt.strftime("%Y-%m-%d")
    assert _chart(repo, dates.tolist(), readings)[0] == 40


def test_the_chart_is_the_same_for_iso_and_day_first_text_dates(qapp, repo, readings, tmp_path) -> None:
    from app.data.sqlite_repo import SqliteRepo

    days = pd.date_range("2024-01-01", periods=40, freq="D")
    other = SqliteRepo(db_path=tmp_path / "other.dhub")
    try:
        iso = _chart(repo, days.strftime("%Y-%m-%d").tolist(), readings, shuffle=True)
        day_first = _chart(other, days.strftime("%d/%m/%Y").tolist(), readings, shuffle=True)
        assert iso == day_first and iso[0] == 40
    finally:
        other.close()
