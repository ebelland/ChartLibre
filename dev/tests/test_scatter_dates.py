"""A scatter plot of dated x draws the points on a date axis."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app.charts.base import SeriesData  # noqa: E402
from app.charts.scatter import ScatterAxisRenderer  # noqa: E402
from app.data.series_frame import SeriesFrame  # noqa: E402


def test_dates_as_text_are_drawn_as_dates() -> None:
    """Dates stored as text, the way SQLite hands them back: every one used
    to become NaN in pd.to_numeric and the series was dropped."""
    ax = Figure().add_subplot()
    frame = SeriesFrame({"x": ["2015-01-02", "2015-01-05", "2015-01-06"], "y": [109.3, 106.2, 106.3]})
    ScatterAxisRenderer().render_axis(ax, [SeriesData(name="close", df=frame, style={})], {})

    points = ax.collections[0].get_offsets()
    assert len(points) == 3
    days = np.array(["2015-01-02", "2015-01-05", "2015-01-06"], dtype="datetime64[D]")
    expected = [mdates.date2num(day) for day in days]
    np.testing.assert_allclose(np.asarray(points)[:, 0], expected)
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)


def test_numbers_stay_numbers() -> None:
    ax = Figure().add_subplot()
    frame = SeriesFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    ScatterAxisRenderer().render_axis(ax, [SeriesData(name="s", df=frame, style={})], {})
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_offsets())[:, 0], [1.0, 2.0, 3.0])
    assert not isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)


def _empty_x() -> SeriesFrame:
    """What a series whose query matched nothing hands the renderer."""
    return SeriesFrame({"x": np.array([], dtype=object), "y": np.array([], dtype=object)})


def test_a_series_with_no_rows_does_not_turn_the_axis_into_dates() -> None:
    """AMC Growth: one series' filter matched no row (a stray space in its
    condition); the axis went to dates and the real series, with x in minutes,
    read as days since 1970."""
    ax = Figure().add_subplot()
    full = SeriesFrame({"x": [0.0, 30.0, 60.0], "y": [0.0, 0.02, 0.03]})
    series = [SeriesData(name="gone", df=_empty_x(), style={}), SeriesData(name="s", df=full, style={})]
    ScatterAxisRenderer().render_axis(ax, series, {})
    assert not isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    assert not isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_offsets())[:, 0], [0.0, 30.0, 60.0])


def test_all_null_x_stays_numeric() -> None:
    from app.utils.coercion import coerce_axis

    frame = SeriesFrame.from_rows(("x", "y"), [(None, 1), (None, 2)])
    assert coerce_axis(frame["x"])[1] is False
    assert coerce_axis(SeriesFrame({"x": ["a", "b"]})["x"])[1] is False  # not numbers, not dates


def test_a_skipped_series_with_dates_does_not_date_the_axis() -> None:
    """Dated x but no y to pair them with: nothing is drawn from it, so it
    must not decide how the other series' x is shown."""
    ax = Figure().add_subplot()
    dated_no_y = SeriesFrame({"x": ["2015-01-02", "2015-01-05"], "y": [np.nan, np.nan]})
    full = SeriesFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    series = [SeriesData(name="a", df=dated_no_y, style={}), SeriesData(name="b", df=full, style={})]
    ScatterAxisRenderer().render_axis(ax, series, {})
    assert not isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)


def test_markersize_is_in_points_per_series_and_axis() -> None:
    frame = SeriesFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    ax = Figure().add_subplot()
    ScatterAxisRenderer().render_axis(
        ax, [SeriesData(name="s", df=frame, style={"markersize": 8.0})], {}
    )
    np.testing.assert_allclose(ax.collections[0].get_sizes(), [64.0])  # area = size squared

    ax = Figure().add_subplot()  # the axis' value, and a series' own over it
    series = [
        SeriesData(name="a", df=frame, style={}),
        SeriesData(name="b", df=frame, style={"markersize": 3.0}),
    ]
    ScatterAxisRenderer().render_axis(ax, series, {"axis_kwargs": {"markersize": 5.0}})
    np.testing.assert_allclose(ax.collections[0].get_sizes(), [25.0])
    np.testing.assert_allclose(ax.collections[1].get_sizes(), [9.0])
