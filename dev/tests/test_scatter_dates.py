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
