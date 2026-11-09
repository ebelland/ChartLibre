"""A click on a line whose x are dates reads its point (it raised TypeError on datetime)."""
import datetime as dt
from types import SimpleNamespace

from matplotlib.figure import Figure


def test_line_xy_with_dates_are_numbers():
    fig = Figure()
    ax = fig.add_subplot()
    (line,) = ax.plot([dt.datetime(2020, 1, 1), dt.datetime(2020, 2, 1)], [1.0, 2.0])
    xy = line.get_xydata()
    assert isinstance(float(xy[1][0]), float) and float(xy[1][1]) == 2.0
    assert SimpleNamespace  # the panel reads the same get_xydata
