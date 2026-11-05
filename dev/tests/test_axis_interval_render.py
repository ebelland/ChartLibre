"""Scale: Interval - a major tick every step on a linear axis; left automatic where a step means nothing."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from app.plugin_api.charts import axis_options  # noqa: E402
from app.plugin_api.charts.render_figure import _apply_intervals  # noqa: E402


def test_the_interval_is_read_as_a_positive_step() -> None:
    assert axis_options.manual_interval({"x_interval": "2.5"}, "x") == 2.5
    for no_step in (None, "", 0, -1, "ten", float("inf")):
        assert axis_options.manual_interval({"x_interval": no_step}, "x") is None
    assert axis_options.defaults()["y_interval"] is None


def test_ticks_every_interval_on_a_linear_axis() -> None:
    figure, ax = plt.subplots()
    ax.plot([0, 10], [0, 100])
    ax.set_xlim(0, 10)
    _apply_intervals(ax, {"x_interval": 2.5})
    assert isinstance(ax.xaxis.get_major_locator(), MultipleLocator)
    shown = [tick for tick in ax.get_xticks() if 0 <= tick <= 10]  # Matplotlib keeps one beyond each end
    assert np.allclose(shown, [0, 2.5, 5, 7.5, 10])
    assert not isinstance(ax.yaxis.get_major_locator(), MultipleLocator)  # y left automatic
    plt.close(figure)


def test_no_interval_on_a_log_axis_categories_or_too_many_ticks() -> None:
    figure, ax = plt.subplots()
    ax.bar(["a", "b", "c"], [1, 10, 100])
    ax.set_yscale("log")
    _apply_intervals(ax, {"x_interval": 1, "y_interval": 5})
    assert not isinstance(ax.xaxis.get_major_locator(), MultipleLocator)  # categories keep their ticks
    assert not isinstance(ax.yaxis.get_major_locator(), MultipleLocator)  # a log scale keeps its own
    plt.close(figure)

    figure, ax = plt.subplots()
    ax.set_xlim(0, 1e6)
    _apply_intervals(ax, {"x_interval": 1})  # a million ticks: left automatic
    assert not isinstance(ax.xaxis.get_major_locator(), MultipleLocator)
    plt.close(figure)
