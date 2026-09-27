"""boxplot.notch, showmeans and meanline in a style reach the box plot.

The renderer draws with ax.bxp, which never reads rcParams, and used to pass
its own False for each - so the style editor's switches changed nothing.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app.charts.base import SeriesData  # noqa: E402
from app.charts.box import BoxAxisRenderer  # noqa: E402
from app.data.series_frame import SeriesFrame  # noqa: E402


def _draw(options: dict, rc: dict):
    with mpl.rc_context(rc):
        ax = Figure().add_subplot()
        frame = SeriesFrame({"value": np.random.default_rng(1).normal(0, 1, 60)})
        BoxAxisRenderer().render_axis(ax, [SeriesData(name="s", df=frame, style={})], options)
    return ax


def _mean_lines(ax) -> int:
    return sum(1 for line in ax.lines if line.get_linestyle() == "--")


def test_the_style_decides_when_the_chart_does_not() -> None:
    plain = _draw({}, {})
    styled = _draw({}, {"boxplot.showmeans": True, "boxplot.meanline": True, "boxplot.notch": True})
    assert _mean_lines(plain) == 0
    assert _mean_lines(styled) == 1
    # A notched box has more vertices than a plain rectangle.
    assert len(styled.patches[0].get_path().vertices) > len(plain.patches[0].get_path().vertices)


def test_the_chart_s_own_choice_wins_over_the_style() -> None:
    ax = _draw({"showmeans": False}, {"boxplot.showmeans": True, "boxplot.meanline": True})
    assert _mean_lines(ax) == 0
