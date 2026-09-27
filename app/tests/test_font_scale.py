"""todo N-12: one factor for every font of a figure, and one per axis."""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from app.charts.render_figure import _apply_font_scale, font_scale_option  # noqa: E402


def _figure_with_one_axis():
    figure = Figure()
    ax = figure.add_subplot()
    ax.set_title("t", fontsize=10)
    ax.set_xlabel("x", fontsize=10)
    ax.plot([0, 1], [0, 1], label="line")
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    figure.suptitle("s", fontsize=10)
    return figure, ax


def test_the_figure_and_axis_factors_multiply() -> None:
    figure, ax = _figure_with_one_axis()
    axis = SimpleNamespace(id=1, options={"font_scale": 1.5})
    descriptor = SimpleNamespace(options={"font_scale": 2.0})

    _apply_font_scale(figure, descriptor, [axis], {1: ax})

    assert ax.title.get_fontsize() == 30
    assert ax.xaxis.label.get_fontsize() == 30
    assert ax.get_legend().get_texts()[0].get_fontsize() == 30
    assert ax.xaxis.get_major_ticks()[0].label1.get_fontsize() == 30
    assert figure._suptitle.get_fontsize() == 20


def test_unset_or_silly_values_mean_no_change() -> None:
    assert font_scale_option({}) == 1.0
    assert font_scale_option({"font_scale": "big"}) == 1.0
    assert font_scale_option({"font_scale": 99}) == 3.0
