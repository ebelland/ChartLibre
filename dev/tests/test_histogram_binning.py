"""Histogram bin methods: a typed count, or NumPy's rules worked out once over every data set."""
from __future__ import annotations

import numpy as np
import pytest

from app.charts.histogram import BIN_METHODS, MAX_BINS, shared_bins

RNG = np.random.default_rng(7)
A = RNG.normal(0, 1, 500)
B = RNG.normal(2, 1, 300)


def test_count_is_the_typed_number_and_the_default() -> None:
    assert shared_bins([A, B], "count", 20, None) == 20
    assert shared_bins([A, B], "count", 5000, None) == MAX_BINS  # never more than the most


@pytest.mark.parametrize("method", [m for m, estimator in BIN_METHODS.items() if estimator])
def test_each_rule_gives_numpys_edges_over_all_the_values(method: str) -> None:
    edges = shared_bins([A, B], method, 20, None)
    expected = np.histogram_bin_edges(np.concatenate([A, B]), bins=BIN_METHODS[method])
    assert isinstance(edges, np.ndarray) and np.allclose(edges, expected)


def test_the_rules_follow_the_range_and_nothing_breaks_on_too_little_data() -> None:
    edges = shared_bins([A, B], "freedman-diaconis", 20, (0.0, 2.0))
    assert edges[0] == 0.0 and edges[-1] == 2.0
    assert shared_bins([np.array([3.0, 3.0, 3.0])], "scott", 12, None) == 12  # no spread: the count
    assert shared_bins([np.array([1.0])], "auto", 12, None) == 12
    assert shared_bins([A], "no such rule", 12, None) == 12


def test_a_rule_asking_for_too_many_bins_is_capped() -> None:
    spike = np.concatenate([np.zeros(10_000), np.linspace(0, 1e6, 50)])  # IQR nearly zero: tiny bins
    edges = shared_bins([spike], "freedman-diaconis", 20, None)
    assert isinstance(edges, np.ndarray) and edges.size - 1 <= MAX_BINS


def test_the_renderer_draws_with_a_rule(qapp=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.charts.histogram import HistogramAxisRenderer
    from app.data.series_frame import SeriesFrame
    from app.plugin_api.charts.base import SeriesData

    figure, ax = plt.subplots()
    series = SeriesData(name="a", df=SeriesFrame({"value": A}), style={})
    HistogramAxisRenderer().render_axis(ax, [series], {"axis_kwargs": {"bin_method": "sturges"}})
    bars = [patch for patch in ax.patches]
    assert len(bars) == np.histogram_bin_edges(A, bins="sturges").size - 1
    plt.close(figure)
