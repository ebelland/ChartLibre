"""The calculations the table dialogs used to make inline now live in app.analysis: rankings, the biplot's
scale and label spacing, the time gap of a line chart, the strongest lags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analysis import linear_model as lm
from app.analysis import multivariate as mv
from app.analysis import partition as pt
from app.analysis import pls as pl
from app.analysis import time_series as tsa


def test_strongest_pairs_puts_the_smallest_p_first_and_missing_last() -> None:
    pairs = pd.DataFrame({"Pair": ["a", "b", "c"], "Prob>|t|": [0.2, np.nan, 0.01]})
    assert list(mv.strongest_pairs(pairs)["Pair"]) == ["c", "a", "b"]


def test_biplot_scale_fits_the_loadings_to_the_scores_spread() -> None:
    loadings = np.array([[0.5, 0.0], [0.0, -1.0]])
    scores = np.array([[2.0, -4.0], [1.0, 3.0]])
    scale, reach = mv.biplot_scale(loadings, scores)
    assert reach == 4.0
    assert np.isclose(scale, 0.85 * 4.0 / 1.0)


def test_spread_labels_moves_labels_that_would_cover_each_other_apart() -> None:
    tips = np.array([[1.0, 1.0], [1.0, 1.01], [-1.0, 0.0]])
    labels = mv.spread_labels(tips, reach=1.0)
    assert len(labels) == 3
    (x0, y0), (x1, y1), _ = labels
    # Clear of each other: not both closer than the label's width and its height (reach 1.0).
    assert abs(x0 - x1) >= 0.25 or abs(y0 - y1) >= 0.09 - 1e-9


def test_rankings_put_the_most_important_first() -> None:
    vip = pd.DataFrame({"Variable": ["a", "b"], "VIP": [0.7, 1.4]})
    assert pl.ranked_vip(vip)["Variable"].iloc[0] == "b"

    contributions = pd.DataFrame({"Column": ["a", "b"], "Portion": [0.2, 0.6]})
    assert pt.ranked_contributions(contributions)["Column"].iloc[0] == "b"

    effects = pd.DataFrame({"Term": ["a", "b", "c"], "LogWorth": [1.0, np.nan, 3.0]})
    ranked = lm.ranked_effects(effects, "LogWorth")
    assert list(ranked["Term"]) == ["c", "a", "b"]


def test_gap_options_is_five_typical_steps_for_dated_rows() -> None:
    daily = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D").astype(str))
    assert tsa.gap_options(daily) == {"gap_threshold": f"{5 * 86400}s"}
    assert tsa.gap_options(daily.iloc[:1]) == {}


def test_strongest_lags_keeps_the_largest_absolute_ccf_first() -> None:
    cross = pd.DataFrame({"Lag": [0, 1, 2], "Prewhitened CCF": [0.1, -0.9, 0.4], "CCF": [0.1, -0.9, 0.4]})
    assert list(tsa.strongest_lags(cross, count=2)["Lag"]) == [1, 2]
