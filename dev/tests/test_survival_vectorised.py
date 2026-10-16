"""Survival: the vectorised Kaplan-Meier and log-rank give what the plain loops gave (PF-1).

The loops scanned every row for every event time - O(rows x event times), a
second for a log-rank test on 20,000 distinct times. The reference below is
those loops, kept here to check the faster code against.
"""
from __future__ import annotations

import time as clock

import numpy as np
import pytest
from scipy import stats

from app.analysis.diagnostics import kaplan_meier, log_rank
from app.plugin_api.charts.base import SeriesData
from app.charts.survival import KaplanMeierAxisRenderer
from app.data.series_frame import SeriesFrame


def _reference_km(time: np.ndarray, event: np.ndarray):
    observed = event != 0
    distinct = np.unique(time[observed])
    at_risk = np.array([np.sum(time >= t) for t in distinct], dtype=float)
    deaths = np.array([np.sum((time == t) & observed) for t in distinct], dtype=float)
    return distinct, at_risk, deaths, np.cumprod(1.0 - deaths / at_risk)


def _reference_log_rank(groups):
    all_times = np.concatenate([t for t, _e in groups])
    all_events = np.concatenate([e != 0 for _t, e in groups])
    k = len(groups)
    obs_exp = np.zeros(k)
    cov = np.zeros((k, k))
    for t in np.unique(all_times[all_events]):
        at_risk = np.array([np.sum(times >= t) for times, _e in groups], dtype=float)
        deaths = np.array([np.sum((times == t) & (events != 0)) for times, events in groups], dtype=float)
        tr, td = at_risk.sum(), deaths.sum()
        obs_exp += deaths - td * at_risk / tr
        if tr > 1:
            share = at_risk / tr
            cov += td * (tr - td) / (tr - 1) * (np.diag(share) - np.outer(share, share))
    r = obs_exp[:-1]
    return float(r @ np.linalg.solve(cov[:-1, :-1], r))


def _data(rng: np.random.Generator, n: int, decimals: int | None):
    time = rng.exponential(10.0, n)
    if decimals is not None:
        time = np.round(time, decimals)  # few decimals: many ties
    return time, (rng.random(n) < 0.7).astype(float)


@pytest.mark.parametrize("decimals", [0, 1, None])
def test_kaplan_meier_matches_the_loops(decimals: int | None) -> None:
    time, event = _data(np.random.default_rng(1), 400, decimals)
    curve = kaplan_meier(time, event)
    distinct, at_risk, deaths, survival = _reference_km(time, event)
    np.testing.assert_allclose(curve.time[1:], distinct)
    np.testing.assert_allclose(curve.at_risk[1:], at_risk)
    np.testing.assert_allclose(curve.events[1:], deaths)
    np.testing.assert_allclose(curve.survival[1:], survival)
    # The height at a censored time is the last step at or before it (1 before the first).
    for t, height in zip(curve.censored_time, curve.censored_survival):
        before = survival[distinct <= t]
        assert height == (before[-1] if before.size else 1.0)


@pytest.mark.parametrize("decimals", [0, None])
def test_log_rank_matches_the_loops(decimals: int | None) -> None:
    rng = np.random.default_rng(2)
    groups = [_data(rng, 150, decimals) for _ in range(3)]
    result = log_rank(groups)
    expected = _reference_log_rank(groups)
    assert result.statistic == pytest.approx(expected, rel=1e-9)
    assert result.dof == 2
    assert result.pvalue == pytest.approx(stats.chi2.sf(expected, 2))


def test_log_rank_without_events_still_says_so() -> None:
    groups = [(np.array([1.0, 2.0, 3.0]), np.zeros(3)), (np.array([1.5, 2.5]), np.zeros(2))]
    with pytest.raises(ValueError, match="no events"):
        log_rank(groups)


def test_large_data_is_fast() -> None:
    rng = np.random.default_rng(3)
    groups = [_data(rng, 30_000, None) for _ in range(3)]
    start = clock.perf_counter()
    kaplan_meier(*groups[0])
    log_rank(groups)
    assert clock.perf_counter() - start < 2.0  # the loops took ~20 s here


def test_groups_become_curves_in_order_of_appearance_and_missing_ones_are_skipped() -> None:
    frame = SeriesFrame({
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "event": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "group": np.array(["b", "a", "b", None, "a", float("nan")], dtype=object),
    })
    curves = KaplanMeierAxisRenderer()._curves([SeriesData(name="s", df=frame, style={}, roles={})])
    assert [label for label, *_ in curves] == ["b", "a"]
    assert curves[0][2].tolist() == [1.0, 3.0] and curves[1][2].tolist() == [2.0, 5.0]
