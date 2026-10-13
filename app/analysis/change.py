"""Quick answer: "Has it changed over time?" - a trend, a change point, before and after, in words (Q-03).

For someone who watches a measure over time and wants to know whether it
moved. Two questions a statistician would ask, each with the standard
nonparametric test - no assumption of normality, robust to a few odd values:

* a steady trend: the Mann-Kendall test, with Sen's slope (the median of
  every pairwise slope) for how fast;
* a sudden change: the Pettitt test, which finds the single point where the
  values before differ most from those after; the two sides are then
  compared (medians, Mann-Whitney).

The order is a time or numeric column, or the rows' own order. Values close
in time often resemble each other, which makes both tests see change too
eagerly: the answer says so when the lag-1 autocorrelation is high.
Sentences are templates (translated by the dialog) with values, as in
app.analysis.answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.analysis.answers import ALPHA, NO, YES

#: Above this many values the O(n²) trend statistics run on an even subsample.
MAX_PAIRS_N = 4000
#: Above this lag-1 autocorrelation (of the detrended values) the answer warns.
AUTOCORRELATION_WARNING = 0.3
ROW_ORDER = ""


@dataclass(slots=True)
class ChangeAnswer:
    question: str
    verdict: str
    sentences: list[tuple[str, dict[str, Any]]]
    #: The values in order: Order (as given), Value, Trend (Sen's line), Segment (Before/After), Segment mean.
    series: pd.DataFrame
    trend_p: float
    #: Sen's slope per unit of the order (per day for dates, per row for the row order).
    slope: float
    change_p: float
    #: Position (0-based, in order) of the last value before the change.
    change_index: int
    rows: int
    temporal: bool
    #: "drift" (a steady trend), "step" (a single change) or "" (no change shown).
    shape: str = ""


def mann_kendall(values: np.ndarray) -> tuple[float, float]:
    """(S, two-sided p) of the Mann-Kendall trend test, ties corrected."""
    x = np.asarray(values, dtype=float)
    n = x.size
    s = 0.0
    for i in range(n - 1):
        s += float(np.sign(x[i + 1:] - x[i]).sum())
    _unique, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    variance = (n * (n - 1) * (2 * n + 5) - float(np.sum(ties * (ties - 1) * (2 * ties + 5)))) / 18.0
    if variance <= 0:
        return s, 1.0
    z = (s - np.sign(s)) / np.sqrt(variance)
    return s, float(2.0 * stats.norm.sf(abs(z)))


def sen_slope(order: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """(slope, intercept): the median pairwise slope, the line through the medians."""
    result = stats.theilslopes(values, order)
    return float(result.slope), float(result.intercept)


def pettitt(values: np.ndarray) -> tuple[int, float]:
    """(last index before the change, approximate p) of the Pettitt test, from the ranks in O(n log n)."""
    x = np.asarray(values, dtype=float)
    n = x.size
    ranks = stats.rankdata(x)
    u = 2.0 * np.cumsum(ranks)[:-1] - np.arange(1, n) * (n + 1)
    at = int(np.argmax(np.abs(u)))
    k = float(np.abs(u[at]))
    p = float(min(1.0, 2.0 * np.exp(-6.0 * k * k / (n ** 3 + n ** 2))))
    return at, p


def _order(data: pd.DataFrame, time: str) -> tuple[np.ndarray, bool]:
    """The order as numbers (days for dates) and whether it is a date."""
    values = data[time]
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().all():
        return numbers.to_numpy(float), False
    dates = pd.to_datetime(values, errors="coerce", format="mixed")
    if dates.notna().all():
        return ((dates - dates.min()) / pd.Timedelta(days=1)).to_numpy(float), True
    raise ValueError(f"{time} is neither numbers nor dates.")


def has_changed(frame: pd.DataFrame, measure: str, time: str = ROW_ORDER) -> ChangeAnswer:
    """Has *measure* changed along *time* (a column, or the rows' order when empty)?"""
    columns = [measure, time] if time else [measure]
    data = frame[columns].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    if len(data) < 8:
        raise ValueError(f"{len(data)} values are too few to see a change: at least 8 are needed.")
    if time:
        order, temporal = _order(data, time)
        sort = np.argsort(order, kind="stable")
        data, order = data.iloc[sort].reset_index(drop=True), order[sort]
    else:
        order, temporal = np.arange(len(data), dtype=float), False
    values = data[measure].to_numpy(float)
    n = values.size

    pick = np.linspace(0, n - 1, MAX_PAIRS_N).round().astype(int) if n > MAX_PAIRS_N else np.arange(n)
    _s, trend_p = mann_kendall(values[pick])
    slope, intercept = sen_slope(order[pick], values[pick])
    at, change_p = pettitt(values)
    before, after = values[: at + 1], values[at + 1:]
    compare_p = float(stats.mannwhitneyu(before, after, alternative="two-sided").pvalue) \
        if before.size and after.size else 1.0
    trend_line = intercept + slope * order
    step_line = np.where(np.arange(n) <= at, before.mean(), after.mean() if after.size else before.mean())
    trend = trend_p < ALPHA
    change = change_p < ALPHA and compare_p < ALPHA
    # Both tests react to both shapes - a steady drift has a "best" change
    # point, a step has a trend. When both speak, the shape that fits the
    # values better is the answer.
    steps = change and (not trend or np.sum((values - step_line) ** 2) < np.sum((values - trend_line) ** 2))
    drifts = trend and not steps
    residual = values - (step_line if steps else trend_line)
    autocorrelation = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if np.ptp(residual) > 0 else 0.0

    label = time
    when_values = data[time] if time else pd.Series(np.arange(1, n + 1))
    unit = "day" if temporal else ("row" if not time else "unit")
    span = float(order[-1] - order[0]) if n > 1 else 0.0
    before_median, after_median = float(np.median(before)), float(np.median(after)) if after.size else float("nan")
    values_dict: dict[str, Any] = {
        "measure": measure, "time": label, "slope": slope, "unit": unit,
        "p_trend": trend_p, "p_change": change_p, "n": n, "total": slope * span,
        "when": str(when_values.iloc[at + 1]) if at + 1 < n else str(when_values.iloc[-1]),
        "before": before_median, "after": after_median,
        "percent": (after_median - before_median) / abs(before_median) * 100.0 if before_median else float("nan"),
    }
    sentences: list[tuple[str, dict[str, Any]]] = []

    def along(template: str) -> str:
        """The row-order wording when there is no time column."""
        return template if time else template.replace("over “{time}”", "along the rows")

    if drifts:
        verdict = YES
        sentences.append((along("Yes: “{measure}” rises over “{time}”, by about {slope:.3g} per {unit} ({total:+.3g} "
                                "over the whole span)." if slope > 0 else
                                "Yes: “{measure}” falls over “{time}”, by about {slope:.3g} per {unit} ({total:+.3g} "
                                "over the whole span)."), values_dict))
        if change:
            sentences.append(("The values drift rather than jump: a steady trend fits them better than a single "
                              "change.", values_dict))
    elif steps:
        verdict = YES
        sentences.append(("Yes: it changed around {when}: the median of “{measure}” went from {before:.4g} to "
                          "{after:.4g} ({percent:+.0f}%)." if np.isfinite(values_dict["percent"]) else
                          "Yes: it changed around {when}: the median of “{measure}” went from {before:.4g} to "
                          "{after:.4g}.", values_dict))
        if trend:
            sentences.append(("The values jump rather than drift: a single change fits them better than a steady "
                              "trend.", values_dict))
    else:
        verdict = NO
        sentences.append((along("No change is shown: these data do not say that “{measure}” has changed over "
                                "“{time}”."), values_dict))
        sentences.append(("That is not proof that it stayed the same: with more values a small change might "
                          "show.", values_dict))
    sentences.append(("How it was decided: the Mann-Kendall trend test (p = {p_trend:.3g}) and the Pettitt "
                      "change-point test (p = {p_change:.3g}), on {n} values; below 0.05 is called a change.",
                      values_dict))
    if n > MAX_PAIRS_N:
        sentences.append(("The trend was measured on {count} values spread evenly over the {n}.",
                          {**values_dict, "count": MAX_PAIRS_N}))
    if autocorrelation > AUTOCORRELATION_WARNING:
        sentences.append(("Values close in time resemble each other (autocorrelation {r:.2f}): that makes both "
                          "tests see change more readily than they should - treat the answer with caution.",
                          {**values_dict, "r": autocorrelation}))

    segment = np.where(np.arange(n) <= at, "Before", "After")
    series = pd.DataFrame({
        "Order": data[time].to_numpy() if time else np.arange(1, n + 1),
        "Value": values,
        "Trend": trend_line,
        "Segment": segment,
        "Segment median": np.where(segment == "Before", before_median, after_median),
    })
    return ChangeAnswer("Has it changed over time?", verdict, sentences, series, trend_p, slope, change_p, at, n,
                        temporal, "drift" if drifts else "step" if steps else "")
