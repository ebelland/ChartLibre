"""Control charts: the arithmetic behind the Control chart operation.

Shewhart charts for measurements (individuals and moving range, X-bar with R
or S) and for counts (p, np, c, u), the Nelson rules that flag what stays
inside the limits, and the second pass that recomputes the limits without the
flagged points. Plain arrays in, no Qt, nothing logged (todo R-01): what the
caller should tell the user comes back as notes.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Variables charts - a measurement per point.
CHART_INDIVIDUALS = "Individuals (I-MR)"
CHART_MOVING_RANGE = "Moving range (MR)"
CHART_XBAR_R = "X-bar and R"
CHART_XBAR_S = "X-bar and S"

# Attribute charts - a count per point.
CHART_P = "p (fraction defective)"
CHART_NP = "np (count defective)"
CHART_C = "c (defects per unit)"
CHART_U = "u (defects per unit, variable size)"


#: Unbiasing constants by subgroup size, from the standard SPC tables.
#: n -> (d2, d3, c4, A2, D3, D4, B3, B4)
#:
#: d2/d3 relate the mean range to sigma; c4 does the same for the mean standard
#: deviation. A2, D3, D4, B3, B4 are the shortcuts that fold those into limit
#: formulas directly, and are what the published tables give.
SPC_CONSTANTS: dict[int, tuple[float, float, float, float, float, float, float, float]] = {
    2:  (1.128, 0.853, 0.7979, 1.880, 0.000, 3.267, 0.000, 3.267),
    3:  (1.693, 0.888, 0.8862, 1.023, 0.000, 2.574, 0.000, 2.568),
    4:  (2.059, 0.880, 0.9213, 0.729, 0.000, 2.282, 0.000, 2.266),
    5:  (2.326, 0.864, 0.9400, 0.577, 0.000, 2.114, 0.000, 2.089),
    6:  (2.534, 0.848, 0.9515, 0.483, 0.000, 2.004, 0.030, 1.970),
    7:  (2.704, 0.833, 0.9594, 0.419, 0.076, 1.924, 0.118, 1.882),
    8:  (2.847, 0.820, 0.9650, 0.373, 0.136, 1.864, 0.185, 1.815),
    9:  (2.970, 0.808, 0.9693, 0.337, 0.184, 1.816, 0.239, 1.761),
    10: (3.078, 0.797, 0.9727, 0.308, 0.223, 1.777, 0.284, 1.716),
    11: (3.173, 0.787, 0.9754, 0.285, 0.256, 1.744, 0.321, 1.679),
    12: (3.258, 0.778, 0.9776, 0.266, 0.283, 1.717, 0.354, 1.646),
    13: (3.336, 0.770, 0.9794, 0.249, 0.307, 1.693, 0.382, 1.618),
    14: (3.407, 0.763, 0.9810, 0.235, 0.328, 1.672, 0.406, 1.594),
    15: (3.472, 0.756, 0.9823, 0.223, 0.347, 1.653, 0.428, 1.572),
    20: (3.735, 0.729, 0.9869, 0.180, 0.415, 1.585, 0.510, 1.490),
    25: (3.931, 0.709, 0.9896, 0.153, 0.459, 1.541, 0.565, 1.435),
}


@dataclass(slots=True)
class Violation:
    """Every rule broken at one point.

    All of them, not just the first: the rule numbers are historical, not a
    severity ranking, so picking one to report means picking arbitrarily. A
    stretch of points that both sits on one side of the centre (rule 2) and
    hugs it (rule 7) is telling you two different things, and reporting only
    the lower-numbered one hides the more interesting half.
    """

    index: int
    x: float
    y: float
    rules: tuple[int, ...]
    descriptions: tuple[str, ...]

    @property
    def rule(self) -> int:
        """The lowest rule number, for sorting and for a compact display."""
        return min(self.rules) if self.rules else 0


def attribute_limits(
    chart: str,
    counts: np.ndarray,
    sizes: np.ndarray,
    sigma_limit: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(plotted_statistic, centre, upper, lower, metadata)``.

    ``counts`` is the series' y (defectives for p/np, defects for c/u).
    ``sizes`` is the sample size per point; the c chart ignores it.
    """
    counts = np.asarray(counts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    point_count = counts.size
    meta: dict[str, Any] = {}

    if chart == CHART_P:
        total_n = float(sizes.sum())
        pbar = float(counts.sum() / total_n) if total_n else 0.0
        statistic = np.divide(counts, sizes, out=np.zeros_like(counts), where=sizes > 0)
        spread = sigma_limit * np.sqrt(
            np.divide(pbar * (1.0 - pbar), sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
        center, upper, lower = pbar, pbar + spread, pbar - spread
        meta["p-bar"] = pbar

    elif chart == CHART_NP:
        n = float(np.mean(sizes)) if sizes.size else 0.0
        pbar = float(counts.sum() / (n * point_count)) if n and point_count else 0.0
        statistic = counts
        spread = sigma_limit * np.sqrt(max(n * pbar * (1.0 - pbar), 0.0))
        center = n * pbar
        upper = np.full(point_count, center + spread)
        lower = np.full(point_count, center - spread)
        meta["p-bar"] = pbar
        meta["sample size"] = n

    elif chart == CHART_C:
        cbar = float(np.mean(counts)) if counts.size else 0.0
        statistic = counts
        spread = sigma_limit * np.sqrt(max(cbar, 0.0))
        center = cbar
        upper = np.full(point_count, cbar + spread)
        lower = np.full(point_count, cbar - spread)
        meta["c-bar"] = cbar

    elif chart == CHART_U:
        total_n = float(sizes.sum())
        ubar = float(counts.sum() / total_n) if total_n else 0.0
        statistic = np.divide(counts, sizes, out=np.zeros_like(counts), where=sizes > 0)
        spread = sigma_limit * np.sqrt(
            np.divide(ubar, sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
        center, upper, lower = ubar, ubar + spread, ubar - spread
        meta["u-bar"] = ubar

    else:  # pragma: no cover - the combo cannot hold anything else
        raise ValueError(f"unknown attribute chart {chart!r}")

    upper = np.broadcast_to(np.asarray(upper, dtype=float), (point_count,)).copy()
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (point_count,)).copy()
    # A count is non-negative; an LCL below zero would never signal.
    np.clip(lower, 0.0, None, out=lower)
    # A proportion is bounded above by 1.
    if chart == CHART_P:
        np.clip(upper, None, 1.0, out=upper)
    return np.asarray(statistic, dtype=float), float(center), upper, lower, meta


def spc_constants(size: int) -> tuple[tuple[float, ...], bool]:
    """Return the SPC constants for a subgroup size, and whether exact.

    Sizes between tabulated entries take the nearest smaller row rather
    than interpolating: the tables are what every other tool uses, and an
    interpolated d2 would put these limits subtly at odds with them.
    """
    if size in SPC_CONSTANTS:
        return SPC_CONSTANTS[size], True

    candidates = [n for n in SPC_CONSTANTS if n <= size]
    nearest = max(candidates) if candidates else min(SPC_CONSTANTS)
    return SPC_CONSTANTS[nearest], False


def rebuild_attribute_band(
    chart: str, center: float, sizes: np.ndarray, sigma_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The band around a revised centre, at every point's own sample size."""
    if chart in (CHART_P,):
        spread = sigma_limit * np.sqrt(
            np.divide(
                center * (1.0 - center), sizes,
                out=np.zeros_like(sizes), where=sizes > 0,
            )
        )
    elif chart == CHART_U:
        spread = sigma_limit * np.sqrt(
            np.divide(center, sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
    elif chart == CHART_NP:
        n = float(np.mean(sizes)) if sizes.size else 0.0
        p = center / n if n else 0.0
        spread = np.full(
            sizes.size, sigma_limit * np.sqrt(max(n * p * (1.0 - p), 0.0))
        )
    else:  # CHART_C
        spread = np.full(sizes.size, sigma_limit * np.sqrt(max(center, 0.0)))

    upper = center + spread
    lower = np.clip(center - spread, 0.0, None)
    if chart == CHART_P:
        upper = np.clip(upper, None, 1.0)
    return upper, lower


def sigma_from_limits(
    upper: np.ndarray, center: float, sigma_limit: float,
) -> np.ndarray:
    """Back out the per-point sigma the band was drawn at.

    The attribute formulas produce a band directly rather than a sigma,
    but the zone lines and the zone rules are both phrased in sigma, so
    it is recovered here rather than special-cased in four places.
    """
    if sigma_limit <= 0:
        return np.zeros_like(upper)
    return (np.asarray(upper, dtype=float) - center) / sigma_limit


def limits_from_individuals(
    values: np.ndarray,
    sigma_limit: float,
) -> tuple[float, float, float, float]:
    """Centre and limits for an individuals chart.

    Sigma comes from the average moving range over d2(2), NOT from the
    standard deviation of the values. A process that has shifted has a
    large overall standard deviation because it shifted, so limits built
    from it would be wide enough to swallow the shift.
    """
    moving_range = np.abs(np.diff(values))
    mean_range = float(np.mean(moving_range)) if moving_range.size else 0.0
    d2 = SPC_CONSTANTS[2][0]
    sigma = mean_range / d2 if d2 else 0.0
    center = float(np.mean(values))
    return center, center + sigma_limit * sigma, center - sigma_limit * sigma, sigma


def individuals_chart(
    x_values: np.ndarray,
    y_values: np.ndarray,
    sigma_limit: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, dict[str, Any]]:
    center, upper, lower, sigma = limits_from_individuals(y_values, sigma_limit)
    return (
        x_values,
        y_values,
        center,
        upper,
        lower,
        sigma,
        1,
        {"estimator": "average moving range / d2(2)"},
    )


def moving_range_chart(
    x_values: np.ndarray,
    y_values: np.ndarray,
    sigma_limit: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, dict[str, Any]]:
    moving_range = np.abs(np.diff(y_values))
    mean_range = float(np.mean(moving_range)) if moving_range.size else 0.0
    _d2, _d3, _c4, _a2, d3_limit, d4_limit, _b3, _b4 = SPC_CONSTANTS[2]

    # The MR chart uses D3/D4 rather than centre +/- k sigma: a range is
    # non-negative and its distribution is skewed, so symmetric limits
    # would put the lower one below zero and never signal.
    upper = d4_limit * mean_range
    lower = d3_limit * mean_range
    sigma = (upper - mean_range) / sigma_limit if sigma_limit else 0.0

    return (
        # One shorter than the source: the first point has no predecessor.
        x_values[1:],
        moving_range,
        mean_range,
        upper,
        lower,
        sigma,
        2,
        {"estimator": "D3/D4 on the moving range"},
    )


def limits_from_subgroup_stats(
    means: np.ndarray,
    dispersions: np.ndarray,
    size: int,
    sigma_limit: float,
    chart: str,
) -> tuple[float, float, float, float]:
    """Centre and limits for X-bar, from within-subgroup dispersion."""
    constants, _exact = spc_constants(size)
    d2, _d3, c4, _a2, _d3l, _d4l, _b3, _b4 = constants

    center = float(np.mean(means))
    mean_dispersion = float(np.mean(dispersions)) if dispersions.size else 0.0

    if chart == CHART_XBAR_S:
        sigma = mean_dispersion / c4 if c4 else 0.0
    else:
        sigma = mean_dispersion / d2 if d2 else 0.0

    # The standard error of a subgroup mean, which is what the X-bar chart
    # plots - not sigma itself. Using sigma would give limits far too wide
    # and a chart that never signals.
    #
    # Derived from d2 rather than applied as the tabulated A2 shortcut,
    # because A2 has the 3 of "three sigma" baked into it and this dialog
    # lets the limit be set to something else. The two agree to about
    # 1e-3 of the limit - the difference is the rounding in the published
    # d2 and A2, not a disagreement about the method.
    standard_error = sigma / np.sqrt(size) if size else 0.0
    return (
        center,
        center + sigma_limit * standard_error,
        center - sigma_limit * standard_error,
        sigma,
    )


def subgrouped_chart(
    x_values: np.ndarray,
    y_values: np.ndarray,
    chart: str,
    subgroup: int,
    sigma_limit: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, dict[str, Any]]:
    size = max(2, int(subgroup))
    count = y_values.size // size
    if count < 2:
        raise ValueError(
            f"a subgroup size of {size} gives {count} subgroup(s); "
            f"at least 2 are needed"
        )

    used = count * size
    remainder = y_values.size - used
    grouped = y_values[:used].reshape(count, size)

    means = grouped.mean(axis=1)
    if chart == CHART_XBAR_S:
        # ddof=1: the within-subgroup standard deviation is an estimate
        # from a sample, and c4 is tabulated for the ddof=1 form.
        dispersions = grouped.std(axis=1, ddof=1)
    else:
        dispersions = grouped.max(axis=1) - grouped.min(axis=1)

    center, upper, lower, sigma = limits_from_subgroup_stats(
        means, dispersions, size, sigma_limit, chart
    )

    # One x per subgroup: the midpoint of the points it covers, so the
    # chart still lines up with the source's axis.
    subgroup_x = x_values[:used].reshape(count, size).mean(axis=1)

    _constants, exact = spc_constants(size)
    meta: dict[str, Any] = {
        "estimator": (
            "average within-subgroup s / c4"
            if chart == CHART_XBAR_S
            else "average within-subgroup range / d2"
        ),
        "subgroups": count,
    }
    if remainder:
        meta["dropped"] = remainder
    if not exact:
        meta["constants"] = f"approximated for n={size}"

    return subgroup_x, means, center, upper, lower, sigma, size, meta


def find_violations(
    values: np.ndarray,
    positions: np.ndarray,
    center: float,
    sigma: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    use_nelson: bool,
) -> list[Violation]:
    """Return every rule broken, most fundamental first.

    Rule 1 is always applied; the rest are the Nelson run rules, which
    catch what stays inside the limits. They are what makes a control
    chart more than an outlier test, and also why a chart of a stable
    process still shows the occasional flag: eight rules each with a
    false-alarm rate compound.

    Which of them are *legal* depends on the chart. Rules 2-4 read only
    the values and the centre line, so they hold on any chart. Rules 5-8
    are phrased in equal-width one- and two-sigma zones, which a chart
    whose sigma moves point to point - a p or u chart on an uneven sample
    size - does not have; applying them there would invent a zone
    boundary per point and flag patterns that mean nothing. So they are
    skipped exactly when sigma is not constant, which is also why sigma
    is carried as an array rather than a scalar.
    """
    found: list[Violation] = []
    size = values.size

    def flag(index: int, rule: int, description: str) -> None:
        found.append(
            Violation(
                index=int(index),
                x=float(positions[index]),
                y=float(values[index]),
                rules=(rule,),
                descriptions=(description,),
            )
        )

    # Rule 1: outside the control limits.
    for index in np.flatnonzero((values > upper) | (values < lower)):
        flag(int(index), 1, "beyond the control limits")

    if not use_nelson or not sigma.size or float(np.max(sigma)) <= 0.0:
        return deduplicate(found)

    above = values > center
    below = values < center

    # Rule 2: nine in a row on one side of the centre - a shift.
    for start in range(size - 8):
        window = slice(start, start + 9)
        if above[window].all() or below[window].all():
            flag(start + 8, 2, "nine in a row on one side of the centre")

    # Rule 3: six in a row steadily increasing or decreasing - a trend.
    differences = np.diff(values)
    for start in range(size - 5):
        window = differences[start : start + 5]
        if window.size == 5 and ((window > 0).all() or (window < 0).all()):
            flag(start + 5, 3, "six in a row trending in one direction")

    # Rule 4: fourteen alternating up and down - overcontrol.
    if size >= 14:
        signs = np.sign(differences)
        for start in range(size - 13):
            window = signs[start : start + 13]
            if window.size == 13 and np.all(window[:-1] * window[1:] < 0):
                flag(start + 13, 4, "fourteen alternating up and down")

    # Rules 5-8 need equal-width zones; see the docstring.
    if float(np.ptp(sigma)) > 0.0:
        return deduplicate(found)
    constant_sigma = float(sigma[0])

    # Rule 5: two of three beyond two sigma, same side.
    two_sigma_up = center + 2.0 * constant_sigma
    two_sigma_down = center - 2.0 * constant_sigma
    for start in range(size - 2):
        window = values[start : start + 3]
        if (window > two_sigma_up).sum() >= 2 or (window < two_sigma_down).sum() >= 2:
            flag(start + 2, 5, "two of three beyond two sigma on one side")

    # Rule 6: four of five beyond one sigma, same side.
    one_sigma_up = center + constant_sigma
    one_sigma_down = center - constant_sigma
    for start in range(size - 4):
        window = values[start : start + 5]
        if (window > one_sigma_up).sum() >= 4 or (window < one_sigma_down).sum() >= 4:
            flag(start + 4, 6, "four of five beyond one sigma on one side")

    # Rule 7: fifteen in a row within one sigma - too good, which usually
    # means the limits are wrong or the data has been smoothed.
    within = (values < one_sigma_up) & (values > one_sigma_down)
    for start in range(size - 14):
        if within[start : start + 15].all():
            flag(start + 14, 7, "fifteen in a row hugging the centre line")

    # Rule 8: eight in a row all beyond one sigma, either side.
    outside = ~within
    for start in range(size - 7):
        if outside[start : start + 8].all():
            flag(start + 7, 8, "eight in a row beyond one sigma, both sides")

    return deduplicate(found)


def deduplicate(found: Sequence[Violation]) -> list[Violation]:
    """One entry per point, carrying every rule that point broke.

    Merged rather than filtered: a point is one thing to investigate, so
    it should be one row, but which rules it broke is exactly what tells
    you what to look for.
    """
    merged: dict[int, Violation] = {}
    for violation in found:
        current = merged.get(violation.index)
        if current is None:
            merged[violation.index] = violation
            continue

        rules = dict(zip(current.rules, current.descriptions))
        rules.update(zip(violation.rules, violation.descriptions))
        ordered = sorted(rules)
        merged[violation.index] = Violation(
            index=current.index,
            x=current.x,
            y=current.y,
            rules=tuple(ordered),
            descriptions=tuple(rules[rule] for rule in ordered),
        )

    return [merged[index] for index in sorted(merged)]


#: The variables charts that average several readings into each point.
SUBGROUPED_CHARTS: frozenset[str] = frozenset({CHART_XBAR_R, CHART_XBAR_S})
#: The attribute charts - a count per point.
ATTRIBUTE_CHARTS: frozenset[str] = frozenset({CHART_P, CHART_NP, CHART_C, CHART_U})
#: The attribute charts that read a sample size per point.
SIZE_CHARTS: frozenset[str] = frozenset({CHART_P, CHART_NP, CHART_U})

#: Below this an attribute chart's limits are too soft to trust; the report
#: says so rather than refusing to draw them.
RECOMMENDED_SUBGROUPS_DEFAULT = 20

_TOO_FEW_LEFT = "too few points would remain after excluding the flagged ones; the limits use every point."


@dataclass(slots=True)
class ControlChart:
    """A computed chart: what is plotted, its limits point by point, and the flags."""

    x: np.ndarray
    y: np.ndarray
    center: float
    upper: float
    lower: float
    sigma: float
    upper_band: np.ndarray
    lower_band: np.ndarray
    sigma_band: np.ndarray
    subgroup_size: int
    violations: list[Violation]
    details: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)


def variables_chart(
    chart: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma_limit: float = 3.0,
    nelson: bool = True,
    exclude_violations: bool = False,
    subgroup: int = 5,
) -> ControlChart:
    """An individuals, moving-range or X-bar chart of the measurements *y*."""
    if chart in SUBGROUPED_CHARTS:
        built = subgrouped_chart(x, y, chart, subgroup, sigma_limit)
    elif chart == CHART_MOVING_RANGE:
        built = moving_range_chart(x, y, sigma_limit)
    else:
        built = individuals_chart(x, y, sigma_limit)

    plot_x, plot_y, center, upper, lower, sigma, size, meta = built
    upper_band = np.full(plot_y.size, upper)
    lower_band = np.full(plot_y.size, lower)
    sigma_band = np.full(plot_y.size, sigma)
    violations = find_violations(plot_y, plot_x, center, sigma_band, upper_band, lower_band, nelson)
    notes: list[str] = []

    if exclude_violations and violations:
        keep = np.ones(plot_y.size, dtype=bool)
        keep[[violation.index for violation in violations]] = False
        if int(keep.sum()) >= 3:
            # Recompute from the surviving points only. Deliberately a second
            # pass over the same estimator rather than a trimmed sigma: the
            # point is to exclude assignable causes, not to make the
            # estimator robust to them.
            sub_y = plot_y[keep]
            if chart in SUBGROUPED_CHARTS:
                center, upper, lower, sigma = limits_from_subgroup_stats(
                    sub_y, meta.get("dispersion_kept", sub_y), size, sigma_limit, chart
                )
            else:
                center, upper, lower, sigma = limits_from_individuals(sub_y, sigma_limit)
            upper_band = np.full(plot_y.size, upper)
            lower_band = np.full(plot_y.size, lower)
            sigma_band = np.full(plot_y.size, sigma)
            meta["excluded"] = int(plot_y.size - keep.sum())
            violations = find_violations(plot_y, plot_x, center, sigma_band, upper_band, lower_band, nelson)
        else:
            notes.append(_TOO_FEW_LEFT)

    return ControlChart(
        plot_x, plot_y, center, upper, lower, sigma, upper_band, lower_band, sigma_band,
        size, violations, meta, tuple(notes),
    )


def attribute_chart(
    chart: str,
    x: np.ndarray,
    counts: np.ndarray,
    sizes: np.ndarray,
    *,
    sigma_limit: float = 3.0,
    nelson: bool = True,
    exclude_violations: bool = False,
    recommended_subgroups: int = RECOMMENDED_SUBGROUPS_DEFAULT,
) -> ControlChart:
    """A p, np, c or u chart of the *counts*, with *sizes* per point."""
    statistic, center, upper_band, lower_band, meta = attribute_limits(chart, counts, sizes, sigma_limit)
    sigma_band = sigma_from_limits(upper_band, center, sigma_limit)
    violations = find_violations(statistic, x, center, sigma_band, upper_band, lower_band, nelson)
    notes: list[str] = []

    if exclude_violations and violations:
        keep = np.ones(statistic.size, dtype=bool)
        keep[[violation.index for violation in violations]] = False
        if int(keep.sum()) >= 3:
            # Trial limits, then revised limits - the second pass every SPC
            # text runs once an assignable cause has been removed. The band is
            # rebuilt at full length, against every point's own sample size.
            _kept, center, _upper_kept, _lower_kept, meta = attribute_limits(
                chart, counts[keep], sizes[keep], sigma_limit
            )
            upper_band, lower_band = rebuild_attribute_band(chart, center, sizes, sigma_limit)
            sigma_band = sigma_from_limits(upper_band, center, sigma_limit)
            meta["excluded"] = int(statistic.size - keep.sum())
            violations = find_violations(statistic, x, center, sigma_band, upper_band, lower_band, nelson)
        else:
            notes.append(_TOO_FEW_LEFT)

    if x.size < recommended_subgroups:
        meta["note"] = (
            f"only {x.size} subgroups; "
            f"{recommended_subgroups}+ give trustworthy limits"
        )
    if chart in SIZE_CHARTS:
        meta["mean sample size"] = float(np.mean(sizes))
        if float(sizes.min()) != float(sizes.max()):
            meta["limits"] = "vary with the sample size"

    return ControlChart(
        x, statistic, center,
        float(np.mean(upper_band)), float(np.mean(lower_band)), float(np.mean(sigma_band)),
        upper_band, lower_band, sigma_band, 1, violations, meta, tuple(notes),
    )
