"""The numbers behind the diagnostic plots (todo R-06), without any Qt.

Each chart in app/charts that draws a diagnostic - Q-Q and P-P, Kaplan-Meier,
the interaction plot, the mosaic - asks one function here for its points
and draws them. Kept apart from the drawing for the same reason as the rest
of app/analysis: the arithmetic is what can be wrong, and it is checked in
dev/tests/test_analysis_diagnostics.py against textbook values and against
statsmodels, without a figure.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from app.analysis import NUMERICAL_FAILURES

#: The reference distributions a probability plot offers: scipy names.
PROBABILITY_DISTRIBUTIONS: tuple[str, ...] = (
    "norm", "lognorm", "expon", "gamma", "weibull_min", "logistic", "laplace", "gumbel_r", "t", "uniform",
)


# ----------------------------------------------------------------------
# Q-Q and P-P
# ----------------------------------------------------------------------
def plotting_positions(n: int) -> np.ndarray:
    """The probabilities the *n* sorted values stand for: R's ``ppoints``.

    ``(i - a) / (n + 1 - 2a)`` with ``a = 3/8`` up to ten values and ``1/2``
    beyond - Blom's choice for small samples, Hazen's for the rest - so a
    Q-Q plot here matches ``qqnorm`` in R point for point.
    """
    if n <= 0:
        return np.empty(0)
    a = 3.0 / 8.0 if n <= 10 else 0.5
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


@dataclass(slots=True)
class ProbabilityPlot:
    """The points of a Q-Q or P-P plot, its reference line and its band."""

    x: np.ndarray
    y: np.ndarray
    distribution: str
    #: The fitted parameters, in scipy's order (shapes, loc, scale).
    params: tuple[float, ...]
    #: The reference line, y = slope * x + intercept; None when not drawn.
    slope: float | None = None
    intercept: float | None = None
    #: Pointwise confidence band around the line, at each x; empty if none.
    lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Kolmogorov-Smirnov distance between the sample and the fit.
    ks_statistic: float = float("nan")


def fit_reference(values: np.ndarray, distribution: str) -> tuple[Any, tuple[float, ...]]:
    """Return the scipy distribution and its parameters fitted to *values*.

    The normal is fitted the way a Q-Q plot reads it - mean and standard
    deviation (n - 1) - and every other family by maximum likelihood. Raises
    ValueError for a name scipy does not have or a sample it cannot fit.
    """
    family = getattr(stats, distribution, None)
    if not isinstance(family, stats.rv_continuous):
        raise ValueError(f"No continuous distribution named {distribution!r}.")
    if distribution == "norm":
        return family, (float(np.mean(values)), float(np.std(values, ddof=1)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = tuple(float(value) for value in family.fit(values))
    except NUMERICAL_FAILURES as exc:
        raise ValueError(f"The {distribution} distribution could not be fitted: {exc}") from exc
    if not all(np.isfinite(params)):
        raise ValueError(f"The {distribution} distribution could not be fitted to this sample.")
    return family, params


def _sample(values: np.ndarray, minimum: int = 3) -> np.ndarray:
    sample = np.sort(np.asarray(values, dtype=float))
    sample = sample[np.isfinite(sample)]
    if sample.size < minimum:
        raise ValueError(f"A probability plot needs at least {minimum} finite values.")
    if np.ptp(sample) == 0:
        raise ValueError("Every value is the same: there is no distribution to compare.")
    return sample


def _order_statistic_band(n: int, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the i-th of *n* uniform order statistics falls, at *confidence*.

    U(i) follows Beta(i, n - i + 1), so its quantiles bound the probability
    each sorted value stands for; mapped through a fitted distribution they
    become the band a Q-Q plot's points should stay inside.
    """
    order = np.arange(1, n + 1)
    tail = (1.0 - confidence) / 2.0
    return stats.beta.ppf(tail, order, n - order + 1), stats.beta.ppf(1.0 - tail, order, n - order + 1)


def qq_plot(
    values: np.ndarray,
    distribution: str = "norm",
    *,
    standardized: bool = True,
    line: str = "quartiles",
    confidence: float = 0.95,
) -> ProbabilityPlot:
    """Sample quantiles against the quantiles of a fitted *distribution*.

    *standardized* puts the theoretical quantiles in the distribution's
    standard form (loc 0, scale 1 - z-scores for the normal), the classic
    Q-Q plot; otherwise they are in the data's units, and a perfect fit lies
    on y = x. *line*: "quartiles" (through the first and third quartiles, R's
    ``qqline``), "least squares", "identity" or "none". *confidence* in
    (0, 1) adds the pointwise band from the order statistics; 0 leaves it out.
    """
    sample = _sample(values)
    family, params = fit_reference(sample, distribution)
    n = sample.size
    probabilities = plotting_positions(n)
    shapes = params[:-2]
    reference = family(*params)
    theoretical = family(*shapes).ppf(probabilities) if standardized else reference.ppf(probabilities)
    finite = np.isfinite(theoretical)
    if not np.all(finite):
        raise ValueError(f"The {distribution} quantiles are not finite for this sample.")

    slope: float | None
    intercept: float | None
    if line == "quartiles":
        q_sample = np.quantile(sample, [0.25, 0.75])
        q_theory = (family(*shapes) if standardized else reference).ppf([0.25, 0.75])
        slope = float((q_sample[1] - q_sample[0]) / (q_theory[1] - q_theory[0]))
        intercept = float(q_sample[0] - slope * q_theory[0])
    elif line == "least squares":
        slope, intercept = (float(value) for value in np.polyfit(theoretical, sample, 1))
    elif line == "identity":
        if standardized:
            slope, intercept = float(params[-1]), float(params[-2])  # scale and loc
        else:
            slope, intercept = 1.0, 0.0
    else:
        slope = intercept = None

    lower = upper = np.empty(0)
    if 0.0 < confidence < 1.0:
        low_p, high_p = _order_statistic_band(n, confidence)
        lower, upper = reference.ppf(low_p), reference.ppf(high_p)

    return ProbabilityPlot(
        x=theoretical, y=sample, distribution=distribution, params=params,
        slope=slope, intercept=intercept, lower=lower, upper=upper,
        ks_statistic=float(stats.kstest(sample, reference.cdf).statistic),
    )


def pp_plot(values: np.ndarray, distribution: str = "norm", *, confidence: float = 0.95) -> ProbabilityPlot:
    """The fitted CDF at each sorted value against the sample's own probability.

    x is F(x(i)) under the fitted *distribution*, y the plotting position of
    x(i); a good fit lies on the diagonal. The band is where F(X(i)) falls for
    a sample the distribution really did produce.
    """
    sample = _sample(values)
    family, params = fit_reference(sample, distribution)
    reference = family(*params)
    probabilities = plotting_positions(sample.size)
    lower = upper = np.empty(0)
    if 0.0 < confidence < 1.0:
        lower, upper = _order_statistic_band(sample.size, confidence)
    return ProbabilityPlot(
        x=reference.cdf(sample), y=probabilities, distribution=distribution, params=params,
        slope=1.0, intercept=0.0, lower=lower, upper=upper,
        ks_statistic=float(stats.kstest(sample, reference.cdf).statistic),
    )


# ----------------------------------------------------------------------
# Kaplan-Meier
# ----------------------------------------------------------------------
@dataclass(slots=True)
class SurvivalCurve:
    """A Kaplan-Meier estimate as a step function, starting at (0, 1).

    ``survival[i]`` holds from ``time[i]`` up to the next time. ``lower`` and
    ``upper`` are the pointwise confidence limits (Greenwood's variance on
    the log(-log) scale, which keeps them inside 0..1).
    """

    time: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    #: Censored observations, and the curve's height at each.
    censored_time: np.ndarray
    censored_survival: np.ndarray
    #: The first time the curve reaches 0.5 or below; NaN if it never does.
    median: float
    n: int


def kaplan_meier(time: np.ndarray, event: np.ndarray, *, confidence: float = 0.95) -> SurvivalCurve:
    """The Kaplan-Meier survival estimate of right-censored *time*.

    *event* is 1 (or True) where the event was observed and 0 where the
    observation was censored - the subject left the study still event-free.
    Rows with a missing time or event are dropped; a negative time is an
    error, not data.
    """
    times = np.asarray(time, dtype=float)
    observed = np.asarray(event, dtype=float)
    keep = np.isfinite(times) & np.isfinite(observed)
    times, observed = times[keep], observed[keep] != 0
    if times.size == 0:
        raise ValueError("No rows with both a time and an event.")
    if np.any(times < 0):
        raise ValueError("A survival time cannot be negative.")

    distinct = np.unique(times[observed])
    at_risk = np.array([np.sum(times >= t) for t in distinct], dtype=float)
    deaths = np.array([np.sum((times == t) & observed) for t in distinct], dtype=float)
    factors = 1.0 - deaths / at_risk
    survival = np.cumprod(factors)

    with np.errstate(divide="ignore", invalid="ignore"):
        greenwood = np.cumsum(deaths / (at_risk * (at_risk - deaths)))
        z = stats.norm.ppf(0.5 + confidence / 2.0) if 0.0 < confidence < 1.0 else 0.0
        log_survival = np.log(survival)
        spread = z * np.sqrt(greenwood) / np.abs(log_survival)
        lower = survival ** np.exp(spread)
        upper = survival ** np.exp(-spread)
    # Where S is 1 the log-log scale has no width; where S is 0 the limits are 0.
    lower = np.where(np.isfinite(lower), lower, np.where(survival > 0, survival, 0.0))
    upper = np.where(np.isfinite(upper), upper, np.where(survival > 0, survival, 0.0))

    step_time = np.concatenate(([0.0], distinct))
    step_survival = np.concatenate(([1.0], survival))

    def height(t: float) -> float:
        index = int(np.searchsorted(step_time, t, side="right")) - 1
        return float(step_survival[max(index, 0)])

    censored = np.sort(times[~observed])
    below = np.nonzero(survival <= 0.5)[0]
    return SurvivalCurve(
        time=step_time,
        survival=step_survival,
        lower=np.concatenate(([1.0], lower)),
        upper=np.concatenate(([1.0], upper)),
        at_risk=np.concatenate(([float(times.size)], at_risk)),
        events=np.concatenate(([0.0], deaths)),
        censored_time=censored,
        censored_survival=np.array([height(t) for t in censored]),
        median=float(distinct[below[0]]) if below.size else float("nan"),
        n=int(times.size),
    )


@dataclass(slots=True)
class LogRank:
    """The log-rank test of equal survival across groups."""

    statistic: float
    dof: int
    pvalue: float


def log_rank(groups: list[tuple[np.ndarray, np.ndarray]]) -> LogRank:
    """Compare survival across *groups*, each a (time, event) pair.

    Mantel's log-rank statistic: at every event time, the deaths seen in each
    group against those expected if the groups shared one hazard, summed
    over time with the hypergeometric covariance, chi-squared on k - 1
    degrees of freedom.
    """
    if len(groups) < 2:
        raise ValueError("The log-rank test compares two or more groups.")
    cleaned: list[tuple[np.ndarray, np.ndarray]] = []
    for time, event in groups:
        t = np.asarray(time, dtype=float)
        e = np.asarray(event, dtype=float)
        keep = np.isfinite(t) & np.isfinite(e)
        cleaned.append((t[keep], e[keep] != 0))
    all_times = np.concatenate([t for t, _e in cleaned])
    all_events = np.concatenate([e for _t, e in cleaned])
    event_times = np.unique(all_times[all_events])
    k = len(cleaned)
    observed_minus_expected = np.zeros(k)
    covariance = np.zeros((k, k))
    for t in event_times:
        at_risk = np.array([np.sum(times >= t) for times, _e in cleaned], dtype=float)
        deaths = np.array([np.sum((times == t) & events) for times, events in cleaned], dtype=float)
        total_risk, total_deaths = at_risk.sum(), deaths.sum()
        if total_risk <= 0:
            continue
        observed_minus_expected += deaths - total_deaths * at_risk / total_risk
        if total_risk > 1:
            share = at_risk / total_risk
            factor = total_deaths * (total_risk - total_deaths) / (total_risk - 1)
            covariance += factor * (np.diag(share) - np.outer(share, share))
    reduced = observed_minus_expected[:-1]
    try:
        statistic = float(reduced @ np.linalg.solve(covariance[:-1, :-1], reduced))
    except np.linalg.LinAlgError as exc:
        raise ValueError("The groups have no events to compare.") from exc
    dof = k - 1
    return LogRank(statistic=statistic, dof=dof, pvalue=float(stats.chi2.sf(statistic, dof)))


# ----------------------------------------------------------------------
# Interaction plot
# ----------------------------------------------------------------------
@dataclass(slots=True)
class CellMeans:
    """The mean response in each (level of x, trace) cell."""

    levels: list[Any]
    traces: list[Any]
    #: [trace, level] arrays; NaN for an empty cell.
    mean: np.ndarray
    sd: np.ndarray
    count: np.ndarray

    def half_width(self, kind: str, confidence: float = 0.95) -> np.ndarray:
        """Error bar half-widths: "sd", "se", "ci" (Student t) or "none" (zeros)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            se = self.sd / np.sqrt(self.count)
            if kind == "sd":
                return np.nan_to_num(self.sd)
            if kind == "se":
                return np.nan_to_num(se)
            if kind == "ci":
                t = stats.t.ppf(0.5 + confidence / 2.0, np.maximum(self.count - 1, 1))
                return np.nan_to_num(t * se)
        return np.zeros_like(self.mean)


def _ordered_levels(values: np.ndarray) -> list[Any]:
    """Distinct values in a reading order: numeric ascending, else first seen."""
    distinct = list(dict.fromkeys(value for value in values.tolist() if value is not None and value == value))
    try:
        numbers = [float(value) for value in distinct]
    except (TypeError, ValueError):
        return distinct
    return [value for _number, value in sorted(zip(numbers, distinct), key=lambda pair: pair[0])]


def cell_means(x: np.ndarray, y: np.ndarray, trace: np.ndarray | None = None) -> CellMeans:
    """Mean, standard deviation and count of *y* in every (x, trace) cell."""
    import pandas as pd

    frame = pd.DataFrame({
        "x": np.asarray(x, dtype=object),
        "y": np.asarray(y, dtype=float),
        "trace": "" if trace is None else np.asarray(trace, dtype=object),
    })
    frame = frame[np.isfinite(frame["y"].to_numpy()) & frame["x"].notna() & frame["trace"].notna()]
    if frame.empty:
        raise ValueError("No rows with both a level and a numeric response.")
    levels = _ordered_levels(frame["x"].to_numpy())
    traces = _ordered_levels(frame["trace"].to_numpy())
    grouped = frame.groupby(["trace", "x"], sort=False)["y"].agg(["mean", "std", "count"])
    shape = (len(traces), len(levels))
    mean, sd, count = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape)
    for (trace_value, level), row in grouped.iterrows():  # pyright: ignore[reportGeneralTypeIssues]
        i, j = traces.index(trace_value), levels.index(level)
        mean[i, j], sd[i, j], count[i, j] = row["mean"], row["std"], row["count"]
    return CellMeans(levels=levels, traces=traces, mean=mean, sd=sd, count=count)


# ----------------------------------------------------------------------
# Mosaic
# ----------------------------------------------------------------------
@dataclass(slots=True)
class MosaicTile:
    """One rectangle of a mosaic, in axes units (0..1 both ways)."""

    column: Any
    row: Any
    x: float
    y: float
    width: float
    height: float
    count: float
    #: Pearson residual (observed - expected) / sqrt(expected) under independence.
    residual: float


@dataclass(slots=True)
class Mosaic:
    tiles: list[MosaicTile]
    columns: list[Any]
    rows: list[Any]
    #: Centre of each column, for the tick labels.
    column_centres: list[float]
    chi_square: float
    dof: int
    pvalue: float


def mosaic(first: np.ndarray, second: np.ndarray, weight: np.ndarray | None = None, *, gap: float = 0.01) -> Mosaic:
    """The tiles of a two-way mosaic: columns by *first*, split by *second*.

    Each column is as wide as its share of the total, and each tile in it as
    tall as the share of *second* within that column, so tile area is the
    joint proportion. *weight* is a count per row, for an already-counted
    table. *gap* is the space between columns and between tiles. Also returns
    Pearson's chi-squared test of independence, which the residuals shade.
    """
    first = np.asarray(first, dtype=object)
    second = np.asarray(second, dtype=object)
    counts_in = np.ones(first.shape) if weight is None else np.asarray(weight, dtype=float)
    present = np.array(
        [a is not None and a == a and b is not None and b == b for a, b in zip(first.tolist(), second.tolist())],
        dtype=bool,
    ) & np.isfinite(counts_in) & (counts_in >= 0)
    first, second, counts_in = first[present], second[present], counts_in[present]
    columns, rows = _ordered_levels(first), _ordered_levels(second)
    if not columns or not rows:
        raise ValueError("A mosaic needs two categorical columns with values.")
    table = np.zeros((len(columns), len(rows)))
    column_index = {value: index for index, value in enumerate(columns)}
    row_index = {value: index for index, value in enumerate(rows)}
    for a, b, count in zip(first.tolist(), second.tolist(), counts_in.tolist()):
        table[column_index[a], row_index[b]] += count
    total = table.sum()
    if total <= 0:
        raise ValueError("The counts add up to nothing.")

    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = np.where(expected > 0, (table - expected) / np.sqrt(expected), 0.0)
    chi_square = float(np.sum(residuals**2))
    dof = (len(columns) - 1) * (len(rows) - 1)
    pvalue = float(stats.chi2.sf(chi_square, dof)) if dof > 0 else float("nan")

    width_left = 1.0 - gap * (len(columns) - 1)
    tiles: list[MosaicTile] = []
    centres: list[float] = []
    x = 0.0
    for i, column in enumerate(columns):
        column_total = table[i].sum()
        width = width_left * column_total / total
        centres.append(x + width / 2.0)
        height_left = 1.0 - gap * (len(rows) - 1)
        y = 0.0
        for j, row in enumerate(rows):
            height = height_left * table[i, j] / column_total if column_total > 0 else 0.0
            tiles.append(MosaicTile(column, row, x, y, width, height, float(table[i, j]), float(residuals[i, j])))
            y += height + gap
        x += width + gap
    return Mosaic(tiles=tiles, columns=columns, rows=rows, column_centres=centres,
                  chi_square=chi_square, dof=dof, pvalue=pvalue)
