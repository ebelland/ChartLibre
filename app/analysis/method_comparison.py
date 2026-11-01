"""Method comparison: do two methods measuring the same samples agree? (ID12; CLSI EP09.)

- Bland-Altman: the differences against the means - the bias (the mean difference) with its interval, the
  limits of agreement (bias +- 1.96 sd, where 95% of the differences fall) with theirs, and whether the
  difference grows with the level (a proportional bias).
- Deming regression: a straight line when both methods have measurement error; *ratio* is the ratio of the
  two methods' error variances (Y's over X's), 1 when they are alike (orthogonal regression). Its interval
  by the jackknife (Linnet).
- Passing-Bablok: the median of the slopes between every two samples - no assumption on the errors, robust
  to outliers (Passing & Bablok 1983), its interval from the ranks of those slopes.

The methods agree when the line's interval holds slope 1 and intercept 0 and the bias is small enough for
the use the results are put to - a judgement the numbers support, not one they make.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class Line:
    """A fitted line: its slope and intercept with their confidence intervals."""

    method: str
    slope: float
    intercept: float
    slope_ci: tuple[float, float]
    intercept_ci: tuple[float, float]

    def row(self) -> dict[str, Any]:
        return {"Method": self.method, "Slope": self.slope, "Slope lower": self.slope_ci[0],
                "Slope upper": self.slope_ci[1], "Intercept": self.intercept,
                "Intercept lower": self.intercept_ci[0], "Intercept upper": self.intercept_ci[1],
                "Slope 1 inside": self.slope_ci[0] <= 1.0 <= self.slope_ci[1],
                "Intercept 0 inside": self.intercept_ci[0] <= 0.0 <= self.intercept_ci[1]}


@dataclass(slots=True)
class Agreement:
    """Bland-Altman: the bias and the limits of agreement, each with its interval."""

    n: int
    bias: float
    bias_ci: tuple[float, float]
    sd: float
    lower: float
    upper: float
    lower_ci: tuple[float, float]
    upper_ci: tuple[float, float]
    #: The difference regressed on the mean: its slope and p-value (a proportional bias when small).
    trend_slope: float
    trend_p: float


@dataclass(slots=True)
class Comparison:
    y: str
    x: str
    agreement: Agreement
    lines: list[Line]
    #: Per sample: X, Y, Mean, Difference.
    rows: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def bland_altman(y: np.ndarray, x: np.ndarray, *, confidence: float = 0.95) -> Agreement:
    """The differences y - x against the means (x + y) / 2 (Bland & Altman 1986, 1999)."""
    difference, mean = y - x, (y + x) / 2.0
    n = difference.size
    if n < 3:
        raise ValueError("Method comparison needs at least 3 samples measured by both methods.")
    bias, sd = float(np.mean(difference)), float(np.std(difference, ddof=1))
    t = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    z = float(stats.norm.ppf(0.975))  # the limits of agreement hold 95% of the differences
    lower, upper = bias - z * sd, bias + z * sd
    se_bias = sd / math.sqrt(n)
    se_limit = sd * math.sqrt(3.0 / n)  # Bland & Altman 1999: the limit's standard error, about sqrt(3 sd^2 / n)
    trend = stats.linregress(mean, difference)
    return Agreement(n, bias, (bias - t * se_bias, bias + t * se_bias), sd, lower, upper,
                     (lower - t * se_limit, lower + t * se_limit), (upper - t * se_limit, upper + t * se_limit),
                     float(trend.slope), float(trend.pvalue))


def _deming_fit(x: np.ndarray, y: np.ndarray, ratio: float) -> tuple[float, float]:
    mx, my = float(np.mean(x)), float(np.mean(y))
    sxx, syy = float(np.sum((x - mx) ** 2)), float(np.sum((y - my) ** 2))
    sxy = float(np.sum((x - mx) * (y - my)))
    if sxy == 0:
        raise ValueError("Deming regression needs the two methods to vary together.")
    slope = (syy - ratio * sxx + math.sqrt((syy - ratio * sxx) ** 2 + 4.0 * ratio * sxy ** 2)) / (2.0 * sxy)
    return slope, my - slope * mx


def deming(x: np.ndarray, y: np.ndarray, *, ratio: float = 1.0, confidence: float = 0.95) -> Line:
    """Deming regression of y on x, *ratio* the error variances' ratio (y's / x's); jackknife intervals."""
    if ratio <= 0:
        raise ValueError("The error variance ratio must be above 0.")
    slope, intercept = _deming_fit(x, y, ratio)
    n = x.size
    leave_one_out = np.array([_deming_fit(np.delete(x, i), np.delete(y, i), ratio) for i in range(n)])
    t = float(stats.t.ppf(0.5 + confidence / 2.0, n - 2))
    intervals = []
    for column, full in ((0, slope), (1, intercept)):
        pseudo = n * full - (n - 1) * leave_one_out[:, column]
        estimate, se = float(np.mean(pseudo)), float(np.std(pseudo, ddof=1) / math.sqrt(n))
        intervals.append((estimate - t * se, estimate + t * se))
    return Line(f"Deming (error ratio {ratio:g})", slope, intercept, intervals[0], intervals[1])


def passing_bablok(x: np.ndarray, y: np.ndarray, *, confidence: float = 0.95) -> Line:
    """Passing-Bablok regression (1983): the shifted median of the pairwise slopes, its interval by their ranks."""
    n = x.size
    i, j = np.triu_indices(n, k=1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    keep = ~((dx == 0) & (dy == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(dx[keep] == 0, np.sign(dy[keep]) * np.inf, dy[keep] / dx[keep])
    slopes = np.sort(slopes[slopes != -1.0])  # a slope of exactly -1 says nothing about agreement
    big_n = slopes.size
    if big_n < 2:
        raise ValueError("Passing-Bablok needs samples at different levels.")
    k = int(np.sum(slopes < -1.0))  # the shift: slopes below -1 are counted from the other end

    def ranked(position: float) -> float:  # 1-based, shifted by k, clipped to the slopes there are
        index = int(min(max(round(position) + k, 1), big_n)) - 1
        return float(slopes[index])

    if big_n % 2:
        slope = ranked((big_n + 1) / 2)
    else:
        slope = (ranked(big_n / 2) + ranked(big_n / 2 + 1)) / 2.0
    c = float(stats.norm.ppf(0.5 + confidence / 2.0)) * math.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    m1 = round((big_n - c) / 2.0)
    m2 = big_n - m1 + 1
    slope_low, slope_high = ranked(m1), ranked(m2)
    intercept = float(np.median(y - slope * x))
    intercept_ci = (float(np.median(y - slope_high * x)), float(np.median(y - slope_low * x)))
    return Line("Passing-Bablok", slope, intercept, (slope_low, slope_high), intercept_ci)


def compare(frame: pd.DataFrame, y: str, x: str, *, ratio: float = 1.0, confidence: float = 0.95) -> Comparison:
    """*y* (the method tested) against *x* (the reference), on the samples both measured."""
    data = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    xv, yv = data[x].to_numpy(float), data[y].to_numpy(float)
    notes = []
    if len(data) < len(frame):
        notes.append(f"{len(frame) - len(data)} sample(s) without both results were left out.")
    if len(data) < 40:
        notes.append(f"{len(data)} samples: CLSI EP09 asks for at least 40 over the measuring range.")
    agreement = bland_altman(yv, xv, confidence=confidence)
    lines = [deming(xv, yv, ratio=ratio, confidence=confidence), passing_bablok(xv, yv, confidence=confidence)]
    if agreement.trend_p < 0.05:
        notes.append("The difference grows with the level (a proportional bias): a single bias and its limits "
                     "describe the methods less well than the regression lines.")
    rows = pd.DataFrame({x: xv, y: yv, "Mean": (xv + yv) / 2.0, "Difference": yv - xv})
    return Comparison(y, x, agreement, lines, rows, notes)
