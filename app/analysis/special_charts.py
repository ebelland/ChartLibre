"""The control charts beyond Shewhart's: small shifts (EWMA, CUSUM), the laboratory's (Levey-Jennings with the
Westgard rules) and rare events (g and t).

- EWMA: each point the weighted mean of the newest value (weight lambda) and the last mean - a shift of half a
  sigma that an individuals chart takes dozens of points to flag shows within a few. Limits narrow at first,
  then settle (Montgomery, Introduction to Statistical Quality Control, 9.2).
- CUSUM, tabular: the running sums of the deviations above target + k sigma and below target - k sigma, each
  reset at zero; a sum beyond h sigma signals a shift (Montgomery 9.1). Plotted as C+ and -C-.
- Levey-Jennings: a control material's results against its known mean and standard deviation (not estimated
  from the moving range, as on an individuals chart), judged by Westgard's rules: 1-3s, 2-2s, R-4s, 4-1s and
  10-x reject the run, 1-2s warns.
- g chart: the number of units (days, procedures) between two rare events, geometric - Benneyan's limits.
- t chart: the time between rare events, made near-normal by the power 1/3.6 (Nelson), charted as
  individuals, its limits taken back to the time's scale.

Each returns the control_charts module's ControlChart, so the dialog draws, reports and flags them the same way.
"""
from __future__ import annotations

import math

import numpy as np

from app.analysis.control_charts import (
    SPC_CONSTANTS,
    ControlChart,
    Violation,
    find_violations,
)

CHART_EWMA = "EWMA (small shifts)"
CHART_CUSUM = "CUSUM (small shifts)"
CHART_LEVEY_JENNINGS = "Levey-Jennings (Westgard rules)"
CHART_G = "g (units between rare events)"
CHART_T = "t (time between rare events)"
SPECIAL_CHARTS = (CHART_EWMA, CHART_CUSUM, CHART_LEVEY_JENNINGS, CHART_G, CHART_T)

#: Nelson's power for the time between events: Weibull-distributed times come out nearly normal.
T_POWER = 1.0 / 3.6


def _mr_sigma(values: np.ndarray) -> float:
    """The short-term sigma: the average moving range over d2(2), as on an individuals chart."""
    ranges = np.abs(np.diff(values))
    return float(np.mean(ranges)) / SPC_CONSTANTS[2][0] if ranges.size else 0.0


def _beyond(values: np.ndarray, x: np.ndarray, upper: np.ndarray, lower: np.ndarray, label: str) -> list[Violation]:
    return [Violation(int(i), float(x[i]), float(values[i]), (1,), (label,))
            for i in np.flatnonzero((values > upper) | (values < lower))]


def ewma_chart(x: np.ndarray, y: np.ndarray, *, lam: float = 0.2, width: float = 3.0,
               target: float | None = None, sigma: float | None = None) -> ControlChart:
    """z_i = lam y_i + (1 - lam) z_(i-1), z_0 the target; limits target +- width sigma sqrt(lam/(2-lam) (1-(1-lam)^2i))."""
    if not 0.0 < lam <= 1.0:
        raise ValueError("The EWMA weight lambda must be in (0, 1].")
    y = np.asarray(y, float)
    center = float(np.mean(y)) if target is None else float(target)
    s = _mr_sigma(y) if sigma is None else float(sigma)
    z = np.empty(y.size)
    previous = center
    for i, value in enumerate(y):
        previous = lam * value + (1.0 - lam) * previous
        z[i] = previous
    steps = np.arange(1, y.size + 1)
    half = width * s * np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * steps)))
    upper, lower = center + half, center - half
    violations = _beyond(z, x, upper, lower, "EWMA beyond its limit: a shift in the mean")
    asymptote = width * s * math.sqrt(lam / (2.0 - lam))
    return ControlChart(x, z, center, center + asymptote, center - asymptote, s, upper, lower, half / width, 1,
                        violations, {"lambda": lam, "L": width, "sigma": s, "estimator": "average moving range / d2(2)"})


def cusum_chart(x: np.ndarray, y: np.ndarray, *, k: float = 0.5, h: float = 5.0, target: float | None = None,
                sigma: float | None = None) -> ControlChart:
    """Tabular CUSUM with reference value k sigma and decision interval h sigma: y is C+, details "low" is -C-."""
    y = np.asarray(y, float)
    center = float(np.mean(y)) if target is None else float(target)
    s = _mr_sigma(y) if sigma is None else float(sigma)
    high, low = np.empty(y.size), np.empty(y.size)
    c_plus = c_minus = 0.0
    for i, value in enumerate(y):
        c_plus = max(0.0, value - (center + k * s) + c_plus)
        c_minus = max(0.0, (center - k * s) - value + c_minus)
        high[i], low[i] = c_plus, -c_minus
    decision = h * s
    upper, lower = np.full(y.size, decision), np.full(y.size, -decision)
    violations = [Violation(int(i), float(x[i]), float(high[i] if high[i] > decision else low[i]), (1,),
                            ("CUSUM above h: the mean shifted up" if high[i] > decision
                             else "CUSUM below -h: the mean shifted down",))
                  for i in np.flatnonzero((high > decision) | (low < -decision))]
    return ControlChart(x, high, 0.0, decision, -decision, s, upper, lower, np.full(y.size, s), 1, violations,
                        {"k": k, "h": h, "target": center, "sigma": s, "low": low})


#: Westgard's rules: the name, and whether it rejects the run (else it warns).
WESTGARD = (("1-3s", True), ("2-2s", True), ("R-4s", True), ("4-1s", True), ("10-x", True), ("1-2s", False))


def westgard(values: np.ndarray, mean: float, sd: float) -> list[tuple[int, str, bool]]:
    """Every rule each result breaks: (index, rule, rejects). Rules on consecutive results within the material."""
    z = (np.asarray(values, float) - mean) / sd if sd > 0 else np.zeros(len(values))
    found: list[tuple[int, str, bool]] = []
    for i in range(z.size):
        if abs(z[i]) > 3:
            found.append((i, "1-3s", True))
        if i >= 1 and ((z[i] > 2 and z[i - 1] > 2) or (z[i] < -2 and z[i - 1] < -2)):
            found.append((i, "2-2s", True))
        if i >= 1 and ((z[i] > 2 and z[i - 1] < -2) or (z[i] < -2 and z[i - 1] > 2)):
            found.append((i, "R-4s", True))
        if i >= 3 and (np.all(z[i - 3:i + 1] > 1) or np.all(z[i - 3:i + 1] < -1)):
            found.append((i, "4-1s", True))
        if i >= 9 and (np.all(z[i - 9:i + 1] > 0) or np.all(z[i - 9:i + 1] < 0)):
            found.append((i, "10-x", True))
        if abs(z[i]) > 2 and not any(index == i and rejects for index, _rule, rejects in found):
            found.append((i, "1-2s", False))
    return found


def levey_jennings_chart(x: np.ndarray, y: np.ndarray, *, mean: float | None = None,
                         sd: float | None = None) -> ControlChart:
    """The results against mean +- 1, 2, 3 sd - the material's known ones, else the data's own - and Westgard."""
    y = np.asarray(y, float)
    center = float(np.mean(y)) if mean is None else float(mean)
    s = float(np.std(y, ddof=1)) if sd is None else float(sd)
    if not s > 0:
        raise ValueError("Levey-Jennings needs a standard deviation above 0.")
    by_point: dict[int, list[tuple[str, bool]]] = {}
    for index, rule, rejects in westgard(y, center, s):
        by_point.setdefault(index, []).append((rule, rejects))
    violations = [Violation(index, float(x[index]), float(y[index]),
                            tuple(1 if rejects else 2 for _rule, rejects in rules),
                            tuple(f"Westgard {rule}: {'reject the run' if rejects else 'warning'}" for rule, rejects in rules))
                  for index, rules in sorted(by_point.items())]
    rejected = sum(any(rejects for _rule, rejects in rules) for rules in by_point.values())
    size = y.size
    return ControlChart(x, y, center, center + 3 * s, center - 3 * s, s, np.full(size, center + 3 * s),
                        np.full(size, center - 3 * s), np.full(size, s), 1, violations,
                        {"mean": center, "sd": s, "source": "given" if sd is not None else "the data's",
                         "runs rejected": rejected})


def g_chart(x: np.ndarray, between: np.ndarray, *, width: float = 3.0) -> ControlChart:
    """Units between rare events: centre the mean g, limits g +- width sqrt(g (g + 1)) (geometric), the lower >= 0."""
    g = np.asarray(between, float)
    if np.any(g < 0):
        raise ValueError("A g chart counts units between events: no negative counts.")
    center = float(np.mean(g))
    spread = math.sqrt(center * (center + 1.0))
    upper, lower = center + width * spread, max(0.0, center - width * spread)
    upper_band, lower_band = np.full(g.size, upper), np.full(g.size, lower)
    violations = _beyond(g, x, upper_band, lower_band, "Beyond the limit: events unusually far apart, or close")
    return ControlChart(x, g, center, upper, lower, spread, upper_band, lower_band, np.full(g.size, spread), 1,
                        violations, {"distribution": "geometric", "mean between events": center})


def t_chart(x: np.ndarray, times: np.ndarray, *, width: float = 3.0, nelson: bool = False) -> ControlChart:
    """Time between rare events: y^(1/3.6) charted as individuals, the centre and limits taken back to time."""
    t = np.asarray(times, float)
    if np.any(t <= 0):
        raise ValueError("A t chart needs times between events above 0.")
    transformed = t ** T_POWER
    sigma = _mr_sigma(transformed)
    center = float(np.mean(transformed))
    upper, lower = center + width * sigma, max(0.0, center - width * sigma)
    flags = find_violations(transformed, x, center, np.full(t.size, sigma), np.full(t.size, upper),
                            np.full(t.size, lower), nelson)
    violations = [Violation(v.index, v.x, float(t[v.index]), v.rules, v.descriptions) for v in flags]
    back = 1.0 / T_POWER
    return ControlChart(x, t, center ** back, upper ** back, lower ** back, sigma, np.full(t.size, upper ** back),
                        np.full(t.size, lower ** back), np.full(t.size, sigma), 1, violations,
                        {"transform": "time^(1/3.6), individuals", "median time": center ** back})
