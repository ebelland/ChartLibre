"""Process capability against specification limits: Cp, Cpk, Pp, Ppk, Cpm and the share out of specification.

Computed by the control chart, which owns the two things capability needs:
whether the process is stable at all (an unstable process has no capability
to speak of) and the within sigma - the average moving range / d2 for an
individuals chart, the subgroups' range / d2 or s / c4 for an X-bar chart.

* Cp = (USL - LSL) / 6 sigma_within; Cpk the smaller of (mean - LSL) and
  (USL - mean) over 3 sigma_within, also given as Cpl and Cpu;
* Pp, Ppk the same with the overall standard deviation of the values;
* Cpm = (USL - LSL) / 6 sqrt(s^2 + (mean - target)^2), with a target;
* the share below LSL and above USL: observed, and expected under a normal
  with each sigma, also in parts per million.

One limit is enough: only the one-sided indices are given then.

A process that is not normal (a lognormal roughness, a Weibull strength) is
judged by the percentile method (ISO 22514-4, Clements): a distribution is
fitted - lognormal, Weibull, gamma, or whichever fits best by AIC - and its
0.135th and 99.865th percentiles stand for mean -/+ 3 sigma, its median for
the mean. Only the overall indices exist then (Pp, Ppk, Ppl, Ppu): a within
sigma has no meaning for a fitted distribution. The expected share out of
specification comes from the fitted distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class Capability:
    lsl: float | None
    usl: float | None
    target: float | None
    mean: float
    sigma_within: float
    sigma_overall: float
    #: Index -> value: Cp, Cpk, Cpl, Cpu, Pp, Ppk, Ppl, Ppu, Cpm (those the limits allow).
    indices: dict[str, float]
    #: Rows Below LSL, Above USL, Total outside: observed %, expected % within and overall, PPM.
    outside: pd.DataFrame
    #: "normal", or the distribution fitted for the percentile method.
    distribution: str = "normal"


def parse_limit(text: object) -> float | None:
    """A typed limit, or None for an empty field; a decimal comma is accepted."""
    text = str(text or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"'{text}' is not a number.") from exc


#: The distributions the percentile method fits: name -> scipy distribution (positive values; loc fixed at 0).
NONNORMAL: dict[str, Any] = {"lognormal": stats.lognorm, "weibull": stats.weibull_min, "gamma": stats.gamma}
DISTRIBUTIONS: tuple[str, ...] = ("normal", "best", *NONNORMAL)


def best_distribution(values: np.ndarray) -> str:
    """The distribution with the lowest AIC among the normal and NONNORMAL (these for positive values only)."""
    scores = {"normal": _aic(stats.norm, stats.norm.fit(values), values)}
    if np.all(values > 0):
        for name, dist in NONNORMAL.items():
            try:
                scores[name] = _aic(dist, dist.fit(values, floc=0), values)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
    return min(scores, key=scores.get)


def _aic(dist: Any, params: tuple, values: np.ndarray) -> float:
    free = len(params) - (1 if dist is not stats.norm else 0)  # loc is fixed at 0 for the positive ones
    return 2 * free - 2 * float(np.sum(dist.logpdf(values, *params)))


def capability(values: np.ndarray, *, sigma_within: float, lsl: float | None = None, usl: float | None = None,
               target: float | None = None, distribution: str = "normal") -> Capability:
    """The indices of *values* against the limits, *sigma_within* from the control chart.

    *distribution* other than "normal" ("best", or a NONNORMAL name): the percentile method's indices.
    """
    if lsl is None and usl is None:
        raise ValueError("Capability needs a lower or an upper specification limit.")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("The lower specification limit must be below the upper one.")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or not sigma_within > 0:
        raise ValueError("Capability needs at least two values that vary.")
    mean, overall = float(np.mean(values)), float(np.std(values, ddof=1))
    if distribution == "best":
        distribution = best_distribution(values)
    if distribution != "normal":
        return _percentile_capability(values, lsl, usl, target, mean, float(sigma_within), overall, distribution)
    indices: dict[str, float] = {}
    for prefix, sigma in (("C", sigma_within), ("P", overall)):
        lower = (mean - lsl) / (3 * sigma) if lsl is not None else None
        upper = (usl - mean) / (3 * sigma) if usl is not None else None
        if lsl is not None and usl is not None:
            indices[f"{prefix}p"] = (usl - lsl) / (6 * sigma)
        indices[f"{prefix}pk"] = min(v for v in (lower, upper) if v is not None)
        if lower is not None:
            indices[f"{prefix}pl"] = lower
        if upper is not None:
            indices[f"{prefix}pu"] = upper
    if target is not None and lsl is not None and usl is not None:
        indices["Cpm"] = (usl - lsl) / (6 * float(np.sqrt(overall ** 2 + (mean - target) ** 2)))

    def expected(sigma: float, below: bool) -> float:
        if below:
            return float(stats.norm.cdf((lsl - mean) / sigma)) if lsl is not None else 0.0
        return float(stats.norm.sf((usl - mean) / sigma)) if usl is not None else 0.0

    rows = []
    for label, below in (("Below LSL", True), ("Above USL", False)):
        if below:
            observed = float(np.mean(values < lsl)) if lsl is not None else 0.0
        else:
            observed = float(np.mean(values > usl)) if usl is not None else 0.0
        rows.append((label, observed, expected(sigma_within, below), expected(overall, below)))
    rows.append(("Total outside", *(sum(row[i] for row in rows) for i in (1, 2, 3))))
    outside = pd.DataFrame(rows, columns=["Portion", "Observed %", "Expected % (within)", "Expected % (overall)"])
    for column in outside.columns[1:]:
        outside[column] = 100.0 * outside[column]
    outside["Expected PPM (overall)"] = outside["Expected % (overall)"] * 1e4
    return Capability(lsl=lsl, usl=usl, target=target, mean=mean, sigma_within=float(sigma_within),
                      sigma_overall=overall, indices=indices, outside=outside)


def _percentile_capability(values: np.ndarray, lsl: float | None, usl: float | None, target: float | None,
                           mean: float, sigma_within: float, overall: float, name: str) -> Capability:
    """The percentile method (ISO 22514-4): Pp, Ppk... from a fitted *name* distribution's percentiles."""
    if name not in NONNORMAL:
        raise ValueError(f"Unknown distribution {name!r}: one of {', '.join(DISTRIBUTIONS)}.")
    if not np.all(values > 0):
        raise ValueError(f"The {name} distribution needs every value above zero.")
    dist = NONNORMAL[name]
    frozen = dist(*dist.fit(values, floc=0))
    low, median, high = (float(frozen.ppf(q)) for q in (0.00135, 0.5, 0.99865))
    indices: dict[str, float] = {}
    lower = (median - lsl) / (median - low) if lsl is not None else None
    upper = (usl - median) / (high - median) if usl is not None else None
    if lsl is not None and usl is not None:
        indices["Pp"] = (usl - lsl) / (high - low)
    indices["Ppk"] = min(v for v in (lower, upper) if v is not None)
    if lower is not None:
        indices["Ppl"] = lower
    if upper is not None:
        indices["Ppu"] = upper
    rows = []
    for label, below in (("Below LSL", True), ("Above USL", False)):
        limit = lsl if below else usl
        if limit is None:
            observed = expected = 0.0
        else:
            observed = float(np.mean(values < limit)) if below else float(np.mean(values > limit))
            expected = float(frozen.cdf(limit)) if below else float(frozen.sf(limit))
        rows.append((label, observed, float("nan"), expected))
    rows.append(("Total outside", *(sum(row[i] for row in rows) for i in (1, 2, 3))))
    outside = pd.DataFrame(rows, columns=["Portion", "Observed %", "Expected % (within)", "Expected % (overall)"])
    for column in outside.columns[1:]:
        outside[column] = 100.0 * outside[column]
    outside["Expected PPM (overall)"] = outside["Expected % (overall)"] * 1e4
    return Capability(lsl=lsl, usl=usl, target=target, mean=mean, sigma_within=sigma_within, sigma_overall=overall,
                      indices=indices, outside=outside.drop(columns=["Expected % (within)"]), distribution=name)
