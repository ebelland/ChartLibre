"""Distribution of one column, as JMP's platform: quantiles, moments, normality, capability (AN-02).

The histogram, the Q-Q plot, the normality tests and the distribution fits
already exist as charts and series operations; this puts what a person asks
of one column in one report:

* the quantile table, 0% to 100% with the tails JMP shows (0.5%, 2.5%,
  10%, ...), each at the (n + 1)p position - the definition JMP and Minitab
  use, numpy's ``weibull`` method;
* the moments: mean, standard deviation, its standard error and the mean's
  95% confidence interval, skewness, kurtosis, coefficient of variation;
* the normality tests of app.analysis.statistics;
* process capability against specification limits, either or both:
  Cp, Cpk (and Cpl, Cpu) from the within sigma - the average moving range
  / d2 = 1.128, for individual measurements - and Pp, Ppk from the overall
  standard deviation; Cpm when there is a target; the share out of
  specification observed and expected under a normal, in parts per million.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from app.analysis import statistics as st

#: The quantiles shown, in percent, highest first (as JMP lists them).
QUANTILES: tuple[float, ...] = (100.0, 99.5, 97.5, 90.0, 75.0, 50.0, 25.0, 10.0, 2.5, 0.5, 0.0)
_QUANTILE_NAMES = {100.0: "maximum", 75.0: "quartile", 50.0: "median", 25.0: "quartile", 0.0: "minimum"}
#: d2 for subgroups of two: the moving range's expected value in sigmas.
D2_MOVING_RANGE = 1.128


@dataclass(slots=True)
class Capability:
    lsl: float | None
    usl: float | None
    target: float | None
    #: Index -> value: Cp, Cpk, Cpl, Cpu, Pp, Ppk, Ppl, Ppu, Cpm (those the limits allow).
    indices: dict[str, float]
    #: Rows: Below LSL, Above USL, Total outside - observed % and expected % / PPM (within, overall).
    outside: pd.DataFrame
    sigma_within: float
    sigma_overall: float


@dataclass(slots=True)
class DistributionResult:
    column: str
    n: int
    missing: int
    quantiles: pd.DataFrame
    moments: pd.DataFrame
    normality: pd.DataFrame
    capability: Capability | None = None
    notes: list[str] = field(default_factory=list)


def quantile_table(values: np.ndarray) -> pd.DataFrame:
    """Percent, its name (median, quartile...), and the value at (n + 1)p."""
    levels = np.asarray(QUANTILES) / 100.0
    found = np.quantile(values, levels, method="weibull")
    return pd.DataFrame({
        "Percent": [f"{q:g}%" for q in QUANTILES],
        "": [_QUANTILE_NAMES.get(q, "") for q in QUANTILES],
        "Value": found,
    })


def moments_table(values: np.ndarray, missing: int) -> pd.DataFrame:
    described = st.describe(values)
    mean, sd, n = described["mean"], described["std"], described["n"]
    rows = [
        ("Mean", mean),
        ("Std Dev", sd),
        ("Std Err Mean", described["sem"]),
        ("Upper 95% Mean", described["ci95_high"]),
        ("Lower 95% Mean", described["ci95_low"]),
        ("N", float(n)),
        ("N Missing", float(missing)),
        ("Sum", float(np.sum(values))),
        ("Variance", described["var"]),
        ("Skewness", described["skewness"]),
        ("Kurtosis", described["kurtosis"]),
        ("CV (%)", 100.0 * sd / mean if mean else np.nan),
    ]
    return pd.DataFrame(rows, columns=["Moment", "Value"])


def sigma_within(values: np.ndarray) -> float:
    """Short-term sigma of individual measurements: average moving range / d2."""
    if values.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(values))) / D2_MOVING_RANGE)


def capability(values: np.ndarray, *, lsl: float | None = None, usl: float | None = None,
               target: float | None = None) -> Capability:
    """Capability indices and the share out of specification; at least one limit."""
    if lsl is None and usl is None:
        raise ValueError("Capability needs a lower or an upper specification limit.")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("The lower specification limit must be below the upper one.")
    mean = float(np.mean(values))
    within, overall = sigma_within(values), float(np.std(values, ddof=1))
    indices: dict[str, float] = {}
    for prefix, sigma in (("C", within), ("P", overall)):
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

    def expected(sigma: float, side: str) -> float:
        if side == "below":
            return float(stats.norm.cdf((lsl - mean) / sigma)) if lsl is not None else 0.0
        return float(stats.norm.sf((usl - mean) / sigma)) if usl is not None else 0.0

    rows = []
    for label, side in (("Below LSL", "below"), ("Above USL", "above")):
        observed = float(np.mean(values < lsl)) if side == "below" and lsl is not None else (
            float(np.mean(values > usl)) if side == "above" and usl is not None else 0.0)
        rows.append((label, observed, expected(within, side), expected(overall, side)))
    total = tuple(sum(row[i] for row in rows) for i in (1, 2, 3))
    rows.append(("Total outside", *total))
    outside = pd.DataFrame(rows, columns=["Portion", "Observed %", "Expected % (within)", "Expected % (overall)"])
    for column in outside.columns[1:]:
        outside[column] = 100.0 * outside[column]
    outside["Expected PPM (overall)"] = outside["Expected % (overall)"] * 1e4
    return Capability(lsl=lsl, usl=usl, target=target, indices=indices, outside=outside,
                      sigma_within=within, sigma_overall=overall)


def distribution(frame: pd.DataFrame, column: str, *, lsl: float | None = None, usl: float | None = None,
                 target: float | None = None) -> DistributionResult:
    """The report of one numeric column; capability when a limit is given."""
    raw = pd.to_numeric(frame[column], errors="coerce")
    values = raw.dropna().to_numpy(float)
    missing = int(raw.isna().sum())
    if values.size < 2:
        raise ValueError(f"'{column}' has {values.size} numeric value(s); a distribution needs at least two.")
    notes = []
    if missing:
        notes.append(f"{missing} missing or non-numeric value(s) left out.")
    normality = pd.DataFrame(st.normality_tests(values))
    result = DistributionResult(column=column, n=int(values.size), missing=missing,
                                quantiles=quantile_table(values), moments=moments_table(values, missing),
                                normality=normality, notes=notes)
    if lsl is not None or usl is not None:
        result.capability = capability(values, lsl=lsl, usl=usl, target=target)
        if values.size < 30:
            notes.append("Fewer than 30 values: the capability indices are rough estimates.")
        normal_p = pd.to_numeric(normality.get("pvalue", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(normal_p) and float(normal_p.min()) < 0.05:
            notes.append("A normality test rejects the normal at 5%: the expected shares out of "
                         "specification, which assume it, may be off - compare them with the observed ones.")
    return result
