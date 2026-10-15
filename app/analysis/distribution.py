"""Distribution of one column, as JMP's platform: quantiles, moments, normality (AN-02).

The histogram, the Q-Q plot, the normality tests and the distribution fits
already exist as charts and series operations; this puts what a person asks
of one column in one report:

* the quantile table, 0% to 100% with the tails JMP shows (0.5%, 2.5%,
  10%, ...), each at the (n + 1)p position - the definition JMP and Minitab
  use, numpy's ``weibull`` method;
* the moments: mean, standard deviation, its standard error and the mean's
  95% confidence interval, skewness, kurtosis, coefficient of variation;
* the normality tests of app.analysis.statistics.

Process capability is the control chart's (app.analysis.capability): it
needs a stable process and the chart's within sigma.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analysis import statistics as st

#: The quantiles shown, in percent, highest first (as JMP lists them).
QUANTILES: tuple[float, ...] = (100.0, 99.5, 97.5, 90.0, 75.0, 50.0, 25.0, 10.0, 2.5, 0.5, 0.0)
_QUANTILE_NAMES = {100.0: "maximum", 75.0: "quartile", 50.0: "median", 25.0: "quartile", 0.0: "minimum"}


@dataclass(slots=True)
class DistributionResult:
    column: str
    n: int
    missing: int
    quantiles: pd.DataFrame
    moments: pd.DataFrame
    normality: pd.DataFrame
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



def distribution(frame: pd.DataFrame, column: str) -> DistributionResult:
    """The report of one numeric column."""
    raw = pd.to_numeric(frame[column], errors="coerce")
    values = raw.dropna().to_numpy(float)
    missing = int(raw.isna().sum())
    if values.size < 2:
        raise ValueError(f"'{column}' has {values.size} numeric value(s); a distribution needs at least two.")
    notes = []
    if missing:
        notes.append(f"{missing} missing or non-numeric value(s) left out.")
    normality = pd.DataFrame(st.normality_tests(values))
    return DistributionResult(column=column, n=int(values.size), missing=missing,
                              quantiles=quantile_table(values), moments=moments_table(values, missing),
                              normality=normality, notes=notes)
