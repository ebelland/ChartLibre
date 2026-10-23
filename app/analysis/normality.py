"""Quick answer: "Is it normal?" - does a measure follow the normal (Gaussian) distribution, in words.

Asked before a t-test, an ANOVA, a capability or a control chart, all of
which assume it. The Shapiro-Wilk test (the most powerful of the usual ones;
Anderson-Darling past its 5000-value limit), the shape that departs (skewed,
heavy or light tails) and the two cautions a statistician would add: with
few values the test rarely objects, with many it objects to departures too
small to matter. The Q-Q plot shows which.

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

#: Shapiro-Wilk's p-value is accurate up to this many values.
SHAPIRO_MAX = 5000
#: Below this many values a "yes" is weak evidence; above LARGE a "no" may not matter.
SMALL, LARGE = 20, 200
#: |skewness| and |excess kurtosis| past which the shape is named.
SKEWED, TAILED = 0.5, 1.0


@dataclass(slots=True)
class NormalityAnswer:
    question: str
    #: YES (can be taken as normal) or NO.
    verdict: str
    sentences: list[tuple[str, dict[str, Any]]]
    #: The normal Q-Q plot: Theoretical (normal quantile), Sample (sorted value), Line (mean + sd * quantile).
    qq: pd.DataFrame
    test: str
    pvalue: float
    skewness: float
    kurtosis: float
    rows: int


def is_normal(frame: pd.DataFrame, measure: str) -> NormalityAnswer:
    """Can *measure* be taken as normally distributed?"""
    values = pd.to_numeric(frame[measure], errors="coerce").dropna().to_numpy(float)
    n = values.size
    if n < 3:
        raise ValueError(f"{n} values are too few to judge a distribution: at least 3 are needed.")
    if np.ptp(values) == 0:
        raise ValueError(f"every value of {measure} is the same: there is no distribution to judge.")
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    skewness, kurtosis = float(stats.skew(values, bias=False)), float(stats.kurtosis(values, bias=False))
    if n <= SHAPIRO_MAX:
        test, pvalue = "Shapiro-Wilk", float(stats.shapiro(values).pvalue)
    else:
        result = stats.anderson(values, dist="norm")
        test = "Anderson-Darling"
        # Its 5% critical value; the p-value is reported as below or above it.
        critical = float(result.critical_values[list(result.significance_level).index(5.0)])
        pvalue = 0.01 if result.statistic > critical else 0.5

    shapes = []
    if skewness > SKEWED:
        shapes.append("skewed to the right (a long tail of high values)")
    elif skewness < -SKEWED:
        shapes.append("skewed to the left (a long tail of low values)")
    if kurtosis > TAILED:
        shapes.append("heavy tails (more extreme values than a normal)")
    elif kurtosis < -TAILED:
        shapes.append("light tails (flatter than a normal, as a uniform)")
    values_dict = {"measure": measure, "n": n, "test": test, "p": pvalue, "skew": skewness, "kurt": kurtosis,
                   "shape": "; ".join(shapes)}

    sentences: list[tuple[str, dict[str, Any]]] = []
    normal = pvalue > ALPHA
    if normal:
        verdict = YES
        sentences.append(("Yes: {measure} can be taken as normal ({test} p = {p:.3g}).", values_dict))
    else:
        verdict = NO
        sentences.append(("No: {measure} is not normal ({test} p = {p:.3g}).", values_dict))
        if shapes:
            sentences.append(("Its shape: {shape}.", values_dict))
    sentences.append(("Skewness {skew:.2f} and excess kurtosis {kurt:.2f}; a normal has 0 for both.", values_dict))
    if normal and n < SMALL:
        sentences.append(("With {n} values the test seldom objects: a 'yes' here is weak evidence.", values_dict))
    if not normal and n >= LARGE:
        sentences.append(("With {n} values even a small departure is detected: if the points follow the "
                          "Q-Q plot's line closely, t-tests and ANOVA are safe at this size.", values_dict))
    sentences.append(("On the Q-Q plot a normal sample follows the straight line; a curve is skew, "
                      "an S is tails.", {}))

    sample = np.sort(values)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))  # Blom's positions
    qq = pd.DataFrame({"Theoretical": theoretical, "Sample": sample, "Line": mean + sd * theoretical})
    return NormalityAnswer(
        question=f"Is {measure} normal?", verdict=verdict, sentences=sentences, qq=qq, test=test,
        pvalue=pvalue, skewness=skewness, kurtosis=kurtosis, rows=n,
    )
