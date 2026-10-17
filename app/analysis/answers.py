"""Quick answers: a question in everyday words, the right test chosen, the answer in a sentence.

For someone who is not a statistician. ``differences`` answers "is there a
difference between these groups?": it looks at the data (numbers or
categories, how many groups, normal or not, how many rows) and picks the
test a statistician would - Welch's t or ANOVA, Mann-Whitney or
Kruskal-Wallis, chi-squared or Fisher - then says what it found, how large
it is, and what it does not mean.

The sentences are templates (English, translated by the dialog) with their
values, so the answer can be read in the window's language.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.analysis.statistics import cohens_d, tukey_hsd, welch_anova
from app.utils.config import get_constant

#: Below this p-value a difference is called shown.
ALPHA = get_constant("quick_answer_alpha", 0.05)

YES, NO = "yes", "no"


@dataclass(slots=True)
class Answer:
    """What was asked, what was found, and how."""

    question: str
    #: YES (a difference is shown) or NO (none is).
    verdict: str
    #: (template, values): the answer, then what it rests on, then the cautions.
    sentences: list[tuple[str, dict[str, Any]]]
    #: The test used, its p-value and its effect size (with its name).
    test: str
    pvalue: float
    effect: float
    effect_name: str
    #: One row per group: n and its summary (mean, median, std dev - or the shares).
    groups: pd.DataFrame
    #: The pairs found different, for more than two groups.
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    numeric: bool = True


def _size(value: float, small: float, medium: float, large: float) -> str:
    value = abs(value)
    if value >= large:
        return "large"
    if value >= medium:
        return "medium"
    if value >= small:
        return "small"
    return "negligible"


def _normal(values: np.ndarray) -> bool:
    """Normal enough for a t-test or ANOVA: many rows, or Shapiro-Wilk does not object."""
    if len(values) >= 30:
        return True
    if len(values) < 3 or np.ptp(values) == 0:
        return False
    return bool(stats.shapiro(values).pvalue > ALPHA)


def differences(frame: pd.DataFrame, measure: str, group: str) -> Answer:
    """Is *measure* different between the groups of *group*?"""
    data = frame[[measure, group]].dropna()
    data = data.assign(**{group: data[group].astype(str)})
    numbers = pd.to_numeric(data[measure], errors="coerce")
    numeric = bool(numbers.notna().all()) and data[measure].nunique() > 2
    if data[group].nunique() < 2:
        raise ValueError(f"{group} has fewer than two groups.")
    if numeric:
        return _numeric_differences(data.assign(**{measure: numbers}), measure, group)
    return _category_differences(data, measure, group)


def _numeric_differences(data: pd.DataFrame, measure: str, group: str) -> Answer:
    groups = {name: part[measure].to_numpy(dtype=float) for name, part in data.groupby(group, sort=True)}
    small = [name for name, values in groups.items() if len(values) < 2]
    groups = {name: values for name, values in groups.items() if len(values) >= 2}
    if len(groups) < 2:
        raise ValueError("At least two groups need two values or more.")
    names = list(groups)
    arrays = list(groups.values())
    normal = all(_normal(values) for values in arrays)
    summary = pd.DataFrame({
        "Group": names,
        "N": [len(v) for v in arrays],
        "Mean": [float(np.mean(v)) for v in arrays],
        "Median": [float(np.median(v)) for v in arrays],
        "Std Dev": [float(np.std(v, ddof=1)) for v in arrays],
    })
    if len(arrays) == 2:
        if normal:
            result = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
            test, pvalue = "Welch t-test", float(result.pvalue)
        else:
            result = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
            test, pvalue = "Mann-Whitney U test", float(result.pvalue)
        effect, effect_name = cohens_d(arrays[0], arrays[1]), "Cohen's d"
        size = _size(effect, 0.2, 0.5, 0.8)
    else:
        if normal:
            welch = welch_anova(arrays)
            test, pvalue = "Welch ANOVA", float(welch[3]) if welch else float(stats.f_oneway(*arrays).pvalue)
        else:
            test, pvalue = "Kruskal-Wallis test", float(stats.kruskal(*arrays).pvalue)
        everything = np.concatenate(arrays)
        between = sum(len(v) * (np.mean(v) - everything.mean()) ** 2 for v in arrays)
        total = float(np.sum((everything - everything.mean()) ** 2))
        effect, effect_name = (between / total if total > 0 else 0.0), "eta²"
        size = _size(effect, 0.01, 0.06, 0.14)

    centre = "Mean" if normal else "Median"
    ordered = summary.sort_values(centre)
    low, high = ordered.iloc[0], ordered.iloc[-1]
    values = {"measure": measure, "group": group, "high": high["Group"], "low": low["Group"],
              "high_value": float(high[centre]), "low_value": float(low[centre]), "p": pvalue,
              "size": size, "test": test, "centre": centre.lower(), "n": int(summary["N"].sum()),
              "groups": len(names)}
    sentences: list[tuple[str, dict[str, Any]]] = []
    pairs = pd.DataFrame()
    if pvalue < ALPHA:
        verdict = YES
        sentences.append(("Yes: {measure} differs between the groups of {group}.", values))
        sentences.append(("{high} has the highest {centre} ({high_value:.4g}), {low} the lowest ({low_value:.4g}).",
                          values))
        sentences.append(("The difference is {size}.", values))
        if len(names) > 2:
            rows = [row for row in tukey_hsd(groups) if row.get("pvalue", 1.0) < ALPHA]
            pairs = pd.DataFrame([{"Pair": row["pair"], "Difference": row["difference"], "p": row["pvalue"]}
                                  for row in rows])
            if not pairs.empty:
                sentences.append(("The groups that differ: {pairs}.", {"pairs": ", ".join(pairs["Pair"])}))
    else:
        verdict = NO
        sentences.append(("No difference is shown: these data do not say that {measure} changes with {group}.",
                          values))
        sentences.append(("That is not proof that the groups are the same: with more rows a small difference "
                          "might show.", values))
    sentences.append(("How it was decided: {test}, p = {p:.3g} (below 0.05 is called a difference), "
                      "on {n} rows in {groups} groups.", values))
    if not normal:
        sentences.append(("The values are not normally distributed, so the test compares medians and ranks "
                          "rather than means.", values))
    if small:
        sentences.append(("Left out, with fewer than two values: {names}.", {"names": ", ".join(small)}))
    if summary["N"].min() < 5:
        sentences.append(("Some groups have fewer than five values: treat the answer with caution.", values))
    return Answer("Is there a difference between these groups?", verdict, sentences, test, pvalue, float(effect),
                  effect_name, summary, pairs, numeric=True)


def _category_differences(data: pd.DataFrame, measure: str, group: str) -> Answer:
    table = pd.crosstab(data[group], data[measure].astype(str))
    counts = table.to_numpy()
    expected = stats.contingency.expected_freq(counts)
    if counts.shape == (2, 2) and expected.min() < 5:
        test, pvalue = "Fisher exact test", float(stats.fisher_exact(counts).pvalue)
    else:
        test, pvalue = "Chi-squared test", float(stats.chi2_contingency(counts, correction=False).pvalue)
    chi2 = float(stats.chi2_contingency(counts, correction=False).statistic)
    n = float(counts.sum())
    k = min(counts.shape) - 1
    effect = math.sqrt(chi2 / (n * k)) if n and k else 0.0
    shares = table.div(table.sum(axis=1), axis=0)
    top = shares.columns[np.argmax(table.sum(axis=0).to_numpy())]
    most = shares[top].idxmax()
    least = shares[top].idxmin()
    summary = shares.reset_index().rename(columns={group: "Group"})
    summary.insert(1, "N", table.sum(axis=1).to_numpy())
    values = {"measure": measure, "group": group, "level": top, "high": most, "low": least,
              "high_value": 100 * float(shares.loc[most, top]), "low_value": 100 * float(shares.loc[least, top]),
              "p": pvalue, "size": _size(effect, 0.1, 0.3, 0.5), "test": test, "n": int(n),
              "groups": len(table.index)}
    if pvalue < ALPHA:
        verdict = YES
        sentences = [
            ("Yes: the share of each {measure} differs between the groups of {group}.", values),
            ("{level} is most frequent in {high} ({high_value:.3g}%) and least in {low} ({low_value:.3g}%).", values),
            ("The difference is {size}.", values),
        ]
    else:
        verdict = NO
        sentences = [
            ("No difference is shown: these data do not say that {measure} changes with {group}.", values),
            ("That is not proof that the groups are the same: with more rows a small difference might show.", values),
        ]
    sentences.append(("How it was decided: {test}, p = {p:.3g} (below 0.05 is called a difference), "
                      "on {n} rows in {groups} groups.", values))
    if expected.min() < 5 and test != "Fisher exact test":
        sentences.append(("Some expected counts are below five: treat the answer with caution.", values))
    return Answer("Is there a difference between these groups?", verdict, sentences, test, pvalue, effect,
                  "Cramér's V", summary, numeric=False)
