"""Descriptive statistics and hypothesis tests on plain arrays.

The engine behind the Statistics series operation. Nothing here knows about
Qt, the repository or a chart: every function takes numbers and returns
numbers, so it can be tested against reference values and called from a
script exactly as the dialog calls it.

A test result is a row dictionary - ``test``, ``n``, ``statistic``,
``pvalue``, ``note`` - which is what the report prints. A test that cannot
run on the data it was given returns a row with only a note saying why,
rather than raising: one unusable series must not blank the whole report.
"""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from app.utils.i18n import _

#: The alternatives SciPy's tests accept.


# ----------------------------------------------------------------------
# Result rows
# ----------------------------------------------------------------------
def result_row(test: str, n: int, statistic: Any, pvalue: Any, note: str = "") -> dict[str, Any]:
    """One line of a test table."""
    return {"test": test, "n": int(n), "statistic": statistic, "pvalue": pvalue, "note": note}


def note_row(test: str, n: int, note: str) -> dict[str, Any]:
    """A test that could not run, and why."""
    return {"test": test, "n": int(n), "statistic": np.nan, "pvalue": np.nan, "note": note}


def finite(values: Any) -> np.ndarray:
    """*values* as a float array with NaN and infinities removed."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _statistic(result: Any) -> float:
    return float(result[0])


def _pvalue(result: Any) -> float:
    return float(result[1])


def _named_pvalue(result: Any) -> float:
    return float(getattr(result, "pvalue", np.nan))


# ----------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------
def describe(values: Any, *, trim_proportion: float = 0.1) -> dict[str, Any]:
    """Location, spread and shape of one sample; ``{"n": 0}`` when empty."""
    values = finite(values)
    n = int(values.size)
    if n == 0:
        return {"n": 0}

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    sem = float(stats.sem(values, nan_policy="omit")) if n > 1 else np.nan
    ci_low = ci_high = np.nan
    if n > 1 and np.isfinite(sem):
        ci = stats.t.interval(0.95, df=n - 1, loc=float(np.mean(values)), scale=sem)
        ci_low, ci_high = float(ci[0]), float(ci[1])

    mode_res = stats.mode(values, keepdims=False)
    try:
        mode_value = float(mode_res.mode)
        mode_count = int(mode_res.count)
    except (TypeError, ValueError):
        mode_value = np.nan
        mode_count = 0

    positive = values[values > 0]
    entropy = np.nan
    if positive.size > 0 and float(np.sum(positive)) > 0:
        probs = positive / float(np.sum(positive))
        entropy = float(stats.entropy(probs))

    return {
        "n": n,
        "mean": float(np.mean(values)),
        "trimmed_mean": float(stats.trim_mean(values, proportiontocut=trim_proportion)),
        "gmean": float(stats.gmean(positive)) if positive.size == n else np.nan,
        "hmean": float(stats.hmean(positive)) if positive.size == n else np.nan,
        "median": float(median),
        "mode": mode_value,
        "mode_count": mode_count,
        "std": float(np.std(values, ddof=1)) if n > 1 else np.nan,
        "var": float(np.var(values, ddof=1)) if n > 1 else np.nan,
        "sem": sem,
        "min": float(np.min(values)),
        "q1": float(q1),
        "q3": float(q3),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "iqr": float(stats.iqr(values)),
        "mad": float(stats.median_abs_deviation(values, scale=1.4826)),
        "skewness": float(stats.skew(values, bias=False)) if n > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(values, bias=False)) if n > 3 else np.nan,
        "entropy": entropy,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


# ----------------------------------------------------------------------
# One sample
# ----------------------------------------------------------------------
def one_sample_tests(values: Any, *, popmean: float = 0.0, alternative: str = "two-sided") -> list[dict[str, Any]]:
    """Is the sample's centre *popmean*? t, Wilcoxon signed-rank and sign test."""
    values = finite(values)
    tests: list[dict[str, Any]] = []

    if values.size > 1:
        res = stats.ttest_1samp(values, popmean=popmean, alternative=alternative)
        tests.append(result_row(_("One-sample t-test"), values.size, _statistic(res), _pvalue(res), f"mean = {popmean:g}"))

    diff = values - float(popmean)
    nonzero = diff[np.abs(diff) > 0]
    if nonzero.size > 0:
        try:
            res = stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
            tests.append(result_row(_("Wilcoxon signed-rank"), nonzero.size, _statistic(res), _pvalue(res), f"median = {popmean:g}"))
        except ValueError as exc:
            tests.append(note_row(_("Wilcoxon signed-rank"), nonzero.size, str(exc)))

        positives = int(np.sum(diff > 0))
        trials = positives + int(np.sum(diff < 0))
        if trials > 0:
            res = stats.binomtest(positives, trials, p=0.5, alternative=alternative)
            tests.append(result_row(_("Sign test"), trials, positives, _named_pvalue(res), "positive signs"))
    return tests


# ----------------------------------------------------------------------
# Normality
# ----------------------------------------------------------------------
def anderson_normality(
    values: Any,
    *,
    method: str = "interpolate",
    resamples: int = 9999,
    batch: int | None = None,
) -> tuple[Any, str]:
    """Anderson-Darling against the normal, with the p-value *method* asked for."""
    values = finite(values)
    if method == "monte_carlo":
        monte_carlo = stats.MonteCarloMethod(n_resamples=int(resamples), batch=batch)
        return (
            stats.anderson(values, dist="norm", method=monte_carlo),
            f"Monte Carlo p-value; resamples={int(resamples):,}; batch={batch or 'Auto'}",
        )
    try:
        return stats.anderson(values, dist="norm", method="interpolate"), "Interpolated p-value"
    except ValueError:
        return stats.anderson(values, dist="norm", method="interpolated"), "Interpolated p-value"
    except TypeError:
        # Older SciPy versions do not have the method parameter.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return (
                stats.anderson(values, dist="norm"),
                "Critical values only; installed SciPy does not support p-value method",
            )


def normality_tests(
    values: Any,
    *,
    anderson_method: str = "interpolate",
    resamples: int = 9999,
    batch: int | None = None,
) -> list[dict[str, Any]]:
    """Shapiro-Wilk, D'Agostino, skewness, kurtosis, Jarque-Bera, KS and Anderson-Darling."""
    values = finite(values)
    n = int(values.size)
    tests: list[dict[str, Any]] = []

    def run(name: str, func) -> None:
        try:
            res = func(values)
            tests.append(result_row(name, n, _statistic(res), _pvalue(res)))
        except Exception as exc:  # SciPy raises several types for degenerate samples
            tests.append(note_row(name, n, str(exc)))

    if n >= 3:
        run(_("Shapiro-Wilk normality"), stats.shapiro)
    if n >= 8:
        run("D'Agostino-Pearson normality", stats.normaltest)
        run("Skewness test", stats.skewtest)
    if n >= 5:
        run(_("Kurtosis test"), stats.kurtosistest)
    if n >= 2:
        run(_("Jarque-Bera normality"), stats.jarque_bera)
        std = float(np.std(values, ddof=1))
        if std > 0:
            z = (values - float(np.mean(values))) / std
            try:
                res = stats.kstest(z, "norm")
                tests.append(result_row(_("Kolmogorov-Smirnov vs normal"), n, _statistic(res), _pvalue(res), "standardized sample"))
            except Exception as exc:
                tests.append(note_row(_("Kolmogorov-Smirnov vs normal"), n, str(exc)))
        try:
            res, note = anderson_normality(values, method=anderson_method, resamples=resamples, batch=batch)
            tests.append(
                result_row(
                    "Anderson-Darling normality",
                    n,
                    float(getattr(res, "statistic", np.nan)),
                    float(getattr(res, "pvalue", np.nan)),
                    note,
                )
            )
        except Exception as exc:
            tests.append(note_row(_("Anderson-Darling normality"), n, str(exc)))
    return tests


# ----------------------------------------------------------------------
# Two samples measured on the same units (paired)
# ----------------------------------------------------------------------
def paired_values(
    left_x: Any, left_y: Any, right_x: Any, right_y: Any
) -> tuple[np.ndarray, np.ndarray, str]:
    """Pair two samples on their shared x values, or by row order when none match."""
    left_by_x: dict[float, float] = {}
    for x_value, y_value in zip(left_x, left_y, strict=False):
        if np.isfinite(x_value) and np.isfinite(y_value):
            left_by_x.setdefault(float(x_value), float(y_value))

    paired_left: list[float] = []
    paired_right: list[float] = []
    for x_value, y_value in zip(right_x, right_y, strict=False):
        key = float(x_value)
        if key in left_by_x and np.isfinite(y_value):
            paired_left.append(left_by_x[key])
            paired_right.append(float(y_value))
    if paired_left:
        return np.asarray(paired_left), np.asarray(paired_right), "common X values"

    left_y, right_y = np.asarray(left_y, dtype=float), np.asarray(right_y, dtype=float)
    n = min(left_y.size, right_y.size)
    return left_y[:n], right_y[:n], "row order, truncated to common length"


def paired_tests(a: Any, b: Any, *, alignment: str = "", alternative: str = "two-sided") -> list[dict[str, Any]]:
    """Paired t, Wilcoxon signed-rank, sign test and normality of the differences."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n == 0:
        return [note_row(_("Paired tests"), 0, "No paired observations.")]

    diff = a - b
    tests = [result_row(_("Paired difference summary"), n, float(np.mean(diff)), np.nan, f"mean difference; {alignment}")]
    if n > 1:
        res = stats.ttest_rel(a, b, alternative=alternative)
        tests.append(result_row(_("Paired t-test"), n, _statistic(res), _pvalue(res), alignment))

    nonzero = diff[np.abs(diff) > 0]
    if nonzero.size > 0:
        try:
            res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
            tests.append(result_row(_("Wilcoxon signed-rank paired test"), nonzero.size, _statistic(res), _pvalue(res), alignment))
        except ValueError as exc:
            tests.append(note_row(_("Wilcoxon signed-rank paired test"), nonzero.size, str(exc)))
        positives = int(np.sum(diff > 0))
        trials = positives + int(np.sum(diff < 0))
        if trials > 0:
            res = stats.binomtest(positives, trials, p=0.5, alternative=alternative)
            tests.append(result_row(_("Sign test"), trials, positives, _named_pvalue(res), f"positive signs; {alignment}"))

    if n >= 3:
        try:
            res = stats.shapiro(diff)
            tests.append(result_row(_("Normality of paired differences"), n, _statistic(res), _pvalue(res), alignment))
        except Exception as exc:
            tests.append(note_row(_("Normality of paired differences"), n, str(exc)))
    return tests


def correlation_tests(a: Any, b: Any, *, alignment: str = "") -> list[dict[str, Any]]:
    """Pearson, Spearman, Kendall and the least-squares slope of b on a."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 2:
        return [note_row(_("Correlation tests"), n, "Need at least two paired observations.")]

    tests: list[dict[str, Any]] = []
    for name, func in (
        ("Pearson correlation", stats.pearsonr),
        ("Spearman rank correlation", stats.spearmanr),
        ("Kendall tau", stats.kendalltau),
    ):
        try:
            res = func(a, b)
            tests.append(result_row(name, n, _statistic(res), _pvalue(res), alignment))
        except Exception as exc:
            tests.append(note_row(name, n, str(exc)))
    try:
        res = stats.linregress(a, b)
        # LinregressResult: slope is [0], p-value [3] - its fields are not
        # annotated, so they are read by position.
        tests.append(result_row(_("Linear regression slope"), n, float(res[0]), float(res[3]), alignment))
    except Exception as exc:
        tests.append(note_row(_("Linear regression slope"), n, str(exc)))
    return tests


# ----------------------------------------------------------------------
# Two independent samples
# ----------------------------------------------------------------------
def cohens_d(a: Any, b: Any) -> float:
    """Standardised mean difference with the pooled standard deviation."""
    a, b = finite(a), finite(b)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    pooled = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if pooled <= 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def independent_tests(a: Any, b: Any, *, alternative: str = "two-sided") -> list[dict[str, Any]]:
    """Two groups of different units: are their centres the same?

    Welch's t first, because it does not assume equal variances and loses
    almost nothing when they are equal - the reason it is the default in R
    and the one SPSS and JMP print beside Student's. Levene's test says
    whether the difference between the two matters here; Mann-Whitney is
    the rank-based answer when the data are far from normal.
    """
    a, b = finite(a), finite(b)
    na, nb = int(a.size), int(b.size)
    n = na + nb
    if na < 2 or nb < 2:
        return [note_row(_("Independent-sample tests"), n, "Each group needs at least two values.")]

    sizes = f"n = {na} + {nb}"
    d = cohens_d(a, b)
    tests = [
        result_row(
            _("Difference of means"),
            n,
            float(np.mean(a) - np.mean(b)),
            np.nan,
            f"{sizes}; Cohen's d = {d:.3g}" if np.isfinite(d) else sizes,
        )
    ]
    res = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    welch_df = float(getattr(res, "df"))  # TtestResult.df is real; its stub does not list it
    tests.append(result_row(_("Welch t-test"), n, _statistic(res), _pvalue(res), f"df = {welch_df:.4g}"))
    res = stats.ttest_ind(a, b, equal_var=True, alternative=alternative)
    tests.append(result_row(_("Student t-test (equal variances)"), n, _statistic(res), _pvalue(res), f"df = {na + nb - 2}"))
    try:
        res = stats.mannwhitneyu(a, b, alternative=alternative)
        tests.append(result_row(_("Mann-Whitney U"), n, _statistic(res), _pvalue(res)))
    except ValueError as exc:
        tests.append(note_row(_("Mann-Whitney U"), n, str(exc)))
    try:
        res = stats.levene(a, b, center="median")
        tests.append(result_row(_("Levene equal variances"), n, _statistic(res), _pvalue(res), "Brown-Forsythe (median)"))
    except ValueError as exc:
        tests.append(note_row(_("Levene equal variances"), n, str(exc)))
    return tests


# ----------------------------------------------------------------------
# Several groups
# ----------------------------------------------------------------------
def group_tests(groups: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Do two or more groups share one mean? ANOVA, Welch's ANOVA, Kruskal-Wallis.

    Effect sizes go in the note: eta squared for the ANOVA (the share of the
    variance the grouping explains), epsilon squared for Kruskal-Wallis.
    """
    samples = {name: finite(values) for name, values in groups.items()}
    samples = {name: values for name, values in samples.items() if values.size}
    k = len(samples)
    n = int(sum(values.size for values in samples.values()))
    if k < 2:
        return [note_row(_("Group comparison"), n, "Needs at least two groups with values.")]
    if any(values.size < 2 for values in samples.values()):
        return [note_row(_("Group comparison"), n, "Every group needs at least two values.")]

    arrays = list(samples.values())
    tests: list[dict[str, Any]] = []

    res = stats.f_oneway(*arrays)
    everything = np.concatenate(arrays)
    grand = float(np.mean(everything))
    between = float(sum(v.size * (np.mean(v) - grand) ** 2 for v in arrays))
    total = float(np.sum((everything - grand) ** 2))
    eta2 = between / total if total > 0 else np.nan
    tests.append(
        result_row(
            _("One-way ANOVA"), n, _statistic(res), _pvalue(res),
            f"df = {k - 1}, {n - k}; eta² = {eta2:.3g}",
        )
    )

    welch = welch_anova(arrays)
    if welch is not None:
        f_value, df1, df2, p_value = welch
        tests.append(result_row(_("Welch ANOVA (unequal variances)"), n, f_value, p_value, f"df = {df1:.4g}, {df2:.4g}"))

    try:
        res = stats.kruskal(*arrays)
        epsilon2 = float(res.statistic) / ((n ** 2 - 1) / (n + 1)) if n > 1 else np.nan
        tests.append(result_row(_("Kruskal-Wallis"), n, _statistic(res), _pvalue(res), f"df = {k - 1}; epsilon² = {epsilon2:.3g}"))
    except ValueError as exc:
        tests.append(note_row(_("Kruskal-Wallis"), n, str(exc)))

    try:
        res = stats.levene(*arrays, center="median")
        tests.append(result_row(_("Levene equal variances"), n, _statistic(res), _pvalue(res), "Brown-Forsythe (median)"))
    except ValueError as exc:
        tests.append(note_row(_("Levene equal variances"), n, str(exc)))
    return tests


def welch_anova(arrays: Sequence[np.ndarray]) -> tuple[float, float, float, float] | None:
    """Welch's heteroscedastic one-way ANOVA: (F, df1, df2, p), or None.

    Written out rather than taken from a library: SciPy has none, and the
    formula (Welch 1951) is short. None when a group has no variance, where
    its weight would be infinite.
    """
    k = len(arrays)
    sizes = np.array([a.size for a in arrays], dtype=float)
    means = np.array([np.mean(a) for a in arrays], dtype=float)
    variances = np.array([np.var(a, ddof=1) for a in arrays], dtype=float)
    if k < 2 or np.any(variances <= 0) or np.any(sizes < 2):
        return None
    weights = sizes / variances
    total_weight = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / total_weight)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2)) / (k - 1)
    lam = float(np.sum((1 - weights / total_weight) ** 2 / (sizes - 1)))
    denominator = 1 + 2 * (k - 2) * lam / (k ** 2 - 1)
    f_value = numerator / denominator
    df1 = float(k - 1)
    df2 = float((k ** 2 - 1) / (3 * lam)) if lam > 0 else np.inf
    p_value = float(stats.f.sf(f_value, df1, df2))
    return float(f_value), df1, df2, p_value


def tukey_hsd(groups: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Tukey's honestly significant difference: every pair of groups.

    Rows carry ``pair``, ``difference`` (first minus second), ``pvalue``
    already adjusted for the number of pairs, and the 95% confidence
    interval of the difference.
    """
    samples = {name: finite(values) for name, values in groups.items()}
    samples = {name: values for name, values in samples.items() if values.size >= 2}
    names = list(samples)
    if len(names) < 2:
        return []
    result = stats.tukey_hsd(*samples.values())
    interval = result.confidence_interval(confidence_level=0.95)
    rows: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(names)), 2):
        rows.append(
            {
                "pair": f"{names[i]} - {names[j]}",
                "difference": float(result.statistic[i, j]),
                "pvalue": float(result.pvalue[i, j]),
                "ci_low": float(interval.low[i, j]),
                "ci_high": float(interval.high[i, j]),
            }
        )
    return rows
