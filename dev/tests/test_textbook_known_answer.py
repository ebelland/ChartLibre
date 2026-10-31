"""Known answers where NIST has no dataset (ID40): the tests checked against R's printed results.

The data are R's own datasets - ``sleep`` (Student, 1908), ``PlantGrowth`` (Dobson, 1983) and the
first of Anscombe's quartet - and the expected values are what R prints for them (``t.test``,
``aov``, ``oneway.test``, ``kruskal.test``, ``TukeyHSD``, ``cor``, ``lm``), to the digits R shows.
"""
from __future__ import annotations

import pytest

from app.analysis import statistics as st

#: ``sleep``: extra hours of sleep of ten patients under two drugs.
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]

#: ``PlantGrowth``: dried weight of plants, a control and two treatments.
PLANT_GROWTH = {
    "ctrl": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14],
    "trt1": [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69],
    "trt2": [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26],
}

#: Anscombe's first data set: y = 3.00 + 0.500 x, r = 0.816.
ANSCOMBE_X = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE_Y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]


def _row(rows: list[dict], test: str) -> dict:
    return next(row for row in rows if row["test"] == test)


def test_paired_t_on_student_sleep() -> None:
    """t.test(extra ~ group, paired = TRUE): t = -4.0621, df = 9, p-value = 0.002833."""
    row = _row(st.paired_tests(SLEEP_1, SLEEP_2), "Paired t-test")
    assert row["statistic"] == pytest.approx(-4.0621, abs=5e-5)
    assert row["pvalue"] == pytest.approx(0.002833, abs=5e-7)


def test_welch_t_on_student_sleep() -> None:
    """t.test(extra ~ group): t = -1.8608, df = 17.776, p-value = 0.07939."""
    row = _row(st.independent_tests(SLEEP_1, SLEEP_2), "Welch t-test")
    assert row["statistic"] == pytest.approx(-1.8608, abs=5e-5)
    assert row["pvalue"] == pytest.approx(0.07939, abs=5e-6)
    assert row["note"] == "df = 17.78"


def test_anova_welch_and_kruskal_on_plant_growth() -> None:
    """aov: F = 4.846 on 2 and 27 df, p = 0.0159; oneway.test: F = 5.181, df = 2, 17.128,
    p = 0.01739; kruskal.test: chi-squared = 7.9882, df = 2, p = 0.01842."""
    rows = st.group_tests(PLANT_GROWTH)
    anova = _row(rows, "One-way ANOVA")
    assert anova["statistic"] == pytest.approx(4.846, abs=5e-4)
    assert anova["pvalue"] == pytest.approx(0.01591, abs=5e-6)
    welch = _row(rows, "Welch ANOVA (unequal variances)")
    assert welch["statistic"] == pytest.approx(5.181, abs=5e-4)
    assert welch["pvalue"] == pytest.approx(0.01739, abs=5e-6)
    assert "17.13" in welch["note"]
    kruskal = _row(rows, "Kruskal-Wallis")
    assert kruskal["statistic"] == pytest.approx(7.9882, abs=5e-5)
    assert kruskal["pvalue"] == pytest.approx(0.01842, abs=5e-6)


def test_tukey_on_plant_growth() -> None:
    """TukeyHSD(aov(weight ~ group)): trt1-ctrl -0.371 (p 0.3909), trt2-ctrl 0.494 (p 0.1980),
    trt2-trt1 0.865 (p 0.0120); R's differences are second minus first."""
    rows = {row["pair"]: row for row in st.tukey_hsd(PLANT_GROWTH)}
    for pair, difference, pvalue in (("ctrl - trt1", 0.371, 0.3909), ("ctrl - trt2", -0.494, 0.1980),
                                     ("trt1 - trt2", -0.865, 0.0120)):
        assert rows[pair]["difference"] == pytest.approx(difference, abs=5e-4)
        assert rows[pair]["pvalue"] == pytest.approx(pvalue, abs=5e-4)


def test_correlation_and_slope_on_anscombe() -> None:
    """cor(x1, y1) = 0.8164205; lm(y1 ~ x1): slope 0.5001, Pr(>|t|) = 0.00217."""
    rows = st.correlation_tests(ANSCOMBE_X, ANSCOMBE_Y)
    assert _row(rows, "Pearson correlation")["statistic"] == pytest.approx(0.8164205, abs=5e-8)
    slope = _row(rows, "Linear regression slope")
    assert slope["statistic"] == pytest.approx(0.5001, abs=5e-5)
    assert slope["pvalue"] == pytest.approx(0.00217, abs=5e-6)


#: ``InsectSprays``: insects counted on plots treated with six sprays.
INSECT_SPRAYS = {
    "A": [10, 7, 20, 14, 14, 12, 10, 23, 17, 20, 14, 13], "B": [11, 17, 21, 11, 16, 14, 17, 17, 19, 21, 7, 13],
    "C": [0, 1, 7, 2, 3, 1, 2, 1, 3, 0, 1, 4], "D": [3, 5, 12, 6, 4, 3, 5, 5, 5, 5, 2, 4],
    "E": [3, 5, 3, 5, 3, 6, 1, 1, 3, 2, 6, 4], "F": [11, 9, 15, 22, 15, 16, 13, 10, 26, 26, 24, 13],
}

#: ``friedman.test``'s RoundingTimes (Hollander & Wolfe): 22 players, three ways of rounding first base.
ROUNDING_TIMES = [
    [5.40, 5.50, 5.55], [5.85, 5.70, 5.75], [5.20, 5.60, 5.50], [5.55, 5.50, 5.40], [5.90, 5.85, 5.70],
    [5.45, 5.55, 5.60], [5.40, 5.40, 5.35], [5.45, 5.50, 5.35], [5.25, 5.15, 5.00], [5.85, 5.80, 5.70],
    [5.25, 5.20, 5.10], [5.65, 5.55, 5.45], [5.60, 5.35, 5.45], [5.05, 5.00, 4.95], [5.50, 5.50, 5.40],
    [5.45, 5.55, 5.50], [5.55, 5.55, 5.35], [5.45, 5.50, 5.55], [5.50, 5.45, 5.25], [5.65, 5.60, 5.40],
    [5.70, 5.65, 5.55], [6.30, 6.30, 6.25],
]


def test_kruskal_and_fligner_on_insect_sprays() -> None:
    """kruskal.test: chi-squared = 54.691, df = 5, p = 1.511e-10; fligner.test: med chi-squared = 14.483,
    df = 5, p = 0.01282."""
    rows = st.group_tests(INSECT_SPRAYS)
    kruskal = _row(rows, "Kruskal-Wallis")
    assert kruskal["statistic"] == pytest.approx(54.691, abs=5e-4)
    assert kruskal["pvalue"] == pytest.approx(1.511e-10, rel=5e-4)
    fligner = _row(rows, "Fligner-Killeen equal variances")
    assert fligner["statistic"] == pytest.approx(14.483, abs=5e-4)
    assert fligner["pvalue"] == pytest.approx(0.01282, abs=5e-6)
    assert _row(rows, "Mood median test")["pvalue"] < 1e-6  # C, D, E all under the grand median


def test_friedman_on_rounding_times() -> None:
    """friedman.test(RoundingTimes): Friedman chi-squared = 11.143, df = 2, p-value = 0.003805."""
    rows = st.repeated_tests(ROUNDING_TIMES)
    friedman = _row(rows, "Friedman test")
    assert friedman["statistic"] == pytest.approx(11.143, abs=5e-4)
    assert friedman["pvalue"] == pytest.approx(0.003805, abs=5e-7)
    assert _row(rows, "Page trend test")["pvalue"] > 0.5  # the times fall, they do not rise


def test_prediction_and_tolerance_intervals() -> None:
    """Prediction: mean +- t(0.975, n-1) s sqrt(1 + 1/n). Tolerance, 95% / 95%, n = 10: the exact k of
    the tables is 3.379, Howe's approximation 3.382."""
    values = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
    prediction, tolerance = (row for row in st.one_sample_tests(values) if "interval" in row["test"])
    assert prediction["statistic"] == pytest.approx(2.262157 * (1.1 ** 0.5), abs=5e-6)
    assert tolerance["statistic"] == pytest.approx(3.379, rel=1e-3)


def test_aligned_values_pair_on_shared_x() -> None:
    table, alignment = st.aligned_values([([1, 2, 3], [10, 20, 30]), ([3, 2, 9], [33, 22, 99]), ([2, 3], [2.5, 3.5])])
    assert alignment == "common X values" and table.tolist() == [[20, 22, 2.5], [30, 33, 3.5]]
