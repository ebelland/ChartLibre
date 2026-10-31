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
