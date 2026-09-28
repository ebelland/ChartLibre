"""The statistics engine, checked against published and independent values.

No Qt here: that the calculations run without it is the point of
app.analysis (todo.txt R-01).
"""
from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from app.analysis import statistics as st

# Shell length / muscle scar measurements of the mussel Mytilus trossulus at
# five sites (McDonald et al. 1991) - the example in SciPy's f_oneway
# documentation, which gives F = 7.1210, p = 0.0002812.
MUSSELS = {
    "Tillamook": [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836],
    "Newport": [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725],
    "Petersburg": [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105],
    "Magadan": [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689],
    "Tvarminne": [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045],
}


def _row(rows: list[dict], name: str) -> dict:
    return next(row for row in rows if row["test"] == name)


def test_one_way_anova_matches_the_published_example() -> None:
    anova = _row(st.group_tests(MUSSELS), "One-way ANOVA")
    assert anova["statistic"] == pytest.approx(7.1210, abs=1e-4)
    assert anova["pvalue"] == pytest.approx(0.0002812, rel=1e-3)
    assert anova["n"] == 39


def test_welch_anova_agrees_with_statsmodels() -> None:
    arrays = [np.asarray(v) for v in MUSSELS.values()]
    f_value, df1, df2, p_value = st.welch_anova(arrays)
    reference = anova_oneway(arrays, use_var="unequal")
    assert f_value == pytest.approx(reference.statistic, rel=1e-9)
    assert df2 == pytest.approx(reference.df[1], rel=1e-9)
    assert p_value == pytest.approx(reference.pvalue, rel=1e-9)
    assert df1 == 4


def test_tukey_hsd_agrees_with_statsmodels() -> None:
    rows = st.tukey_hsd(MUSSELS)
    assert len(rows) == 10  # five groups, every pair once
    values = np.concatenate([np.asarray(v) for v in MUSSELS.values()])
    labels = np.concatenate([[name] * len(v) for name, v in MUSSELS.items()])
    reference = pairwise_tukeyhsd(values, labels)
    ours = {tuple(sorted(row["pair"].split(" - "))): row["pvalue"] for row in rows}
    for group1, group2, p_adj in zip(
        reference.groupsunique[reference._multicomp.pairindices[0]],
        reference.groupsunique[reference._multicomp.pairindices[1]],
        reference.pvalues,
        strict=True,
    ):
        assert ours[tuple(sorted((group1, group2)))] == pytest.approx(p_adj, abs=1e-3)


def test_independent_tests_report_welch_student_and_mann_whitney() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(10, 1, 40), rng.normal(11, 3, 25)
    rows = st.independent_tests(a, b)
    names = [row["test"] for row in rows]
    assert names[:5] == [
        "Difference of means",
        "Welch t-test",
        "Student t-test (equal variances)",
        "Mann-Whitney U",
        "Levene equal variances",
    ]
    assert _row(rows, "Difference of means")["statistic"] == pytest.approx(a.mean() - b.mean())
    # Very different spreads: Levene should notice.
    assert _row(rows, "Levene equal variances")["pvalue"] < 0.01


def test_cohens_d_of_a_one_sd_shift_is_one() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sd = np.std(a, ddof=1)
    assert st.cohens_d(a + sd, a) == pytest.approx(1.0)


def test_an_unusable_sample_gives_a_note_not_an_exception() -> None:
    assert st.independent_tests([1.0], [2.0, 3.0])[0]["note"]
    assert st.group_tests({"only": [1.0, 2.0]})[0]["note"]
    assert st.describe([np.nan]) == {"n": 0}


def test_describe_and_one_sample_on_known_values() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    desc = st.describe(values)
    assert desc["mean"] == pytest.approx(5.0)
    assert desc["std"] == pytest.approx(np.std(values, ddof=1))
    assert desc["mode"] == 4.0 and desc["mode_count"] == 3
    t_test = _row(st.one_sample_tests(values, popmean=5.0), "One-sample t-test")
    assert t_test["statistic"] == pytest.approx(0.0)
