"""G-02: what the wizard offers for the columns picked - only what they allow, phrased with their names."""
from __future__ import annotations

import pandas as pd
import pytest

from app.plugin_api.table_operations import guided_plan as gp
from app.plugin_api.table_operations import table_profile


@pytest.fixture
def profile():
    frame = pd.DataFrame({
        "mass": [3.1, 4.2, 5.0, 3.8, 4.4, 4.9] * 5,
        "flipper": [180.0, 190, 210, 185, 200, 215] * 5,
        "bill": [39.0, 41, 47, 38, 45, 49] * 5,
        "species": ["A", "B", "C"] * 10,
        "island": ["x", "y"] * 15,
        "day": pd.date_range("2024-01-01", periods=30).astype(str),
    })
    return table_profile.profile("penguins", frame)


def _keys(plan) -> list[str]:
    return [q.key for q in plan.questions]


def test_one_measure_shows_its_distribution(profile) -> None:
    plan = gp.plan(profile, ["mass"])
    assert [c.chart_type for c in plan.charts] == ["Histogram", "Box Plot", "Q-Q Plot"]
    assert _keys(plan) == ["analysis:Distribution"] and plan.questions[0].roles == {"response": ["mass"]}
    assert plan.results == []  # nothing to ask


def test_a_measure_and_a_group_ask_about_differences(profile) -> None:
    plan = gp.plan(profile, ["mass", "species"])
    assert [c.chart_type for c in plan.charts] == ["Box Plot", "Violin Plot", "Bar Chart"]
    first = plan.questions[0]
    assert first.key == "analysis:Are there differences?"
    assert first.roles == {"response": ["mass"], "group": ["species"]} and first.kinds == {"species": "nominal"}
    assert first.question.format(**first.values) == "Does “mass” change between the groups of “species”?"


def test_two_measures_ask_which_is_the_result(profile) -> None:
    plan = gp.plan(profile, ["flipper", "mass"])
    assert plan.results == ["flipper", "mass"] and plan.result == "mass"
    assert plan.questions[0].roles == {"response": ["mass"], "factor": ["flipper"]}
    other = gp.plan(profile, ["flipper", "mass"], result="flipper")
    assert other.questions[0].roles == {"response": ["flipper"], "factor": ["mass"]}


def test_groups_dates_and_many_measures(profile) -> None:
    assert [c.chart_type for c in gp.plan(profile, ["species"]).charts] == ["Pareto Chart", "Bar Chart"]
    assert _keys(gp.plan(profile, ["species", "island"])) == ["analysis:Fit Y by X"]
    timed = gp.plan(profile, ["day", "mass"])
    assert timed.charts[0].chart_type == "Time Series" and _keys(timed) == ["analysis:Has it changed over time?"]
    three = gp.plan(profile, ["mass", "flipper", "bill"])
    assert [c.chart_type for c in three.charts] == ["Pair Plot", "Scatter Plot"]
    assert _keys(three) == ["analysis:What has the most impact?", "analysis:Multivariate",
                            "analysis:Principal Components"]
    mixed = gp.plan(profile, ["mass", "species", "island"])
    assert mixed.charts and mixed.charts[0].roles == {"value": "mass", "group": "species"}


def test_the_chart_queries_read_counts_and_means(profile) -> None:
    pareto = gp.plan(profile, ["species"]).charts[0]
    sql, roles = gp.chart_sql("penguins", pareto)
    assert sql == ('SELECT "species" AS "X", COUNT(*) AS "Y" FROM "penguins" WHERE "species" IS NOT NULL '
                   'GROUP BY "species" ORDER BY "Y" DESC')
    means = gp.plan(profile, ["mass", "species"]).charts[2]
    assert 'AVG("mass") AS "Y"' in gp.chart_sql("penguins", means)[0] and roles == {"X": "X", "Y": "Y"}
