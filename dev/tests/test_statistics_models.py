"""Statistics on tables by role (todo R-04): contingency, two-way ANOVA, survival."""
from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analysis import statistics as st
from app.data.select_sql import projection_source
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.statistics_dialog import SeriesStatisticsDialog

# Agresti (2007), table 2.5: party identification by gender.
PARTY = {("female", "Democrat"): 762, ("female", "Independent"): 327, ("female", "Republican"): 468,
         ("male", "Democrat"): 484, ("male", "Independent"): 239, ("male", "Republican"): 477}

# Freireich's 6-MP trial: weeks in remission, relapse seen (1) or censored (0).
TREATED = [(6, 1), (6, 1), (6, 1), (6, 0), (7, 1), (9, 0), (10, 1), (10, 0), (11, 0), (13, 1), (16, 1),
           (17, 0), (19, 0), (20, 0), (22, 1), (23, 1), (25, 0), (32, 0), (32, 0), (34, 0), (35, 0)]
PLACEBO = [(t, 1) for t in (1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23)]


def _party_rows() -> tuple[list[str], list[str]]:
    gender = [g for (g, _p), count in PARTY.items() for _ in range(count)]
    party = [p for (_g, p), count in PARTY.items() for _ in range(count)]
    return gender, party


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------
def test_contingency_matches_agrestis_party_identification() -> None:
    table = st.contingency_table(*_party_rows())
    assert table.to_numpy().tolist() == [[762, 327, 468], [484, 239, 477]]
    rows = {row["test"]: row for row in st.contingency_tests(table.to_numpy())}
    assert rows["Pearson chi-squared"]["statistic"] == pytest.approx(30.07, abs=0.005)
    assert rows["Likelihood-ratio G"]["statistic"] == pytest.approx(30.02, abs=0.005)
    assert "df = 2" in rows["Pearson chi-squared"]["note"]
    assert rows["Fisher exact"]["pvalue"] == pytest.approx(1 / 10_000)  # Monte Carlo floor
    # Seeded: the same p-value every run.
    again = {row["test"]: row for row in st.contingency_tests(table.to_numpy())}
    assert again["Fisher exact"]["pvalue"] == rows["Fisher exact"]["pvalue"]


def test_a_two_by_two_table_gets_yates_and_fishers_odds_ratio() -> None:
    rows = {row["test"]: row for row in st.contingency_tests(np.array([[3, 1], [1, 3]]))}
    assert rows["Fisher exact"]["pvalue"] == pytest.approx(0.4857, abs=1e-4)  # the lady tasting tea
    assert rows["Fisher exact"]["statistic"] == pytest.approx(9.0)
    assert rows["Chi-squared, Yates' correction"]["statistic"] == pytest.approx(
        stats.chi2_contingency([[3, 1], [1, 3]], correction=True)[0]
    )
    assert "read Fisher" in rows["Pearson chi-squared"]["note"]


def test_weights_count_a_table_already_counted() -> None:
    gender, party = zip(*PARTY)
    table = st.contingency_table(gender, party, list(PARTY.values()))
    assert table.to_numpy().tolist() == [[762, 327, 468], [484, 239, 477]]


def test_two_way_anova_is_type_ii_and_equals_type_i_when_balanced() -> None:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    rng = np.random.default_rng(7)
    a = np.repeat(["low", "mid", "high"], 20)
    b = np.tile(np.repeat(["F", "M"], 10), 3)
    y = rng.normal(10.0, 1.0, 60) + (a == "high") * 2.0 + ((a == "high") & (b == "M")) * 1.5
    result = st.two_way_anova(a, b, y)
    assert result["balanced"] and result["n"] == 60
    sequential = anova_lm(ols("y ~ C(a) * C(b)", data=pd.DataFrame({"a": a, "b": b, "y": y})).fit(), typ=1)
    by_source = {row["source"]: row for row in result["table"]}
    for source, key in (("A", "C(a)"), ("B", "C(b)"), ("A × B", "C(a):C(b)")):
        assert by_source[source]["F"] == pytest.approx(sequential.loc[key, "F"])
        assert by_source[source]["pvalue"] == pytest.approx(sequential.loc[key, "PR(>F)"])
    residual = by_source["Residual"]["ss"]
    assert by_source["A"]["partial_eta2"] == pytest.approx(by_source["A"]["ss"] / (by_source["A"]["ss"] + residual))
    assert by_source["Residual"]["df"] == 54


def test_two_way_anova_refuses_an_empty_cell() -> None:
    with pytest.raises(ValueError, match="no values"):
        st.two_way_anova(["a", "a", "b", "b"], ["x", "y", "x", "x"], [1.0, 2.0, 3.0, 4.0])


def test_survival_summary_of_the_6mp_trial() -> None:
    times = [t for t, _e in TREATED + PLACEBO]
    events = [e for _t, e in TREATED + PLACEBO]
    summary = st.survival_summary(times, events, ["6-MP"] * 21 + ["placebo"] * 21)
    treated, placebo = summary["groups"]
    assert (treated["n"], treated["events"], treated["censored"], treated["median"]) == (21, 9, 12, 23.0)
    assert treated["median_low"] == 13.0 and np.isnan(treated["median_high"])
    assert (placebo["median"], placebo["events"]) == (8.0, 21)
    assert summary["logrank"]["statistic"] == pytest.approx(16.79, abs=0.01)


def test_projection_source_names_the_column_behind_an_alias() -> None:
    sql = 'SELECT species AS x, "body mass" AS y, CAST(sex AS TEXT) AS trace FROM penguins'
    assert projection_source(sql, "x") == "species"
    assert projection_source(sql, "y") == "body mass"
    assert projection_source(sql, "trace") == "CAST(sex AS TEXT)"
    assert projection_source(sql, "z") == ""


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
@pytest.fixture
def project(qapp, repo: SqliteRepo) -> Iterator[dict[str, int]]:
    gender, party = _party_rows()
    repo.import_dataframe(pd.DataFrame({"gender": gender, "party": party}), table_name="party", normalize_columns=False)
    repo.import_dataframe(
        pd.DataFrame([(arm, t, e) for arm, rows in (("6-MP", TREATED), ("placebo", PLACEBO)) for t, e in rows],
                     columns=["arm", "weeks", "relapse"]),
        table_name="trial", normalize_columns=False,
    )
    rng = np.random.default_rng(3)
    repo.import_dataframe(
        pd.DataFrame({"dose": np.repeat(["low", "high"], 20), "sex": np.tile(["F", "M"], 20),
                      "response": rng.normal(5.0, 1.0, 40)}),
        table_name="doses", normalize_columns=False,
    )
    figures: dict[str, int] = {}
    for name, chart, sql, roles in (
        ("contingency", "Mosaic Plot", "SELECT gender AS x, party AS y FROM party", {"x": "x", "y": "y"}),
        ("survival", "Kaplan-Meier", 'SELECT weeks AS time, relapse AS event, arm AS "group" FROM trial',
         {"time": "time", "event": "event", "group": "group"}),
        ("two_way", "Interaction Plot", "SELECT dose AS x, response AS y, sex AS trace FROM doses",
         {"x": "x", "y": "y", "trace": "trace"}),
        ("normality", "Histogram", "SELECT response AS value FROM doses", {"value": "value"}),
    ):
        figure_id = int(repo.create_figure_descriptor(name=name))
        axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart, title=name,
                                                  x_label="", y_label="", options={}))
        repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=name, sql_query=sql, roles=roles, style={})
        figures[name] = figure_id
    yield figures


def _run(repo: SqliteRepo, figure_id: int, model: str, *, chart: str = "") -> str:
    dialog = SeriesStatisticsDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData(model))
    if chart:
        dialog.create_chart_check.setChecked(True)
        dialog.chart_type_combo.setCurrentIndex(dialog.chart_type_combo.findData(chart))
    assert dialog.apply()
    page = dialog.format_results(dialog.compute_results())
    dialog.close()
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


def _charts(repo: SqliteRepo, figure_id: int) -> list[str]:
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    return [axis.name for axis in (descriptor.axes if descriptor else [])]


@pytest.mark.parametrize(
    ("model", "chart", "expected"),
    [
        ("contingency", "Mosaic Plot", ["gender / party", "Pearson chi-squared", "30.07", "762"]),
        ("survival", "Kaplan-Meier", ["6-MP", "23", "13 .. ∞", "Log-rank", "16.79"]),
        ("two_way", "Interaction Plot", ["dose × sex", "Partial eta²", "Residual"]),
    ],
)
def test_each_model_reports_and_draws_its_chart(repo: SqliteRepo, project: dict[str, int], model: str, chart: str,
                                               expected: list[str]) -> None:
    text = _run(repo, project[model], model, chart=chart)
    for snippet in expected:
        assert snippet in text, snippet
    assert _charts(repo, project[model]).count(chart) == 2  # the series' own axis and the one Statistics added
    descriptor = repo.load_figure_descriptor(figure_id=project[model])
    added = descriptor.axes[-1].series[0]  # pyright: ignore[reportOptionalMemberAccess]
    assert len(repo.query_df(added.sql_query)) > 0


def test_a_series_without_the_columns_is_named_not_failed(repo: SqliteRepo, project: dict[str, int]) -> None:
    text = _run(repo, project["normality"], "survival")
    assert "Not analysed" in text and "time and event" in text


def test_normality_offers_the_q_q_and_p_p_plots(repo: SqliteRepo, project: dict[str, int]) -> None:
    _run(repo, project["normality"], "normality", chart="Q-Q Plot")
    assert "Q-Q Plot" in _charts(repo, project["normality"])
