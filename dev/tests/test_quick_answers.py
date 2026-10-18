"""Quick answers: the right test chosen, the answer in a sentence, the chart drawn."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app.analysis import answers
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.differences_dialog import DifferencesDialog


def _groups(shift: float = 2.0, n: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "line": np.repeat(["A", "B", "C"], n),
        "weight": np.r_[rng.normal(10, 1, n), rng.normal(10, 1, n), rng.normal(10 + shift, 1, n)],
    })
    frame["grade"] = np.where(frame["weight"] > 11, "pass", "fail")
    return frame


def test_a_shifted_group_is_found_and_named() -> None:
    answer = answers.differences(_groups(), "weight", "line")
    assert answer.verdict == answers.YES and answer.test == "Welch ANOVA"
    assert answer.sentences[1][1]["high"] == "C"
    assert set(answer.pairs["Pair"]) == {"A - C", "B - C"}
    assert answer.effect_name == "eta²" and answer.effect > 0.14  # large


def test_no_difference_says_what_it_does_not_mean() -> None:
    answer = answers.differences(_groups(shift=0.0), "weight", "line")
    assert answer.verdict == answers.NO
    assert any("not proof" in template for template, _values in answer.sentences)


def test_two_skewed_groups_are_compared_by_ranks() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"g": np.repeat(["x", "y"], 20),
                          "t": np.r_[rng.lognormal(0, 1.2, 20), rng.lognormal(1, 1.2, 20)]})
    answer = answers.differences(frame, "t", "g")
    assert answer.test == "Mann-Whitney U test" and answer.effect_name == "Cohen's d"


def test_categories_are_compared_by_their_shares() -> None:
    answer = answers.differences(_groups(), "grade", "line")
    assert not answer.numeric and answer.test == "Chi-squared test" and answer.verdict == answers.YES


def _wait(qapp, dialog) -> None:
    deadline = time.monotonic() + 30
    while dialog._task is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog._task is None


@pytest.mark.parametrize(("measure", "chart"), [("weight", "Box Plot"), ("grade", "Mosaic Plot")])
def test_the_dialog_answers_and_ok_draws_the_comparison(qapp, repo: SqliteRepo, measure: str, chart: str) -> None:
    repo.import_dataframe(_groups(), table_name="lines", normalize_columns=False)
    dialog = DifferencesDialog(repo=repo, table="lines")
    try:
        dialog.roles_widget.set_casting({"response": [measure], "group": ["line"]})
        assert not dialog.problems()
        dialog.preview()
        _wait(qapp, dialog)
        assert "Yes:" in dialog.format_results(dialog._result)
        dialog.ok()
        _wait(qapp, dialog)
        assert "_lines_differences" in set(repo.list_user_tables()["Table"])
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == [chart]
    finally:
        dialog.close()
