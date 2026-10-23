"""Quick answer Q-02 "What has the most impact?": the factor put in the data comes first, the noise last."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.analysis.answers import NO, YES
from app.analysis.impact import impact
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.impact_dialog import ImpactDialog


def _data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "temp": rng.uniform(0, 100, n),
        "press": rng.uniform(0, 10, n),
        "supplier": rng.choice(list("ABCD"), n),
        "noise": rng.normal(size=n),
    })
    frame["y"] = 0.3 * frame.temp + np.where(frame.supplier.isin(["A", "C"]), 8, 0) + rng.normal(0, 3, n)
    frame["ok"] = np.where(frame.temp > 50, "pass", "fail")
    return frame


def test_the_driver_comes_first_with_its_direction_and_the_noise_has_little_impact() -> None:
    answer = impact(_data(), "y", ["noise", "press", "supplier", "temp"])
    assert answer.verdict == YES
    assert list(answer.ranking["Factor"][:2]) == ["temp", "supplier"]
    assert abs(answer.ranking["Impact %"].sum() - 100.0) < 0.5
    assert answer.ranking.set_index("Factor").loc["temp", "Direction"] == "rises"
    words = [template.format(**values) for template, values in answer.sentences]
    assert words[0].startswith("“temp” has the most impact on “y”")
    assert any("Little or no impact" in line and "noise" in line and "press" in line for line in words)
    assert answer.explained > 0.7


def test_a_categorical_result_and_no_impact() -> None:
    data = _data()
    answer = impact(data, "ok", ["temp", "press"])
    assert answer.ranking["Factor"][0] == "temp" and not answer.numeric
    noise = impact(data.assign(y=np.random.default_rng(5).normal(size=len(data))), "y", ["press", "noise"])
    assert noise.verdict == NO


def _wait(qapp, dialog: ImpactDialog) -> None:
    deadline = time.monotonic() + 60
    while dialog.job_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not dialog.job_running


def test_ok_writes_the_ranking_and_draws_a_pareto(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(_data(), table_name="plant", normalize_columns=False)
    dialog = ImpactDialog(repo=repo, table="plant")
    try:
        dialog.roles_widget.set_casting({"response": ["y"], "factor": ["temp", "press", "supplier", "noise"]})
        assert not dialog.problems()
        dialog.ok()
        _wait(qapp, dialog)
        ranking = repo.table_frame("_plant_impact")
        assert ranking["Factor"].iloc[0] == "temp"
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Pareto Chart"]
    finally:
        dialog.close()
