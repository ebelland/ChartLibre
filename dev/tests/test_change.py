"""Quick answer Q-03 "Has it changed over time?": a step, a drift, nothing - each told apart."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from app.analysis.answers import NO, YES
from app.analysis.change import has_changed, mann_kendall, pettitt
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.change_dialog import ChangeDialog

RNG = np.random.default_rng(0)
N = 60
DAYS = pd.date_range("2026-01-01", periods=N).astype(str)
STEP = np.r_[RNG.normal(10, 1, 30), RNG.normal(13, 1, 30)]
DRIFT = 0.05 * np.arange(N) + RNG.normal(0, 0.5, N)
NOISE = RNG.normal(0, 1, N)


def test_the_tests_agree_with_their_definitions() -> None:
    walk = RNG.normal(size=50).cumsum()
    # Mann-Kendall is Kendall's tau against time; without ties the p-values nearly agree.
    assert abs(mann_kendall(walk)[1] - stats.kendalltau(np.arange(50), walk).pvalue) < 0.01
    at, p = pettitt(STEP)
    assert at == 29 and p < 1e-6


def test_a_step_a_drift_and_noise() -> None:
    step = has_changed(pd.DataFrame({"giorno": DAYS, "y": STEP}), "y", "giorno")
    assert (step.verdict, step.shape, step.temporal) == (YES, "step", True)
    assert step.sentences[0][0].startswith("Yes: it changed around {when}")
    assert step.sentences[0][1]["when"] == "2026-01-31"
    drift = has_changed(pd.DataFrame({"y": DRIFT}), "y")
    assert (drift.verdict, drift.shape) == (YES, "drift") and abs(drift.slope - 0.05) < 0.01
    assert "along the rows" in drift.sentences[0][0]
    flat = has_changed(pd.DataFrame({"y": NOISE}), "y")
    assert (flat.verdict, flat.shape) == (NO, "")


def test_the_order_column_sorts_the_values() -> None:
    shuffled = pd.DataFrame({"t": np.arange(N), "y": DRIFT}).sample(frac=1.0, random_state=3)
    answer = has_changed(shuffled, "y", "t")
    assert list(answer.series["Order"]) == list(range(N)) and answer.shape == "drift"


def test_ok_draws_the_values_with_the_change(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(pd.DataFrame({"giorno": DAYS, "y": STEP}), table_name="line", normalize_columns=False)
    dialog = ChangeDialog(repo=repo, table="line")
    try:
        dialog.roles_widget.set_casting({"response": ["y"], "time": ["giorno"]})
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 30
        while dialog.job_running and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert {"Order", "Value", "Trend", "Segment", "Segment median"} <= set(repo.table_frame("_line_change").columns)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Time Series"]
        assert len(figure.axes[0].series) == 3  # the values, the median before, the median after
        from matplotlib.figure import Figure

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert len(drawn.axes[0].lines) >= 3
    finally:
        dialog.close()
