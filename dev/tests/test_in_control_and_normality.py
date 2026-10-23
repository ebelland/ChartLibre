"""Quick answers "Is it in control?" and "Is it normal?": a stable and a shifted process, a normal and a skewed sample."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.analysis.answers import NO, YES
from app.analysis.in_control import is_in_control
from app.analysis.normality import is_normal
from app.data.sqlite_repo import SqliteRepo

RNG = np.random.default_rng(1)
N = 60
DAYS = pd.date_range("2026-01-01", periods=N).astype(str)
# A record with no point past 3 sigma: one in five stable records of 60 has one, by chance alone.
STABLE = np.random.default_rng(2).normal(10, 1, N)
SHIFTED = np.r_[RNG.normal(10, 1, 40), RNG.normal(14, 1, 20)]


def test_a_stable_process_is_in_control_and_a_shifted_one_is_not() -> None:
    assert is_in_control(pd.DataFrame({"y": STABLE}), "y").verdict == YES
    answer = is_in_control(pd.DataFrame({"t": DAYS, "y": SHIFTED}), "y", "t")
    assert answer.verdict == NO and answer.signals > 0
    assert answer.series["Signal"].notna().sum() == answer.signals
    # Points after the shift signal; so do the first forty, all below a centre the shift pulled up.
    assert answer.series["Signal"].iloc[40:].notna().any()
    assert {"Order", "Value", "CL", "UCL", "LCL", "Signal"} == set(answer.series.columns)


def test_a_normal_sample_is_normal_and_a_skewed_one_is_named_skewed() -> None:
    assert is_normal(pd.DataFrame({"y": RNG.normal(0, 1, 200)}), "y").verdict == YES
    skewed = is_normal(pd.DataFrame({"y": RNG.exponential(1.0, 200)}), "y")
    assert skewed.verdict == NO and skewed.skewness > 0.5
    assert any("skewed to the right" in str(values.get("shape", "")) for _t, values in skewed.sentences)
    assert np.all(np.diff(skewed.qq["Sample"]) >= 0) and len(skewed.qq) == 200


def _run(qapp, dialog) -> None:
    dialog.ok()
    deadline = time.monotonic() + 30
    while dialog._task is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_both_dialogs_draw_their_chart(qapp, repo: SqliteRepo) -> None:
    from app.table_operations.in_control_dialog import InControlDialog
    from app.table_operations.normality_dialog import NormalityDialog

    repo.import_dataframe(pd.DataFrame({"giorno": DAYS, "y": SHIFTED}), table_name="line", normalize_columns=False)
    for dialog_type, roles, table, axis_name in (
        (InControlDialog, {"response": ["y"], "time": ["giorno"]}, "_line_control", "Time Series"),
        (NormalityDialog, {"response": ["y"]}, "_line_normality", "Scatter Plot"),
    ):
        dialog = dialog_type(repo=repo, table="line")
        try:
            dialog.roles_widget.set_casting(roles)
            assert not dialog.problems()
            _run(qapp, dialog)
            assert not repo.table_frame(table).empty
            figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
            assert [axis.name for axis in figure.axes] == [axis_name]
        finally:
            dialog.close()
