"""Quick answers "Is it in control?" and "Is it normal?": a stable and a shifted process, a normal and a skewed sample."""
from __future__ import annotations

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


def test_in_control_draws_its_chart_and_writes_no_table(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.in_control_dialog import InControlDialog

    repo.import_dataframe(pd.DataFrame({"giorno": DAYS, "y": SHIFTED}), table_name="line", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Line"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="line", x_label="giorno", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Line",
                                  sql_query='SELECT giorno AS x, y AS y FROM "line"', roles={"x": "x", "y": "y"},
                                  style={})
    dialog = InControlDialog(repo=repo, figure_id=figure_id)
    try:
        (result,) = dialog.compute_results()
        assert result.answer.verdict == NO and result.answer.temporal
        dialog.ok()
        assert set(repo.list_user_tables()["Table"]) == {"line"}
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Time Series"]
        assert [series.name for series in figure.axes[0].series] == ["UCL", "LCL", "CL", "Line", "Signals"]
        from matplotlib.figure import Figure

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert len(drawn.axes[0].lines) >= 5
        steps = dialog.next_steps(result)
        assert [step.operation for step in steps] == ["Has it changed over time?", "Is it normal?"]
        assert steps[0].table == "line" and steps[0].roles == {"response": ["y"], "time": ["giorno"]}
    finally:
        dialog.close()
