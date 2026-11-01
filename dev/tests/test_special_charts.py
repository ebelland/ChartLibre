"""ID12: EWMA, CUSUM, Levey-Jennings with Westgard's rules, and the g and t charts for rare events."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import special_charts as sc
from app.data.sqlite_repo import SqliteRepo

#: Montgomery, Introduction to Statistical Quality Control, Table 9.1: 20 values from N(10, 1), then 10 from N(11, 1).
MONTGOMERY = np.array([9.45, 7.99, 9.29, 11.66, 12.16, 10.18, 8.04, 11.46, 9.20, 10.34, 9.03, 11.47, 10.51, 9.40,
                       10.08, 9.37, 10.62, 10.31, 8.52, 10.84, 10.90, 9.33, 12.29, 11.50, 10.60, 11.08, 10.38,
                       11.62, 11.31, 10.52])
X = np.arange(1.0, 31.0)


def test_the_tabular_cusum_signals_the_shift_at_sample_29_as_montgomery_does() -> None:
    chart = sc.cusum_chart(X, MONTGOMERY, k=0.5, h=5.0, target=10.0, sigma=1.0)
    # C+_1 = max(0, 9.45 - 10.5) = 0, C-_1 = max(0, 9.5 - 9.45) = 0.05
    assert chart.y[0] == 0.0 and chart.details["low"][0] == pytest.approx(-0.05)
    assert [v.index + 1 for v in chart.violations][:1] == [29]
    assert chart.y[28] == pytest.approx(5.28, abs=0.01)
    assert chart.upper == 5.0 and chart.lower == -5.0


def test_the_ewma_starts_at_the_target_its_limits_widen_to_their_asymptote() -> None:
    chart = sc.ewma_chart(X, MONTGOMERY, lam=0.1, width=2.7, target=10.0, sigma=1.0)
    assert chart.y[0] == pytest.approx(9.945)  # 0.1 * 9.45 + 0.9 * 10
    assert chart.upper_band[0] == pytest.approx(10.27, abs=1e-3)  # 2.7 * sqrt(0.1/1.9 * (1 - 0.81))
    assert chart.upper == pytest.approx(10.0 + 2.7 * np.sqrt(0.1 / 1.9))
    assert np.all(np.diff(chart.upper_band) >= 0)
    assert chart.violations and min(v.index for v in chart.violations) >= 20  # after the shift only
    with pytest.raises(ValueError, match="lambda"):
        sc.ewma_chart(X, MONTGOMERY, lam=0.0)


@pytest.mark.parametrize(("z", "rule", "index"), [
    ([0, 3.5], "1-3s", 1),
    ([0, 2.5, 2.4], "2-2s", 2),
    ([0, 2.5, -2.3], "R-4s", 2),
    ([0, 1.2, 1.5, 1.1, 1.3], "4-1s", 4),
    ([-0.5] + [0.4] * 10, "10-x", 10),
    ([0, 2.4, 0], "1-2s", 1),
])
def test_each_westgard_rule(z: list[float], rule: str, index: int) -> None:
    found = sc.westgard(np.array(z) * 2.0 + 100.0, 100.0, 2.0)
    assert (index, rule) in [(i, r) for i, r, _rejects in found]
    rejects = dict(sc.WESTGARD)[rule]
    assert all(rej == rejects for i, r, rej in found if r == rule)


def test_levey_jennings_uses_the_materials_assigned_mean_and_sd() -> None:
    values = np.array([100.0, 101.0, 99.0, 106.5, 100.5])
    chart = sc.levey_jennings_chart(np.arange(5.0), values, mean=100.0, sd=2.0)
    assert (chart.center, chart.upper, chart.lower) == (100.0, 106.0, 94.0)
    assert [(v.index, v.descriptions[0]) for v in chart.violations] == [(3, "Westgard 1-3s: reject the run")]
    assert chart.details["runs rejected"] == 1


def test_the_g_chart_has_the_geometric_limits_and_the_t_chart_the_back_transformed_ones() -> None:
    g = sc.g_chart(np.arange(6.0), np.array([10.0, 30.0, 20.0, 15.0, 25.0, 20.0]))
    assert g.center == 20.0 and g.upper == pytest.approx(20 + 3 * np.sqrt(20 * 21)) and g.lower == 0.0
    times = np.array([12.0, 30.0, 8.0, 21.0, 15.0, 40.0, 9.0, 18.0, 26.0, 14.0])
    t = sc.t_chart(np.arange(10.0), times)
    transformed = times ** (1 / 3.6)
    assert t.center == pytest.approx(np.mean(transformed) ** 3.6)
    assert t.lower < t.center < t.upper and np.array_equal(t.y, times)
    with pytest.raises(ValueError):
        sc.t_chart(np.arange(2.0), np.array([1.0, 0.0]))


def test_the_dialog_draws_a_cusum_with_both_sums(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.control_chart_dialog import SeriesControlChartDialog

    repo.import_dataframe(pd.DataFrame({"n": X, "v": MONTGOMERY}), table_name="shift", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Shift"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="v", x_label="n", y_label="v", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="v", style={},
                                  sql_query="SELECT n AS x, v AS y FROM shift", roles={"x": "x", "y": "y"})
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.source_chart_radio.setChecked(True)
        dialog.series_selector.select_all_series()
        dialog.model_combo.setCurrentText(sc.CHART_CUSUM)
        dialog._parameter_form_spec.set_values({"process_mean": "10", "process_sigma": "1"})
        (result,) = dialog.compute_results()
        assert [v.index + 1 for v in result.violations][:1] == [29]
        assert "cusum_low" in result.to_df().columns
        assert dialog.apply()
    finally:
        dialog.close()
    names = {str(series.name) for axis in repo.load_figure_descriptor(figure_id).axes for series in axis.series}
    assert any(name.endswith("- C-") for name in names) and any("UCL" in name for name in names)
