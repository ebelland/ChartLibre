"""Distribution (AN-02) and Fit Y by X (AN-03): the engines against hand-worked values, and both windows' OK."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analysis import distribution as dist
from app.analysis import fit_y_by_x as fyx
from app.analysis import statistics as st


def test_quantiles_sit_at_n_plus_one_p() -> None:
    table = dist.quantile_table(np.arange(1.0, 10.0))
    values = dict(zip(table["Percent"], table["Value"]))
    assert values["50%"] == 5.0 and values["25%"] == 2.5 and values["75%"] == 7.5
    assert values["0%"] == 1.0 and values["100%"] == 9.0


def test_capability_follows_its_definitions() -> None:
    values = np.array([9.0, 11.0] * 50)  # moving range 2 everywhere
    result = dist.capability(values, lsl=4.0, usl=16.0, target=10.0)
    within = 2.0 / dist.D2_MOVING_RANGE
    overall = float(np.std(values, ddof=1))
    assert result.indices["Cp"] == pytest.approx(12.0 / (6 * within))
    assert result.indices["Cpk"] == pytest.approx(6.0 / (3 * within))
    assert result.indices["Pp"] == pytest.approx(12.0 / (6 * overall))
    assert result.indices["Cpm"] == pytest.approx(12.0 / (6 * overall))  # mean on target


def test_one_limit_gives_the_one_sided_indices_only() -> None:
    result = dist.capability(np.linspace(0, 1, 50), usl=2.0)
    assert set(result.indices) == {"Cpk", "Cpu", "Ppk", "Ppu"}
    with pytest.raises(ValueError):
        dist.capability(np.linspace(0, 1, 50))


def test_the_analysis_is_chosen_by_the_types() -> None:
    assert fyx.analysis_for(True, True) == fyx.BIVARIATE
    assert fyx.analysis_for(True, False) == fyx.ONEWAY
    assert fyx.analysis_for(False, False) == fyx.CONTINGENCY
    assert fyx.analysis_for(False, True) == fyx.LOGISTIC


def test_games_howell_is_close_to_tukey_when_variances_and_sizes_are_equal() -> None:
    rng = np.random.default_rng(3)
    groups = {"a": rng.normal(0, 1, 40), "b": rng.normal(0.5, 1, 40), "c": rng.normal(1, 1, 40)}
    tukey = pd.DataFrame(st.tukey_hsd(groups))
    games = fyx.games_howell(groups)
    assert np.allclose(games["difference"], tukey["difference"])
    assert np.allclose(games["pvalue"], tukey["pvalue"], atol=0.01)


def test_dunn_on_two_groups_is_the_mann_whitney_z_test() -> None:
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 30), rng.normal(0.6, 1, 25)
    table = fyx.dunn({"a": a, "b": b})
    expected = stats.mannwhitneyu(a, b, method="asymptotic", use_continuity=False).pvalue
    assert table["pvalue (unadjusted)"][0] == pytest.approx(expected)


def test_holm_steps_down() -> None:
    assert np.allclose(fyx.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    group = np.repeat(["low", "mid", "high"], 30)
    x = rng.normal(0, 1, 90)
    y = 2 * x + np.repeat([0.0, 1.0, 2.0], 30) + rng.normal(0, 0.5, 90)
    return pd.DataFrame({"x": x, "y": y, "group": group, "flag": np.where(y > 1, "yes", "no")})


def _ok(qapp, dialog) -> None:
    dialog.ok()
    deadline = time.monotonic() + 30
    while dialog._task is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_both_windows_write_their_tables_and_a_figure(qapp, repo, frame) -> None:
    from app.table_operations.distribution_dialog import DistributionDialog
    from app.table_operations.fit_y_by_x_dialog import FitYByXDialog

    repo.import_dataframe(frame, table_name="data", normalize_columns=False)
    kinds = {"x": "continuous", "y": "continuous", "group": "nominal", "flag": "nominal"}

    dialog = DistributionDialog(repo=repo, table="data")
    dialog.roles_widget.set_casting({"response": ["y"]}, kinds)
    dialog._parameter_form.set_values({"lsl": "-6", "usl": "8"})
    _ok(qapp, dialog)
    assert dialog.created_figure_ids
    capability = repo.query_df('SELECT * FROM "data_distribution_capability"')
    assert {"Cp", "Cpk", "Pp", "Ppk"} <= set(capability["Index"])
    dialog.close()

    dialog = FitYByXDialog(repo=repo, table="data")
    dialog.roles_widget.set_casting({"response": ["y"], "factor": ["x", "group"]}, kinds)
    _ok(qapp, dialog)
    assert dialog.created_figure_ids
    names = repo.list_table_names()
    assert "data_y_by_x_linear_fit" in names and "data_y_by_group_games_howell_unequal_variances" in names
    dialog.close()

    dialog = FitYByXDialog(repo=repo, table="data")
    dialog.roles_widget.set_casting({"response": ["flag"], "factor": ["x", "group"]}, kinds)
    _ok(qapp, dialog)
    names = repo.list_table_names()
    assert "data_flag_by_x_whole_model_test" in names and "data_flag_by_group_tests" in names
    dialog.close()
