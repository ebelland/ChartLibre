"""Partition: the tree finds the structure put in the data; the dialog splits, prunes and writes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app.analysis.partition import NOMINAL, PartitionSpec, partition
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.partition_dialog import PartitionDialog

KINDS = {"temp": "continuous", "press": "continuous", "supplier": NOMINAL, "y": "continuous", "ok": NOMINAL}


def _data(n: int = 400) -> pd.DataFrame:
    """y jumps at temp 60 and with suppliers A and C; press is noise."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "temp": rng.uniform(0, 100, n),
        "press": rng.uniform(0, 10, n),
        "supplier": rng.choice(list("ABCD"), n),
    })
    frame["y"] = (np.where(frame.temp > 60, 20, 5) + np.where(frame.supplier.isin(["A", "C"]), 3, 0)
                  + rng.normal(0, 1, n))
    frame["ok"] = np.where((frame.temp > 50) & (frame.supplier != "B"), "pass", "fail")
    return frame


def test_a_regression_tree_splits_where_the_response_jumps() -> None:
    result = partition(_data(), PartitionSpec("y", ["temp", "press", "supplier"], KINDS, splits=3,
                                              validation_fraction=0.3))
    assert result.splits == 3 and len(result.leaves) == 4
    assert result.contributions["Column"].iloc[0] == "temp"
    assert result.contributions.set_index("Column").loc["press", "Splits"] == 0
    assert any("supplier ∈ {A, C}" in rule for rule in result.leaves["Rule"])
    assert result.summary["R² (validation)"] > 0.95
    assert list(result.history["Splits"]) == [1, 2, 3]


def test_a_classification_tree_stops_when_validation_stops_improving() -> None:
    result = partition(_data(), PartitionSpec("ok", ["temp", "press", "supplier"], KINDS, auto=True,
                                              validation_fraction=0.3))
    assert result.splits == 2  # temp, then supplier B apart
    assert result.summary["Misclassification (validation)"] == 0
    assert result.confusion is not None and int(np.trace(result.confusion.to_numpy())) == result.summary["Rows (training)"]
    assert result.probabilities is not None and list(result.probabilities.columns) == ["Prob(fail)", "Prob(pass)"]


def test_a_validation_column_marks_the_rows_kept_out() -> None:
    frame = _data()
    frame["valid"] = (np.arange(len(frame)) % 4 == 0).astype(int)
    result = partition(frame, PartitionSpec("y", ["temp", "supplier"], KINDS, splits=2, validation_column="valid"))
    assert result.summary["Rows (validation)"] == 100
    assert (result.role == "Validation").sum() == 100


def _wait(qapp, dialog: PartitionDialog) -> None:
    deadline = time.monotonic() + 60
    while dialog.job_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not dialog.job_running, "the computation did not finish"


@pytest.fixture
def dialog(qapp, repo: SqliteRepo):
    repo.import_dataframe(_data(), table_name="process", normalize_columns=False)
    built = PartitionDialog(repo=repo, table="process")
    built.roles_widget.set_casting({"response": ["y"], "factor": ["temp", "press", "supplier"]}, KINDS)
    yield built
    built.close()


def test_split_prune_and_ok_write_the_leaves_and_the_figure(qapp, repo: SqliteRepo, dialog: PartitionDialog) -> None:
    assert not dialog.problems()
    dialog._parameter_form.set_values({"splits": 1})
    dialog._step(+1)  # Split
    _wait(qapp, dialog)
    assert dialog._result.splits == 2
    dialog._step(-1)  # Prune
    _wait(qapp, dialog)
    assert dialog._result.splits == 1
    report = dialog.format_results(dialog._result)
    assert "Leaf report" in report and "Column contributions" in report

    dialog.ok()
    _wait(qapp, dialog)
    tables = set(repo.list_user_tables()["Table"])
    assert {"_process_partition", "_process_partition_leaves", "_process_partition_history",
            "_process_partition_contributions", "_process_partition_graph"} <= tables
    rows = repo.table_frame("_process_partition")
    assert {"Leaf", "Predicted y", "Partition role", "Partition x"} <= set(rows.columns)
    assert set(rows["Leaf"].dropna().astype(int)) == {1, 2}
    # JMP's graph: each leaf a band as wide as its rows, left to right.
    bands = rows.groupby("Leaf")["Partition x"].agg(["min", "max"])
    assert 0 < bands.loc[1, "min"] and bands.loc[1, "max"] < bands.loc[2, "min"] and bands.loc[2, "max"] < 1

    figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
    assert [axis.name for axis in figure.axes] == ["Scatter Plot", "Pareto Chart", "Scatter Plot", "Scatter Plot"]
    graph = figure.axes[0]
    assert len(graph.options["annotations"]) == 3  # All rows, and a rule per leaf
    assert graph.options["lines"][0]["orientation"] == "horizontal"
    from matplotlib.figure import Figure

    from app.plugin_api.charts.render_figure import render_figure_from_descriptor

    drawn = Figure()
    render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
    assert all(axes.collections or axes.lines or axes.patches for axes in drawn.axes)


def test_box_plots_by_leaf(qapp, repo: SqliteRepo, dialog: PartitionDialog) -> None:
    dialog._parameter_form.set_values({"splits": 2})
    for key, check in dialog._chart_checks.items():
        check.setChecked(key == "box")
    dialog.ok()
    _wait(qapp, dialog)
    figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
    assert [axis.name for axis in figure.axes] == ["Box Plot"]
    groups = repo.query_df(figure.axes[0].series[0].sql_query)["group"]
    assert list(dict.fromkeys(groups)) == ["1", "2", "3"]


def test_automatic_sets_the_split_count_it_chose(qapp, dialog: PartitionDialog) -> None:
    dialog.roles_widget.set_casting({"response": ["ok"], "factor": ["temp", "press", "supplier"]}, KINDS)
    dialog._parameter_form.set_values({"validation_fraction": 0.3})
    dialog._automatic()
    _wait(qapp, dialog)
    assert dialog._result.splits == 2
    assert dialog.parameter_values()["splits"] == 2


@pytest.mark.parametrize("method", ["forest", "boosted"])
def test_a_forest_or_boosted_trees_fit_and_ok_writes_no_leaves(qapp, repo: SqliteRepo, dialog: PartitionDialog,
                                                              method: str) -> None:
    dialog._parameter_form.set_values({"method": method, "trees": 40, "validation_fraction": 0.3})
    dialog.preview()
    _wait(qapp, dialog)
    result = dialog._result
    assert result.method == method and result.leaves.empty
    assert result.summary["R² (validation)"] > 0.9
    assert result.contributions["Column"].iloc[0] == "temp"
    assert list(result.history.columns)[0] == "Trees"
    assert "Leaf report" not in dialog.format_results(result)

    dialog.ok()
    _wait(qapp, dialog)
    tables = set(repo.list_user_tables()["Table"])
    assert "_process_partition_leaves" not in tables and "_process_partition_history" in tables
    assert "Leaf" not in repo.table_frame("_process_partition").columns
    figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
    assert "Scatter Plot" in [axis.name for axis in figure.axes]  # the fit by number of trees


def test_y_cannot_also_be_an_x(dialog: PartitionDialog) -> None:
    dialog.roles_widget.set_casting({"response": ["y"], "factor": ["y", "temp"]}, KINDS)
    assert any("both Y and X" in problem for problem in dialog.problems())


def test_permutation_importance_and_partial_dependence_for_any_method(qapp, repo: SqliteRepo) -> None:
    """AN-18: the forest included - shuffling temp costs most, press nothing; the dependence is drawn."""
    for method in ("tree", "forest"):
        result = partition(_data(), PartitionSpec(response="y", factors=["temp", "press", "supplier"], kinds=KINDS,
                                                  splits=4, method=method, validation_fraction=0.3))
        assert list(result.importance["Column"][:2]) == ["temp", "supplier"]
        assert abs(result.importance.set_index("Column").loc["press", "Importance"]) < 0.02
        temp = result.dependence[result.dependence["Factor"] == "temp"]
        assert temp["Predicted"].iloc[-1] > temp["Predicted"].iloc[0] + 10  # y jumps from 5 to 20 at temp 60

    repo.import_dataframe(_data(), table_name="plant", normalize_columns=False)
    dialog = PartitionDialog(repo=repo, table="plant")
    try:
        dialog.roles_widget.set_casting({"response": ["y"], "factor": ["temp", "press", "supplier"]}, KINDS)
        for key, check in dialog._chart_checks.items():
            check.setChecked(key in ("importance", "dependence"))
        dialog.ok()
        _wait(qapp, dialog)
        assert {"_plant_partition_importance", "_plant_partition_dependence"} <= set(repo.list_user_tables()["Table"])
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        names = [axis.name for axis in figure.axes]
        assert names[0] == "Pareto Chart" and "Bar Chart" in names and "Scatter Plot" in names
        from matplotlib.figure import Figure

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert all(axes.collections or axes.lines or axes.patches for axes in drawn.axes)
    finally:
        dialog.close()


def test_a_classification_tree_reports_and_draws_its_roc_curve(qapp, repo: SqliteRepo) -> None:
    """AN-19: AUC on the validation rows, the curve with the chance diagonal."""
    result = partition(_data(), PartitionSpec(response="ok", factors=["temp", "press", "supplier"], kinds=KINDS,
                                              splits=3, validation_fraction=0.3))
    assert list(result.auc) == ["fail"] and result.auc["fail"] > 0.9
    assert "AUC fail (validation)" in result.summary

    repo.import_dataframe(_data(), table_name="plant", normalize_columns=False)
    dialog = PartitionDialog(repo=repo, table="plant")
    try:
        dialog.roles_widget.set_casting({"response": ["ok"], "factor": ["temp", "press", "supplier"]}, KINDS)
        for key, check in dialog._chart_checks.items():
            check.setChecked(key == "roc")
        dialog.ok()
        _wait(qapp, dialog)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.title for axis in figure.axes] == ["ROC curve"]
        assert [s.name for s in figure.axes[0].series][-1] == "Chance"
    finally:
        dialog.close()
