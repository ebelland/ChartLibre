"""Fit Model on a DOE table: the design casts the roles, Preview reports, OK writes and draws."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.doe_dialog import DOEExperimentDialog
from app.table_operations.column_roles import CONTINUOUS, NOMINAL, guess_kind
from app.table_operations.fit_model_dialog import FitModelDialog


def _wait(qapp, dialog: FitModelDialog) -> None:
    """Until the background computation is done (the first statsmodels import takes seconds)."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        qapp.processEvents()
        if dialog._task is None:
            return
        time.sleep(0.01)
    raise AssertionError("the computation did not finish")


@pytest.fixture
def doe_table(qapp, tmp_path: Path):
    """A Box-Behnken table with its response filled in: y = 5 + 2A - B + AB + A²."""
    repo = SqliteRepo(db_path=tmp_path / "fit.dhub")
    doe = DOEExperimentDialog(repo)
    doe.factor_count.setValue(3)
    doe.model_combo.setCurrentIndex(doe.model_combo.findData("box_behnken"))
    doe.randomize.setChecked(False)
    doe.accept()
    name = doe.created_table_name
    assert name is not None
    frame = repo.table_frame(name)
    rng = np.random.default_rng(3)
    a, b = frame["Factor_1"], frame["Factor_2"]
    frame["Response_1"] = 5 + 2 * a - b + a * b + a * a + rng.normal(0, 0.05, len(frame))
    frame["Quality"] = np.where(np.arange(len(frame)) % 3 == 0, "poor", "good")  # not separable by the factors
    repo.delete_table(name)
    repo.import_dataframe(frame.drop(columns=[c for c in ("Hide", "Selected") if c in frame]), table_name=name,
                          normalize_columns=False)
    repo.set_table_info(name, "doe", DOEExperimentDialog.design_record(doe._read_request()))
    yield repo, name
    repo.close()


def _select(dialog: FitModelDialog, *names: str) -> None:
    columns = dialog.roles_widget._column_list
    columns.clearSelection()
    for row in range(columns.count()):
        item = columns.item(row)
        item.setSelected(item.data(Qt.ItemDataRole.UserRole) in names)


def test_the_design_casts_y_and_picks_the_model(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        assert dialog.roles_widget.casting().columns("response") == ["Response_1"]
        labels = [dialog._term_list.item(i).text() for i in range(dialog._term_list.count())]
        assert "Factor_1*Factor_2" in labels and "Factor_1*Factor_1" in labels  # a response surface
        assert [f.name for f in dialog._factors()] == ["Factor_1", "Factor_2", "Factor_3"]
        assert not dialog.problems()
    finally:
        dialog.close()


def test_preview_reports_and_ok_writes_the_results_and_the_figure(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog.preview()
        _wait(qapp, dialog)
        fit = dialog._result[0]
        estimates = fit.estimates.set_index("Term")["Estimate"]
        assert estimates["Factor_1"] == pytest.approx(2, abs=0.1)
        assert estimates["Factor_1*Factor_2"] == pytest.approx(1, abs=0.1)
        assert estimates["Factor_1*Factor_1"] == pytest.approx(1, abs=0.15)
        assert "Effect summary" in dialog.format_results(dialog._result)

        dialog._chart_checks["qq"].setChecked(True)
        dialog._chart_checks["interaction"].setChecked(True)
        applied: list[bool] = []
        dialog.applied.connect(lambda: applied.append(True))
        dialog.ok()
        _wait(qapp, dialog)
        assert applied == [True]
        tables = list(repo.list_user_tables()["Table"])
        assert {f"{name}_fit", f"{name}_fit_effects", f"{name}_fit_profile"} <= set(tables)
        results = repo.table_frame(f"{name}_fit")
        assert {"Predicted Response_1", "Residual Response_1"} <= set(results.columns)
        assert len(dialog.created_figure_ids) == 1
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert figure is not None
        assert [axis.name for axis in figure.axes] == [
            "Scatter Plot", "Pareto Chart", "Scatter Plot", "Q-Q Plot", "Scatter Plot", "Interaction Plot",
        ]
        assert (figure.nrows, figure.ncols) == (3, 2)
        profile_axis = figure.axes[4]
        assert [s.name for s in profile_axis.series] == ["Factor_1", "Factor_2", "Factor_3"]
        # Every chart draws its data: a series reads its columns under its roles' names.
        from matplotlib.figure import Figure

        from app.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert all(axes.collections or axes.lines or axes.patches for axes in drawn.axes)
        notes = repo.get_figure_options(dialog.created_figure_ids[0])["view"]["notes_html"]
        assert "Analysis of variance" in notes and "Response_1" in notes
        history = repo.query_df('SELECT operation FROM "__operations__"')
        assert list(history["operation"]) == ["Fit Model"]
    finally:
        dialog.close()


def test_effects_are_built_from_the_selected_columns(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        _select(dialog, "Factor_1", "Factor_3")
        dialog._apply_macro("main", edited=True)
        assert dialog._terms == [("Factor_1",), ("Factor_3",)]
        dialog._cross_selected()
        assert ("Factor_1", "Factor_3") in dialog._terms
        _select(dialog, "Factor_2")
        dialog._add_selected()
        dialog._cross_selected()
        assert ("Factor_2",) in dialog._terms and ("Factor_2", "Factor_2") in dialog._terms
        dialog._term_list.item(0).setSelected(True)
        dialog._remove_selected()
        assert ("Factor_1",) not in dialog._terms
        # A response cannot be a factor too.
        _select(dialog, "Response_1")
        dialog._add_selected()
        assert any("both a response and a factor" in p for p in dialog.problems())
    finally:
        dialog.close()


def test_column_types_are_guessed_from_the_data() -> None:
    assert guess_kind(pd.Series(["a", "b", "a"])) == NOMINAL
    assert guess_kind(pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3])) == NOMINAL  # three repeated levels
    assert guess_kind(pd.Series([0.1, 0.5, 2.3, 7.7, 1.2])) == CONTINUOUS


def test_emphasis_ticks_the_charts_and_a_logistic_fit_reports_chi_squares(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog._emphasis.setCurrentIndex(dialog._emphasis.findData("screening"))
        assert dialog.charts() == ["pareto", "profile"]
        assert "jmp.com/support/help" in dialog._doc_link.text()

        # A nominal Y needs a logistic personality.
        dialog.roles_widget.set_casting({"response": ["Quality"]})
        dialog._apply_macro("main", edited=True)
        assert any("nominal" in p for p in dialog.problems())
        dialog._personality.setCurrentIndex(dialog._personality.findData("nominal"))
        assert not dialog.problems()
        assert not dialog._chart_checks["actual"].isEnabled()
        assert dialog._family.isHidden() and dialog._emphasis.isHidden()

        dialog.preview()
        _wait(qapp, dialog)
        fit = dialog._result[0]
        assert fit.personality == "nominal" and fit.levels == ["good", "poor"]
        report = dialog.format_results(dialog._result)
        assert "Whole model test" in report and "Prob &gt; ChiSq" in report

        dialog.ok()
        _wait(qapp, dialog)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Pareto Chart", "Scatter Plot"]
        effects = repo.table_frame(f"{name}_fit_effects")
        assert "Prob > ChiSq" in effects.columns
    finally:
        dialog.close()


def test_a_poisson_glm_and_forward_stepwise(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog.roles_widget.set_casting({"response": ["Response_1"]})  # the window remembers the last Y
        dialog._personality.setCurrentIndex(dialog._personality.findData("stepwise"))
        assert not dialog._direction.isHidden()
        dialog._direction.setCurrentIndex(dialog._direction.findData("forward"))
        dialog.preview()
        _wait(qapp, dialog)
        kept = {lm_term for lm_term in dialog._result[0].terms}
        assert ("Factor_1",) in kept and ("Factor_3",) not in kept

        dialog._personality.setCurrentIndex(dialog._personality.findData("glm"))
        dialog._family.setCurrentIndex(dialog._family.findData("gamma"))
        dialog.preview()
        _wait(qapp, dialog)
        assert dialog._result[0].p_column == "Prob > ChiSq"
    finally:
        dialog.close()
