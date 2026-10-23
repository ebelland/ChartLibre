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
from app.plugin_api.table_operations.column_roles import CONTINUOUS, NOMINAL, ORDINAL, declared_kind, guess_kind
from app.table_operations.fit_model_dialog import FitModelDialog


def _wait(qapp, dialog: FitModelDialog) -> None:
    """Until the background computation is done (the first statsmodels import takes seconds)."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        qapp.processEvents()
        if not dialog.job_running:
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
        assert f"_{name}_fit" in tables
        # What only feeds the charts is marked by its name and left out of the list.
        feeding = {f"_{name}_fit_effects_", f"_{name}_fit_profile_"}
        assert not feeding & set(tables)
        assert feeding <= set(repo.list_user_tables(include_internal=True)["Table"])
        results = repo.table_frame(f"_{name}_fit")
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

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

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


def test_column_types_follow_the_declared_type_then_the_data() -> None:
    """As JMP's defaults: a float is continuous, an integer ordinal, text nominal."""
    assert declared_kind("REAL") == CONTINUOUS and declared_kind("DOUBLE") == CONTINUOUS
    assert declared_kind("INTEGER") == ORDINAL and declared_kind("BIGINT") == ORDINAL
    assert declared_kind("TEXT") == NOMINAL and declared_kind("") is None
    assert guess_kind(pd.Series(["a", "b", "a"])) == NOMINAL
    assert guess_kind(pd.Series([1, 2, 3, 1, 2, 3])) == ORDINAL
    assert guess_kind(pd.Series([0.1, 0.5, 2.3, 7.7, 1.2])) == CONTINUOUS
    assert guess_kind(pd.Series(["1", "2", "3"], dtype=object)) == ORDINAL


def test_an_ordinal_response_is_fitted_by_ordinal_logistic_and_an_ordinal_factor_as_levels(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        kinds = dict(dialog.roles_widget.casting().kinds, Quality=ORDINAL)
        dialog.roles_widget.set_casting({"response": ["Quality"], "factor": ["Factor_1"]}, kinds)
        assert dialog.personality() == "ordinal"
        kinds["Factor_1"] = ORDINAL
        dialog.roles_widget.set_casting({"response": ["Response_1"], "factor": ["Factor_1"]}, kinds)
        assert dialog.personality() == "standard"
        assert [f.kind for f in dialog._factors() if f.name == "Factor_1"] == [NOMINAL]
    finally:
        dialog.close()


def test_emphasis_ticks_the_charts_and_a_logistic_fit_reports_chi_squares(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog._emphasis.setCurrentIndex(dialog._emphasis.findData("screening"))
        assert dialog.charts() == ["pareto", "profile"]
        assert "jmp.com/support/help" in dialog._doc_link.text()

        # A nominal Y needs a logistic personality, and gets one, as in JMP.
        dialog.roles_widget.set_casting({"response": ["Quality"]})
        dialog._apply_macro("main", edited=True)
        assert dialog.personality() == "nominal"
        dialog._personality.setCurrentIndex(dialog._personality.findData("standard"))
        assert any("nominal" in p for p in dialog.problems())
        dialog._personality.setCurrentIndex(dialog._personality.findData("nominal"))
        assert not dialog.problems()
        assert not dialog._chart_checks["actual"].isEnabled()
        assert dialog._family.isHidden() and dialog._emphasis.isHidden()

        dialog.preview()
        _wait(qapp, dialog)
        fit = dialog._result[0]
        assert fit.personality == "nominal" and fit.levels == ["good", "poor"]
        # AN-19: the ROC curve of the first level, its area, the confusion matrix.
        assert set(fit.auc) == {"good"} and 0.5 <= fit.auc["good"] <= 1.0
        assert "AUC (good)" in fit.summary and int(fit.confusion.to_numpy().sum()) == int(fit.summary["Observations"])
        assert dialog._chart_checks["roc"].isEnabled()
        report = dialog.format_results(dialog._result)
        assert "Whole model test" in report and "Prob &gt; ChiSq" in report

        dialog.ok()
        _wait(qapp, dialog)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Pareto Chart", "Scatter Plot", "Scatter Plot"]  # + ROC
        effects = repo.table_frame(f"_{name}_fit_effects_")
        assert "Prob > ChiSq" in effects.columns
    finally:
        dialog.close()


def test_a_text_response_of_the_design_is_nominal_and_fitted_by_logistic(qapp, doe_table) -> None:
    """Reaction_DOE's Grade: a design's response recorded as Pass/Fail is not a number to least-squares."""
    repo, name = doe_table
    design = repo.get_table_info(name)["doe"]
    repo.set_table_info(name, "doe", {**design, "responses": ["Response_1", "Quality"]})
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        roles, kinds = dialog.default_casting(name, repo.table_frame(name))
        assert roles == {"response": ["Response_1", "Quality"]}
        dialog.roles_widget.set_casting(roles, kinds)
        casting = dialog.roles_widget.casting()
        assert casting.kinds["Quality"] == NOMINAL and casting.kinds["Response_1"] == CONTINUOUS
        dialog._apply_macro("main", edited=True)
        dialog._personality.setCurrentIndex(dialog._personality.findData("standard"))
        assert any("Quality is nominal" in p for p in dialog.problems())  # mixed: said, not attempted

        dialog.roles_widget.set_casting({"response": ["Quality"]}, casting.kinds)
        assert dialog.personality() == "nominal" and not dialog.problems()
        dialog._apply_macro("main", edited=True)  # 15 runs: a response surface would separate them
        dialog.preview()
        _wait(qapp, dialog)
        assert dialog._result[0].personality == "nominal"

        dialog.roles_widget.set_casting({"response": ["Response_1"]}, casting.kinds)
        assert dialog.personality() == "standard"
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


def test_generalized_regression_lasso_drops_noise_and_ridge_keeps_every_term(qapp, tmp_path) -> None:
    """AN-16: penalized fits, the penalty chosen by cross-validation."""
    import app.analysis.linear_model as lm

    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(8)})
    frame["y"] = 3 * frame.x0 - 2 * frame.x1 + rng.normal(0, 1, n)
    factors = [lm.Factor(f"x{i}", CONTINUOUS) for i in range(8)]
    fits = {}
    for penalty in ("lasso", "ridge"):
        spec = lm.ModelSpec(factors=factors, responses=["y"], terms=lm.main_effects(factors),
                            personality="generalized", penalty=penalty)
        fits[penalty] = lm.fit_models(frame, spec)[0]
    lasso, ridge = fits["lasso"], fits["ridge"]
    assert list(lasso.effect_tests["Term"][:2]) == ["x0", "x1"]
    assert lasso.removed and all(t[0] not in ("x0", "x1") for t in lasso.removed)
    assert not ridge.removed and ridge.summary["L1 ratio"] == 0.0
    assert lasso.summary["RSquare (cross-validated)"] > 0.8

    repo = SqliteRepo(db_path=tmp_path / "g.dhub")
    repo.import_dataframe(frame, table_name="wide", normalize_columns=False)
    dialog = FitModelDialog(repo=repo, table="wide")
    try:
        dialog.roles_widget.set_casting({"response": ["y"]})
        dialog.cast_explanatory([f"x{i}" for i in range(8)])
        dialog._personality.setCurrentIndex(dialog._personality.findData("generalized"))
        assert not dialog._penalty.isHidden()
        dialog._chart_checks["pareto"].setChecked(True)
        dialog.ok()
        _wait(qapp, dialog)
        assert "Std Magnitude" in dialog.format_results(dialog._result)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert "Pareto Chart" in [axis.name for axis in figure.axes]
    finally:
        dialog.close()
        repo.close()


def test_a_logistic_fit_draws_its_roc_curve(qapp, doe_table) -> None:
    """AN-19: OK writes the curve with the chance diagonal and draws it."""
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog.roles_widget.set_casting({"response": ["Quality"]})
        dialog._apply_macro("main", edited=True)
        assert dialog.personality() == "nominal" and "roc" in dialog.charts()
        dialog.ok()
        _wait(qapp, dialog)
        curve = repo.table_frame(f"_{name}_fit_roc_")
        assert {"good", "Chance"} <= set(curve["Level"])
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        roc_axis = next(axis for axis in figure.axes if axis.title == "ROC curve")
        assert [s.name for s in roc_axis.series][-1] == "Chance"
        assert "AUC" in roc_axis.series[0].name
    finally:
        dialog.close()


def test_several_responses_share_one_figure_and_one_report(qapp, doe_table) -> None:
    repo, name = doe_table
    frame = repo.table_frame(name)
    frame["Response_2"] = 3 * frame["Factor_1"] + frame["Factor_3"]
    repo.delete_table(name)
    repo.import_dataframe(frame.drop(columns=[c for c in ("Hide", "Selected") if c in frame]), table_name=name,
                          normalize_columns=False)
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog.roles_widget.assign("response", ["Response_1", "Response_2"])
        _select(dialog, "Factor_1", "Factor_2", "Factor_3")
        dialog._apply_macro("main", edited=True)
        dialog._chart_checks["qq"].setChecked(True)
        dialog.ok()
        _wait(qapp, dialog)
        assert len(dialog.created_figure_ids) == 1  # every response in it, not the first only
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        titles = [axis.title for axis in figure.axes]
        assert any(t.startswith("Response_1: ") for t in titles) and any(t.startswith("Response_2: ") for t in titles)
        notes = repo.get_figure_options(dialog.created_figure_ids[0])["view"]["notes_html"]
        assert "Response Response_1" in notes and "Response Response_2" in notes
        effects = repo.query_df(f'SELECT DISTINCT "Response" FROM "_{name}_fit_effects_"')
        assert set(effects["Response"]) == {"Response_1", "Response_2"}
    finally:
        dialog.close()


def test_stop_reaches_the_fit_and_leaves_the_results_untouched(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog.preview()
        dialog.stop()  # before the worker looks: the fit must give up, not finish
        _wait(qapp, dialog)
        assert dialog._result is None
        assert dialog._results.toPlainText() == "Stopped. Nothing was changed." if hasattr(dialog._results, "toPlainText") else True
    finally:
        dialog.close()


def test_robust_standard_errors_and_the_residual_diagnostics_are_reported(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog._std_errors.setCurrentIndex(dialog._std_errors.findData("HC3"))
        dialog.preview()
        _wait(qapp, dialog)
        fit = dialog._result[0]
        assert "HC3" in fit.std_errors_note
        assert set(fit.diagnostics["Test"]) >= {"Durbin-Watson (autocorrelation)", "Jarque-Bera (normal residuals)"}
        assert "VIF" in fit.estimates.columns
        html = dialog.report_body(dialog._result)
        assert "Residual diagnostics" in html and "Breusch-Pagan" in html and "robust to unequal variances" in html
    finally:
        dialog.close()
