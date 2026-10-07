"""Fit Model on a DOE table: the design casts the roles, Preview reports, OK writes and draws."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    repo.delete_table(name)
    repo.import_dataframe(frame.drop(columns=[c for c in ("Hide", "Selected") if c in frame]), table_name=name,
                          normalize_columns=False)
    repo.set_table_info(name, "doe", DOEExperimentDialog.design_record(doe._read_request()))
    yield repo, name
    repo.close()


def test_the_design_casts_the_roles_and_picks_the_macro(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        casting = dialog.roles_widget.casting()
        assert casting.columns("response") == ["Response_1"]
        assert casting.columns("factors") == ["Factor_1", "Factor_2", "Factor_3"]
        labels = [dialog._term_list.item(i).text() for i in range(dialog._term_list.count())]
        assert "Factor_1*Factor_2" in labels and "Factor_1*Factor_1" in labels  # a response surface
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

        applied: list[bool] = []
        dialog.applied.connect(lambda: applied.append(True))
        dialog.ok()
        _wait(qapp, dialog)
        assert applied == [True]
        tables = list(repo.list_user_tables()["Table"])
        assert f"{name}_fit" in tables and f"{name}_fit_effects" in tables
        results = repo.table_frame(f"{name}_fit")
        assert {"Predicted Response_1", "Residual Response_1"} <= set(results.columns)
        assert len(dialog.created_figure_ids) == 1
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert figure is not None and [axis.name for axis in figure.axes] == [
            "Scatter Plot", "Horizontal Bar Chart", "Scatter Plot", "Q-Q Plot",
        ]
        history = repo.query_df('SELECT operation FROM "__operations__"')
        assert list(history["operation"]) == ["Fit Model"]
    finally:
        dialog.close()


def test_the_effects_tab_adds_crosses_and_removes(qapp, doe_table) -> None:
    repo, name = doe_table
    dialog = FitModelDialog(repo=repo, table=name)
    try:
        dialog._apply_macro("main", edited=True)
        assert dialog._terms == [("Factor_1",), ("Factor_2",), ("Factor_3",)]
        dialog._factor_list.item(0).setSelected(True)
        dialog._factor_list.item(2).setSelected(True)
        dialog._cross_selected()
        assert ("Factor_1", "Factor_3") in dialog._terms
        dialog._term_list.item(0).setSelected(True)
        dialog._remove_selected()
        assert ("Factor_1",) not in dialog._terms
        # Taking a factor out of its role takes its terms with it.
        dialog.roles_widget.unassign("factors", ["Factor_3"])
        assert all("Factor_3" not in term for term in dialog._terms)
    finally:
        dialog.close()


def test_column_types_are_guessed_from_the_data() -> None:
    assert guess_kind(pd.Series(["a", "b", "a"])) == NOMINAL
    assert guess_kind(pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3])) == NOMINAL  # three repeated levels
    assert guess_kind(pd.Series([0.1, 0.5, 2.3, 7.7, 1.2])) == CONTINUOUS
