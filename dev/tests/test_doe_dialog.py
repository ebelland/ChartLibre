"""Design of Experiments: every model makes its matrix, and OK makes it a table."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.doe_dialog import DOEExperimentDialog

#: Runs for three factors, one replicate, the default centre points.
_RUNS = {
    "full_factorial": 8,
    "two_level": 8,
    "box_behnken": 15,
    "ccd_face": 17,
    "ccd_circumscribed": 17,
    "latin_hypercube": 20,
}


@pytest.fixture
def dialog(qapp, tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "doe.dhub")
    built = DOEExperimentDialog(repo)
    built.factor_count.setValue(3)
    yield built
    built.close()
    repo.close()


@pytest.mark.parametrize("model", sorted(_RUNS))
def test_each_model_makes_its_matrix_within_the_bounds(dialog: DOEExperimentDialog, model: str) -> None:
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData(model))
    frame = dialog.generate_matrix(dialog._read_request())
    assert list(frame.columns) == ["Id", "Factor_1", "Factor_2", "Factor_3", "Response_1"]
    assert len(frame) == _RUNS[model]
    factors = frame[["Factor_1", "Factor_2", "Factor_3"]]
    limit = 1.0 if model != "ccd_circumscribed" else 2.0  # its star points reach past the bounds
    assert factors.abs().to_numpy().max() <= limit + 1e-9


def test_every_latin_hypercube_algorithm_runs(dialog: DOEExperimentDialog) -> None:
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData("latin_hypercube"))
    for index in range(dialog.lhs_algorithm.count()):
        dialog.lhs_algorithm.setCurrentIndex(index)
        frame = dialog.generate_matrix(dialog._read_request())
        assert len(frame) == 20 and frame["Factor_1"].between(-1, 1).all()


def test_ok_writes_the_matrix_as_a_new_table(dialog: DOEExperimentDialog) -> None:
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData("full_factorial"))
    dialog.accept()
    assert dialog.created_table_name is not None
    assert dialog._repo.series_row_count(f'SELECT * FROM "{dialog.created_table_name}"') == 8


def test_the_table_keeps_its_design_through_rename_and_copy(dialog: DOEExperimentDialog) -> None:
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData("box_behnken"))
    dialog.accept()
    repo, name = dialog._repo, dialog.created_table_name
    assert name is not None
    design = repo.get_table_info(name)["doe"]
    assert design["model"] == "box_behnken"
    assert [f["name"] for f in design["factors"]] == ["Factor_1", "Factor_2", "Factor_3"]
    assert design["factors"][0] == {"name": "Factor_1", "kind": "numeric", "low": -1.0, "high": 1.0}
    assert design["responses"] == ["Response_1"]

    repo.rename_table(name, "trial")
    assert repo.get_table_info("trial")["doe"] == design
    assert repo.get_table_info(name) == {}
    copy = repo.duplicate_table("trial")
    assert repo.get_table_info(copy)["doe"] == design
    repo.delete_table("trial")
    assert repo.get_table_info("trial") == {}
