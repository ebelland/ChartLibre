"""Series Operation Builder: scaffold a new operation file (todo.txt P3-6).

OPERATIONS_DIR is monkeypatched to a tmp_path in every test here - this
tool writes real .py files, and app/series_operations/ is not a place a
test run should ever leave litter in, success or failure.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs import series_operation_builder_dialog as module
from app.dialogs.dev_tools_common import class_name, import_check, slug
from app.dialogs.series_operation_builder_dialog import (
    SeriesOperationBuilderDialog,
    render_stub_source,
)


@pytest.fixture(autouse=True)
def operations_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(module, "OPERATIONS_DIR", tmp_path)
    return tmp_path


# ----------------------------------------------------------------------
# The generated source
# ----------------------------------------------------------------------
def test_the_generated_source_parses() -> None:
    source = render_stub_source(
        class_name="EnvelopeDetectorDialog",
        name="Envelope Detector",
        description="Detect the amplitude envelope.",
        slug="envelope_detector",
    )
    ast.parse(source)
    assert "class EnvelopeDetectorDialog(SeriesOperationDialogBase):" in source
    assert "class EnvelopeDetectorDialogResult" in source


def test_the_generated_class_is_importable_and_discoverable(operations_dir: Path) -> None:
    source = render_stub_source(
        class_name="EnvelopeDetectorDialog", name="Envelope Detector",
        description="d", slug="envelope_detector",
    )
    target = operations_dir / "envelope_detector_dialog.py"
    target.write_text(source, encoding="utf-8")

    assert import_check(target, "EnvelopeDetectorDialog") == ""
    assert module.SeriesOperationBuilderDialog._operation_name_taken("Envelope Detector")


def test_the_generated_operation_runs_end_to_end(
    operations_dir: Path, qapp, tmp_path: Path
) -> None:
    """Not just importable - construct it against real data and run it."""
    import importlib.util
    import sys

    source = render_stub_source(
        class_name="EnvelopeDetectorDialog", name="Envelope Detector",
        description="d", slug="envelope_detector",
    )
    target = operations_dir / "envelope_detector_dialog.py"
    target.write_text(source, encoding="utf-8")

    module_name = "_series_operation_builder_smoke_check"
    spec = importlib.util.spec_from_file_location(module_name, target)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = loaded
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.modules.pop(module_name, None)
    dialog_cls = loaded.EnvelopeDetectorDialog

    db_path = tmp_path / "smoke.dhub"
    repo = SqliteRepo(db_path=db_path)
    repo.import_dataframe(
        pd.DataFrame({"t": np.arange(30.0), "v": np.sin(np.arange(30) / 5.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="t", y_label="v", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT t AS x, v AS y FROM w", roles={"x": "x", "y": "y"}, style={},
    )

    built = dialog_cls(repo=repo, figure_id=figure_id)
    try:
        built.series_selector.reload(select_all_series=True)
        results = built.compute_results()
        assert len(results) == 1
        np.testing.assert_allclose(results[0].y, np.sin(np.arange(30) / 5.0))
        assert built.format_results(results)
        assert built.generated_style_filter == {
            "generated_envelope_detector": True,
            "envelope_detector_dialog": "envelope_detector",
        }
    finally:
        built.close()


# ----------------------------------------------------------------------
# The dialog itself
# ----------------------------------------------------------------------
@pytest.fixture
def dialog(qapp, operations_dir: Path) -> SeriesOperationBuilderDialog:
    built = SeriesOperationBuilderDialog()
    yield built
    built.close()


def test_the_file_name_follows_the_operation_name_until_edited_by_hand(dialog) -> None:
    dialog._name_edit.setText("Envelope Detector")
    dialog._on_name_edited("Envelope Detector")
    assert dialog._file_edit.text() == "envelope_detector_dialog.py"

    dialog._file_edit.setText("custom_name.py")
    dialog._on_file_edited_by_user("custom_name.py")
    dialog._name_edit.setText("Something Else")
    dialog._on_name_edited("Something Else")

    assert dialog._file_edit.text() == "custom_name.py"


def test_an_empty_name_is_rejected(dialog) -> None:
    dialog._file_edit.setText("something.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_a_missing_description_is_rejected(dialog) -> None:
    dialog._name_edit.setText("N")
    dialog._file_edit.setText("n.py")

    assert dialog._validate() != ""


def test_an_existing_file_name_is_rejected(dialog, operations_dir: Path) -> None:
    (operations_dir / "taken.py").write_text("# already here\n", encoding="utf-8")
    dialog._name_edit.setText("N")
    dialog._file_edit.setText("taken.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_a_name_already_used_by_an_operation_is_rejected(dialog, operations_dir: Path) -> None:
    (operations_dir / "existing_dialog.py").write_text(
        render_stub_source(
            class_name="ExistingDialog", name="Existing", description="d", slug="existing"
        ),
        encoding="utf-8",
    )
    dialog._name_edit.setText("Existing")
    dialog._file_edit.setText("other.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_creating_a_valid_operation_writes_and_opens_the_file(
    dialog, operations_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(module, "open_in_editor", lambda path: opened.append(str(path)))
    shown: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        module, "show_message",
        lambda _parent, message_id, **fields: shown.append((message_id, fields)),
    )

    dialog._name_edit.setText("Envelope Detector")
    dialog._on_name_edited("Envelope Detector")
    dialog._description_edit.setText("Detect the amplitude envelope.")

    dialog._on_create()

    target = operations_dir / "envelope_detector_dialog.py"
    assert target.exists()
    ast.parse(target.read_text(encoding="utf-8"))
    assert opened == [str(target)]
    assert shown == [
        ("dev.series_operation_created", {"name": "Envelope Detector", "path": str(target)})
    ]
