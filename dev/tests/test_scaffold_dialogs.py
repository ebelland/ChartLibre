"""The Developer menu's three scaffolding tools write a file their scanner finds."""
from __future__ import annotations

from pathlib import Path

import pytest

import app.dialogs.dev_tools_common as common
from app.dialogs.function_creator_dialog import FunctionCreatorDialog
from app.dialogs.renderer_helper_dialog import RendererHelperDialog
from app.dialogs.series_operation_builder_dialog import SeriesOperationBuilderDialog


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """What the tools would have shown or opened, instead of showing it."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(common, "show_message",
                        lambda parent, key, **kw: seen.append((key, str(kw.get("error") or kw.get("detail") or ""))))
    monkeypatch.setattr(common, "open_in_editor", lambda path: seen.append(("opened", path.name)))
    return seen


@pytest.mark.parametrize(
    ("dialog_class", "name", "file_name", "created"),
    [
        (FunctionCreatorDialog, "Stretched Exponential Test", "stretched_exponential_test.py", "dev.function_created"),
        (SeriesOperationBuilderDialog, "Envelope Detector Test", "envelope_detector_test_dialog.py",
         "dev.series_operation_created"),
        (RendererHelperDialog, "Ridgeline Test", "ridgeline_test.py", "dev.renderer_created"),
    ],
)
def test_a_tool_writes_a_file_its_scanner_finds_and_will_not_overwrite_it(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shown: list[tuple[str, str]],
    dialog_class, name: str, file_name: str, created: str,
) -> None:
    monkeypatch.setattr(dialog_class, "USER_DIR", tmp_path)
    dialog = dialog_class()
    dialog._name_edit.setText(name)
    dialog._on_name_edited(name)
    assert dialog._file_edit.text() == file_name

    dialog._on_create()  # no description yet
    assert shown == [("dev.validation_error", "Description cannot be empty.")]

    dialog._description_edit.setText("A test.")
    shown.clear()
    dialog._on_create()
    assert shown == [(created, ""), ("opened", file_name)]
    assert (tmp_path / file_name).is_file()

    again = dialog_class()
    again._name_edit.setText(name)
    again._on_name_edited(name)
    again._description_edit.setText("A test.")
    shown.clear()
    again._on_create()
    assert shown and "already exists" in shown[0][1]


def test_a_built_in_name_is_refused_and_a_typed_file_name_is_kept(qapp, shown: list[tuple[str, str]]) -> None:
    dialog = RendererHelperDialog()
    dialog._name_edit.setText("Scatter Plot")
    dialog._on_name_edited("Scatter Plot")
    dialog._description_edit.setText("x")
    dialog._on_create()
    assert shown == [("dev.validation_error", '"Scatter Plot" is already used by an existing chart type.')]

    dialog._file_edit.setText("custom.py")
    dialog._on_file_edited_by_user("custom.py")
    dialog._on_name_edited("Something else")
    assert dialog._file_edit.text() == "custom.py"


@pytest.mark.parametrize("dialog_class", [FunctionCreatorDialog, SeriesOperationBuilderDialog, RendererHelperDialog])
def test_every_text_a_tool_shows_is_found_for_the_catalogues(dialog_class) -> None:
    from app.utils.i18n import TRANSLATED_ATTRIBUTES, source_translator_calls

    found = source_translator_calls()
    for attribute in TRANSLATED_ATTRIBUTES & set(vars(dialog_class)):
        assert getattr(dialog_class, attribute) in found, attribute
