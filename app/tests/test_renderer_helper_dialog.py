"""Renderer Helper: scaffold a new chart-type file (todo.txt P3-6).

CHARTS_DIR is monkeypatched to a tmp_path in every test here - this tool
writes real .py files, and app/charts/ is not a place a test run should
ever leave litter in, success or failure.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.dialogs import renderer_helper_dialog as module
from app.dialogs.renderer_helper_dialog import (
    RendererHelperDialog,
    _class_name,
    _import_check,
    _parse_roles,
    _renderer_name_taken,
    _slug,
    render_stub_source,
)


@pytest.fixture(autouse=True)
def charts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(module, "CHARTS_DIR", tmp_path)
    return tmp_path


# ----------------------------------------------------------------------
# Name -> file/class name derivation
# ----------------------------------------------------------------------
def test_slug_from_a_title() -> None:
    assert _slug("Ridgeline Plot") == "ridgeline_plot"


def test_slug_strips_punctuation() -> None:
    assert _slug("3D Bar-ish Chart!") == "3d_bar_ish_chart"


def test_class_name_from_a_title() -> None:
    assert _class_name("Ridgeline Plot", suffix="AxisRenderer") == "RidgelinePlotAxisRenderer"


def test_class_name_starting_with_a_digit_gets_a_prefix() -> None:
    # "3DThing" is not a legal identifier - a class name may not start with a digit.
    name = _class_name("3D Thing", suffix="AxisRenderer")
    assert name[0].isalpha()
    assert ast.parse(f"class {name}: pass")


def test_parse_roles_splits_on_commas_and_whitespace() -> None:
    assert _parse_roles("x, y,  z") == ["x", "y", "z"]
    assert _parse_roles("") == []


# ----------------------------------------------------------------------
# The generated source
# ----------------------------------------------------------------------
def test_the_generated_source_parses() -> None:
    source = render_stub_source(
        class_name="RidgelinePlotAxisRenderer",
        name="Ridgeline Plot",
        category="User",
        description="Overlapping density curves per category.",
        link="https://matplotlib.org/",
        required_roles=["x", "y"],
        optional_roles=["category"],
    )
    ast.parse(source)
    assert 'Name: str = ' in source
    assert "'Ridgeline Plot'" in source or '"Ridgeline Plot"' in source


def test_x_y_roles_get_a_working_plot_stub() -> None:
    source = render_stub_source(
        class_name="C", name="N", category="User", description="d", link="",
        required_roles=["x", "y"], optional_roles=[],
    )
    assert "ax.plot(" in source
    assert "TODO: draw sd onto ax" not in source


def test_other_roles_get_a_todo_stub() -> None:
    source = render_stub_source(
        class_name="C", name="N", category="User", description="d", link="",
        required_roles=["value"], optional_roles=[],
    )
    assert "ax.plot(" not in source
    assert 'sd.df["value"]' in source


def test_the_generated_class_is_importable_and_discoverable(charts_dir: Path) -> None:
    source = render_stub_source(
        class_name="RidgelinePlotAxisRenderer", name="Ridgeline Plot",
        category="User", description="d", link="",
        required_roles=["x", "y"], optional_roles=[],
    )
    target = charts_dir / "ridgeline_plot.py"
    target.write_text(source, encoding="utf-8")

    error = _import_check(target, "RidgelinePlotAxisRenderer")

    assert error == ""
    assert _renderer_name_taken("Ridgeline Plot") is True
    assert _renderer_name_taken("Something Else Entirely") is False


def test_import_check_reports_a_broken_file(charts_dir: Path) -> None:
    target = charts_dir / "broken.py"
    target.write_text("this is not valid python (((", encoding="utf-8")

    error = _import_check(target, "Whatever")

    assert error != ""


# ----------------------------------------------------------------------
# The dialog itself
# ----------------------------------------------------------------------
@pytest.fixture
def dialog(qapp, charts_dir: Path) -> RendererHelperDialog:
    built = RendererHelperDialog()
    yield built
    built.close()


def test_the_file_name_follows_the_chart_name_until_edited_by_hand(dialog) -> None:
    dialog._name_edit.setText("Ridgeline Plot")
    dialog._on_name_edited("Ridgeline Plot")
    assert dialog._file_edit.text() == "ridgeline_plot.py"

    dialog._file_edit.setText("custom_name.py")
    dialog._on_file_edited_by_user("custom_name.py")
    dialog._name_edit.setText("Something Else")
    dialog._on_name_edited("Something Else")

    assert dialog._file_edit.text() == "custom_name.py"


def test_category_defaults_to_user(dialog) -> None:
    assert dialog._category_combo.currentText() == "User"


def test_an_empty_name_is_rejected(dialog) -> None:
    dialog._file_edit.setText("something.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_a_missing_description_is_rejected(dialog) -> None:
    dialog._name_edit.setText("N")
    dialog._file_edit.setText("n.py")

    assert dialog._validate() != ""


def test_an_existing_file_name_is_rejected(dialog, charts_dir: Path) -> None:
    (charts_dir / "taken.py").write_text("# already here\n", encoding="utf-8")
    dialog._name_edit.setText("N")
    dialog._file_edit.setText("taken.py")
    dialog._description_edit.setText("d")

    assert "taken.py" in dialog._validate() or "already exists" in dialog._validate()


def test_a_name_already_used_by_a_renderer_is_rejected(dialog, charts_dir: Path) -> None:
    (charts_dir / "existing.py").write_text(
        render_stub_source(
            class_name="ExistingAxisRenderer", name="Existing",
            category="User", description="d", link="",
            required_roles=["x", "y"], optional_roles=[],
        ),
        encoding="utf-8",
    )
    dialog._name_edit.setText("Existing")
    dialog._file_edit.setText("other.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_creating_a_valid_renderer_writes_and_opens_the_file(
    dialog, charts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(module, "open_in_editor", lambda path: opened.append(str(path)))
    # show_message opens a real, blocking QMessageBox - never let a test
    # leave one on screen; record what it was called with instead.
    shown: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        module, "show_message",
        lambda _parent, message_id, **fields: shown.append((message_id, fields)),
    )

    dialog._name_edit.setText("Ridgeline Plot")
    dialog._on_name_edited("Ridgeline Plot")
    dialog._description_edit.setText("Overlapping density curves.")
    dialog._required_roles_edit.setText("x, y")

    dialog._on_create()

    target = charts_dir / "ridgeline_plot.py"
    assert target.exists()
    ast.parse(target.read_text(encoding="utf-8"))
    assert opened == [str(target)]
    assert shown == [("dev.renderer_created", {"name": "Ridgeline Plot", "path": str(target)})]
    assert dialog.result() == RendererHelperDialog.DialogCode.Accepted
