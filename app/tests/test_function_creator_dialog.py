"""Function Creator: scaffold a new fit-function file (todo.txt P3-4).

FUNCTIONS_DIR is monkeypatched to a tmp_path in every test here - this
tool writes real .py files, and app/functions/ is not a place a test run
should ever leave litter in, success or failure.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from app.dialogs import function_creator_dialog as module
from app.dialogs.dev_tools_common import class_name, import_check
from app.dialogs.function_creator_dialog import (
    FunctionCreatorDialog,
    render_stub_source,
)
from app.scanners.functions_scanner import FunctionScanner


@pytest.fixture(autouse=True)
def functions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(module, "FUNCTIONS_DIR", tmp_path)
    return tmp_path


# ----------------------------------------------------------------------
# The generated source
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "params, x, p, expected",
    [
        (["a"], [0.0, 5.0], [3.0], [3.0, 3.0]),
        (["intercept", "slope"], [0.0, 1.0, 2.0], [1.0, 2.0], [1.0, 3.0, 5.0]),
        (["a", "b", "c"], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 6.0, 17.0]),
    ],
)
def test_execute_is_a_real_polynomial_of_the_right_degree(
    params: list[str], x: list[float], p: list[float], expected: list[float]
) -> None:
    source = render_stub_source(
        class_name="TestFunction", name="Test", description="d", params=params
    )
    ast.parse(source)

    namespace: dict = {}
    exec(compile(source, "<test>", "exec"), namespace)
    cls = namespace["TestFunction"]

    result = cls.execute(np.asarray(x), np.asarray(p))
    np.testing.assert_allclose(result, expected)
    assert cls.params == params
    assert cls.p0 == [1.0] * len(params)


def test_the_generated_class_is_importable_and_discoverable(functions_dir: Path) -> None:
    source = render_stub_source(
        class_name="TestFunction", name="Test Function XYZ", description="d",
        params=["a", "b"],
    )
    target = functions_dir / "test_function.py"
    target.write_text(source, encoding="utf-8")

    assert import_check(target, "TestFunction") == ""
    assert module.FunctionCreatorDialog._function_name_taken("Test Function XYZ")


def test_a_scanner_pointed_at_the_folder_finds_it_and_runs_it(functions_dir: Path) -> None:
    source = render_stub_source(
        class_name="TestFunction", name="Test Function ABC", description="d",
        params=["intercept", "slope"],
    )
    target = functions_dir / "test_function.py"
    target.write_text(source, encoding="utf-8")

    scanner = FunctionScanner(extra_roots=(functions_dir,))
    matches = [spec for spec in scanner.specs() if spec.name == "Test Function ABC"]
    assert len(matches) == 1
    cls = scanner.load_class(matches[0].as_catalog_payload())
    np.testing.assert_allclose(
        cls.execute(np.array([0.0, 1.0]), np.array([1.0, 2.0])), [1.0, 3.0]
    )


# ----------------------------------------------------------------------
# The dialog itself
# ----------------------------------------------------------------------
@pytest.fixture
def dialog(qapp, functions_dir: Path) -> FunctionCreatorDialog:
    built = FunctionCreatorDialog()
    yield built
    built.close()


def test_parameters_default_to_intercept_and_slope(dialog) -> None:
    assert dialog._params_edit.text() == "intercept, slope"


def test_the_file_name_follows_the_function_name_until_edited_by_hand(dialog) -> None:
    dialog._name_edit.setText("Stretched Exponential")
    dialog._on_name_edited("Stretched Exponential")
    assert dialog._file_edit.text() == "stretched_exponential.py"

    dialog._file_edit.setText("custom_name.py")
    dialog._on_file_edited_by_user("custom_name.py")
    dialog._name_edit.setText("Something Else")
    dialog._on_name_edited("Something Else")

    assert dialog._file_edit.text() == "custom_name.py"


def test_an_empty_name_is_rejected(dialog) -> None:
    dialog._file_edit.setText("something.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_no_parameters_is_rejected(dialog) -> None:
    dialog._name_edit.setText("N")
    dialog._file_edit.setText("n.py")
    dialog._description_edit.setText("d")
    dialog._params_edit.setText("")

    assert dialog._validate() != ""


def test_a_name_already_used_is_rejected(dialog, functions_dir: Path) -> None:
    (functions_dir / "existing.py").write_text(
        render_stub_source(
            class_name="ExistingFunction", name="Existing", description="d",
            params=["a"],
        ),
        encoding="utf-8",
    )
    dialog._name_edit.setText("Existing")
    dialog._file_edit.setText("other.py")
    dialog._description_edit.setText("d")

    assert dialog._validate() != ""


def test_creating_a_valid_function_writes_and_opens_the_file(
    dialog, functions_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(module, "open_in_editor", lambda path: opened.append(str(path)))
    shown: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        module, "show_message",
        lambda _parent, message_id, **fields: shown.append((message_id, fields)),
    )

    dialog._name_edit.setText("Stretched Exponential")
    dialog._on_name_edited("Stretched Exponential")
    dialog._description_edit.setText("A stretched exponential decay.")

    dialog._on_create()

    target = functions_dir / "stretched_exponential.py"
    assert target.exists()
    ast.parse(target.read_text(encoding="utf-8"))
    assert opened == [str(target)]
    assert shown == [
        ("dev.function_created", {"name": "Stretched Exponential", "path": str(target)})
    ]
