"""Function Creator: scaffold a new fit-function file (todo.txt P3-4).

A fit function lives entirely in one file, as a class that directly
subclasses base_function (app/functions/base.py) - functions_scanner.py
AST-scans app/functions/ (built in, including the hand-written example
app/functions/user_functions.py's own sample_user_function - the template
this dialog's default follows) and user/functions/ (user-authored, kept
out of the app's own source tree - see
app.utils.config.USER_FUNCTIONS_DIR) together, with no registration step.
This dialog is a short form for Name, Description and a parameter list
that writes a new, immediately-usable file: execute(x, p) already
evaluates a real polynomial in the declared parameters (a constant for
one, linear for two - matching sample_user_function exactly - quadratic
for three, and so on), left for the user to replace with the real
formula. See base_function's own docstring for the full contract.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFormLayout, QLineEdit, QWidget

from app.developer_helpers.developer_helper_base import DeveloperDialogBase
from app.utils.config import USER_FUNCTIONS_DIR
from app.utils.i18n import _

#: Where functions_scanner.py itself looks for the built-in functions -
#: used here only to check a new name against them, never written to.
BUILTIN_FUNCTIONS_DIR: Path = Path(__file__).resolve().parent.parent / "functions"

#: Where this dialog writes - a user-authored function, kept separate from
#: the app's own shipped source. Created on first use.
FUNCTIONS_DIR: Path = USER_FUNCTIONS_DIR


def _parse_params(text: str) -> list[str]:
    """Split a comma-separated parameter-names field into clean names."""
    return [param.strip() for param in text.split(",") if param.strip()]


def _execute_body(params: list[str]) -> str:
    """A working execute(x, p) - a polynomial of degree len(params)-1.

    One parameter is a constant, two is exactly sample_user_function's own
    intercept + slope, three is a parabola, and so on - always something
    real to evaluate and plot immediately, never a bare TODO that leaves
    the new function looking broken before it has been touched at all.
    """
    terms = ["p[0]"] + [
        f"p[{index}] * x**{index}" for index in range(1, len(params))
    ]
    return " + ".join(terms)


def render_stub_source(
    *, class_name: str, name: str, description: str, params: list[str]
) -> str:
    """Return the full source text of the scaffolded function file."""
    p0 = ", ".join("1.0" for _param in params)
    params_literal = ", ".join(repr(param) for param in params)
    execute_body = _execute_body(params)
    param_lines = "\n".join(f"    p[{i}]   {param}" for i, param in enumerate(params)) or (
        "    (none declared)"
    )

    return f'''"""{name} fit function.

TODO: describe what this function models and when to reach for it, the
way app/functions/user_functions.py's own sample_user_function does -
read it alongside this one, and base_function's own docstring
(app/functions/base.py) documents the full contract.

Parameters:
{param_lines}

This file was scaffolded by the Function Creator (Developer menu). It is
already a valid, discoverable function - functions_scanner.py picks up
any class in app/functions/ or user/functions/ that directly subclasses
base_function, with no registration step - and execute(x, p) below
already evaluates a real polynomial in these parameters. Replace it with
the actual formula, update expression to match (it is the only thing in
the fit dialog that says what the parameters mean before a fit has run),
then delete this paragraph.
"""
from __future__ import annotations

import numpy as np

from app.functions.base import base_function


class {class_name}(base_function):
    name = {name!r}
    category = "User functions"
    description = {description!r}
    # TODO: update to match execute() below - shown in the fit dialog
    # before anything has been fitted, so it is what tells a person what
    # p[0], p[1], ... actually mean.
    expression = "<b>{name}</b><br>y = {execute_body}"
    p0 = [{p0}]
    params = [{params_literal}]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return {execute_body}

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        """TODO: read a starting point off the data, or delete this and
        let the caller search for one instead - see base_function's own
        docstring for why returning None is a real, useful answer."""
        del x, y
        return None
'''


class FunctionCreatorDialog(DeveloperDialogBase):
    """Scaffold a new base_function file from a short form."""

    TITLE = "Function Creator"
    ICON = "function_creator"
    CARD_NAME = "functionCreatorCard"
    HINT = (
        "Writes a new file under user/functions/ with a class already "
        "filled in from these fields - no registration step, the "
        "scanner picks it up as soon as the file exists. execute(x, p) "
        "already evaluates a real polynomial in the declared "
        "parameters; replace it with the actual formula."
    )
    NAME_PLACEHOLDER = "e.g. Stretched Exponential"
    FILE_PLACEHOLDER = "e.g. stretched_exponential.py"
    DESCRIPTION_PLACEHOLDER = "One line, shown in the fit dialog's model list."
    CLASS_SUFFIX = "Function"
    CREATE_ACTION = "create_function"
    BUILTIN_DIR = BUILTIN_FUNCTIONS_DIR
    USER_DIR = FUNCTIONS_DIR
    BASE_CLASS_NAME = "base_function"
    VALUE_ATTR = "name"
    REQUIRE_VALUE_ATTR = False
    CREATED_MESSAGE = "dev.function_created"
    FAILED_MESSAGE = "dev.function_create_failed"
    BAD_FILE_NAME = "File name must be a valid Python file name, e.g. my_function.py."
    FILE_EXISTS = "A file named \"{file}\" already exists under user/functions/."
    NAME_TAKEN = "\"{name}\" is already used by an existing function."
    NOT_DISCOVERED = (
        "The class imported, but functions_scanner did not discover it as "
        "a function - check that it subclasses base_function directly."
    )

    def add_fields(self, form: QFormLayout, card: QWidget) -> None:
        super().add_fields(form, card)
        self._params_edit = QLineEdit(card)
        self._params_edit.setText("intercept, slope")
        self._params_edit.setPlaceholderText(_("comma-separated, e.g. intercept, slope"))
        form.addRow(_("Parameters:"), self._params_edit)

    def validate_fields(self) -> str:
        if not _parse_params(self._params_edit.text()):
            return _("At least one parameter is needed.")
        return ""

    def render_source(self, *, class_name: str, name: str, description: str) -> str:
        return render_stub_source(
            class_name=class_name, name=name, description=description,
            params=_parse_params(self._params_edit.text()),
        )
