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

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.dev_tools_common import (
    class_name as _class_name,
    discover_both_roots,
    import_check as _generic_import_check,
    open_in_editor,
    slug as _slug,
)
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils.config import USER_FUNCTIONS_DIR
from app.utils.i18n import _
from app.utils.messages import show_message

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


class FunctionCreatorDialog(QDialog):
    """Scaffold a new base_function file from a short form."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Function Creator"))
        self.setWindowIcon(load_icon("function_creator"))
        self.setModal(True)

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        card = CardFrame(self, "functionCreatorCard")
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_("Function Creator"), card))

        hint = QLabel(
            _(
                "Writes a new file under user/functions/ with a class already "
                "filled in from these fields - no registration step, the "
                "scanner picks it up as soon as the file exists. execute(x, p) "
                "already evaluates a real polynomial in the declared "
                "parameters; replace it with the actual formula."
            ),
            card,
        )
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._name_edit = QLineEdit(card)
        self._name_edit.setPlaceholderText(_("e.g. Stretched Exponential"))
        self._name_edit.textEdited.connect(self._on_name_edited)
        form.addRow(_("Name:"), self._name_edit)

        self._file_edit = QLineEdit(card)
        self._file_edit.setPlaceholderText(_("e.g. stretched_exponential.py"))
        self._file_edit.textEdited.connect(self._on_file_edited_by_user)
        form.addRow(_("File name:"), self._file_edit)

        self._description_edit = QLineEdit(card)
        self._description_edit.setPlaceholderText(
            _("One line, shown in the fit dialog's model list.")
        )
        form.addRow(_("Description:"), self._description_edit)

        self._params_edit = QLineEdit(card)
        self._params_edit.setText("intercept, slope")
        self._params_edit.setPlaceholderText(_("comma-separated, e.g. intercept, slope"))
        form.addRow(_("Parameters:"), self._params_edit)

        card_layout.addLayout(form)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        create_action_button(
            parent=self,
            action_id="create_function",
            action=self._on_create,
            layout=action_row,
        )
        card_layout.addLayout(action_row)

        root.addWidget(card, 1)

        self._file_edited_by_user = False

    # ------------------------------------------------------------------
    # Auto-deriving the file name from the function's Name
    # ------------------------------------------------------------------
    def _on_name_edited(self, text: str) -> None:
        if self._file_edited_by_user:
            return
        self._file_edit.setText(f"{_slug(text)}.py" if text.strip() else "")

    def _on_file_edited_by_user(self, _text: str) -> None:
        self._file_edited_by_user = True

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def _on_create(self) -> None:
        problem = self._validate()
        if problem:
            show_message(self, "dev.validation_error", detail=problem)
            return

        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        class_name = _class_name(name, suffix="Function", fallback="Custom")
        description = self._description_edit.text().strip()
        params = _parse_params(self._params_edit.text())

        source = render_stub_source(
            class_name=class_name, name=name, description=description, params=params
        )

        target = FUNCTIONS_DIR / file_name
        try:
            FUNCTIONS_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        except OSError as exc:
            applogger.exception("Could not write the new function file.")
            show_message(self, "dev.validation_error", detail=str(exc))
            return

        error = self._import_check(target, class_name)
        if error:
            show_message(
                self,
                "dev.function_create_failed",
                path=str(target),
                error=error,
            )
            return

        applogger.info(
            "Function Creator: wrote %s (%s, name=%r).", target, class_name, name
        )
        show_message(self, "dev.function_created", name=name, path=str(target))
        open_in_editor(target)
        self.accept()

    def _validate(self) -> str:
        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        description = self._description_edit.text().strip()
        params = _parse_params(self._params_edit.text())

        if not name:
            return _("Name cannot be empty.")
        if not description:
            return _("Description cannot be empty.")
        if not file_name.endswith(".py") or not _slug(file_name[:-3]):
            return _("File name must be a valid Python file name, e.g. my_function.py.")
        if (FUNCTIONS_DIR / file_name).exists():
            return _(
                "A file named \"{file}\" already exists under user/functions/."
            ).format(file=file_name)
        if not params:
            return _("At least one parameter is needed.")
        if self._function_name_taken(name):
            return _(
                "\"{name}\" is already used by an existing function."
            ).format(name=name)
        return ""

    @staticmethod
    def _function_name_taken(name: str) -> bool:
        entries = discover_both_roots(
            builtin_root=BUILTIN_FUNCTIONS_DIR,
            user_root=FUNCTIONS_DIR,
            base_class_name="base_function",
            value_attr="name",
            require_value_attr=False,
        )
        return any(entry["value"] == name for entry in entries)

    @staticmethod
    def _import_check(path: Path, class_name: str) -> str:
        """Import *path* fresh and confirm it discovers as a function."""
        error = _generic_import_check(path, class_name)
        if error:
            return error

        entries = discover_both_roots(
            builtin_root=BUILTIN_FUNCTIONS_DIR,
            user_root=FUNCTIONS_DIR,
            base_class_name="base_function",
            value_attr="name",
            require_value_attr=False,
        )
        if not any(entry["name"] == class_name for entry in entries):
            return _(
                "The class imported, but functions_scanner did not discover it as "
                "a function - check that it subclasses base_function directly."
            )
        return ""
