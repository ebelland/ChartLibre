"""A stored function, editable: its formula, its parameters with their values, and its range - one plain form.

Function, New Chart and Chart properties > Series each edited some of these their own way; this is the one way
(dev/plans/function_editor_plan.txt):

    Kind          Expression or Code
    Description   a line or two
    Parameters    each with its value: a=1, b=2 - an expression's parameters are the names written here, a
                  Code's are its def's arguments (each needs its value here)
    Expression    the text: a few lines, growing as it is written - checked as text, never run to be checked
                  (todo ID47)
    x range       from / to / points / spacing; y range the same, shown only when the formula uses y

The values and the range are a chart's (its series' query keeps them: sympy_store.source_sql); the formula is
the project's, shared by every chart of the function. New Chart shows the values and the range only.
"""
from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.data.repo.function_sql import LINEAR, LOG, GridRange
from app.styles.style import stdSizeAndlayout
from app.utils.i18n import _

EXPRESSION, CODE = "expression", "code"
#: The coordinates: never parameters.
COORDINATES = ("x", "y", "z")
#: How far a value or a range may go: wide, and still a number a spin box can show.
_LIMIT = 1e12
#: The formula's lines: shown at first, and at most before it scrolls.
_TEXT_LINES = (3, 12)


class _NumberBox(QDoubleSpinBox):
    """A number as it is written - 2, 0.25, 1e-06 - not 2.000000."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt's name
        return self.locale().toString(float(value), "g", 10)


def number_box(parent: QWidget, value: float) -> QDoubleSpinBox:
    box = _NumberBox(parent)
    box.setRange(-_LIMIT, _LIMIT)
    box.setDecimals(10)
    box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    box.setValue(float(value))
    # Qt sizes a spin box for its longest possible number - twelve digits and ten decimals here, a third of
    # the panel: as wide as a usual number instead, and wider when there is room; typed, not stepped.
    box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    box.setMinimumWidth(box.fontMetrics().horizontalAdvance("-00000.00") + 12)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return box


class _GrowingText(QPlainTextEdit):
    """A few lines at first, taller as the text grows - up to a limit, then it scrolls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.textChanged.connect(self._fit)
        self._fit()

    def _fit(self) -> None:
        lines = max(1, int(self.document().documentLayout().documentSize().height()))
        shown = min(max(lines, _TEXT_LINES[0]), _TEXT_LINES[1])
        margins = self.contentsMargins()
        frame = int(self.document().documentMargin() * 2) + margins.top() + margins.bottom()
        self.setFixedHeight(shown * self.fontMetrics().lineSpacing() + frame + 2)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        self._fit()  # wrapped lines change with the width


class RangeRow(QWidget):
    """From, to - and under them how many points and how they are spaced: two short lines, not one wide one."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        stdSizeAndlayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        ends, sampling = QHBoxLayout(), QHBoxLayout()
        self.start = number_box(self, 0.0)
        self.stop = number_box(self, 10.0)
        self.points = QSpinBox(self)
        self.points.setRange(2, 100_000)
        self.points.setToolTip(_("Points"))
        self.spacing = QComboBox(self)
        self.spacing.addItem(_("Linear"), LINEAR)
        self.spacing.addItem(_("Log"), LOG)
        ends.addWidget(self.start, 1)
        ends.addWidget(QLabel(_("to"), self))
        ends.addWidget(self.stop, 1)
        sampling.addWidget(self.points)
        sampling.addWidget(QLabel(_("points"), self))
        sampling.addWidget(self.spacing)
        sampling.addStretch(1)
        layout.addLayout(ends)
        layout.addLayout(sampling)
        for box in (self.start, self.stop):
            box.valueChanged.connect(self.changed)
        self.points.valueChanged.connect(self.changed)
        self.spacing.currentIndexChanged.connect(self.changed)

    def set_grid(self, grid: GridRange) -> None:
        for widget in (self.start, self.stop, self.points, self.spacing):
            widget.blockSignals(True)
        try:
            self.start.setValue(grid.start)
            self.stop.setValue(grid.stop)
            self.points.setValue(int(grid.points))
            self.spacing.setCurrentIndex(max(0, self.spacing.findData(grid.spacing)))
        finally:
            for widget in (self.start, self.stop, self.points, self.spacing):
                widget.blockSignals(False)

    def grid(self) -> GridRange:
        return GridRange(self.start.value(), self.stop.value(), self.points.value(),
                         str(self.spacing.currentData() or LINEAR))


def _names_in(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _code_def(code: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(code, mode="exec")
    node = next((item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name), None)
    if node is None:
        raise ValueError(_("The code must define def {name}(...).").format(name=name))
    return node


def parse_assignments(text: str) -> tuple[dict[str, float], str]:
    """"a=1, b=2.5" as {"a": 1.0, "b": 2.5}, and what is wrong with it ("" when nothing)."""
    values: dict[str, float] = {}
    for part in (piece.strip() for piece in str(text or "").split(",")):
        if not part:
            continue
        name, equals, number = (piece.strip() for piece in part.partition("="))
        if not name.isidentifier():
            return values, _("'{part}' is not a parameter's name.").format(part=part)
        if name in COORDINATES:
            return values, _("x, y and z are the coordinates, never parameters.")
        if not equals or not number:
            return values, _("{name} needs its value: {name}=1.").format(name=name)
        if name in values:
            return values, _("{name} is given twice.").format(name=name)
        try:
            value = float(number)
        except ValueError:
            return values, _("{name}: '{number}' is not a number.").format(name=name, number=number)
        if not math.isfinite(value):
            return values, _("{name}: '{number}' is not a number.").format(name=name, number=number)
        values[name] = value
    return values, ""


def format_assignments(values: Mapping[str, float]) -> str:
    return ", ".join(f"{name}={float(value):g}" for name, value in values.items())


class FunctionEditor(QFrame):
    """A stored function's formula, its parameters with their values and its range, in one form."""

    #: Anything edited.
    changed = Signal()

    def __init__(self, parent: QWidget | None = None, *, formula: bool = True) -> None:
        super().__init__(parent)
        self._name = ""
        self._stored: dict[str, Any] = {}
        form = QFormLayout(self)
        stdSizeAndlayout(form)
        form.setContentsMargins(0, 0, 0, 0)
        self.kind = QComboBox(self)
        self.kind.addItem(_("Expression"), EXPRESSION)
        self.kind.addItem(_("Code"), CODE)
        self.description = QPlainTextEdit(self)
        self.description.setPlaceholderText(_("Optional description"))
        self.description.setFixedHeight(2 * self.description.fontMetrics().lineSpacing() + 14)
        self.symbols = QLineEdit(self)
        self.symbols.setPlaceholderText("a=1, b=2")
        self._symbols_note = QLabel(self)
        self._symbols_note.setWordWrap(True)
        self._symbols_note.setProperty("muted", True)
        self.text = _GrowingText(self)
        self._text_label = QLabel(_("Expression:"), self)
        self.x = RangeRow(self)
        self.y = RangeRow(self)
        form.addRow(_("Kind:"), self.kind)
        form.addRow(_("Description:"), self.description)
        form.addRow(_("Parameters:"), self.symbols)
        form.addRow("", self._symbols_note)
        form.addRow(self._text_label, self.text)
        form.addRow(_("x range:"), self.x)
        form.addRow(_("y range:"), self.y)
        self._form = form

        self.kind.currentIndexChanged.connect(self._formula_changed)
        self.symbols.textChanged.connect(self._formula_changed)
        self.text.textChanged.connect(self._formula_changed)
        self.description.textChanged.connect(self.changed)
        self.x.changed.connect(self.changed)
        self.y.changed.connect(self.changed)
        self._formula = True
        self.set_formula_visible(formula)
        self._formula_changed()

    # -- What is shown -----------------------------------------------------

    def set_formula_visible(self, visible: bool) -> None:
        """New Chart sets a chart's values and range, not the formula: Kind, Description and the text out."""
        self._formula = visible
        for widget in (self.kind, self.description, self.text):
            self._form.setRowVisible(widget, visible)

    def set_read_only(self, read_only: bool) -> None:
        """The formula shown but not editable - a function this computer does not trust (todo ID47)."""
        for widget in (self.description, self.text):
            widget.setReadOnly(read_only)
        self.symbols.setReadOnly(read_only)
        self.kind.setEnabled(not read_only)

    # -- Loading -----------------------------------------------------------

    def load(self, function: Mapping[str, Any], values: Mapping[str, float] | None = None,
             x: GridRange | None = None, y: GridRange | None = None) -> None:
        """*function* (a __sympy__ row) at *values* and over *x* / *y* - its stored ones where not given."""
        from app.data_operations.sympy_store import parameter_names, stored_values

        self._stored = dict(function)
        self._name = str(function.get("function_name", ""))
        widgets = (self.kind, self.symbols, self.text, self.description)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            kind = str(function.get("source_type", EXPRESSION)).lower()
            self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
            self.text.setPlainText(str(function.get("code_string", "")))
            self.description.setPlainText(str(function.get("description", "")))
            known = {**stored_values(function), **dict(values or {})}
            if kind == CODE:
                try:
                    names = [arg.arg for arg in _code_def(self.text.toPlainText(), self._name).args.args]
                except (SyntaxError, ValueError):
                    names = parameter_names(str(function.get("symbols", "")))
            else:
                names = parameter_names(str(function.get("symbols", "")))
            self.symbols.setText(format_assignments({name: known.get(name, 1.0) for name in names}))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.text._fit()
        self.x.set_grid(x or GridRange(float(function.get("from", 0.0)), float(function.get("to", 10.0)),
                                       int(float(function.get("points", 200)))))
        self.y.set_grid(y or GridRange(float(function.get("y_from", 0.0)), float(function.get("y_to", 10.0)),
                                       int(float(function.get("y_points", 200)))))
        self._formula_changed()

    # -- Reading -----------------------------------------------------------

    def name(self) -> str:
        return self._name

    def source_type(self) -> str:
        return str(self.kind.currentData() or EXPRESSION)

    def declared_parameters(self) -> list[str]:
        """The parameters: a Code's def arguments, an expression's names in Parameters; ValueError if wrong."""
        if self.source_type() == CODE:
            node = _code_def(self.text.toPlainText(), self._name)
            names = [arg.arg for arg in node.args.args]
            if set(COORDINATES).intersection(names):
                raise ValueError(_("x, y and z are the coordinates, never parameters."))
            return names
        values, problem = parse_assignments(self.symbols.text())
        if problem:
            raise ValueError(problem)
        return list(values)

    def ndim(self) -> int:
        """2 for z = f(x, y), 1 for y = f(x) - read from the text, never run."""
        try:
            if self.source_type() == CODE:
                return 2 if "y" in _names_in(_code_def(self.text.toPlainText(), self._name)) else 1
            tree = ast.parse(self.text.toPlainText().replace("^", "**") or "0", mode="eval")
        except (SyntaxError, ValueError):
            return 1
        return 2 if "y" in _names_in(tree) else 1

    def problem(self) -> str:
        """What is wrong - the parameters, or the formula as text - or "": nothing run to find out."""
        values, problem = parse_assignments(self.symbols.text())
        if problem:
            return problem
        if not self._formula:
            missing = [name for name in self._names_known() if name not in values]
            return _("{names}: needs a value.").format(names=", ".join(missing)) if missing else ""
        text = self.text.toPlainText().strip()
        if not text:
            return _("The formula is empty.")
        try:
            if self.source_type() == CODE:
                missing = [name for name in self.declared_parameters() if name not in values]
                if missing:
                    return _("{names}: the def's parameters need their values in Parameters ({first}=1).").format(
                        names=", ".join(missing), first=missing[0])
                return ""
            import sympy  # its names - sin, exp, pi - are the formula's own, not parameters

            used = _names_in(ast.parse(text.replace("^", "**"), mode="eval"))
            unknown = sorted(name for name in used - set(values) - set(COORDINATES) if not hasattr(sympy, name))
            if unknown:
                return _("{names}: not among the parameters - add it with its value ({first}=1).").format(
                    names=", ".join(unknown), first=unknown[0])
        except (SyntaxError, ValueError) as exc:
            return str(exc)
        return ""

    def parameters(self) -> dict[str, float]:
        """Each parameter declared and its value; one with no value written takes its stored one, else 1."""
        values, _problem = parse_assignments(self.symbols.text())
        try:
            names = self.declared_parameters() if self._formula else self._names_known()
        except (SyntaxError, ValueError):
            names = list(values)
        stored = self._stored_values()
        return {name: values.get(name, stored.get(name, 1.0)) for name in names}

    def x_range(self) -> GridRange:
        return self.x.grid()

    def y_range(self) -> GridRange | None:
        return self.y.grid() if self.ndim() == 2 else None

    def function_data(self) -> dict[str, Any]:
        """The function as sympy_store.save takes it: the formula, its values and its range as defaults."""
        x, y = self.x.grid(), self.y.grid()
        values = self.parameters()
        return {
            "function_name": self._name, "source_type": self.source_type(),
            "symbols": ",".join(values) if self.source_type() == EXPRESSION else "",
            "code_string": self.text.toPlainText(), "description": self.description.toPlainText(),
            "from": str(x.start), "to": str(x.stop), "points": str(x.points),
            "y_from": str(y.start), "y_to": str(y.stop), "y_points": str(y.points),
            "values": values,
        }

    # -- Keeping the parts in step ----------------------------------------

    def _names_known(self) -> list[str]:
        """The stored function's parameters - what New Chart, which does not edit the formula, may set."""
        from app.data_operations.sympy_store import payload

        try:
            return list(payload(self._stored)["params"]) if self._stored.get("code_string") else []
        except (SyntaxError, ValueError, KeyError):
            return []

    def _stored_values(self) -> dict[str, float]:
        from app.data_operations.sympy_store import stored_values

        return stored_values(self._stored)

    def _formula_changed(self, *_args: Any) -> None:
        code = self.source_type() == CODE
        self._text_label.setText(_("Code:") if code else _("Expression:"))
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont if code else
                                                   QFontDatabase.SystemFont.GeneralFont))
        self.text.setPlaceholderText(f"def {self._name or 'f'}(a, b):\n    f_numeric = a*x**2 + b\n"
                                     "    return f_numeric" if code else "a*x**2 + b")
        problem = self.problem()
        explanation = (_("The def's parameters, each with its value: a=1, b=2.") if code else
                       _("Each parameter with its value: a=1, b=2. x, y and z are the coordinates."))
        self._symbols_note.setText(problem or explanation)
        if bool(self._symbols_note.property("problem")) != bool(problem):
            self._symbols_note.setProperty("problem", bool(problem))
            self._symbols_note.style().unpolish(self._symbols_note)
            self._symbols_note.style().polish(self._symbols_note)
        self._form.setRowVisible(self.y, self.ndim() == 2)
        self.changed.emit()
