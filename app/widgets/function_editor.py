"""A stored function, editable: its formula, its parameters' values and its range - one widget, three tabs.

Function, New Chart and Chart properties > Series each edited some of these their own way; this is the one way
(dev/plans/function_editor_plan.txt):

    Formula      Expression or Code: its description, its parameters (an expression's) and its text - checked
                 as text, never run to be checked (todo ID47).
    Parameters   one row per declared parameter and its value; the rows are made from the declaration each
                 time, so they cannot drift apart from it (a row missing was a call short of an argument).
    Range        x from / to / points / spacing; y the same, shown only when the formula uses y.

The values and the range are a chart's (its series' query keeps them: sympy_store.source_sql); the formula is
the project's, shared by every chart of the function.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
    return box


class RangeRow(QWidget):
    """From, to, how many points and how they are spaced."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        stdSizeAndlayout(layout)
        self.start = number_box(self, 0.0)
        self.stop = number_box(self, 10.0)
        self.points = QSpinBox(self)
        self.points.setRange(2, 100_000)
        self.points.setToolTip(_("Points"))
        self.spacing = QComboBox(self)
        self.spacing.addItem(_("Linear"), LINEAR)
        self.spacing.addItem(_("Log"), LOG)
        layout.addWidget(self.start, 1)
        layout.addWidget(QLabel(_("to"), self))
        layout.addWidget(self.stop, 1)
        layout.addWidget(self.points)
        layout.addWidget(self.spacing)
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


class FunctionEditor(QFrame):
    """A stored function's formula, its parameters' values and its range, in three tabs."""

    #: Anything edited.
    changed = Signal()

    def __init__(self, parent: QWidget | None = None, *, formula: bool = True) -> None:
        super().__init__(parent)
        self._name = ""
        self._stored: dict[str, Any] = {}
        self.tabs = QTabWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        # Formula
        self.formula_page = QWidget(self.tabs)
        form = QFormLayout(self.formula_page)
        stdSizeAndlayout(form)
        self.kind = QComboBox(self.formula_page)
        self.kind.addItem(_("Expression"), EXPRESSION)
        self.kind.addItem(_("Code"), CODE)
        self.description = QPlainTextEdit(self.formula_page)
        self.description.setPlaceholderText(_("Optional description"))
        self.description.setFixedHeight(56)
        self.symbols = QLineEdit(self.formula_page)
        self.symbols.setPlaceholderText("a, b")
        self.text = QPlainTextEdit(self.formula_page)
        self.text.setMinimumHeight(90)
        form.addRow(_("Kind:"), self.kind)
        form.addRow(_("Description:"), self.description)
        form.addRow(_("Parameters:"), self.symbols)
        self._text_label = QLabel(_("Expression:"), self.formula_page)
        form.addRow(self._text_label, self.text)
        note = QLabel(_("x, y and z are the coordinates, never parameters."), self.formula_page)
        note.setWordWrap(True)
        note.setProperty("muted", True)
        form.addRow(note)
        self._form = form
        self.tabs.addTab(self.formula_page, _("Formula"))

        # Parameters
        self.values = QTableWidget(0, 2, self.tabs)
        self.values.setHorizontalHeaderLabels([_("Parameter"), _("Value")])
        self.values.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.values.verticalHeader().setVisible(False)
        self.tabs.addTab(self.values, _("Parameters"))

        # Range
        range_page = QWidget(self.tabs)
        ranges = QFormLayout(range_page)
        stdSizeAndlayout(ranges)
        self.x = RangeRow(range_page)
        self.y = RangeRow(range_page)
        ranges.addRow(_("x:"), self.x)
        ranges.addRow(_("y:"), self.y)
        self._ranges = ranges
        self.tabs.addTab(range_page, _("Range"))

        self.kind.currentIndexChanged.connect(self._formula_changed)
        self.symbols.textChanged.connect(self._formula_changed)
        self.text.textChanged.connect(self._formula_changed)
        self.description.textChanged.connect(self.changed)
        self.values.itemChanged.connect(lambda _item: self.changed.emit())
        self.x.changed.connect(self.changed)
        self.y.changed.connect(self.changed)
        self.set_formula_visible(formula)
        self._formula_changed()

    # -- What is shown -----------------------------------------------------

    def set_formula_visible(self, visible: bool) -> None:
        """New Chart edits a chart's values and range, not the formula: its tab out."""
        self.tabs.setTabVisible(self.tabs.indexOf(self.formula_page), visible)

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
        from app.data_operations.sympy_store import stored_values

        self._stored = dict(function)
        self._name = str(function.get("function_name", ""))
        widgets = (self.kind, self.symbols, self.text, self.description, self.values)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            kind = str(function.get("source_type", EXPRESSION)).lower()
            self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
            self.symbols.setText(", ".join(name for name in str(function.get("symbols", "")).replace(" ", "")
                                           .split(",") if name and name not in COORDINATES))
            self.text.setPlainText(str(function.get("code_string", "")))
            self.description.setPlainText(str(function.get("description", "")))
            self._fill_values({**stored_values(function), **dict(values or {})})
        finally:
            for widget in widgets:
                widget.blockSignals(False)
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
        """The parameters the formula declares: an expression's list, the def's arguments; ValueError if wrong."""
        if self.source_type() == CODE:
            node = _code_def(self.text.toPlainText(), self._name)
            names = [arg.arg for arg in node.args.args]
        else:
            names = [name.strip() for name in self.symbols.text().split(",") if name.strip()]
        if len(set(names)) != len(names) or any(not name.isidentifier() for name in names):
            raise ValueError(_("The parameters must be different names, separated by commas."))
        if set(COORDINATES).intersection(names):
            raise ValueError(_("x, y and z are the coordinates, never parameters."))
        return names

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
        """What is wrong with the formula as text, or "" - nothing run to find out."""
        text = self.text.toPlainText().strip()
        if not text:
            return _("The formula is empty.")
        try:
            names = self.declared_parameters()
            if self.source_type() == EXPRESSION:
                import sympy  # its names - sin, exp, pi - are the formula's own, not parameters

                used = _names_in(ast.parse(text.replace("^", "**"), mode="eval"))
                unknown = sorted(name for name in used - set(names) - set(COORDINATES) if not hasattr(sympy, name))
                if unknown:
                    return _("{names}: not among the parameters.").format(names=", ".join(unknown))
        except (SyntaxError, ValueError) as exc:
            return str(exc)
        return ""

    def parameters(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in range(self.values.rowCount()):
            name, cell = self.values.item(row, 0), self.values.item(row, 1)
            try:
                values[name.text()] = float(cell.text()) if cell is not None else 1.0
            except ValueError:
                values[name.text()] = 1.0
        return values

    def x_range(self) -> GridRange:
        return self.x.grid()

    def y_range(self) -> GridRange | None:
        return self.y.grid() if self.ndim() == 2 else None

    def function_data(self) -> dict[str, Any]:
        """The function as sympy_store.save takes it: the formula, its values and its range as defaults."""
        x, y = self.x.grid(), self.y.grid()
        return {
            "function_name": self._name, "source_type": self.source_type(),
            "symbols": ",".join(self.declared_parameters()) if self.source_type() == EXPRESSION else "",
            "code_string": self.text.toPlainText(), "description": self.description.toPlainText(),
            "from": str(x.start), "to": str(x.stop), "points": str(x.points),
            "y_from": str(y.start), "y_to": str(y.stop), "y_points": str(y.points),
            "values": self.parameters(),
        }

    # -- Keeping the parts in step ----------------------------------------

    def _formula_changed(self, *_args: Any) -> None:
        code = self.source_type() == CODE
        self._form.setRowVisible(self.symbols, not code)
        self._text_label.setText(_("Code:") if code else _("Expression:"))
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont if code else
                                        QFontDatabase.SystemFont.GeneralFont)
        self.text.setFont(font)
        if code and not self.text.toPlainText().strip():
            self.text.setPlaceholderText(f"def {self._name or 'f'}(a, b):\n    f_numeric = a*x**2 + b\n"
                                         "    return f_numeric")
        elif not code:
            self.text.setPlaceholderText("a*x**2 + b")
        try:
            declared = self.declared_parameters()
        except (SyntaxError, ValueError):
            declared = None  # half typed: the rows as they were
        if declared is not None and declared != list(self.parameters()):
            self._fill_values({**self._stored_values(), **self.parameters()}, declared)
        self._ranges.setRowVisible(self.y, self.ndim() == 2)
        self.changed.emit()

    def _stored_values(self) -> dict[str, float]:
        from app.data_operations.sympy_store import stored_values

        return stored_values(self._stored)

    def _fill_values(self, known: Mapping[str, float], names: list[str] | None = None) -> None:
        """A row per parameter declared - *names*, or the declaration - at its known value, else 1."""
        if names is None:
            try:
                names = self.declared_parameters()
            except (SyntaxError, ValueError):
                names = []
        self.values.blockSignals(True)
        try:
            self.values.setRowCount(len(names))
            for row, name in enumerate(names):
                label = QTableWidgetItem(name)
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.values.setItem(row, 0, label)
                self.values.setItem(row, 1, QTableWidgetItem(f"{float(known.get(name, 1.0)):g}"))
        finally:
            self.values.blockSignals(False)
