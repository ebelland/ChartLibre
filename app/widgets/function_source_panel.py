"""A stored function as New Chart's data source: its parameters and its range, as this chart draws it.

The function's stored values and range are where it starts (Q1 of the function data source plan); what is
chosen here goes into the series' own query (sympy_store.source_sql), so each chart of one function keeps its
own parameters and range, and the formula is still read from the project at each draw.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from app.data.repo.function_sql import LINEAR, LOG, GridRange
from app.styles.style import stdSizeAndlayout
from app.utils.i18n import _

#: How far a value or a range may go: wide, and still a number a spin box can show.
_LIMIT = 1e12
#: Parameters side by side on a row, so a function of six of them is two rows, not six.
_PER_ROW = 3


class _NumberBox(QDoubleSpinBox):
    """A number as it is written - 2, 0.25, 1e-06 - not 2.000000."""

    def textFromValue(self, value: float) -> str:
        return self.locale().toString(float(value), "g", 10)


def _number_box(parent: QWidget, value: float) -> QDoubleSpinBox:
    box = _NumberBox(parent)
    box.setRange(-_LIMIT, _LIMIT)
    box.setDecimals(10)
    box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    box.setValue(float(value))
    return box


class _RangeRow(QWidget):
    """From, to, how many points and how they are spaced."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        stdSizeAndlayout(layout)
        self.start = _number_box(self, 0.0)
        self.stop = _number_box(self, 10.0)
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
        self.start.setValue(grid.start)
        self.stop.setValue(grid.stop)
        self.points.setValue(int(grid.points))
        self.spacing.setCurrentIndex(max(0, self.spacing.findData(grid.spacing)))

    def grid(self) -> GridRange:
        return GridRange(self.start.value(), self.stop.value(), self.points.value(),
                         str(self.spacing.currentData() or LINEAR))


class FunctionSourcePanel(QObject):
    """The parameters and range of the stored function chosen as a data source: rows of the dialog's form."""

    changed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._data: dict[str, Any] | None = None
        self._ndim = 1
        self._values: dict[str, QDoubleSpinBox] = {}
        self._form: QFormLayout | None = None
        self._shown = False
        self._params_box = QWidget(parent)
        self._params_grid = QGridLayout(self._params_box)
        stdSizeAndlayout(self._params_grid)
        self._x = _RangeRow(parent)
        self._x.setToolTip(_("Where x goes: from, to, how many points and how they are spaced."))
        self._y = _RangeRow(parent)
        self._y.setToolTip(_("Where y goes, for a function of x and y."))
        self._x.changed.connect(self.changed)
        self._y.changed.connect(self.changed)

    def insert_rows(self, form: QFormLayout, row: int) -> None:
        """Its rows in *form* from *row* on - the parameters, the x range, the y range - hidden until shown."""
        self._form = form
        form.insertRow(row, _("Parameters:"), self._params_box)
        form.insertRow(row + 1, _("x range:"), self._x)
        form.insertRow(row + 2, _("y range:"), self._y)
        self.set_shown(False)

    def set_shown(self, shown: bool) -> None:
        """Its rows shown - the parameters if it has some, the y range for f(x, y) - or all hidden."""
        self._shown = shown
        if self._form is None:
            return
        self._form.setRowVisible(self._params_box, shown and bool(self._values))
        self._form.setRowVisible(self._x, shown)
        self._form.setRowVisible(self._y, shown and self._ndim == 2)

    def is_row_shown(self, widget: QWidget) -> bool:
        return self._form is not None and self._form.isRowVisible(widget)

    def load(self, data: Mapping[str, Any]) -> None:
        """Show *data*'s parameters at their stored values, and its stored range."""
        from app.data_operations import sympy_store  # SymPy: loaded when a function is chosen

        info = sympy_store.payload(data)
        self._data = dict(data)
        self._ndim = int(info["ndim"])
        self.blockSignals(True)
        self._x.blockSignals(True)
        self._y.blockSignals(True)
        try:
            while self._params_grid.count():
                item = self._params_grid.takeAt(0)
                if item is not None and item.widget() is not None:
                    item.widget().deleteLater()
            self._values = {}
            for index, (name, value) in enumerate(zip(info["params"], info["p0"])):
                row, column = divmod(index, _PER_ROW)
                box = _number_box(self._params_box, float(value))
                box.valueChanged.connect(self.changed)
                self._params_grid.addWidget(QLabel(f"{name}:", self._params_box), row, 2 * column)
                self._params_grid.addWidget(box, row, 2 * column + 1)
                self._values[str(name)] = box
            self._x.set_grid(GridRange(float(data.get("from", 0.0)), float(data.get("to", 10.0)),
                                       int(float(data.get("points", 200)))))
            self._y.set_grid(GridRange(float(data.get("y_from", 0.0)), float(data.get("y_to", 10.0)),
                                       int(float(data.get("y_points", 200)))))
        finally:
            self.blockSignals(False)
            self._x.blockSignals(False)
            self._y.blockSignals(False)
        self.set_shown(self._shown)

    def parameters(self) -> dict[str, float]:
        return {name: box.value() for name, box in self._values.items()}

    def source_sql(self) -> str:
        """The query this chart reads: the function at the chosen parameters, over the chosen range."""
        from app.data_operations import sympy_store

        if self._data is None:
            return ""
        y = self._y.grid() if self._ndim == 2 else None
        return sympy_store.source_sql(self._data, self.parameters(), self._x.grid(), y)
