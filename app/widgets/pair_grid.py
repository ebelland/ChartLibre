"""Label and field pairs that reflow to the width they are given.

A property panel can be as narrow as the splitter allows - a little over
200 pixels of content - and a spin box with its stepper needs about a
hundred of them for a number like "16.26 cm". Two pairs side by side
therefore cannot always fit, and a field squeezed to fit clips its own
number, which is worse than a taller panel. So the pairs are laid out in
whichever of three arrangements fits the width, widest first:

* two pairs per row, each label beside its field;
* one pair per row, the label beside its field;
* one pair per row, the label above its field - needs only the wider of the
  two, so it fits wherever the panel can be at all.

Nothing here knows what the fields are: it takes (label, field) widgets and
their minimum widths, which callers set with :func:`app.styles.style.fit_spin_width`.
"""
from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QWidget

#: Space between a label and its field, and between two pairs.
_GAP = 8

#: The arrangements, widest first: (pairs per row, label above its field).
_ARRANGEMENTS: tuple[tuple[int, bool], ...] = ((2, False), (1, False), (1, True))


class PairGrid(QWidget):
    """(label, field) pairs, two to a row when they fit, else one."""

    def __init__(
        self,
        pairs: Sequence[tuple[QWidget, QWidget]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pairs = list(pairs)
        self._arrangement: tuple[int, bool] | None = None
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(_GAP)
        self._grid.setVerticalSpacing(_GAP)
        for label, field in self._pairs:
            label.setParent(self)
            field.setParent(self)
            field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._arrange((1, True))

    def _needed_width(self, arrangement: tuple[int, bool]) -> int:
        """Width the *arrangement* needs so no field is squeezed below its minimum."""
        per_row, stacked = arrangement
        label_width = max(label.sizeHint().width() for label, _field in self._pairs)
        field_width = max(max(field.minimumWidth(), field.minimumSizeHint().width()) for _label, field in self._pairs)
        pair = max(label_width, field_width) if stacked else label_width + _GAP + field_width
        return per_row * pair + (per_row - 1) * _GAP

    def _fitting_arrangement(self, width: int) -> tuple[int, bool]:
        for arrangement in _ARRANGEMENTS:
            per_row, _stacked = arrangement
            if per_row > len(self._pairs):
                continue
            if self._needed_width(arrangement) <= width:
                return arrangement
        return _ARRANGEMENTS[-1]

    def _arrange(self, arrangement: tuple[int, bool]) -> None:
        if arrangement == self._arrangement:
            return
        self._arrangement = arrangement
        per_row, stacked = arrangement
        while self._grid.count():
            self._grid.takeAt(0)
        for column in range(2 * per_row + 1):
            self._grid.setColumnStretch(column, 0)

        row = 0
        for index, (label, field) in enumerate(self._pairs):
            column = index % per_row
            if index and column == 0:
                row += 2 if stacked else 1
            if stacked:
                self._grid.addWidget(label, row, column, Qt.AlignmentFlag.AlignLeft)
                self._grid.addWidget(field, row + 1, column)
                self._grid.setColumnStretch(column, 1)
            else:
                self._grid.addWidget(label, row, 2 * column, Qt.AlignmentFlag.AlignVCenter)
                self._grid.addWidget(field, row, 2 * column + 1)
                self._grid.setColumnStretch(2 * column + 1, 1)
        self.updateGeometry()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._arrange(self._fitting_arrangement(self.width()))
