"""A− / 100 % / A+: grow or shrink every font of a figure or an axis at once.

The value is a factor, stored as the "font_scale" option and applied by
render_figure._apply_font_scale after the renderers have drawn - so it
scales whatever sizes the style and the kwargs chose, rather than
replacing them.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from app.utils.i18n import _

#: The steps A− and A+ move through; 1.0 is "as the style says".
FONT_SCALE_STEPS: tuple[float, ...] = (
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0,
)


class FontScaleControl(QWidget):
    """Two buttons and the current percentage between them."""

    value_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 1.0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._smaller = QToolButton(self)
        self._smaller.setText("A−")
        self._smaller.setToolTip(_("Make every font smaller"))
        self._smaller.clicked.connect(lambda: self._step(-1))
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumWidth(44)
        self._label.setToolTip(_("Double-click to reset to 100 %"))
        self._larger = QToolButton(self)
        self._larger.setText("A+")
        self._larger.setToolTip(_("Make every font larger"))
        self._larger.clicked.connect(lambda: self._step(+1))

        layout.addWidget(self._smaller)
        layout.addWidget(self._label)
        layout.addWidget(self._larger)
        layout.addStretch(1)
        self._show()

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        """Show *value* without announcing it (used when a panel reloads)."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 1.0
        self._value = min(max(number, FONT_SCALE_STEPS[0]), FONT_SCALE_STEPS[-1])
        self._show()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._value != 1.0:
            self._value = 1.0
            self._show()
            self.value_changed.emit(self._value)
        super().mouseDoubleClickEvent(event)

    def _step(self, direction: int) -> None:
        if direction > 0:
            larger = [s for s in FONT_SCALE_STEPS if s > self._value + 1e-9]
            new = larger[0] if larger else self._value
        else:
            smaller = [s for s in FONT_SCALE_STEPS if s < self._value - 1e-9]
            new = smaller[-1] if smaller else self._value
        if new != self._value:
            self._value = new
            self._show()
            self.value_changed.emit(self._value)

    def _show(self) -> None:
        self._label.setText(f"{round(self._value * 100):d} %")
        self._smaller.setEnabled(self._value > FONT_SCALE_STEPS[0] + 1e-9)
        self._larger.setEnabled(self._value < FONT_SCALE_STEPS[-1] - 1e-9)
