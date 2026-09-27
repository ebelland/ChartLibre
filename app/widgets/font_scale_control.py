"""A percentage spin box: grow or shrink every font of a figure or an axis at once.

The value is a factor, stored as the "font_scale" option and applied by
render_figure._apply_font_scale after the renderers have drawn - so it
scales whatever sizes the style and the kwargs chose, rather than
replacing them. A spin box like the Width/Height ones beside it, so the
panel reads as one set of controls.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSpinBox, QWidget

from app.utils.i18n import _

#: Allowed range, in percent; 100 is "as the style says".
FONT_SCALE_MIN_PERCENT: int = 50
FONT_SCALE_MAX_PERCENT: int = 300
FONT_SCALE_STEP_PERCENT: int = 10


class FontScaleControl(QSpinBox):
    """The font scale as a percentage; factor() and value_changed speak factors."""

    value_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(FONT_SCALE_MIN_PERCENT, FONT_SCALE_MAX_PERCENT)
        self.setSingleStep(FONT_SCALE_STEP_PERCENT)
        self.setSuffix(" %")
        self.setValue(100)
        self.setToolTip(_("Make every font larger or smaller at once; 100 % is the style's own size."))
        self.valueChanged.connect(lambda percent: self.value_changed.emit(percent / 100.0))

    def factor(self) -> float:
        """The factor, 1.0 for 100 %."""
        return self.value() / 100.0

    def set_factor(self, value: float) -> None:
        """Show factor *value* without announcing it (used when a panel reloads)."""
        try:
            percent = int(round(float(value) * 100))
        except (TypeError, ValueError):
            percent = 100
        self.blockSignals(True)
        try:
            self.setValue(percent)
        finally:
            self.blockSignals(False)
