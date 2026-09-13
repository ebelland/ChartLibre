# -*- coding: utf-8 -*-
"""app.widgets.line_combo

LineStyleCombo

A PySide6 QComboBox that lists Matplotlib line styles and shows a small
line-preview icon next to each entry, mirroring ``MatplotlibColorCombo``.
The dropdown popup auto-sizes to fit the longest label, independent of how
narrow the combo box itself is.

Notes for strict PySide6 typing / Pylance:
- Use Qt.PenStyle.SolidLine (not Qt.SolidLine)
- Use Qt.GlobalColor.transparent (not Qt.transparent)

This module is intentionally standalone and PEP8 compliant.
"""

from __future__ import annotations

from functools import lru_cache

from matplotlib import rcParams
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.charts.kwarg_spec import DEFAULT
from app.styles.style import create_hidpi_pixmap
from app.widgets.icon_combo import ComboEntry, IconComboBox

_ICON_W, _ICON_H = 28, 14
_LINE_COLOR = "#3a3a3a"
_LINE_WIDTH = 2

#: (label, Matplotlib linestyle code) — same entries as the original
#: hand-rolled combo, now paired with a preview icon. "Default" is first
#: since it is what a brand-new series should start on: it defers to
#: whatever the active style sheet's ``lines.linestyle`` says, rather than
#: this combo silently overriding it with a hard-coded "Solid" the moment a
#: series exists - see app.charts.base.BaseAxisRenderer.series_linestyle.
_ENTRIES: tuple[ComboEntry, ...] = (
    ("Default", DEFAULT),
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dash-dot", "-."),
    ("Dotted", ":"),
    ("None", "none"),
)

#: Matplotlib linestyle code -> Qt pen style. ``None`` means "draw nothing"
#: (used for the "None" / no-line entry). "Default" is absent deliberately:
#: it has no pen of its own - see _line_icon.
_PEN_STYLES: dict[str, Qt.PenStyle | None] = {
    "-": Qt.PenStyle.SolidLine,
    "--": Qt.PenStyle.DashLine,
    "-.": Qt.PenStyle.DashDotLine,
    ":": Qt.PenStyle.DotLine,
    "none": None,
}


def _line_icon(linestyle: str) -> QIcon:
    """Return a preview icon for one Matplotlib linestyle, "Default" included.

    "Default" has no style of its own to preview, so it is drawn as a faint
    version of whatever it actually resolves to right now - rcParams'
    ``lines.linestyle`` - rather than assumed to be solid: a custom
    .mplstyle is free to ship dashed, dotted or no line at all as its own
    default, and a combo claiming "Default" looks like a continuous line
    regardless would be showing the wrong preview for that style sheet.
    Not cached here: the *resolved* style is what gets cached, in
    _resolved_line_icon, so a style sheet switch changing what "Default"
    means invalidates only the entries it actually changed, with nothing to
    clear by hand.
    """
    is_default = linestyle == DEFAULT
    resolved = (
        str(rcParams.get("lines.linestyle", "-") or "-").strip().lower()
        if is_default
        else linestyle
    )
    return _resolved_line_icon(resolved, faint=is_default)


@lru_cache(maxsize=None)
def _resolved_line_icon(linestyle: str, *, faint: bool) -> QIcon:
    """Build (and cache) a preview icon for one concrete Matplotlib linestyle."""
    # Allocated at the display's pixel density: the coordinates below stay
    # logical, but the bitmap has the pixels to be sharp on a Retina screen.
    pixmap = create_hidpi_pixmap(_ICON_W, _ICON_H)

    # "none" is an explicit key here (draw nothing), so .get's fallback only
    # ever applies to a linestyle this combo does not know - an exotic
    # custom dash tuple a style sheet set as its default, say - where solid
    # is the least wrong guess.
    pen_style = _PEN_STYLES.get(linestyle, Qt.PenStyle.SolidLine)
    if pen_style is not None:
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor(_LINE_COLOR))
            pen.setWidth(_LINE_WIDTH)
            pen.setStyle(pen_style)
            if faint:
                # It used to be drawn dashed regardless, which made the
                # entry read as a second, washed-out "Dashed" - the
                # faintness has to be the only difference from the concrete
                # entry it resolved to.
                color = QColor(_LINE_COLOR)
                color.setAlpha(110)
                pen.setColor(color)
            painter.setPen(pen)
            mid_y = _ICON_H // 2
            painter.drawLine(2, mid_y, _ICON_W - 2, mid_y)
        finally:
            painter.end()

    return QIcon(pixmap)


class LineStyleCombo(IconComboBox):
    """Combo box listing Matplotlib line styles with a preview icon.

    The stored value is the Matplotlib linestyle code (``"-"``, ``"--"``,
    ``"-."``, ``":"`` or ``"none"``), matching what was previously hand-
    rolled inline.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            entries=_ENTRIES,
            icon_for=_line_icon,
            icon_size=QSize(_ICON_W, _ICON_H),
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed,)

    def current_linestyle(self) -> str:
        """Return the selected Matplotlib linestyle code."""
        return self.current_value()

    def set_current_linestyle(self, linestyle: str) -> bool:
        """Select the entry matching *linestyle* (e.g. ``"--"``)."""
        return self.set_current_value((linestyle or "").strip())
