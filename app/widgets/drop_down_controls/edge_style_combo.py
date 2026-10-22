# -*- coding: utf-8 -*-
"""Visual selectors for Matplotlib solid join and cap styles."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from app.styles.style import create_hidpi_pixmap
from app.widgets.drop_down_controls.icon_combo import IconComboBox

_ICON = QSize(42, 18)
_JOIN_ENTRIES = (("Miter", "miter"), ("Round", "round"), ("Bevel", "bevel"))
_CAP_ENTRIES = (("Butt", "butt"), ("Round", "round"), ("Projecting", "projecting"))
_JOIN_MAP = {"miter": Qt.PenJoinStyle.MiterJoin, "round": Qt.PenJoinStyle.RoundJoin,
             "bevel": Qt.PenJoinStyle.BevelJoin}
_CAP_MAP = {"butt": Qt.PenCapStyle.FlatCap, "round": Qt.PenCapStyle.RoundCap,
            "projecting": Qt.PenCapStyle.SquareCap}


@lru_cache(maxsize=16)
def join_style_preview_icon(style: str) -> QIcon:
    pixmap = create_hidpi_pixmap(_ICON.width(), _ICON.height())
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#3a3a3a"), 6)
        pen.setJoinStyle(_JOIN_MAP.get(style, Qt.PenJoinStyle.MiterJoin))
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF([QPointF(5, 14), QPointF(21, 4), QPointF(37, 14)]))
    finally:
        painter.end()
    return QIcon(pixmap)


@lru_cache(maxsize=16)
def cap_style_preview_icon(style: str) -> QIcon:
    pixmap = create_hidpi_pixmap(_ICON.width(), _ICON.height())
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        guide = QPen(QColor("#aaaaaa"), 1, Qt.PenStyle.DotLine)
        painter.setPen(guide)
        painter.drawLine(8, 2, 8, 16)
        painter.drawLine(34, 2, 34, 16)
        pen = QPen(QColor("#3a3a3a"), 6)
        pen.setCapStyle(_CAP_MAP.get(style, Qt.PenCapStyle.FlatCap))
        painter.setPen(pen)
        painter.drawLine(8, 9, 34, 9)
    finally:
        painter.end()
    return QIcon(pixmap)


class JoinStyleCombo(IconComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, entries=_JOIN_ENTRIES, icon_for=join_style_preview_icon, icon_size=_ICON)


class CapStyleCombo(IconComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, entries=_CAP_ENTRIES, icon_for=cap_style_preview_icon, icon_size=_ICON)
