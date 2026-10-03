"""Matplotlib's Qt cursors, made safe on macOS.

On macOS Qt has native cursors for some shapes and draws the others - the
wait cursor, the four-way move arrow, the diagonal resize arrows - from an
image. Since macOS 27 turning such an image into a cursor can crash the
whole application inside CoreGraphics (CGImageCreate), below anything
Python can catch: the crash that once came from setOverrideCursor and is
why that call is gone from app/.

Matplotlib asks for two of those shapes by itself: the wait cursor around
any draw that comes after a second's pause (FigureCanvasAgg.draw), and the
move arrow in pan mode. Here they map to native cursors instead - the plain
arrow, and the open hand macOS uses for dragging a view - before any canvas
exists. Elsewhere nothing changes.
"""
from __future__ import annotations

import sys

from matplotlib.backend_tools import Cursors
from matplotlib.backends import backend_qt
from PySide6.QtCore import Qt

#: Shapes macOS's Qt builds from an image rather than taking from the system.
IMAGE_CURSORS: frozenset[Qt.CursorShape] = frozenset({
    Qt.CursorShape.WaitCursor,
    Qt.CursorShape.BusyCursor,
    Qt.CursorShape.SizeAllCursor,
    Qt.CursorShape.SizeBDiagCursor,
    Qt.CursorShape.SizeFDiagCursor,
})

#: What Matplotlib gets instead, on macOS.
NATIVE_REPLACEMENTS: dict[Cursors, Qt.CursorShape] = {
    Cursors.WAIT: Qt.CursorShape.ArrowCursor,
    Cursors.MOVE: Qt.CursorShape.OpenHandCursor,
}


def use_native_cursors(platform: str = sys.platform) -> None:
    """On macOS, point Matplotlib's Qt cursor table at native cursors only."""
    if platform != "darwin":
        return
    backend_qt.cursord.update(NATIVE_REPLACEMENTS)


use_native_cursors()
