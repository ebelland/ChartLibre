"""The chart canvas: a window resize redraws the figure once it pauses (W-04).

FigureCanvasQT.resizeEvent resizes the figure and redraws it at every step
of a resize, so dragging a window corner rendered the whole chart tens of
times a second. Here the first resize of a burst goes through as before - a
programmatic resize (a tab shown, a layout settling) is drawn at once - and
the ones that follow it within BURST_MS only stretch the last image, until
the size has held still for SETTLE_MS: then the figure is resized and drawn
once, and ``resize_settled`` lets the panel run its own geometry pass.
"""
from __future__ import annotations

import time

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QRectF, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QWidget

BURST_MS = 120
SETTLE_MS = 150


class ChartCanvas(FigureCanvasQTAgg):
    resize_settled = Signal()

    def __init__(self, figure) -> None:
        super().__init__(figure)
        self._last_resize = 0.0
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(SETTLE_MS)
        self._settle_timer.timeout.connect(self._settle_resize)

    @property
    def resize_deferred(self) -> bool:
        """True while a burst of resizes waits for its final size."""
        return self._settle_timer.isActive()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        now = time.monotonic()
        in_burst = (now - self._last_resize) * 1000.0 < BURST_MS
        self._last_resize = now
        if not in_burst or not self.isVisible() or not hasattr(self, "renderer"):
            self._settle_timer.stop()
            super().resizeEvent(event)
            return
        # The widget takes its new size; the figure keeps the old one for now.
        QWidget.resizeEvent(self, event)
        self._settle_timer.start()
        self.update()

    def _settle_resize(self) -> None:
        super().resizeEvent(QResizeEvent(self.size(), self.size()))
        self.resize_settled.emit()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        if not self.resize_deferred:
            super().paintEvent(event)
            return
        # Stretch the last rendered image over the new size: no draw here,
        # not even a pending draw_idle - that would render at the old size.
        buffer = memoryview(self.renderer.buffer_rgba())
        height, width = buffer.shape[0], buffer.shape[1]
        image = QImage(buffer, width, height, QImage.Format.Format_RGBA8888)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(QRectF(self.rect()), image, QRectF(0, 0, width, height))
        finally:
            painter.end()
