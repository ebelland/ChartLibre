"""The strip above the chart: back/forward arrows and the current chart's name.

It replaces the chart tabs' own bar, the way a recent Mac app's toolbar
does (Xcode's jump bar, Finder's back/forward): the sidebar lists every
chart, and this strip says which one is showing, steps to the one above or
below, and pops up the whole list from the name. On macOS the empty part
of the strip moves the window, as a native toolbar does.
"""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMenu, QSizePolicy, QToolButton, QWidget

from app.styles.style import IS_MACOS, action_presentation

#: Tall enough to sit level with the macOS traffic lights' strip.
JUMP_BAR_HEIGHT: int = 32


class ChartJumpBar(QFrame):
    """‹ › and a menu button naming the current chart."""

    previous_requested = Signal()
    next_requested = Signal()
    chart_chosen = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chartJumpBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(JUMP_BAR_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(2)

        self._previous = self._arrow("chart_previous", self.previous_requested.emit)
        self._next = self._arrow("chart_next", self.next_requested.emit)
        layout.addWidget(self._previous)
        layout.addWidget(self._next)
        layout.addSpacing(6)

        self._title = QToolButton(self)
        self._title.setObjectName("chartJumpTitle")
        self._title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._title.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._title.setAutoRaise(True)
        self._title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._title.setToolTip(action_presentation("chart_jump")[2])
        self._menu = QMenu(self._title)
        self._title.setMenu(self._menu)
        layout.addWidget(self._title, 0)
        layout.addStretch(1)

        self.set_charts([], -1)

    def _arrow(self, action_id: str, callback) -> QToolButton:
        icon, text, tooltip = action_presentation(action_id)
        button = QToolButton(self)
        button.setObjectName("chartJumpArrow")
        button.setIcon(icon)
        button.setIconSize(QSize(16, 16))
        button.setAutoRaise(True)
        button.setAccessibleName(text)
        button.setToolTip(tooltip)
        button.setFixedSize(28, 24)
        button.clicked.connect(callback)
        return button

    def set_charts(self, names: Sequence[str], current: int) -> None:
        """Show *names* in the menu and *current* as the title."""
        self._menu.clear()
        for index, name in enumerate(names):
            action = self._menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(index == current)
            action.triggered.connect(lambda _checked=False, i=index: self.chart_chosen.emit(i))
        has_current = 0 <= current < len(names)
        self._title.setText(f"{names[current]}  " if has_current else "")
        self._title.setVisible(has_current)
        self._previous.setEnabled(has_current and current > 0)
        self._next.setEnabled(has_current and current < len(names) - 1)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if IS_MACOS and event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)
