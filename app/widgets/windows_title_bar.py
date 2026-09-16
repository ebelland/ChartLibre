"""Custom Windows title bar for frameless windows."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QStyle, QToolButton

from app.utils.i18n import _

if TYPE_CHECKING:
    from app.dialogs.main_window import MainWindow

CUSTOM_TITLE_BAR_HEIGHT: int = 40


class WindowsTitleBar(QFrame):
    """Custom Windows chrome that delegates movement to the window system."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self.setObjectName("windowsTitleBar")
        self.setFixedHeight(CUSTOM_TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(4)
        icon_label = QLabel(self)
        icon_label.setPixmap(window.windowIcon().pixmap(18, 18))
        icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(icon_label)
        self.title_label = QLabel(window.windowTitle(), self)
        self.title_label.setObjectName("titleBarTitle")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.title_label, 1)
        self.minimize_button = self._window_button(QStyle.StandardPixmap.SP_TitleBarMinButton, window.showMinimized, _("Minimize"))
        self.maximize_button = self._window_button(QStyle.StandardPixmap.SP_TitleBarMaxButton, self._toggle_maximized, _("Maximize"))
        self.close_button = self._window_button(QStyle.StandardPixmap.SP_TitleBarCloseButton, window.close, _("Close"), "titleBarCloseButton")
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)
        window.windowTitleChanged.connect(self.title_label.setText)

    def _window_button(self, icon: QStyle.StandardPixmap, callback: Any, tooltip: str, object_name: str = "titleBarButton") -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setAutoRaise(True)
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setFixedSize(46, CUSTOM_TITLE_BAR_HEIGHT)
        button.clicked.connect(callback)
        return button

    def _toggle_maximized(self) -> None:
        self._window.showNormal() if self._window.isMaximized() else self._window.showMaximized()
        icon = QStyle.StandardPixmap.SP_TitleBarNormalButton if self._window.isMaximized() else QStyle.StandardPixmap.SP_TitleBarMaxButton
        self.maximize_button.setIcon(self.style().standardIcon(icon))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
