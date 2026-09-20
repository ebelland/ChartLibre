"""Custom title bar for frameless windows (Windows and macOS)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QStyle, QToolButton

from app.utils.i18n import _

if TYPE_CHECKING:
    from app.dialogs.main_window import MainWindow

#: Windows: icon + title on the left, square min/max/close on the right,
#: each this tall. Compacted from an original 40 to match a native Windows
#: 11 title bar's own height more closely.
CUSTOM_TITLE_BAR_HEIGHT: int = 24

#: macOS: a shorter strip - real AppKit title bars run about this tall -
#: holding nothing but the traffic lights. Compacted alongside the Windows
#: height above, from 28/12/8, for the same native-height match.
_MAC_TITLE_BAR_HEIGHT: int = 24
_MAC_BUTTON_DIAMETER: int = 10
_MAC_BUTTON_SPACING: int = 6

#: (fill, border) per light, in AppKit's own left-to-right order:
#: close, minimize, zoom. Not theme-dependent - a real traffic light
#: does not go dark in Dark Mode either.
_MAC_TRAFFIC_LIGHTS: dict[str, tuple[str, str]] = {
    "close": ("#FF5F57", "#E0443E"),
    "minimize": ("#FEBC2E", "#DEA123"),
    "maximize": ("#28C840", "#1AAB29"),
}


class CustomTitleBar(QFrame):
    """Custom chrome that delegates movement/resizing to the window system.

    Built for Windows first (no native chrome at all under
    Qt.FramelessWindowHint - "no border under Windows and square corners
    under Mac" was the original report) - a left-aligned icon and title,
    square min/max/close buttons on the right, Qt's own standard glyphs.

    macOS went frameless later, the same way (same FramelessWindowHint,
    same QWindow.startSystemMove/startSystemResize this class already
    used for Windows) but not with the same buttons: three small round
    "traffic lights" on the *left*, AppKit's own close/minimize/zoom
    colors and ordering, and no icon or title text at all - a real
    Mac window in this style (Music, Notes, ...) shows none once the
    sidebar itself is doing the same job a title once did.
    """

    def __init__(self, window: MainWindow, *, is_macos: bool) -> None:
        super().__init__(window)
        self._window = window
        # Passed in, not re-imported: main_window.py already decided which
        # of IS_WINDOWS/IS_MACOS triggered building this at all (see
        # _create_central_host) - re-deriving IS_MACOS here independently
        # would silently ignore whatever that caller (or a test
        # monkeypatching it) actually decided, the same reason
        # NavigationBar takes is_macos as a parameter rather than importing
        # it itself.
        self._is_macos = is_macos
        self.setObjectName("customTitleBar")
        self.setFixedHeight(_MAC_TITLE_BAR_HEIGHT if self._is_macos else CUSTOM_TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setSpacing(4)

        self.title_label: QLabel | None = None
        if self._is_macos:
            self._build_mac_controls(layout)
        else:
            self._build_windows_controls(layout)

    # ------------------------------------------------------------------
    # Windows: icon, title, square right-aligned buttons
    # ------------------------------------------------------------------
    def _build_windows_controls(self, layout: QHBoxLayout) -> None:
        layout.setContentsMargins(8, 0, 0, 0)
        icon_label = QLabel(self)
        icon_label.setPixmap(self._window.windowIcon().pixmap(18, 18))
        icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(icon_label)
        self.title_label = QLabel(self._window.windowTitle(), self)
        self.title_label.setObjectName("titleBarTitle")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.title_label, 1)
        self.minimize_button = self._window_button(QStyle.StandardPixmap.SP_TitleBarMinButton, self._window.showMinimized, _("Minimize"))
        self.maximize_button = self._window_button(QStyle.StandardPixmap.SP_TitleBarMaxButton, self._toggle_maximized, _("Maximize"))
        self.close_button = self._window_button(QStyle.StandardPixmap.SP_TitleBarCloseButton, self._window.close, _("Close"), "titleBarCloseButton")
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)
        self._window.windowTitleChanged.connect(self.title_label.setText)

    def _window_button(self, icon: QStyle.StandardPixmap, callback: Any, tooltip: str, object_name: str = "titleBarButton") -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setAutoRaise(True)
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setFixedSize(46, CUSTOM_TITLE_BAR_HEIGHT)
        button.clicked.connect(callback)
        return button

    # ------------------------------------------------------------------
    # macOS: traffic lights, left-aligned, nothing else
    # ------------------------------------------------------------------
    def _build_mac_controls(self, layout: QHBoxLayout) -> None:
        left_inset = max((_MAC_TITLE_BAR_HEIGHT - _MAC_BUTTON_DIAMETER) // 2, 0)
        layout.setContentsMargins(left_inset, 0, 0, 0)
        layout.setSpacing(_MAC_BUTTON_SPACING)
        self.close_button = self._traffic_light("close", self._window.close, _("Close"))
        self.minimize_button = self._traffic_light("minimize", self._window.showMinimized, _("Minimize"))
        self.maximize_button = self._traffic_light("maximize", self._toggle_maximized, _("Zoom"))
        layout.addWidget(self.close_button)
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addStretch(1)

    def _traffic_light(self, kind: str, callback: Any, tooltip: str) -> QToolButton:
        fill, border = _MAC_TRAFFIC_LIGHTS[kind]
        button = QToolButton(self)
        button.setObjectName(f"trafficLight{kind.capitalize()}")
        button.setToolTip(tooltip)
        button.setFixedSize(_MAC_BUTTON_DIAMETER, _MAC_BUTTON_DIAMETER)
        button.setCursor(Qt.CursorShape.ArrowCursor)
        # Inline, not the shared QSS files: these three colors are AppKit's
        # own fixed palette - unlike everything else this app themes, a
        # traffic light does not change with the app's light/dark mode.
        button.setStyleSheet(
            "QToolButton {"
            f" background: {fill}; border: 1px solid {border};"
            f" border-radius: {_MAC_BUTTON_DIAMETER // 2}px;"
            "}"
        )
        button.clicked.connect(callback)
        return button

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------
    def _toggle_maximized(self) -> None:
        self._window.showNormal() if self._window.isMaximized() else self._window.showMaximized()
        if self._is_macos:
            # The zoom light does not change appearance with state, the
            # same way a real one does not.
            return
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
