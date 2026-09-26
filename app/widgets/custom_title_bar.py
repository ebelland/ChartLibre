"""Title strips: a full caption bar on Windows, the sidebar toggle on macOS."""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QWidget,
)

from app.styles.style import icon_from_svg_source
from app.utils.i18n import _

if TYPE_CHECKING:
    from app.dialogs.main_window import MainWindow

#: Windows: icon + title on the left, square min/max/close on the right,
#: each this tall. Compacted from an original 40 to match a native Windows
#: 11 title bar's own height more closely.
CUSTOM_TITLE_BAR_HEIGHT: int = 24

#: macOS: the strip at the top of the rail, level with the native traffic
#: lights. The window is a real NSWindow with a transparent title bar
#: (MainWindow.__init__), so AppKit draws the lights at its standard place:
#: a 32pt title bar, the three buttons ending at x=69.
MAC_TITLE_BAR_HEIGHT: int = 32
#: Where the native lights end, measured from the window's left edge, plus
#: the gap macOS apps leave before the sidebar button.
MAC_TRAFFIC_LIGHTS_END: int = 69
_MAC_SIDEBAR_BUTTON_GAP: int = 12

#: The sidebar toggle's own glyph: a panel with its left column divided
#: off, the same shape every macOS app uses for "hide/show the sidebar".
_SIDEBAR_ICON = (
    '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/>'
    '<line x1="9.5" y1="4.5" x2="9.5" y2="19.5"/>'
)

class CustomTitleBar(QFrame):
    """Custom chrome that delegates movement/resizing to the window system.

    Built for Windows first (no native chrome at all under
    Qt.FramelessWindowHint - "no border under Windows and square corners
    under Mac" was the original report) - a left-aligned icon and title,
    square min/max/close buttons on the right, Qt's own standard glyphs.

    macOS keeps its native window chrome instead: this strip only
    leaves room for AppKit's own traffic lights and puts the sidebar
    toggle beside them, where Claude, Finder and Music have it. It stays
    in that spot whether the rail is expanded or collapsed.
    """

    def __init__(
        self,
        window: MainWindow,
        *,
        is_macos: bool,
        parent: QWidget | None = None,
    ) -> None:
        # *window* is what the buttons act on (close, minimise, zoom,
        # startSystemMove); *parent* is who owns this widget. They are the
        # same for the Windows caption strip, which the window lays out
        # itself, and different on macOS, where the rail holds the strip -
        # passing the window as parent there left Qt and the layout
        # disagreeing about the owner, which crashed the interpreter when
        # Python later collected the window (SIGBUS in deleteChildren).
        super().__init__(parent if parent is not None else window)
        # A weak proxy, not the window itself. The buttons here act on the
        # window, but this widget does not own it - and on macOS it is a
        # grandchild of it (the rail holds the strip), so a strong
        # reference closes a cycle: window -> rail -> strip -> window.
        # Python's collector is then free to break that cycle in an order
        # Qt does not expect, which crashed the interpreter outright
        # (SIGBUS inside QObjectPrivate::deleteChildren, at whichever
        # unrelated line the collector happened to run on). Every call
        # below goes through the proxy unchanged.
        self._window: MainWindow = weakref.proxy(window)
        # Passed in, not re-imported: main_window.py already decided which
        # of IS_WINDOWS/IS_MACOS triggered building this at all (see
        # _create_central_host) - re-deriving IS_MACOS here independently
        # would silently ignore whatever that caller (or a test
        # monkeypatching it) actually decided, the same reason
        # NavigationBar takes is_macos as a parameter rather than importing
        # it itself.
        self._is_macos = is_macos
        self.setObjectName("customTitleBar")
        self.setFixedHeight(MAC_TITLE_BAR_HEIGHT if self._is_macos else CUSTOM_TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.title_label: QLabel | None = None
        self.sidebar_button: QToolButton | None = None

        if self._is_macos:
            self._build_mac_controls(QHBoxLayout(self))
        else:
            layout = QHBoxLayout(self)
            layout.setSpacing(4)
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
        layout.addWidget(self._build_sidebar_button())
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
    # macOS: room for the native traffic lights, then the sidebar toggle
    # ------------------------------------------------------------------
    def _build_mac_controls(self, layout: QHBoxLayout) -> None:
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addSpacing(self._mac_lights_inset())
        layout.addWidget(self._build_sidebar_button(), 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

    def _mac_lights_inset(self) -> int:
        """Room to leave on the left so the toggle clears the traffic lights.

        The lights are placed relative to the window, and this strip sits
        inside the rail's own left margin, so that margin is taken off.
        """
        parent = self.parentWidget()
        layout = parent.layout() if parent is not None else None
        margin = layout.contentsMargins().left() if layout is not None else 0
        return MAC_TRAFFIC_LIGHTS_END - margin + _MAC_SIDEBAR_BUTTON_GAP

    # ------------------------------------------------------------------
    # Shared: the sidebar toggle
    # ------------------------------------------------------------------
    def _build_sidebar_button(self) -> QToolButton:
        """The control that collapses the navigation rail to its icons.

        In the title bar rather than in the rail itself: it has to stay
        reachable once the rail is narrow, and this is the strip that is
        already above it on both platforms.
        """
        button = QToolButton(self)
        button.setObjectName("titleBarSidebarButton")
        button.setAutoRaise(True)
        button.setCheckable(True)
        button.setIcon(icon_from_svg_source(_SIDEBAR_ICON, size=18))
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(26, 22)
        button.setCursor(Qt.CursorShape.ArrowCursor)
        button.setToolTip(_("Hide the navigation labels"))
        button.toggled.connect(self._on_sidebar_toggled)
        self.sidebar_button = button
        return button

    def _on_sidebar_toggled(self, collapsed: bool) -> None:
        self.sidebar_button.setToolTip(
            _("Show the navigation labels")
            if collapsed
            else _("Hide the navigation labels")
        )
        self._window.set_navigation_compact(collapsed)

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------
    def _toggle_maximized(self) -> None:
        self._window.showNormal() if self._window.isMaximized() else self._window.showMaximized()
        if self._is_macos:
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
