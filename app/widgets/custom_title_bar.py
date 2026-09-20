"""Custom title bar for frameless windows (Windows and macOS)."""
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
    QVBoxLayout,
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

#: macOS: a shorter strip - real AppKit title bars run about this tall -
#: holding the traffic lights and the sidebar toggle. Kept at AppKit's own
#: measurements rather than following the Windows height above: the lights
#: are a native control people recognise by size, and the strip also has to
#: fit the sidebar button beside them.
_MAC_TITLE_BAR_HEIGHT: int = 28
_MAC_BUTTON_DIAMETER: int = 12
_MAC_BUTTON_SPACING: int = 8

#: Extra height the strip takes when the sidebar toggle moves onto its own
#: line under the traffic lights - see CustomTitleBar.set_compact.
_MAC_STACKED_TOGGLE_HEIGHT: int = 24

#: The sidebar toggle's own glyph: a panel with its left column divided
#: off, the same shape every macOS app uses for "hide/show the sidebar".
_SIDEBAR_ICON = (
    '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/>'
    '<line x1="9.5" y1="4.5" x2="9.5" y2="19.5"/>'
)

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
        self.setFixedHeight(_MAC_TITLE_BAR_HEIGHT if self._is_macos else CUSTOM_TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.title_label: QLabel | None = None
        self.sidebar_button: QToolButton | None = None
        self._controls_row: QHBoxLayout | None = None

        if self._is_macos:
            # A column, not a row: set_compact below moves the sidebar
            # toggle onto a second line when the rail is too narrow to
            # hold it beside the lights.
            self._stacked_row = QVBoxLayout(self)
            self._stacked_row.setContentsMargins(0, 0, 0, 0)
            self._stacked_row.setSpacing(2)
            controls = QHBoxLayout()
            controls.setSpacing(4)
            self._controls_row = controls
            self._stacked_row.addLayout(controls)
            self._build_mac_controls(controls)
        else:
            layout = QHBoxLayout(self)
            layout.setSpacing(4)
            self._build_windows_controls(layout)

    def set_compact(self, compact: bool) -> None:
        """Stack the sidebar toggle under the traffic lights, or inline it.

        macOS only, and only because of arithmetic: three 12px lights with
        8px between them and an inset already fill a collapsed rail's
        width, so a toggle beside them would be pushed off the edge (it
        was - the lights themselves ended up overlapping). Below them it
        fits, and the rail grows by one row rather than by 40px of width,
        which is the width collapsing was meant to give back in the first
        place.
        """
        if not self._is_macos or self.sidebar_button is None:
            return
        row = self._controls_row
        if row is None:
            return

        row.removeWidget(self.sidebar_button)
        if compact:
            self._stacked_row.addWidget(
                self.sidebar_button, 0, Qt.AlignmentFlag.AlignHCenter
            )
            self.setFixedHeight(_MAC_TITLE_BAR_HEIGHT + _MAC_STACKED_TOGGLE_HEIGHT)
        else:
            row.addWidget(self.sidebar_button)
            row.addStretch(1)
            self.setFixedHeight(_MAC_TITLE_BAR_HEIGHT)
        self.sidebar_button.setVisible(True)

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
        # Immediately right of the lights, where every macOS app that has
        # one puts it, and far enough from them not to be hit by accident.
        layout.addSpacing(_MAC_BUTTON_SPACING * 2)
        layout.addWidget(self._build_sidebar_button())
        layout.addStretch(1)

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
