"""Fluent 2 navigation rail for the main window."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QButtonGroup, QFrame, QSizePolicy, QToolButton, QVBoxLayout

from app.styles.style import action_presentation, icon_from_svg_source
from app.utils.i18n import _

if TYPE_CHECKING:
    from app.dialogs.main_window import MainWindow

#: Windows/Fluent: a fixed-width column of square icon-over-label tiles.
NAV_BAR_WIDTH = 120
NAV_ICON_SIZE = QSize(20, 20)
NAV_ITEM_SIZE = QSize(104, 64)

#: macOS: a wider Apple Music/Finder-style sidebar of icon-beside-label
#: rows, rather than square tiles - there is no native AppKit sidebar
#: control to defer to (see macos_native.qss's own note on #activityRail),
#: so this is this app's best approximation of one.
_MACOS_NAV_BAR_WIDTH = 200
_MACOS_ICON_SIZE = QSize(16, 16)
_MACOS_ROW_HEIGHT = 30

_HOME_ICON = (
    '<path d="M3 11.5 12 4l9 7.5"/>'
    '<path d="M5.5 10.5V21h13V10.5"/>'
    '<path d="M9.5 21v-6h5v6"/>'
)
_TABLE_ICON = (
    '<rect x="3" y="4" width="18" height="16" rx="2"/>'
    '<line x1="3" y1="9" x2="21" y2="9"/>'
    '<line x1="9" y1="4" x2="9" y2="20"/>'
    '<line x1="15" y1="4" x2="15" y2="20"/>'
    '<line x1="3" y1="14.5" x2="21" y2="14.5"/>'
)
_SLIDERS_ICON = (
    '<line x1="4" y1="6" x2="20" y2="6"/>'
    '<circle cx="9" cy="6" r="2"/>'
    '<line x1="4" y1="12" x2="20" y2="12"/>'
    '<circle cx="15" cy="12" r="2"/>'
    '<line x1="4" y1="18" x2="20" y2="18"/>'
    '<circle cx="11" cy="18" r="2"/>'
)
_DATABASE_ICON = (
    '<ellipse cx="12" cy="5" rx="8" ry="3"/>'
    '<path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
    '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'
)
_FILE_ICON = (
    '<path d="M6 2h9l5 5v15H6z"/>'
    '<path d="M15 2v5h5"/>'
)
_DEVELOPER_ICON = (
    '<polyline points="16 18 22 12 16 6"/>'
    '<polyline points="8 6 2 12 8 18"/>'
)


class NavigationBar(QFrame):
    """Wide Fluent rail with icon-over-label navigation tiles."""

    def __init__(self, window: MainWindow, *, is_macos: bool) -> None:
        super().__init__(window)
        self._window: MainWindow = window
        self._is_macos = is_macos
        self.setObjectName("activityRail")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(_MACOS_NAV_BAR_WIDTH if is_macos else NAV_BAR_WIDTH)
        # Vertical Ignored, not Expanding: the tiles below are each
        # setFixedSize (64px tall, deliberately - Fluent-style tiles, not
        # accidental), and stacked vertically that sums to well over the
        # window's own minimum height. Under Expanding, Qt's own
        # qSmartMinSize takes max(sizeHint, minimumSizeHint) as the floor
        # regardless of what setMinimumHeight is given - an explicit minimum
        # only overrides that floor when it is > 0, so setMinimumHeight(0)
        # alone is silently a no-op here. Ignored drops that floor to 0 (Qt
        # still hands the rail whatever height the splitter actually gives
        # it - this only changes what the rail *demands*, not what it
        # *gets*), which is what stops it from forcing the containing frame,
        # and the window through it, at least as tall as its own content.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored)
        self.setMinimumHeight(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(4)

        self.workspace_button = self._tile(
            icon_from_svg_source(_HOME_ICON, size=20),
            _("Workspace"),
            _("Hide the left panel"),
            checkable=True,
        )
        layout.addWidget(self.workspace_button)

        # "File" is a page tile like Data/Database, not a platform-gated
        # auxiliary button: File's actions (New/Open/Import/Save...) used to
        # only exist as this button's own popup menu, off macOS only - see
        # main_window._create_file_page for where they live now, on every
        # platform, the same way Database's own popup dialog became a page.
        self.action_ids = (
            "nav_data",
            "nav_chart_options",
            "nav_series_operations",
            "nav_database",
            "nav_file",
            "nav_developer",
        )

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.button_group.addButton(self.workspace_button, -2)
        self.button_group.idClicked.connect(self._on_group_clicked)
        self.buttons: list[QToolButton] = []
        for index, action_id in enumerate(self.action_ids):
            button = self._navigation_tile(action_id)
            self.button_group.addButton(button, index)
            self.buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)

        if not is_macos:
            self.settings_button = self._catalogue_tile(
                "settings", _("Settings"), checkable=False
            )
            self.settings_button.clicked.connect(window._on_settings)
            layout.addWidget(self.settings_button)

            self.help_button = self._catalogue_tile(
                "user_manual", _("Help"), checkable=False
            )
            self.help_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self.help_button.setMenu(window._help_menu)
            layout.addWidget(self.help_button)
        else:
            self.settings_button = None
            self.help_button = None

        # No inline setStyleSheet here: #activityRail/#navigationItem are
        # styled per platform in fluent_win11.qss (square Fluent tiles) and
        # macos_native.qss (Apple Music/Finder-style sidebar rows) instead,
        # the same split every other piece of bespoke chrome in this app
        # already follows - see that file's own PLATFORM PARITY NOTES.

    def _on_group_clicked(self, button_id: int) -> None:
        if button_id == -2:
            if self._window._left_stack.isVisible():
                self._window._toggle_workspace()
            return
        self._window._set_nav_index(button_id)

    def set_workspace_hidden(self, hidden: bool) -> None:
        self.workspace_button.setChecked(bool(hidden))
        tooltip = _("Show the left panel") if hidden else _("Hide the left panel")
        self.workspace_button.setToolTip(tooltip)
        self.workspace_button.setStatusTip(tooltip)
        self.workspace_button.setAccessibleName(tooltip)

    def select_page(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)

    def _navigation_tile(self, action_id: str) -> QToolButton:
        if action_id == "nav_data":
            return self._tile(
                icon_from_svg_source(_TABLE_ICON, size=20),
                _("Tables"),
                _("Show data tables"),
                checkable=True,
            )
        if action_id == "nav_chart_options":
            _icon, text, tooltip = action_presentation(action_id)
            return self._tile(
                icon_from_svg_source(_SLIDERS_ICON, size=20),
                text,
                tooltip,
                checkable=True,
            )
        if action_id == "nav_database":
            return self._tile(
                icon_from_svg_source(_DATABASE_ICON, size=20),
                _("Database"),
                _("Show database tools"),
                checkable=True,
            )
        if action_id == "nav_file":
            return self._tile(
                icon_from_svg_source(_FILE_ICON, size=20),
                _("File"),
                _("New, open, import and save"),
                checkable=True,
            )
        if action_id == "nav_developer":
            return self._tile(
                icon_from_svg_source(_DEVELOPER_ICON, size=20),
                _("Developer"),
                _("Scaffolding tools and the translation catalogue"),
                checkable=True,
            )
        icon, text, tooltip = action_presentation(action_id)
        return self._tile(icon, text, tooltip, checkable=True)

    def _catalogue_tile(
        self, action_id: str, label: str, *, checkable: bool
    ) -> QToolButton:
        icon, _text, tooltip = action_presentation(action_id)
        return self._tile(icon, label, tooltip, checkable=checkable)

    def _tile(self, icon, text: str, tooltip: str, *, checkable: bool) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("navigationItem")
        button.setAutoRaise(False)
        button.setIcon(icon)
        button.setText(text)
        if self._is_macos:
            # A sidebar row - icon beside a left-aligned label, stretched to
            # the rail's own width - not a square tile: Apple Music/Finder's
            # sidebar is a list, not a grid of icons.
            button.setIconSize(_MACOS_ICON_SIZE)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            button.setFixedHeight(_MACOS_ROW_HEIGHT)
        else:
            button.setIconSize(NAV_ICON_SIZE)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setFixedSize(NAV_ITEM_SIZE)
        button.setToolTip(tooltip)
        button.setStatusTip(tooltip)
        button.setAccessibleName(text)
        button.setCheckable(checkable)
        return button
