"""Fluent 2 navigation rail for the main window."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QButtonGroup, QFrame, QSizePolicy, QToolButton, QVBoxLayout

from app.styles.style import action_presentation, icon_from_svg_source
from app.utils.i18n import _

if TYPE_CHECKING:
    from app.dialogs.main_window import MainWindow

NAV_BAR_WIDTH = 120
NAV_ICON_SIZE = QSize(20, 20)
NAV_ITEM_SIZE = QSize(104, 64)

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


class NavigationBar(QFrame):
    """Wide Fluent rail with icon-over-label navigation tiles."""

    def __init__(self, window: MainWindow, *, is_macos: bool) -> None:
        super().__init__(window)
        self._window: MainWindow = window
        self.setObjectName("activityRail")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(NAV_BAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

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

        if not is_macos:
            self.file_button = self._catalogue_tile("open", _("File"), checkable=False)
            self.file_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self.file_button.setMenu(window._file_menu)
            layout.addWidget(self.file_button)
        else:
            self.file_button = None

        self.action_ids = (
            "nav_data",
            "nav_chart_options",
            "nav_series_operations",
            "nav_database",
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

        self.setStyleSheet("""
            QFrame#activityRail {
                background: palette(window);
                border: none;
                border-right: 1px solid palette(midlight);
            }
            QToolButton#navigationItem {
                background: transparent;
                color: palette(window-text);
                border: none;
                border-radius: 8px;
                padding: 6px 4px;
                font-size: 8pt;
            }
            QToolButton#navigationItem:hover {
                background: palette(midlight);
            }
            QToolButton#navigationItem:pressed {
                background: palette(mid);
            }
            QToolButton#navigationItem:checked {
                background: palette(button);
                color: palette(window-text);
                border-left: 4px solid palette(highlight);
                padding-left: 0px;
                font-weight: 600;
            }
            QToolButton#navigationItem::menu-indicator {
                image: none;
                width: 0px;
            }
        """)

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
        button.setIconSize(NAV_ICON_SIZE)
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setFixedSize(NAV_ITEM_SIZE)
        button.setToolTip(tooltip)
        button.setStatusTip(tooltip)
        button.setAccessibleName(text)
        button.setCheckable(checkable)
        return button
