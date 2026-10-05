"""Generic sectioned navigation bar with an integrated panel host.

The parent supplies an ordered dictionary shaped as::

    {
        "Tools": {
            "workspace": NavBarItem("Workspace", icon, None),
            "tables": NavBarItem("Tables", icon, tables_panel),
        },
        "Charts": {
            "chart:12": NavBarItem("Yield", icon, None),
        },
    }

An item with a panel behaves like a tab: the panel host is made visible and
that exact widget is selected. An item without a panel emits ``action_clicked``.
The parent can rebuild any section without rebuilding the navigation widget.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import _


@dataclass(frozen=True, slots=True)
class NavBarItem:
    """One navigation item.

    ``key`` is supplied by the containing dictionary. ``panel=None`` defines
    an action item; otherwise the item selects the supplied panel.
    """

    title: str
    icon: QIcon
    panel: QWidget | None
    tooltip: str = ""


NavBarItems: TypeAlias = Mapping[str, NavBarItem]
NavBarSections: TypeAlias = Mapping[str, NavBarItems]


class NavigationBar(QWidget):
    """A left rail and a right panel host driven by a sections dictionary."""

    action_clicked = Signal(str)
    panel_changed = Signal(str, object)
    item_clicked = Signal(str)

    def __init__(
        self,
        sections: NavBarSections,
        parent: QWidget | None = None,
        *,
        bar_width: int = 200,
        compact_width: int = 56,
        row_height: int = 36,
        header: QWidget | None = None,
        panel_margins: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__(parent)
        self.header = header
        self.title_bar = header
        self._sections: dict[str, dict[str, NavBarItem]] = {
            str(section): dict(items) for section, items in sections.items()
        }
        self._bar_width = int(bar_width)
        self._compact_width = int(compact_width)
        self._row_height = int(row_height)
        self._compact = False
        self._selected_key: str | None = None
        self._buttons: dict[str, QToolButton] = {}
        self._items: dict[str, NavBarItem] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.bar = QFrame(self)
        self.bar.setObjectName("activityRail")
        self.bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.bar.setFixedWidth(self._bar_width)
        self.bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        root.addWidget(self.bar, 0)

        self._bar_layout = QVBoxLayout(self.bar)
        self._bar_layout.setContentsMargins(8, 8, 8, 8)
        self._bar_layout.setSpacing(4)
        if header is not None:
            self._bar_layout.addWidget(header)

        self._section_scroller = QScrollArea(self.bar)
        self._section_scroller.setWidgetResizable(True)
        self._section_scroller.setFrameShape(QFrame.Shape.NoFrame)
        self._section_scroller.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._section_widget = QWidget(self._section_scroller)
        self._section_layout = QVBoxLayout(self._section_widget)
        self._section_layout.setContentsMargins(0, 0, 0, 0)
        self._section_layout.setSpacing(4)
        self._section_scroller.setWidget(self._section_widget)
        self._bar_layout.addWidget(self._section_scroller, 1)

        self.panels = QFrame(self)
        self.panels.setObjectName("navigationPanels")
        self.panels.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.panels.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panels_layout = QVBoxLayout(self.panels)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(0)
        # Where the header goes while the bar is hidden (set_bar_hidden):
        # above the pages and outside their margins.
        self._header_slot = QVBoxLayout()
        self._header_slot.setContentsMargins(0, 0, 0, 0)
        panels_layout.addLayout(self._header_slot)
        stack_layout = QVBoxLayout()
        stack_layout.setContentsMargins(*panel_margins)
        self.panel_stack = QStackedWidget(self.panels)
        self.panel_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        stack_layout.addWidget(self.panel_stack, 1)
        panels_layout.addLayout(stack_layout, 1)
        root.addWidget(self.panels, 1)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._button_group.buttonClicked.connect(self._button_clicked)

        self._rebuild()
        self.set_panel_visible(False)

    @property
    def selected_key(self) -> str | None:
        return self._selected_key

    @property
    def panel_visible(self) -> bool:
        return not self.panels.isHidden()

    @property
    def bar_width(self) -> int:
        return self.bar.width()

    def sections(self) -> dict[str, dict[str, NavBarItem]]:
        """Return a shallow copy of the current section definitions."""
        return {name: dict(items) for name, items in self._sections.items()}

    def set_sections(self, sections: NavBarSections) -> None:
        """Replace all sections while preserving a still-valid selection."""
        selected = self._selected_key
        self._sections = {
            str(section): dict(items) for section, items in sections.items()
        }
        self._rebuild()
        if selected in self._items:
            self.select(selected)
        elif selected is not None:
            self._selected_key = None
            self.set_panel_visible(False)

    def set_section(self, title: str, items: NavBarItems) -> None:
        """Create or replace one section, for example the current chart list."""
        sections = self.sections()
        sections[str(title)] = dict(items)
        self.set_sections(sections)

    def remove_section(self, title: str) -> None:
        sections = self.sections()
        sections.pop(str(title), None)
        self.set_sections(sections)

    def select(self, key: str) -> bool:
        """Activate one item by stable key.

        Returns ``False`` when the key does not exist. Action items emit only
        ``action_clicked``; panel items show the panel host and select their
        associated widget.
        """
        item = self._items.get(key)
        if item is None:
            return False
        button = self._buttons[key]
        if item.panel is None:
            self.item_clicked.emit(key)
            self.action_clicked.emit(key)
            return True

        button.setChecked(True)
        self._selected_key = key
        self.set_panel_visible(True)
        self.panel_stack.setCurrentWidget(item.panel)
        self.item_clicked.emit(key)
        self.panel_changed.emit(key, item.panel)
        return True

    def set_panel_visible(self, visible: bool) -> None:
        """Show or hide the whole right-hand panel section."""
        self.panels.setVisible(bool(visible))

    def set_compact(self, compact: bool) -> None:
        """Collapse the rail to icons while leaving panel selection unchanged."""
        self._compact = bool(compact)
        self.bar.setFixedWidth(self._compact_width if compact else self._bar_width)
        for key, button in self._buttons.items():
            item = self._items[key]
            button.setText("" if compact else _(item.title))
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonIconOnly
                if compact
                else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
        for index in range(self._section_layout.count()):
            item = self._section_layout.itemAt(index)
            if item is None:
                continue
            widget = item.widget()
            if isinstance(widget, QLabel):
                widget.setVisible(not compact)

    def set_bar_hidden(self, hidden: bool) -> None:
        """Hide the rail itself, or bring it back, leaving the panels as they are.

        The header moves to the top of the panels while the bar is hidden,
        and back afterwards: on macOS it holds the traffic lights and the
        button that brings the bar back, which must stay reachable. The
        caller keeps the panels shown while the bar is hidden - with
        neither, nothing would be left to click.
        """
        hidden = bool(hidden)
        if self.bar.isHidden() == hidden:
            return
        if self.header is not None:
            if hidden:
                self._header_slot.addWidget(self.header)
            else:
                self._bar_layout.insertWidget(0, self.header)
        self.bar.setVisible(not hidden)

    def set_workspace_hidden(self, hidden: bool) -> None:
        """Compatibility helper: Workspace means hiding the panel section."""
        self.set_panel_visible(not hidden)

    def _button_clicked(self, button: QToolButton) -> None:
        key = str(button.property("navKey") or "")
        if key:
            self.select(key)

    def _rebuild(self) -> None:
        old_panels: set[QWidget] = set()
        for index in range(self.panel_stack.count()):
            widget = self.panel_stack.widget(index)
            if widget is not None:
                old_panels.add(widget)
        while self._section_layout.count():
            item = self._section_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                # Hidden now: deleteLater waits for the event loop, and until
                # then the old row would still be drawn under the new one.
                widget.hide()
                widget.deleteLater()

        for button in self._buttons.values():
            self._button_group.removeButton(button)
        self._buttons.clear()
        self._items.clear()

        new_panels: set[QWidget] = set()
        keys_seen: set[str] = set()
        for section_title, items in self._sections.items():
            if section_title:
                label = QLabel(_(section_title), self._section_widget)
                label.setObjectName("navSectionTitle")
                self._section_layout.addWidget(label)
            for key, nav_item in items.items():
                if key in keys_seen:
                    raise ValueError(f"Duplicate navigation key: {key}")
                keys_seen.add(key)
                self._items[key] = nav_item
                button = self._make_button(key, nav_item)
                self._buttons[key] = button
                self._button_group.addButton(button)
                self._section_layout.addWidget(button)
                if nav_item.panel is not None:
                    new_panels.add(nav_item.panel)
                    if self.panel_stack.indexOf(nav_item.panel) < 0:
                        self.panel_stack.addWidget(nav_item.panel)
            self._section_layout.addSpacing(8)
        self._section_layout.addStretch(1)

        for panel in old_panels - new_panels:
            self.panel_stack.removeWidget(panel)
        self.set_compact(self._compact)

    def _make_button(self, key: str, item: NavBarItem) -> QToolButton:
        button = QToolButton(self._section_widget)
        button.setObjectName("navigationItem")
        button.setProperty("navKey", key)
        button.setAutoRaise(False)
        button.setCheckable(item.panel is not None)
        button.setIcon(item.icon)
        button.setIconSize(QSize(20, 20))
        button.setText(_(item.title))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setFixedHeight(self._row_height)
        tooltip = _(item.tooltip or item.title)
        button.setToolTip(tooltip)
        button.setStatusTip(tooltip)
        button.setAccessibleName(_(item.title))
        return button
