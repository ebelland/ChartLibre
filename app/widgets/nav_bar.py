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

A panel is any widget; :class:`NavPanel` is the usual one - a column of
titled frames, each a title above a grey rounded box, scrolling when taller
than the window or split by draggable handles.

The bar also looks after its own width. Placed in a QSplitter, it gives
width to its neighbour and takes it back as its panels are hidden and shown,
as it collapses to icons and as the bar itself is hidden - and it never
leaves both the bar and the panels hidden, which would leave nothing to
click.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import CardFrame, TitledCard, action_presentation, create_action_button, stdSizeAndlayout
from app.utils.i18n import _

#: Qt's "no maximum" width.
_NO_MAXIMUM = 16_777_215


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


@dataclass(frozen=True, slots=True)
class NavButton:
    """One button of a NavPanel frame: an action of the catalogue and what it runs.

    ``text`` and ``tooltip`` replace the catalogue's own words; ``described``
    shows the tooltip under the button, for an action that needs its
    sentence - the button then has a row of its own.
    """

    action_id: str
    handler: Callable[[], object]
    text: str | None = None
    tooltip: str | None = None
    described: bool = False


class NavPanel(QWidget):
    """A navigation panel: titled frames, one under the other.

    Each frame is a title above a grey rounded box (style.TitledCard), the
    way System Settings groups its rows. :meth:`add_frame` adds one and
    returns its box to fill. The column scrolls when it is taller than the
    window; with ``resizable=True`` the frames share the height instead,
    split by handles the user drags - a list above a preview, say.
    """

    def __init__(self, parent: QWidget | None = None, *, resizable: bool = False) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._splitter: QSplitter | None = None
        self._column: QVBoxLayout | None = None
        self._stretched = False
        if resizable:
            self._splitter = QSplitter(Qt.Orientation.Vertical, self)
            self._splitter.setChildrenCollapsible(False)
            layout.addWidget(self._splitter)
            return
        body = QWidget(self)
        # The white page the frames sit on: a plain QWidget paints no
        # stylesheet background without WA_StyledBackground.
        body.setProperty("toolboxPage", True)
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._column = QVBoxLayout(body)
        self._column.setContentsMargins(0, 0, 0, 0)
        self._column.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        scroll.setWidget(body)
        layout.addWidget(scroll)

    @property
    def splitter(self) -> QSplitter | None:
        """The handles between the frames of a resizable panel."""
        return self._splitter

    def add_frame(
        self,
        title: str,
        content: QWidget | None = None,
        *,
        object_name: str | None = None,
        stretch: int = 0,
        margins: tuple[int, int, int, int] | None = None,
    ) -> CardFrame:
        """Add a frame titled *title*, holding *content* if given; return its box.

        *stretch* > 0 lets the frame take the height left over (a list that
        should fill the page); *margins* overrides the box's own padding - a
        list or a table that draws its own edge wants none.
        """
        kwargs = {} if margins is None else {"margins": margins}
        titled = TitledCard(self, title, object_name, **kwargs)
        box_layout = titled.card.layout()
        if content is not None and box_layout is not None:
            box_layout.addWidget(content)
            if stretch and isinstance(box_layout, QVBoxLayout):
                box_layout.setStretchFactor(content, 1)
        if stretch:
            outer = titled.layout()
            if isinstance(outer, QVBoxLayout):
                outer.setStretchFactor(titled.card, 1)
        self.add_widget(titled, stretch=stretch)
        return titled.card

    def add_buttons(
        self,
        title: str,
        buttons: Sequence[NavButton],
        *,
        object_name: str | None = None,
        description: str = "",
    ) -> CardFrame:
        """Add a frame of action buttons: on one row, or one per row when described.

        *description* is a muted sentence under them all.
        """
        box = self.add_frame(title, object_name=object_name)
        layout = box.layout()
        assert isinstance(layout, QVBoxLayout)
        row: QHBoxLayout | None = None
        for button in buttons:
            if row is None or button.described:
                row = QHBoxLayout()
                stdSizeAndlayout(row)
                layout.addLayout(row)
            icon, text, tooltip = action_presentation(button.action_id)
            text = button.text if button.text is not None else text
            tooltip = button.tooltip if button.tooltip is not None else tooltip
            create_action_button(
                parent=box, action_id=button.action_id, action=button.handler,
                layout=row, presentation=(icon, text, tooltip),
            )
            if button.described:
                row.addStretch(1)
                self._muted(tooltip, box, layout)
                row = None
        if row is not None:
            row.addStretch(1)
        if description:
            self._muted(description, box, layout)
        return box

    def add_widget(self, widget: QWidget, *, stretch: int = 0) -> None:
        """Add a widget that is already a frame, or brings its own (a TitledCard)."""
        if self._splitter is not None:
            self._splitter.addWidget(widget)
            self._splitter.setStretchFactor(self._splitter.count() - 1, stretch)
            return
        assert self._column is not None
        if stretch and not self._stretched:
            # The trailing stretch keeps short pages at the top; a frame
            # that takes the spare height replaces it.
            self._column.takeAt(self._column.count() - 1)
            self._stretched = True
        index = self._column.count() if self._stretched else self._column.count() - 1
        self._column.insertWidget(index, widget, stretch)

    @staticmethod
    def _muted(text: str, parent: QWidget, layout: QVBoxLayout) -> None:
        label = QLabel(text, parent)
        label.setWordWrap(True)
        label.setProperty("muted", True)
        layout.addWidget(label)


class NavigationBar(QWidget):
    """A left rail and a right panel host driven by a sections dictionary."""

    action_clicked = Signal(str)
    panel_changed = Signal(str, object)
    item_clicked = Signal(str)
    #: The bar was hidden (True) or shown again (False) - by the caller, or by
    #: the bar itself when hiding the panels would otherwise leave nothing.
    bar_hidden_changed = Signal(bool)

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
        panel_min_width: int = 240,
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
        #: The narrowest the panels may be, and how wide the whole widget was
        #: when they were last hidden - what showing them again restores.
        self._panel_min_width = int(panel_min_width)
        self._restore_width = self._bar_width + self._panel_min_width
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
        # No top margin under a header: on macOS it is the title strip, whose
        # traffic lights and sidebar button sit on the window's top edge.
        self._bar_layout.setContentsMargins(8, 0 if header is not None else 8, 8, 8)
        self._bar_layout.setSpacing(4)
        if header is not None:
            self._bar_layout.addWidget(header)

        # Named for the stylesheets, which keep it transparent: a scroll
        # area paints the window colour by default, a grey block over the
        # bar's own background.
        self._section_scroller = QScrollArea(self.bar)
        self._section_scroller.setObjectName("navSections")
        self._section_scroller.setWidgetResizable(True)
        self._section_scroller.setFrameShape(QFrame.Shape.NoFrame)
        self._section_scroller.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._section_widget = QWidget(self._section_scroller)
        self._section_widget.setObjectName("navSectionsBody")
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
        return self.bar.maximumWidth()

    @property
    def visible_bar_width(self) -> int:
        """The bar's width, or 0 while it is hidden."""
        return 0 if self.bar.isHidden() else self.bar.maximumWidth()

    @property
    def bar_hidden(self) -> bool:
        return self.bar.isHidden()

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

    def button(self, key: str) -> QToolButton | None:
        """The row for *key*, to place a popup beside it; None when there is none."""
        return self._buttons.get(key)

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
        self.set_panel_visible(True)  # reopened at its width, if hidden
        self.panel_stack.setCurrentWidget(item.panel)
        self.item_clicked.emit(key)
        self.panel_changed.emit(key, item.panel)
        return True

    def set_panel_visible(self, visible: bool) -> None:
        """Show or hide the panels, the bar staying as it is.

        Hidden, the widget narrows to the bar, and remembers its width for
        when they are shown again. The bar comes back first if it was hidden
        too: with neither, nothing would be left to click.
        """
        visible = bool(visible)
        if visible == self.panel_visible:
            self._apply_width_limits()
            return
        if not visible and self.bar.isHidden():
            self.set_bar_hidden(False)
        if not visible and self.isVisible():
            self._restore_width = max(self.width(), self._panel_min_width + self.visible_bar_width)
        self.panels.setVisible(visible)
        self._apply_width_limits()
        self._resize_in_splitter(self._restore_width if visible else self.visible_bar_width)

    def toggle_panels(self) -> None:
        """Hide the panels if shown, show them if hidden."""
        self.set_panel_visible(not self.panel_visible)

    def set_compact(self, compact: bool) -> None:
        """Collapse the rail to icons while leaving panel selection unchanged.

        The width the labels took goes to the neighbour in the splitter, or
        collapsing would only widen the empty gap inside the widget.
        """
        before = self.visible_bar_width
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
        moved = self.visible_bar_width - before
        self._restore_width = max(self._restore_width + moved, self._panel_min_width + self.visible_bar_width)
        self._apply_width_limits()
        if moved:
            self._resize_in_splitter(
                self.width() + moved if self.panel_visible else self.visible_bar_width
            )

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
        if hidden and not self.panel_visible:
            self.set_panel_visible(True)  # never both hidden
        width = self.width()
        if self.header is not None:
            if hidden:
                self._header_slot.addWidget(self.header)
            else:
                self._bar_layout.insertWidget(0, self.header)
        self.bar.setVisible(not hidden)
        self._restore_width = max(
            self._restore_width + (-self.bar_width if hidden else self.bar_width),
            self._panel_min_width + self.visible_bar_width,
        )
        self._apply_width_limits()
        self._resize_in_splitter(width - self.bar_width if hidden else width + self.bar_width)
        self.bar_hidden_changed.emit(hidden)

    # -- Width ------------------------------------------------------------

    def _apply_width_limits(self) -> None:
        """Pinned to the bar while the panels are hidden; free above their floor otherwise.

        Worked out from the current state every time rather than adjusted
        step by step: hiding the bar and the panels in one order and showing
        them in the other once left the widget pinned at zero width.
        """
        bar = self.visible_bar_width
        if self.panel_visible:
            self.setMaximumWidth(_NO_MAXIMUM)
            self.setMinimumWidth(self._panel_min_width + bar)
        else:
            self.setMinimumWidth(bar)
            self.setMaximumWidth(bar)

    def _resize_in_splitter(self, width: int) -> None:
        """Become *width* wide in the splitter holding this widget, if any.

        The difference goes to, or comes from, the neighbour, never below
        the neighbour's own minimum width.
        """
        splitter = self.parentWidget()
        if not isinstance(splitter, QSplitter) or not self.isVisible():
            return
        index = splitter.indexOf(self)
        sizes = splitter.sizes()
        if index < 0 or len(sizes) < 2:
            return
        other = index + 1 if index + 1 < len(sizes) else index - 1
        neighbour = splitter.widget(other)
        floor = neighbour.minimumWidth() if neighbour is not None else 0
        width = max(int(width), 0)
        total = max(sizes[index] + sizes[other], width + floor)
        sizes[index], sizes[other] = width, max(total - width, 1)
        splitter.setSizes(sizes)

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
