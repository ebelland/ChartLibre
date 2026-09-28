"""A platform-aware navigation rail: Fluent tiles on Windows, a macOS sidebar.

Reusable outside this application. The rail knows how to *look* like the
platform it is running on and nothing about what its entries mean: the
pages are passed in as :class:`NavPage` records and every click leaves as
a signal, so a host window decides what a tile does. Nothing here reaches
back into the window that owns it except to give the title bar something
to move (macOS), which is what a frameless window needs.

To reuse it:

    rail = NavigationBar(window, is_macos=IS_MACOS, pages=[
        NavPage("files", "Files", "Show the files", icon_svg=FOLDER_SVG),
        NavPage("settings", "Settings", "Preferences"),
    ])
    rail.page_selected.connect(stack.setCurrentIndex)
    rail.workspace_toggled.connect(window.toggle_side_panel)

A page with no ``icon_svg`` falls back to the application's own action
catalogue (``style.action_presentation``), which is this project's way of
naming an icon per platform; a project without one simply passes the SVG.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)

from app.styles.style import action_presentation, icon_from_svg_source
from app.utils.i18n import _
from app.widgets.custom_title_bar import CustomTitleBar

if TYPE_CHECKING:
    from app.dialogs.main_window import MainWindow
    from PySide6.QtWidgets import QWidget
else:
    from PySide6.QtWidgets import QWidget


@dataclass(frozen=True, slots=True)
class NavPage:
    """One page tile: its key, what it reads as, and how it is drawn.

    ``key`` is what identifies the page to the host - this application
    uses its action-catalogue ids, another project can use anything it
    likes. ``icon_svg`` is the body of an SVG (the ``<path>``/``<rect>``
    elements, no wrapper), drawn at the rail's own icon size; leaving it
    None asks the action catalogue for a platform icon instead.
    """

    key: str
    label: str = ""
    tooltip: str = ""
    icon_svg: str | None = None

#: Windows/Fluent: a fixed-width column of square icon-over-label tiles.
#: Widened from the original 104/120 - "Operazioni sulle serie" and
#: "Area di lavoro" wrapped or clipped badly under an icon in a tile
#: that narrow, at any font past this sheet's own 10pt base.
NAV_BAR_WIDTH = 132
NAV_ICON_SIZE = QSize(20, 20)
NAV_ITEM_SIZE = QSize(116, 64)

#: macOS: a wider Apple Music/Finder-style sidebar of icon-beside-label
#: rows, rather than square tiles - there is no native AppKit sidebar
#: control to defer to (see macos_native.qss's own note on #activityRail),
#: so this is this app's best approximation of one.
_MACOS_NAV_BAR_WIDTH = 200
_MACOS_ICON_SIZE = QSize(20, 20)
_MACOS_ROW_HEIGHT = 34

#: Windows: collapsed ("icons only") rail width. macOS has no such mode -
#: its sidebar toggle hides the rail outright (MainWindow).
_WINDOWS_COMPACT_WIDTH = 56

#: This application's own rail. Passed as a default rather than built
#: inside the class, so the class itself carries no knowledge of what
#: ChartLibre's pages are - see the module docstring for reuse. The
#: order is the order of the pages beside it (see
#: main_window._create_left_stack): File first, directly under Workspace,
#: since that is where a session starts.
DEFAULT_PAGES: tuple[NavPage, ...] = (
    NavPage("nav_file", "File", "New, open, import and save"),
    NavPage("nav_data", "Tables", "Show data tables"),
    NavPage("nav_chart_options", "", ""),
    NavPage("nav_series_operations", "", ""),
    NavPage("nav_database", "Database", "Show database tools"),
    NavPage(
        "nav_developer",
        "Developer",
        "Scaffolding tools and the translation catalogue",
    ),
)


def _wrap_tile_label(text: str) -> str:
    """Break a tile label onto two lines at its middlemost space.

    QToolButton has no word-wrap of its own (a Qt limitation, not a
    missed setting) - left alone, a label past the tile's own width just
    elides ("Series ...rations"), unreadable. Breaking it explicitly is
    the same thing WinUI3's own longer tile labels do. Only ever applied
    to this button's own text, never to the shared catalogue string
    (spec.translated_text()) that menus and tooltips reuse unwrapped.
    """
    if " " not in text:
        return text
    words = text.split(" ")
    lengths = [len(word) for word in words]
    total = sum(lengths) + len(words) - 1
    best_index, best_gap, running = 0, total, 0
    for index, length in enumerate(lengths[:-1]):
        running += length + 1
        gap = abs(running - total / 2)
        if gap < best_gap:
            best_index, best_gap = index, gap
    return " ".join(words[: best_index + 1]) + "\n" + " ".join(words[best_index + 1 :])


class NavigationBar(QFrame):
    """A navigation rail: Fluent tiles on Windows, a sidebar list on macOS.

    Host-agnostic: it reports what was clicked and never acts on the
    window itself. See the module docstring for how to reuse it.
    """

    #: A page tile was chosen - the index into ``pages``, so a host can
    #: hand it straight to a QStackedWidget.
    page_selected = Signal(int)
    #: The workspace tile was clicked; the host decides what "workspace"
    #: means (here: collapse the panel beside the rail).
    workspace_toggled = Signal()
    #: A footer tile was clicked, by key ("settings" here). Footer tiles
    #: are actions rather than pages, so they carry no index.
    action_triggered = Signal(str)
    #: A chart was picked in the Charts section - its row, which is its
    #: index among the chart pages the host shows.
    chart_selected = Signal(int)

    def __init__(
        self,
        window: QWidget,
        *,
        is_macos: bool,
        pages: Sequence[NavPage] | None = None,
    ) -> None:
        super().__init__(window)
        # Held only to give the frameless title bar a window to move and
        # to parent the help menu - never to call back into it.
        self._window = window
        self._is_macos = is_macos
        self._compact = False
        self.pages: tuple[NavPage, ...] = tuple(
            DEFAULT_PAGES if pages is None else pages
        )
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
        # macOS: no top margin - the title strip below has to sit level
        # with the native traffic lights, which AppKit draws from y=0.
        layout.setContentsMargins(8, 0 if is_macos else 10, 8, 10)
        layout.setSpacing(4)

        # macOS: the traffic lights sit at the top of the rail itself, on
        # the rail's own grey. A full-window-width strip above the splitter
        # was tried instead and looked wrong for the reason Finder, Music
        # and System Settings all avoid it: it put a grey band across the
        # top of the *white* panels beside the rail, so their background
        # stopped short of the window's edge and every vertical divider
        # ended in a seam a few pixels down from the top. Windows keeps its
        # own full-width caption strip (icon, title, min/max/close), which
        # is what that platform's windows actually look like - see
        # main_window._create_central_host.
        self.title_bar: CustomTitleBar | None = None
        if self._is_macos:
            # parent=self, window=the window its buttons act on: see
            # CustomTitleBar.__init__ on why those must not be the same
            # object here.
            self.title_bar = CustomTitleBar(
                cast("MainWindow", window), is_macos=True, parent=self
            )
            layout.addWidget(self.title_bar)
            layout.addSpacing(4)

        # Two sections, the way a recent Mac sidebar is laid out (Music's
        # Library and Playlists, SF Symbols' own): the tools, then the
        # project's charts - which used to be tabs above the chart.
        self._tools_title = self._section_title(_("Tools"))
        layout.addWidget(self._tools_title)

        self.workspace_button = self._tile(
            # The catalogue's own icon: an SF Symbol on macOS, a Segoe
            # Fluent glyph on Windows (config.json's nav_workspace).
            action_presentation("nav_workspace")[0],
            _("Workspace"),
            _("Hide the left panel"),
            checkable=True,
        )
        layout.addWidget(self.workspace_button)

        #: The page keys, in rail order. Kept as a plain tuple of strings
        #: because callers index pages by key ("which tile is Database?")
        #: far more often than they want the whole record.
        self.action_ids = tuple(page.key for page in self.pages)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.button_group.addButton(self.workspace_button, -2)
        self.button_group.idClicked.connect(self._on_group_clicked)
        self.buttons: list[QToolButton] = []
        for index, page in enumerate(self.pages):
            button = self._navigation_tile(page)
            self.button_group.addButton(button, index)
            self.buttons.append(button)
            layout.addWidget(button)

        layout.addSpacing(10)
        self._charts_title = self._section_title(_("Charts"))
        layout.addWidget(self._charts_title)
        self.chart_list = QListWidget(self)
        self.chart_list.setObjectName("chartList")
        self.chart_list.setFrameShape(QFrame.Shape.NoFrame)
        self.chart_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.chart_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chart_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.chart_list.setIconSize(_MACOS_ICON_SIZE if is_macos else QSize(16, 16))
        self.chart_list.setUniformItemSizes(True)
        self.chart_list.setMinimumHeight(0)
        self.chart_list.currentRowChanged.connect(self._on_chart_row_changed)
        layout.addWidget(self.chart_list, 1)

        if not is_macos:
            self.settings_button = self._catalogue_tile(
                "settings", _("Settings"), checkable=False
            )
            # A signal, not window._on_settings: the rail reports the
            # click and the host decides what settings are.
            self.settings_button.clicked.connect(
                lambda: self.action_triggered.emit("settings")
            )
            layout.addWidget(self.settings_button)

            self.help_button = self._catalogue_tile(
                "user_manual", _("Help"), checkable=False
            )
            self.help_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            layout.addWidget(self.help_button)
        else:
            self.settings_button = None
            self.help_button = None

        # No inline setStyleSheet here: #activityRail/#navigationItem are
        # styled per platform in fluent_win11.qss (square Fluent tiles) and
        # macos_native.qss (Apple Music/Finder-style sidebar rows) instead,
        # the same split every other piece of bespoke chrome in this app
        # already follows - see that file's own PLATFORM PARITY NOTES.

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName("navSectionTitle")
        return label

    def set_charts(self, charts: list[tuple[str, QIcon | None]], current: int) -> None:
        """Show *charts* - (name, icon) - in the Charts section, *current* selected."""
        self.chart_list.blockSignals(True)
        try:
            self.chart_list.clear()
            for name, icon in charts:
                item = QListWidgetItem(name, self.chart_list)
                if icon is not None:
                    item.setIcon(icon)
                item.setToolTip(name)
            if 0 <= current < self.chart_list.count():
                self.chart_list.setCurrentRow(current)
        finally:
            self.chart_list.blockSignals(False)

    def select_chart(self, index: int) -> None:
        """Mark row *index* as the current chart without announcing it."""
        self.chart_list.blockSignals(True)
        try:
            self.chart_list.setCurrentRow(index)
        finally:
            self.chart_list.blockSignals(False)

    def _on_chart_row_changed(self, row: int) -> None:
        if row >= 0:
            self.chart_selected.emit(row)

    def _on_group_clicked(self, button_id: int) -> None:
        """Report the click; what it means is the host's business."""
        if button_id == -2:
            self.workspace_toggled.emit()
            return
        self.page_selected.emit(button_id)

    def set_help_menu(self, menu) -> None:
        """Attach the menu the Help tile pops up (off macOS only).

        Separate from construction because the menu belongs to the host -
        the rail only needs somewhere to hang it.
        """
        if self.help_button is not None:
            self.help_button.setMenu(menu)

    def set_workspace_hidden(self, hidden: bool) -> None:
        self.workspace_button.setChecked(bool(hidden))
        tooltip = _("Show the left panel") if hidden else _("Hide the left panel")
        self.workspace_button.setToolTip(tooltip)
        self.workspace_button.setStatusTip(tooltip)
        self.workspace_button.setAccessibleName(tooltip)

    def set_compact(self, compact: bool) -> None:
        """Show the tiles as icons only, and narrow the rail to match (Windows).

        The labels are dropped rather than elided: a tile showing three
        characters of a word is worse than one showing none - the tooltip
        already carries the full text.
        """
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact

        self.setFixedWidth(_WINDOWS_COMPACT_WIDTH if compact else NAV_BAR_WIDTH)

        for widget in (self._tools_title, self._charts_title, self.chart_list):
            widget.setVisible(not compact)

        for button in self._all_tiles():
            self._apply_tile_mode(button)

    def _all_tiles(self) -> list[QToolButton]:
        """Every tile in the rail, including the ones that are not pages."""
        tiles = [self.workspace_button, *self.buttons]
        for extra in (self.settings_button, self.help_button):
            if extra is not None:
                tiles.append(extra)
        return tiles

    def _apply_tile_mode(self, button: QToolButton) -> None:
        """Set one tile's text, style and size for the current mode.

        The icon keeps its size in both modes.
        """
        full_text = str(button.property("navLabel") or button.text())
        button.setIconSize(_MACOS_ICON_SIZE if self._is_macos else NAV_ICON_SIZE)

        if self._compact:
            button.setText("")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setFixedSize(QSize(_WINDOWS_COMPACT_WIDTH - 16, NAV_ITEM_SIZE.height()))
            return

        button.setText(full_text if self._is_macos else _wrap_tile_label(full_text))
        if self._is_macos:
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setFixedHeight(_MACOS_ROW_HEIGHT)
        else:
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setFixedSize(NAV_ITEM_SIZE)

    def select_page(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)

    def _navigation_tile(self, page: NavPage) -> QToolButton:
        """Build one page tile from its record.

        Where the icon comes from is the one platform difference left:
        macOS draws the page's own outline SVG, matching the hairline
        look of a Finder sidebar, while Windows asks the action catalogue
        for the Segoe Fluent glyph so a Windows icon can never drift from
        its catalogue entry. A page carrying no SVG - or any page at all
        on Windows - falls through to the catalogue; a project reusing
        this class without one just passes ``icon_svg`` for every page.
        """
        icon, text, tooltip = action_presentation(page.key)
        if self._is_macos and page.icon_svg:
            icon = icon_from_svg_source(page.icon_svg, size=20)
        return self._tile(
            icon,
            _(page.label) if page.label else text,
            _(page.tooltip) if page.tooltip else tooltip,
            checkable=True,
        )

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
        button.setText(text if self._is_macos else _wrap_tile_label(text))
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
        # Kept aside so collapsing to icons and expanding again restores the
        # label: accessibleName is not a safe place to read it back from -
        # set_workspace_hidden rewrites that one to the current tooltip.
        button.setProperty("navLabel", text)
        button.setCheckable(checkable)
        return button
