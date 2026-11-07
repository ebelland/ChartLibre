"""Main application window: the navigation bar and its panels on the left, the charts on the right.

The window is assembled here - the panels, the bar, the splitter, the chart
tabs - and its larger concerns follow as sections of the one class: the menus
(the menu bar, the app menu, Undo and Open recent), the Chart properties tabs,
the project (new, open, save, demo, import, undo) and the frame (the frameless
window's edges and dropped files).
"""
from __future__ import annotations

import gc
import json
import logging
import re
import warnings
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pandas as pd
from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import APP_ICON, APP_NAME
from app.ai import settings as assistant_settings
from app.data.demos import PROJECTS_DIR, DemoProject, copy_demo_project
from app.data.repo._common import ensure_read_only_select
from app.data.repo.function_sql import function_names
from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.dialogs.connect_database_dialog import ConnectDatabaseDialog
from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.dialogs.credits_dialog import CreditsDialog
from app.dialogs.import_data_dialog import ImportDataDialog
from app.dialogs.log_viewer_dialog import LogViewerDialog
from app.dialogs.query_builder_dialog import QueryBuilderDialog
from app.logs.logger import applogger, log_events
from app.main_window.custom_title_bar import CustomTitleBar
from app.plugin_api.charts import layout_presets
from app.scanners.data_operation_scanner import import_class_from_file
from app.styles.style import (
    CONTEXT_MENU_ICONS,
    load_icon,
    EXPANDED_CLIENT_AREA_HINT,
    IS_MACOS,
    IS_WINDOWS,
    MARGIN_TOOLBOX_PAGE,
    PANEL_MIN_WIDTH,
    SPACING_DEFAULT,
    SPLITTER_HANDLE_WIDTH,
    CardFrame,
    MenuItem,
    _pyobjc_core_is_safe_to_import,
    action_menu_item,
    action_presentation,
    colored_icon,
    create_menu,
    create_menu_item,
    icon_from_svg_source,
    relax_minimum_width,
    stdSizeAndlayout,
    theme_is_dark,
)
from app.utils import column_summary
from app.utils.coercion import column_of, to_numbers
from app.utils.config import (
    clear_recent_databases,
    get_constant,
    get_recent_databases,
    get_section,
    set_last_database,
    set_section,
)
from app.utils.data_sources import (
    SERVER_DATABASE_QUERY_READERS,
    SERVER_DATABASE_READERS,
    is_importable,
)
from app.utils.dialog_state import restore_window_geometry, save_window_geometry
from app.utils.i18n import _
from app.utils.messages import ask, show_message
from app.utils.screen_fit import fit_on_show
from app.utils.startup import PROJECT_FILE_FILTER
from app.widgets.analysis_panel import AnalysisPanel, plot_operation, plugin_icon
from app.widgets.assistant_panel import AssistantPanel
from app.widgets.workpad_panel import WorkpadPanel
from app.utils import recent_analyses
from app.widgets.catalog_search import CatalogSearch
from app.widgets.chart_jump_bar import ChartJumpBar
from app.widgets.chart_panel import ChartPanel
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget
from app.widgets.chart_properties.figure_properties import FigurePropertiesWidget
from app.widgets.chart_properties.overlay_properties import OverlayPropertiesWidget
from app.widgets.chart_properties.series_properties import SeriesPropertiesWidget
from app.widgets.database_info_panel import DatabaseInfoPanel
from app.widgets.guided_panel import GuidedAnalysisPanel
from app.widgets.nav_bar import NavBarItem, NavButton, NavigationBar, NavPanel
from app.widgets.recent_projects import ProjectsTabs
from app.widgets.table_list import TableListPanel
from app.widgets.table_preview import TablePreviewPanel

if TYPE_CHECKING:
    from app.data.sqlite_repo import SqliteRepo

# config.json keys for the remembered window layout.
STATE_KEY: str = "main_window"
SPLITTERS_SECTION: str = "main_window_splitters"
LOG_VIEWER_STATE_KEY: str = "log_viewer"

# The user manual PDF, built by docs/manual/user_manual.typ and shipped
# alongside the source rather than generated at runtime.
USER_MANUAL_PATH: Path = Path(__file__).resolve().parents[2] / "docs" / "manual" / "user_manual.pdf"

# Narrowest useful chart pane.  Explicit, because the alternative is whatever
# the chart toolbar happens to add up to - and that number silently wins the
# splitter negotiation against the left panel.
CHART_PANE_MIN_WIDTH: int = get_constant("chart_pane_min_width", 260)

# How wide the panel *beside* the navigation rail starts out - the table
# list, the property pages, the data-operation panels. Added to the rail's
# own width rather than used as the whole left pane, since the rail's width
# differs per platform; see _create_main_split.
PANEL_DEFAULT_WIDTH: int = get_constant("panel_default_width", 340)

# How long a picked-point readout stays in the status bar, in milliseconds.
# Long enough to read and write down, short enough that it is gone before it
# can be mistaken for a description of some later chart.
CHART_SELECTION_TIMEOUT_MS: int = get_constant("chart_selection_timeout_ms", 15_000)


# Coalescing window for property-driven chart reloads, in milliseconds.
# Long enough to swallow a spinbox drag, short enough to feel immediate.
PROPERTIES_REDRAW_DEBOUNCE_MS: int = get_constant("properties_redraw_debounce_ms", 120)


# A run of same-label descriptor snapshots inside this many seconds is
# treated as one edit and recorded once. The auto-applying property panels
# fire _snapshot_descriptors on every debounced change; without this a
# single slider drag would leave a stack of identical "Figure properties"
# undo entries in front of the state worth going back to.
SNAPSHOT_COALESCE_SECONDS: float = get_constant("snapshot_coalesce_seconds", 2.0)


def resize_edges(
    pos: QPoint,
    size: QSize,
    margin: int,
    *,
    top: bool = True,
    bottom: bool = True,
    corner: int = 0,
) -> Qt.Edge:
    """Which edge(s) of a *size* rectangle *pos* is within *margin* of.

    *top* and *bottom* say whether that side is an edge of the window at all:
    a strip in the middle of it (the central area above the status bar) has no
    edge there to drag. *corner* widens the left and right bands along the
    bottom row to that many pixels, so the window's corner is as easy to grab
    as the corner of a normal window (a 6 px square is a small target).
    """
    edges = Qt.Edge(0)
    on_bottom = bottom and pos.y() >= size.height() - margin
    side = max(margin, corner) if on_bottom else margin
    if pos.x() <= side:
        edges |= Qt.Edge.LeftEdge
    elif pos.x() >= size.width() - side:
        edges |= Qt.Edge.RightEdge
    if top and pos.y() <= margin:
        edges |= Qt.Edge.TopEdge
    elif on_bottom:
        edges |= Qt.Edge.BottomEdge
    return edges


def _first_line(message: str, limit: int = 160) -> str:
    """The first line of a log message, cut to *limit* characters: a status bar has one line."""
    line = next((part.strip() for part in str(message or "").splitlines() if part.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "\u2026"



def _bold_plus_icon(colour: str, size: int) -> QIcon:
    """A heavy plus in *colour*: the symbol fonts draw theirs too thin to stand out as the main action."""
    from PySide6.QtGui import QColor, QPainter, QPen, QPixmap

    ratio = 3
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(colour), size * ratio * 0.16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    middle, reach = size * ratio / 2, size * ratio * 0.36
    painter.drawLine(int(middle), int(middle - reach), int(middle), int(middle + reach))
    painter.drawLine(int(middle - reach), int(middle), int(middle + reach), int(middle))
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    """The navigation bar and its panels, and the chart tabs beside them.

    The window must stay shrinkable: so that no child pushes a large minimum
    height up to it, the central layout uses zero minimum sizes and every
    panel scrolls.
    """

    _tabs: QTabWidget
    _properties_control: QTabWidget
    _figure_widget: FigurePropertiesWidget
    _axis_widget: AxisPropertiesWidget
    _series_widget: SeriesPropertiesWidget
    _overlay_widget: OverlayPropertiesWidget

    def __init__(self, repo: SqliteRepo, db_path: Path) -> None:
        super().__init__()
        self._repo = repo
        self._db_path = db_path
        applogger.set_status_bar(self.statusBar())
        self._configure_status_bar()
        applogger.debug(f"Initializing main window for database: {db_path}")

        self._update_window_title()
        self.setWindowIcon(icon_from_svg_source(APP_ICON, size=32))
        #: Only set on Windows, where this window lays the caption strip
        #: out itself. On macOS the rail owns the strip and this stays
        #: None - read it through the _custom_title_bar property below,
        #: which asks whichever of the two actually built one. Holding a
        #: second, strong Python reference here to a widget the rail owns
        #: in C++ is what made the interpreter crash (SIGBUS inside
        #: QObjectPrivate::deleteChildren) when the garbage collector
        #: later tore a closed window's object graph down.
        self._own_title_bar: CustomTitleBar | None = None
        if IS_WINDOWS:
            # Frameless: CustomTitleBar (below) replaces the native caption
            # strip one-for-one - icon, title, min/max/close.
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        elif IS_MACOS:
            # A real NSWindow, with its content drawn up under a title bar
            # that paints nothing: macOS itself draws the corners (the
            # system radius, anti-aliased), the shadow, the traffic lights
            # and the resize edges, the way Claude, Finder and Music look.
            # Only Qt's own public flags - no pyobjc on the window - so
            # there is no native pointer here to get wrong. The traffic
            # lights land on the rail's own title strip (NavigationBar).
            with warnings.catch_warnings():
                # PySide6 reports this value under its deprecated alias.
                warnings.simplefilter("ignore", DeprecationWarning)
                self.setWindowFlag(EXPANDED_CLIENT_AREA_HINT, True)
            self.setWindowFlag(Qt.WindowType.NoTitleBarBackgroundHint, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)
        self.resize(1200, 800)
        # 800 is taller than many laptops leave free: fitted when shown.
        fit_on_show(self)
        # The scientific libraries the data operations need, imported in
        # the background once the window is up, so the first operation of
        # the session opens at once (app/utils/warm_up.py).
        from app.utils.warm_up import warm_up_in_background

        QTimer.singleShot(3_000, warm_up_in_background)

        # Debounce for property-driven chart reloads (see _redraw_properties_chart).
        self._properties_redraw_callback: Any | None = None
        # The panel itself, kept alongside the raw Figure the other property
        # widgets get: the fit-mode combo calls panel.set_resize_mode and
        # panel.resize_mode directly, since that state is not a descriptor
        # key routed through figure_options_requested like everything else
        # the figure widget writes.
        self._properties_panel: ChartPanel | None = None
        self._properties_redraw_timer = QTimer(self)
        self._properties_redraw_timer.setSingleShot(True)
        self._properties_redraw_timer.timeout.connect(self._flush_properties_chart_redraw)

        # Last descriptor snapshot, for coalescing a run of auto-applied
        # edits into one undo entry (see _snapshot_descriptors).
        self._last_snapshot_label: str = ""
        self._last_snapshot_at: float = 0.0

        # Files can be dropped on the window: see dropEvent.
        self.setAcceptDrops(True)

        # Keep the whole window shrinkable.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Right side: chart tabs.
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(self._on_chart_tab_changed)
        self._configure_tabs()

        # Left side data widgets.
        self._table_panel = TableListPanel(parent=self, repo=self._repo)
        self._table_panel.tableSelected.connect(self._on_table_selected)
        self._preview = TablePreviewPanel(parent=self, repo=self._repo)
        self._preview.refresh.connect(lambda: self.refresh(reload_preview=False))
        self._table_panel.changed.connect(self.refresh)
        self._preview.chart_requested.connect(self._on_chart_from_columns)
        self._preview.histogram_requested.connect(self._on_histogram_from_column)
        self._preview.selected_rows_changed.connect(self._on_rows_selected)
        self._preview.selected_only_changed.connect(self._on_selected_only_changed)
        self._preview.query_requested.connect(lambda sql: self._on_create_query_table(sql=sql))
        # Left-side pages.
        self._data_page = self._create_data_page()
        self._properties_control = self._create_properties_control()
        self._configure_properties_control()

        # One navigation component owns both the bar and its panels.
        # MainWindow supplies the sections and handles the action-only rows.
        self._build_app_menu()
        self._left_panel: NavigationBar = self._create_navigation()
        # Cmd+[ / Cmd+] on macOS (Qt maps Ctrl to Command), as in Xcode.
        for keys, step in (("Ctrl+[", -1), ("Ctrl+]", +1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(lambda s=step: self._step_chart(s))
        # Cmd/Ctrl+K: search everything the application can do, from anywhere.
        QShortcut(QKeySequence("Ctrl+K"), self).activated.connect(self.focus_search)
        self._configure_left_panel()

        # Main split: left panel + chart tabs.
        self._main_split = self._create_main_split()

        # Wrap the splitter in a plain central widget with a zero-minimum layout.
        self._central_host = self._create_central_host()
        self.setCentralWidget(self._central_host)
        if IS_WINDOWS:
            # Frameless (see setWindowFlag above), so the OS gives us no edge
            # resize handles at all - _resize_edge_at/_update_resize_cursor
            # below fill that in. Watching _central_host, not self: it fills
            # the entire client area (setCentralWidget), margin included, so
            # self - the QMainWindow - never actually sees a mouse event of
            # its own to filter. The 8px contentsMargins in
            # _create_central_host is exactly the band this hit-tests.
            self._central_host.setMouseTracking(True)
            self._central_host.installEventFilter(self)
            # The status bar is outside _central_host, so the window's real
            # bottom edge and corners are the bar's: it needs the same
            # watching, and its labels must not take the mouse (a label
            # swallows the moves the cursor shape depends on). Its own size
            # grip stays off - this replaces it.
            status_bar = self.statusBar()
            status_bar.setMouseTracking(True)
            status_bar.installEventFilter(self)
            for label in (self._status_project, self._status_context, self._status_state, self._status_dot):
                label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Default page: the tables, not whatever happens to be first in the
        # rail. File sits above them now, and opening onto an empty file
        # page would hide the data the window was just opened on.
        self._select_navigation("tables")
        self._check_project_sql()
        self._check_project_functions()
        self._table_panel.reload()
        self._reload_tabs()
        self._update_properties_for_current_chart()

        # Last: the saved geometry must win over the resize() above and over
        # any size hint the freshly populated panels have just produced.
        self._restore_layout()
        # After _restore_layout, not before: collapsing the rail moves the
        # splitter, and a restore running afterwards would put the saved
        # (expanded) split back while the rail stayed narrow.
        if IS_WINDOWS or IS_MACOS:
            self._restore_navigation_compact()

        applogger.debug("Main window initialized")

    #: The status bar's states, light theme then dark: background, top border, text, dot.
    #: A quiet bar in the window's own colour while all is well - the solid blue
    #: strip it was drew the eye to the one place with nothing to read - and a
    #: coloured one only when there is something to see: amber for a warning,
    #: red for an error, until it is dismissed.
    _STATUS_BAR_STATES: ClassVar[dict[str, tuple[tuple[str, str, str, str], tuple[str, str, str, str]]]] = {
        "normal": (("palette(window)", "rgba(0, 0, 0, 0.08)", "#4A5263", "#22A06B"),
                   ("palette(window)", "rgba(255, 255, 255, 0.10)", "#B9C0CB", "#3FCF8E")),
        "success": (("palette(window)", "rgba(0, 0, 0, 0.08)", "#16704A", "#22A06B"),
                    ("palette(window)", "rgba(255, 255, 255, 0.10)", "#5BD69A", "#3FCF8E")),
        "busy": (("palette(window)", "rgba(0, 0, 0, 0.08)", "#1D4FB8", "#2F6FE4"),
                 ("palette(window)", "rgba(255, 255, 255, 0.10)", "#8DB4FF", "#4C8DF6")),
        "working": (("#dbeafe", "#93c5fd", "#1e3a8a", "#2563eb"), ("#1e3a8a", "#2563eb", "#dbeafe", "#93c5fd")),
        "warning": (("#FEF6E4", "#F3D9A4", "#7A4D05", "#B54708"),
                    ("#3A2E12", "#6B5320", "#F5D58A", "#F0B429")),
        "error": (("#FDECEC", "#F4B4B0", "#912018", "#C4320A"),
                  ("#3A1D1D", "#6E2B2B", "#FFB4AB", "#FF6B5E")),
    }
    #: A warning fades back to Ready after this long; an error stays until dismissed.
    _STATUS_WARNING_MS = 8_000

    def _configure_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setObjectName("vscodeStatusBar")
        bar.setSizeGripEnabled(False)
        bar.setFixedHeight(26)
        # The bar's own edge padding, and the only way to get it: QStatusBar
        # is one of the widgets whose QSS box model Qt only honours for
        # background and border, so a "padding" in the sheet below is
        # accepted, ignored, and looks like it worked (see
        # test_status_bar_padding.py, which pins that down so this does not
        # get "simplified" back into the stylesheet). Without it the project
        # label and the state label sit however many pixels off the edge the
        # active style's own internal offset happens to leave - two on
        # Fusion, flush under others - rather than a padding this app chose.
        bar.setContentsMargins(SPACING_DEFAULT, 0, SPACING_DEFAULT, 0)
        self._status_project = QLabel(self._db_path.name if self._db_path else "", bar)
        self._status_context = QLabel(_("ChartLibre"), bar)
        self._status_state = QLabel(_("Ready"), bar)
        self._status_state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._status_state.setObjectName("statusState")
        self._status_dot = QLabel(bar)
        self._status_dot.setObjectName("statusDot")
        self._status_dot.setFixedSize(8, 8)
        # With an error only: the log it came from, and a way back to Ready.
        # Links rather than buttons: a label sits on the bar's text line, where
        # a tool button's own frame put it a few pixels lower.
        self._status_actions = QLabel(bar)
        self._status_actions.setObjectName("statusActions")
        self._status_actions.setTextFormat(Qt.TextFormat.RichText)
        self._status_actions.setText(
            f'<a href="log">{_("Show log")}</a>&nbsp;&nbsp;&nbsp;<a href="dismiss" title="{_("Dismiss")}">\u2715</a>'
        )
        self._status_actions.linkActivated.connect(self._on_status_action)
        self._status_actions.hide()
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self._set_status_state("normal", _("Ready")))
        self._status_current = "normal"
        bar.addWidget(self._status_project, 0)
        bar.addWidget(self._status_context, 1)
        bar.addPermanentWidget(self._status_dot, 0)
        bar.addPermanentWidget(self._status_state, 0)
        bar.addPermanentWidget(self._status_actions, 0)
        self._set_status_state("normal")
        log_events.record_emitted.connect(self._on_log_record_for_status)

    def _on_status_action(self, link: str) -> None:
        """A link in the status bar: open the log, or dismiss the error."""
        if link == "log":
            self._show_log_viewer()
        self._set_status_state("normal", _("Ready"))

    def _on_log_record_for_status(self, _formatted: str, level: int, message: str) -> None:
        """Turn the status bar amber on a warning and red on an error, saying what happened.

        Only what was logged at WARNING or above: an INFO line is not news. A
        warning does not replace an error still showing, which is the one that
        needs reading. Qt's own warnings ("This plugin does not support ...")
        are about the toolkit, not about anything the person did, and stay in
        the log only - a Qt critical is still an error.
        """
        if level < logging.ERROR and str(message).startswith(("Qt warning:", "Qt info:", "Qt debug:")):
            return
        if level >= logging.ERROR:
            self._set_status_state("error", _first_line(message))
        elif level >= logging.WARNING and self._status_current != "error":
            self._set_status_state("warning", _first_line(message), timeout_ms=self._STATUS_WARNING_MS)
        else:
            return
        # The logger also writes every message at the bar's left for a few
        # seconds; this one is said on the right, in colour, and stays - the
        # left keeps the project's name instead of the same words twice.
        QTimer.singleShot(0, self.statusBar().clearMessage)

    def _on_assistant_busy(self, busy: bool) -> None:
        """The bar is coloured while the assistant works; an error already shown stays."""
        if busy and self._status_current != "error":
            self._set_status_state("working", _("The assistant is working..."))
        elif not busy and self._status_current == "working":
            self._set_status_state("normal", _("Ready"))

    def _set_status_state(self, state: str = "normal", message: str | None = None, timeout_ms: int = 0) -> None:
        """Colour the status bar for *state* and show *message* in it; after *timeout_ms*, back to Ready."""
        if state not in self._STATUS_BAR_STATES:
            state = "normal"
        background, border, text, dot = self._STATUS_BAR_STATES[state][1 if theme_is_dark() else 0]
        stylesheet = (
            f"QStatusBar#vscodeStatusBar {{ background: {background}; color: {text}; border: none; "
            f"border-top: 1px solid {border}; }} "
            "QStatusBar#vscodeStatusBar::item { border: none; } "
            f"QStatusBar#vscodeStatusBar QLabel {{ color: {text}; background: transparent; padding: 0 4px; }} "
            f"QStatusBar#vscodeStatusBar QLabel#statusDot {{ background: {dot}; border-radius: 4px; padding: 0; }} "
        )
        self.statusBar().setStyleSheet(stylesheet)
        self._status_current = state
        alarming = state == "error"
        # The links in the bar's own text colour, bold: blue on red would not read.
        self._status_actions.setText(
            f'<a href="log" style="color:{text}; font-weight:600;">{_("Show log")}</a>&nbsp;&nbsp;&nbsp;'
            f'<a href="dismiss" style="color:{text}; text-decoration:none;">\u2715</a>'
        )
        self._status_actions.setVisible(alarming)
        self._status_timer.stop()
        if message is not None:
            self._status_state.setText(message)
            self._status_state.setToolTip(message if alarming or state == "warning" else "")
            if timeout_ms > 0:
                self._status_timer.start(timeout_ms)

    def _update_window_title(self) -> None:
        """Show the active project beside ChartLibre in every title surface.

        No ``self.setToolTip(...)`` here: a tooltip on the main window
        itself is also the fallback tooltip for every child widget that has
        none of its own - Qt forwards an unhandled ToolTip event up the
        parent chain - so setting one here used to make the .dhub path show
        up on almost any control instead of that control's own hint. The
        status bar's project label is the one place this path belongs, and
        it already carries it below.
        """
        project = self._db_path.name if self._db_path else _("Untitled project")
        # macOS: no window title at all. The title bar is transparent and
        # AppKit would draw the text across the rail; the status bar below
        # already names the project.
        self.setWindowTitle("" if IS_MACOS else f"{APP_NAME} | {project}")
        if hasattr(self, "_status_project"):
            self._status_project.setText(project)
            self._status_project.setToolTip(str(self._db_path) if self._db_path else "")

    #: config.json key remembering whether the rail is collapsed to icons.
    NAV_COMPACT_KEY: str = "navigation_compact"

    def set_navigation_compact(self, compact: bool) -> None:
        """Collapse the bar - on macOS hide it outright, elsewhere keep its icons.

        The sidebar button in the title bar calls this. NavigationBar moves
        the splitter and keeps the panel reachable; this only says which of
        the two the platform does, and remembers it.
        """
        compact = bool(compact)
        if IS_MACOS:
            self._left_panel.set_bar_hidden(compact)
        else:
            self._left_panel.set_compact(compact)
        set_section(STATE_KEY, {**get_section(STATE_KEY), self.NAV_COMPACT_KEY: compact})

    def _on_bar_hidden_changed(self, hidden: bool) -> None:
        """macOS: the traffic lights moved with the title strip; the button follows the bar.

        The bar can come back on its own - hiding the panel while the bar is
        hidden would leave nothing - and the sidebar button must then read
        as shown too.
        """
        title_bar = self._left_panel.header
        if isinstance(title_bar, CustomTitleBar):
            title_bar.refresh_lights_inset()
        button = getattr(self._custom_title_bar, "sidebar_button", None)
        if button is not None and button.isCheckable() and button.isChecked() != hidden:
            button.setChecked(hidden)

    def _configure_left_panel(self) -> None:
        """Configure the composite left panel to avoid height lock-up."""
        self._left_panel.setMinimumSize(0, 0)
        self._left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

    @property
    def _custom_title_bar(self) -> CustomTitleBar | None:
        """This window's title bar, wherever it was built.

        Windows builds its own caption strip; on macOS the navigation
        rail holds the traffic lights instead. A property rather than an
        attribute so that the macOS one is only ever reached through its
        real owner - see _own_title_bar on what caching it here cost.
        """
        if self._own_title_bar is not None:
            return self._own_title_bar
        rail = getattr(self, "_left_panel", None)
        return None if rail is None else rail.title_bar

    def _on_assistant_changed(self, figures: list) -> None:
        """The assistant wrote into the project: the tables and charts again, a figure it made shown."""
        if figures:
            self._reload_tabs(select_figure_id=int(figures[-1]))
        self.refresh()

    def _on_settings(self) -> None:
        """Open the application preferences.

        The menus are rebuilt afterwards because the stylesheet may have
        changed underneath them, and because a saved language has to reach the
        one part of the interface that is cheap to rebuild - the rest of it
        picks the new catalogue up at the next start, which the dialog says.
        """
        from app.dialogs.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self)
        if dialog.exec():
            self._build_app_menu()
            self._refresh_recent_section()  # its count may have changed (Settings > General)
        # Shown in the navigation bar or not (Settings > Assistant).
        self._left_panel.set_item_hidden("assistant", not assistant_settings.in_navigation())
        # The key, the consent or the permissions may have changed: the panel says what is missing.
        assistant = getattr(self, "_assistant_panel", None)
        if assistant is not None:
            assistant.refresh()

    def _show_log_viewer(self) -> None:
        """Open a read-only viewer for recent in-memory log records.

        Laid out like every other dialog in the app - shell margins, a titled
        card around the content, a button row at the bottom - rather than a
        bare text box filling the window to its edges.  The records are
        monospaced and not wrapped: a log line is columns (time, level, caller,
        message), and wrapping destroys the alignment that makes it scannable.
        """
        dialog = LogViewerDialog(self)
        restore_window_geometry(dialog, LOG_VIEWER_STATE_KEY)
        dialog.exec()
        save_window_geometry(dialog, LOG_VIEWER_STATE_KEY)

    # ------------------------------------------------------------------
    # Left pages
    # ------------------------------------------------------------------

    def _create_data_page(self) -> NavPanel:
        """Tables above, the selected table's data below, the handle between them dragged."""
        page = NavPanel(self, resizable=True)
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # A small inset: the tables' square headers and scroll corners stay
        # inside the frame's rounded corners instead of squaring them (B-10);
        # the frame's border is the only one (the sheets drop the views').
        tables = page.add_frame(_("Tables"), self._table_panel, object_name="tablesFrame", stretch=1,
                                margins=(4, 4, 4, 4))
        data = page.add_frame(_("Data"), self._preview, object_name="dataFrame", stretch=2, margins=(4, 4, 4, 4))
        # Beside each title, the actions used most, as the chart's own buttons; the last opens the right-click menu.
        self._clipboard_transposed = QAction(_("Rows and columns swapped (transposed)"), self)
        self._clipboard_transposed.setCheckable(True)
        self._clipboard_transposed.setObjectName("clipboardTransposed")
        self._clipboard_transposed.setToolTip(
            _("For data copied one series per row: the clipboard is read with rows and columns exchanged.")
        )
        tables_bar = self._title_buttons([
            ("add", _("New table: blank, from a file, the clipboard, the web or a database"),
             lambda button: self._new_table_menu().exec(button.mapToGlobal(button.rect().bottomLeft()))),
            ("paste", _("Import from clipboard"), lambda _button: self._on_import_clipboard()),
            ("delete", _("Delete the selected tables"), lambda _button: self._table_panel._delete_selected()),
            ("chart_from_columns", _("A new chart of this table"),
             lambda _button: self._table_panel._on_new_plot_tab()),
            ("column_info", _("The table's columns, types and cells; a function's formula; a query's SQL"),
             lambda _button: self._table_panel.edit_selected()),
            ("rename", _("Rename the selected table, saved query or stored function"),
             lambda _button: self._table_panel._rename_table()),
            ("table_more", _("More"), lambda button: self._table_panel.show_menu_at(
                button.mapToGlobal(button.rect().bottomLeft()))),
        ])
        data_bar = self._title_buttons([
            ("column_info", _("Column Info..."), lambda _button: self._preview.column_info()),
            ("chart_from_columns", _("A chart of the selected columns: a histogram of one, else New plot"),
             lambda _button: self._preview.chart()),
            ("copy", _("Copy the selected cells"), lambda _button: self._preview._copy_selection()),
            ("table_more", _("More"), lambda button: self._preview.show_menu_at(
                button.mapToGlobal(button.rect().bottomLeft()))),
        ])
        for card, bar in ((tables, tables_bar), (data, data_bar)):
            titled = card.parentWidget()
            if hasattr(titled, "add_header_widget"):
                titled.add_header_widget(bar)
        assert page.splitter is not None
        self._data_split = page.splitter
        return page

    def _title_buttons(self, entries: list[tuple[str, str, Any]]) -> QFrame:
        """Small icon buttons in one rounded group, as the chart's navigation: (action id, tooltip, call(button))."""
        pill = QFrame(self)
        pill.setObjectName("chartNavPill")
        pill.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(pill)
        row.setContentsMargins(3, 2, 3, 2)
        row.setSpacing(1)
        for action_id, tooltip, call in entries:
            button = QToolButton(pill)
            button.setObjectName("chartToolButton")
            icon = load_icon(action_id)
            # Delete in red, as a destructive action is everywhere on the Mac.
            button.setIcon(colored_icon(icon, "#dc2626", 16) if action_id == "delete" else icon)
            if action_id == "add":
                # The page's main action: the plus in the accent blue, a little larger - marked, not shouting.
                button.setIcon(_bold_plus_icon("#2f6fe0", 16))
            # The chart's sizes: the sheet's pill rule (radius 17, buttons 28) is drawn for them.
            button.setIconSize(QSize(16, 16))
            button.setFixedSize(QSize(28, 28))
            button.setAutoRaise(True)
            button.setToolTip(tooltip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _checked=False, b=button, c=call: c(b))
            row.addWidget(button)
        return pill

    def _on_data_operation_requested(self, operation: dict) -> None:
        """Handle a built-in or runtime data-operation request."""
        icon = plugin_icon(operation)
        if recent_analyses.remember(operation):
            self._refresh_recent_section()

        if operation.get("name") == "NewPlotTabDialog":
            self._on_new_plot_tab(icon)
            return
        from app.scanners.data_operation_scanner import reads_roles

        if reads_roles(operation):
            # One family: an operation on a table's columns opens on the table, its columns cast.
            self._on_table_operation_requested(operation)
            return

        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error(
                "Could not load data operation class: %r",
                operation.get("name"),
            )
            return

        self._open_data_operation(dialog_class, icon)

    def _configure_tabs(self) -> None:
        """Make the chart tabs shrink-friendly in both directions.

        A QTabWidget is as wide as the wider of its tab bar and its current
        page, and both push back by default: the bar lays every tab out at its
        full title width, and the page reports the chart toolbar's width.  With
        that floor in place the splitter has no room left to give the left
        panel, which is why the left panel appeared to ignore its own minimum.
        Eliding the titles and scrolling the bar removes the first half; the
        second is handled inside ChartPanel.
        """
        self._tabs.setObjectName("chartTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)
        self._tabs.setTabsClosable(False)
        self._tabs.setMinimumSize(0, 0)
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        tab_bar = self._tabs.tabBar()
        tab_bar.setUsesScrollButtons(True)
        tab_bar.setExpanding(False)
        tab_bar.setElideMode(Qt.TextElideMode.ElideNone)

        # Without this the bar still asks for the full width of every title.
        tab_bar.setMinimumWidth(300)
        tab_bar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        # The charts are chosen from the jump bar above the chart (see
        # _sync_chart_navigation); the QTabWidget stays as the page stack, its
        # own bar hidden.
        tab_bar.hide()

    def _restore_navigation_compact(self) -> None:
        """Re-apply the remembered collapsed state, button included."""
        if not bool(get_section(STATE_KEY).get(self.NAV_COMPACT_KEY, False)):
            return
        button = getattr(self._custom_title_bar, "sidebar_button", None)
        if button is not None:
            # setChecked drives the toggled signal, which calls
            # set_navigation_compact - the button and the rail cannot
            # disagree about the state this way.
            button.setChecked(True)
        else:
            self.set_navigation_compact(True)

    def _new_table_menu(self) -> QMenu:
        """The ways to make a table, under the Tables title's + (it was a page of the rail)."""
        menu = QMenu(self)
        entries = [
            ("new", _("New Blank Table"), _("Create an empty table with named columns."), self._on_new_blank_table),
            ("query", _("Create Query Table"), _("Create a table from a database query."),
             self._on_create_query_table),
            None,
            ("paste", _("Import from clipboard"), _("Create a table from the table copied to the clipboard."),
             self._on_import_clipboard),
            "transposed",
            None,
            ("import", _("Import from file"), _("Create a table from a file: CSV, Excel, JSON."),
             self._on_import_data),
            ("import", _("Import Database Table"), _("Create a table from a database table."),
             self._on_database_table_import),
            ("web_sources", _("Import from the web"),
             _("Create a table from a web address or a listed open-data source."), self._on_import_web),
        ]
        menu.setToolTipsVisible(True)
        for entry in entries:
            if entry is None:
                menu.addSeparator()
            elif entry == "transposed":
                # An option of Import from clipboard, which reads it when pressed: right under it.
                menu.addAction(self._clipboard_transposed)
            else:
                icon, text, tip, call = entry
                # No icons on the Mac, as its own menus; Windows and Linux show them (CONTEXT_MENU_ICONS).
                action = menu.addAction(load_icon(icon), text) if CONTEXT_MENU_ICONS else menu.addAction(text)
                action.setToolTip(tip)
                action.triggered.connect(lambda _checked=False, c=call: c())
        return menu

    def _show_table(self, table_name: str | None = None) -> None:
        """Go to the Tables page, with *table_name* (when there is one) selected and previewed.

        What follows an import, a new table or a project opened or created:
        the person sees where the data went.
        """
        self._table_panel.reload()
        if table_name:
            self._table_panel.select_table(str(table_name))
            self._preview.set_context(self._repo, str(table_name))
        self._select_navigation("tables")

    def _on_analysis_tool_requested(self, tool: str) -> None:
        """A tool of the Analysis panel: the design of experiments, or New chart on a chart type ("chart:<type>")."""
        if tool == "doe":
            self._on_new_table_from_doe()
        elif tool.startswith("chart:"):
            self._on_new_plot_tab(chart_type=tool.split(":", 1)[1])

    def _on_new_table_from_doe(self) -> None:
        from app.dialogs.doe_dialog import DOEExperimentDialog

        try:
            dialog = DOEExperimentDialog(self._repo, parent=self)
            dialog.applied.connect(self.refresh)
            dialog.exec()
            table_name = getattr(dialog, "created_table_name", None)
            if table_name:
                self._show_table(table_name)
            else:
                self._table_panel.reload()
        except Exception as exc:
            applogger.exception("Failed to create a table from DOE: %s", exc)
            QMessageBox.warning(self, _("DOE table"), str(exc))

    def _on_new_blank_table(self) -> None:
        table_name, accepted = QInputDialog.getText(
            self,
            _("New Blank Table"),
            _("Table name:"),
            text="New_Table",
        )
        if not accepted:
            return
        raw_columns, accepted = QInputDialog.getText(
            self,
            _("New Blank Table"),
            _("Column names, separated by commas:"),
            text="Id, Value",
        )
        if not accepted:
            return

        columns = [value.strip() for value in str(raw_columns).split(",") if value.strip()]
        if not str(table_name).strip() or not columns:
            QMessageBox.warning(
                self,
                _("New Blank Table"),
                _("Enter a table name and at least one column."),
            )
            return
        if len({column.casefold() for column in columns}) != len(columns):
            QMessageBox.warning(
                self,
                _("New Blank Table"),
                _("Column names must be unique."),
            )
            return

        try:
            name = re.sub(
                r"[^A-Za-z0-9_]+", "_", str(table_name).strip()
            ).strip("_") or "New_Table"
            if name[0].isdigit():
                name = f"Table_{name}"
            base, suffix = name, 1
            while self._repo.check_if_table_exists(name):
                name = f"{base}_{suffix}"
                suffix += 1
            frame = pd.DataFrame(
                {column: pd.Series(dtype="object") for column in columns}
            )
            self._repo.import_dataframe(
                frame,
                table_name=name,
                normalize_columns=False,
            )
            self._show_table(name)
        except Exception as exc:
            applogger.exception("Failed to create blank table: %s", exc)
            QMessageBox.warning(self, _("New Blank Table"), str(exc))

    def _create_file_page(self) -> NavPanel:
        """Project, Save (and Export), then the demos and the recent projects, a tab each.

        Import is on the New table page, with the other ways to make a
        table; Project info on the Project page, with what else is about the
        open project. Every button of the panel has the same width.
        """
        page = NavPanel(self)
        page.add_buttons(_("Project"), [
            NavButton("new", self._on_new_file),
            NavButton("open", self._on_open_database),
        ], object_name="fileWorkspaceCard")
        page.add_buttons(_("Save"), [
            NavButton("save", self._on_save),
            NavButton("save_as", self._on_save_as),
            # A row of its own, with its sentence: three buttons did not fit the panel on Windows.
            NavButton("project_report", self._on_project_report, text=_("Export…"), described=True),
        ], object_name="fileSaveCard")
        self._projects_tabs = ProjectsTabs(page)
        self._recent_view = self._projects_tabs.recent
        self._projects_tabs.demos.open_requested.connect(self._on_load_demo)
        self._recent_view.open_requested.connect(self._on_open_recent)
        self._recent_view.clear_requested.connect(self._on_clear_recent)
        # The lists take whatever height the page has left.
        page.add_frame(_("Projects"), self._projects_tabs, object_name="fileProjectsCard", stretch=1)
        # One width for every button of the panel, the widest one's.
        buttons = page.findChildren(QPushButton)
        width = max(button.sizeHint().width() for button in buttons)
        for button in buttons:
            button.setFixedWidth(width)
        return page

    def _create_developer_page(self) -> NavPanel:
        """The Project page: the project's information, history, database and log.

        The Database page's content lives here since that page went; Query
        Builder moved to the Data Operations list. Each tool has a frame
        of its own with its catalogue sentence under the button. The tools for
        contributors are in dev/ChartLibreDeveloper.py.
        """
        page = NavPanel(self)
        page.add_buttons(_("Project information"), [NavButton("project_info", self._on_project_info, described=True)],
                         object_name="projectInfoCard")
        page.add_buttons(_("History"), [NavButton("project_history", self._on_project_history)],
                         object_name="projectHistoryCard")
        self._database_info_panel = DatabaseInfoPanel(self._repo, page, optimize_action=self._on_optimize_db)
        page.add_widget(self._database_info_panel)
        page.add_buttons(action_presentation("log_viewer")[1],
                         [NavButton("log_viewer", self._show_log_viewer, described=True)],
                         object_name="log_viewerCard")
        return page

    def _navigation_icon(action_id: str) -> QIcon:
        icon, _text, _tooltip = action_presentation(action_id)
        return icon

    #: Each rail row's colour, light theme then dark. A rail of grey icons
    #: all alike is read word by word; a colour per row is found by eye, and
    #: stays the same row's in every theme. The hues are the chart style's,
    #: so the window and the charts in it agree.
    _RAIL_COLOURS: ClassVar[dict[str, tuple[str, str]]] = {
        "nav_workspace": ("#5B6474", "#A9B1BD"),
        "nav_file": ("#C26A12", "#F2A65A"),
        "nav_data": ("#2F6FE4", "#6EA0FF"),
        "nav_chart_options": ("#6A4BD6", "#A893FF"),
        "nav_analysis": ("#178A58", "#4DD293"),
        "nav_guided": ("#B7791F", "#F0C04A"),
        "nav_assistant": ("#C2417A", "#F27BB0"),
        "database_info": ("#3E5A8A", "#8FA9D6"),
        "nav_help": ("#5B6474", "#A9B1BD"),
        "settings": ("#5B6474", "#A9B1BD"),
    }

    @classmethod
    def _rail_icon(cls, action_id: str) -> QIcon:
        """The rail's icon for *action_id*, drawn in that row's colour."""
        icon = cls._navigation_icon(action_id)
        colours = cls._RAIL_COLOURS.get(action_id)
        if colours is None:
            return icon
        return colored_icon(icon, colours[1] if theme_is_dark() else colours[0], 20)

    def _create_navigation(self) -> NavigationBar:
        """Create the bar and panel host from one section dictionary."""
        file_panel = self._create_file_page()
        tables_panel = self._data_page
        chart_properties_panel = self._properties_control
        analysis_panel = AnalysisPanel(self)
        # Data operations, table analyses and the DOE in one panel, as JMP's Analyze menu.
        analysis_panel.data_operation_requested.connect(self._on_data_operation_requested)
        analysis_panel.table_operation_requested.connect(self._on_table_operation_requested)
        analysis_panel.tool_requested.connect(self._on_analysis_tool_requested)
        self._rail_search = CatalogSearch(self, collapsed=True, compact=True)
        self._rail_search.entry_chosen.connect(self.open_catalog_entry)
        guided_panel = GuidedAnalysisPanel(self)
        guided_panel.entry_requested.connect(self.open_catalog_entry)
        guided_panel.question_requested.connect(self._open_catalog_entry_with_columns)
        guided_panel.wizard_requested.connect(self._open_guided_wizard)
        guided_panel.set_loader(self._guided_table)
        guided_panel.set_table(self._table_panel.current)
        self._guided_panel = guided_panel
        developer_panel = self._create_developer_page()
        # The assistant: a conversation with an AI about the project (app.ai), its provider a plugin.
        assistant_panel = self._assistant_panel = AssistantPanel(
            self, repo=lambda: self._repo,
            current_figure=lambda: getattr(self._current_chart_panel(), "figure_id", None))
        assistant_panel.project_changed.connect(self._on_assistant_changed)
        assistant_panel.settings_requested.connect(self._on_settings)
        assistant_panel.busy_changed.connect(self._on_assistant_busy)
        # The SymPy workpad: commands on the project's tables (app.workpad).
        workpad_panel = self._workpad_panel = WorkpadPanel(self, repo=lambda: self._repo)
        workpad_panel.project_changed.connect(self._on_assistant_changed)

        sections: dict[str, dict[str, NavBarItem]] = {
            "Tools": {
                "workspace": NavBarItem(
                    _("Workspace"), self._rail_icon("nav_workspace"), None,
                    _("Hide the panel"),
                ),
                "file": NavBarItem(
                    _("File"), self._rail_icon("nav_file"), file_panel,
                    _("New, open, import and save"),
                ),
                "tables": NavBarItem(
                    _("Tables"), self._rail_icon("nav_data"), tables_panel,
                    _("Show data tables"),
                ),
                "chart_properties": NavBarItem(
                    _("Chart properties"), self._rail_icon("nav_chart_options"),
                    chart_properties_panel, _("Edit the current chart"),
                ),
                "analysis": NavBarItem(
                    _("Analysis"), self._rail_icon("nav_analysis"),
                    analysis_panel, _("Every analysis and data operation, by category, as JMP's Analyze menu"),
                ),
                # After Analysis, in amber like a hint: the way in for someone
                # who does not know which analysis to choose.
                "guided": NavBarItem(
                    _("Guided analysis"), self._rail_icon("nav_guided"),
                    guided_panel, _("Find the function for what you want to find out"),
                ),
                "assistant": NavBarItem(
                    _("Assistant"), self._rail_icon("nav_assistant"), assistant_panel,
                    _("Ask an AI about the project's data - and, as Settings allow, have it run analyses or change "
                      "the data"),
                ),
                "workpad": NavBarItem(
                    _("SymPy workpad"), self._rail_icon("nav_analysis"), workpad_panel,
                    _("SymPy commands on the project's tables: get, save, plot"),
                ),
                "developer": NavBarItem(
                    _("Project"), self._rail_icon("database_info"), developer_panel,
                    _("The project's information, history, database and log"),
                ),
            },
        }
        if not IS_MACOS:
            # macOS has Help in its menu bar; elsewhere the rail is the menu.
            sections["Settings"] = {
                "help": NavBarItem(
                    _("Help"), self._rail_icon("nav_help"), None,
                    _("User manual and credits"),
                ),
            }
        recent = self._recent_items()
        if recent:
            sections = {**{k: v for k, v in sections.items() if k != "Settings"}, "Recent": recent,
                        **({"Settings": sections["Settings"]} if "Settings" in sections else {})}
        if IS_WINDOWS:
            sections.setdefault("Settings", {})["settings"] = NavBarItem(
                _("Settings"), self._rail_icon("settings"), None,
                _("Open application settings"),
            )

        header = CustomTitleBar(cast(Any, self), is_macos=True) if IS_MACOS else None
        # The pages keep the same gap from the rail as from the panel's right
        # and bottom edges - flush, every card ran into the rail and the
        # window's edge, which showed on Windows' white page (todo W-01).
        navigation = NavigationBar(
            sections, parent=self, header=header,
            # Less room above the page's first title and below its last frame off macOS:
            # there it was empty space (bugs2.pptx); macOS keeps its title strip's.
            panel_margins=(SPACING_DEFAULT, 12, SPACING_DEFAULT, SPACING_DEFAULT) if IS_MACOS
            else (SPACING_DEFAULT, 4, SPACING_DEFAULT, 4),
            panel_min_width=PANEL_MIN_WIDTH,
        )
        # The search sits in the rail under Workspace; its matches list under it while typing.
        navigation.set_row_widget("workspace", self._rail_search)
        navigation.set_hidden_panels_key("workspace")
        navigation.set_item_hidden("assistant", not assistant_settings.in_navigation())
        navigation.action_clicked.connect(self._on_navigation_action)
        navigation.panel_changed.connect(self._on_navigation_panel_changed)
        navigation.bar_hidden_changed.connect(self._on_bar_hidden_changed)
        return navigation

    def _on_navigation_action(self, key: str) -> None:
        """Handle rows that intentionally have no associated panel."""
        if key == "workspace":
            self._left_panel.set_panel_visible(False)
            return
        if key == "settings":
            self._on_settings()
            return
        if key == "help":
            self._popup_help_menu()
            return
        if key.startswith(self.RECENT_PREFIX):
            name = key[len(self.RECENT_PREFIX):]
            from app.scanners.data_operation_scanner import data_operations

            for operation in data_operations:
                if str(operation.get("name")) == name:
                    self._on_data_operation_requested(operation)
                    return

    #: The rail keys of the Recent section's rows: this prefix and the analysis' class name.
    RECENT_PREFIX = "recent:"

    def _recent_items(self) -> dict[str, NavBarItem]:
        """The Recent section's rows: the analyses opened last, as many as Settings says (none: no section)."""
        from app.scanners.data_operation_scanner import data_operations

        return {
            f"{self.RECENT_PREFIX}{operation['name']}": NavBarItem(
                _(str(operation.get("value") or operation["name"])), plugin_icon(operation), None,
                _(str(operation.get("description") or "")),
            )
            for operation in recent_analyses.operations(list(data_operations))
        }

    def _refresh_recent_section(self) -> None:
        """The Recent section again: after an analysis opened, or its count changed in Settings."""
        navigation = getattr(self, "_left_panel", None)
        if navigation is None:
            return
        sections = {k: v for k, v in navigation.sections().items() if k != "Recent"}
        recent = self._recent_items()
        if recent:
            settings_rows = sections.pop("Settings", None)
            sections["Recent"] = recent
            if settings_rows is not None:
                sections["Settings"] = settings_rows
        navigation.set_sections(sections)

    def _popup_help_menu(self) -> None:
        """The Help menu beside its rail row, opening to the right: the rail is on the left."""
        button = self._left_panel.button("help")
        if button is not None:
            self._help_menu.exec(button.mapToGlobal(button.rect().topRight()))

    def _on_navigation_panel_changed(self, key: str, panel: object) -> None:
        del panel
        if key == "file":
            self._recent_view.refresh()
        applogger.debug("Left navigation item changed to %s", key)

    def _select_navigation(self, key: str) -> None:
        """Select a panel by stable key; no page index is involved."""
        self._left_panel.select(key)

    # ------------------------------------------------------------------
    # Main layout
    # ------------------------------------------------------------------
    def _create_main_split(self) -> QSplitter:
        """Create the main horizontal splitter.

        Three things make the handle track the mouse instead of snapping:

        * the left panel's children have their implicit minimum widths removed
          (``relax_minimum_width``), so the only floor is the one chosen here -
          previously a 24-character combo held the panel at about 300 px and
          the handle simply stopped;
        * neither pane may collapse, so there is no jump to zero at the end of
          the travel;
        * the chart pane is the only stretchy one, so growing the window does
          not silently re-widen the panel the user just narrowed.
        """
        split = QSplitter(Qt.Orientation.Horizontal, self)
        # Named for the style sheets: its handle is the one drawn as a line.
        split.setObjectName("mainSplitter")
        split.setChildrenCollapsible(False)
        split.setOpaqueResize(True)
        split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        split.setMinimumSize(0, 0)
        split.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        relax_minimum_width(self._left_panel)
        # relax_minimum_width cleared every floor; NavigationBar works out its
        # own again (the panels' minimum beside the bar).
        self._left_panel.set_panel_visible(self._left_panel.panel_visible)

        # The chart pane needs the same treatment, and for the same reason:
        # whichever pane keeps a large implicit minimum wins the whole
        # negotiation, and the other one is squeezed past the minimum it asked
        # for.  Both floors are now explicit and comparable.
        relax_minimum_width(self._tabs)
        self._tabs.setMinimumWidth(CHART_PANE_MIN_WIDTH)

        # Flush: now that #chartSurfaceCard paints pure white (see
        # macos_native.qss), a margin here was just empty space with
        # nothing left to separate from - the same white on both sides
        # of it.
        # Windows rounds the pane's corners (fluent_win11.qss): the chart is
        # inset a little so its square edges stay inside the curve.
        self._chart_surface = CardFrame(
            self, "chartSurfaceCard", margins=(3, 1, 3, 3) if IS_WINDOWS else (0, 0, 0, 0),
        )
        self._chart_surface.setProperty("elevated", True)
        self._chart_surface.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        chart_layout = self._chart_surface.layout()
        if not isinstance(chart_layout, QBoxLayout):
            raise RuntimeError("CardFrame did not create a box layout")
        self._jump_bar = ChartJumpBar(self._chart_surface)
        self._jump_bar.previous_requested.connect(lambda: self._step_chart(-1))
        self._jump_bar.next_requested.connect(lambda: self._step_chart(+1))
        self._jump_bar.chart_chosen.connect(self._tabs.setCurrentIndex)
        chart_layout.addWidget(self._jump_bar, 0)
        chart_layout.addWidget(self._tabs, 1)

        split.addWidget(self._left_panel)
        split.addWidget(self._chart_surface)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        # Measured from the rail outwards, not a flat 360: the left pane
        # holds the navigation rail *and* the panel beside it, and the rail
        # is 200px wide on macOS against 132 on Windows. A fixed 360 left
        # the macOS panel 160px of usable width - the table list and the
        # data-operation panels were unreadably narrow in it - while
        # giving Windows 228 for the same panel.
        split.setSizes([self._left_panel.visible_bar_width + PANEL_DEFAULT_WIDTH, 840])

        return split

    # ------------------------------------------------------------------
    # Database switching / reload
    # ------------------------------------------------------------------

    def _reload_tabs(self, select_figure_id: int | None = None) -> None:
        """Reload chart tabs from the current repository.

        *select_figure_id* makes that figure's tab the current one - a chart
        just created is shown, not left behind the one that was open.
        """
        applogger.debug("Reloading chart tabs")
        current_index = self._tabs.currentIndex()

        self._tabs.blockSignals(True)
        try:
            self._tabs.clear()
            for fig_id, name in self._repo.load_figures_from_db():
                # Only the tab that ends up current needs its first render
                # before the window is interactive - see
                # _update_properties_for_current_chart's ensure_rendered()
                # call, which renders whichever tab that turns out to be.
                # A project with many figures otherwise pays for a full
                # render of every one of them just to show the first.
                panel = ChartPanel(self._repo, fig_id, parent=self._tabs, defer_render=True)
                panel.setMinimumSize(0, 0)
                panel.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Expanding,
                )
                panel.delete_requested.connect(self._on_chart_panel_deleted)
                panel.figure_edited.connect(self._refresh_undo_item)
                # A reference line or annotation dropped from the chart's own
                # context menu is written straight to the axis descriptor -
                # reload the property pages so the Overlay panel's Lines and
                # Annotations tables show it without a chart reselection.
                panel.figure_edited.connect(self._update_properties_for_current_chart)
                # Clicking a point reports it in the status bar, which is the
                # only surface in the window that can carry a transient line
                # without moving anything else. It times out rather than
                # staying: the bar is shared with the logger, and a readout
                # that never cleared would sit there describing a chart the
                # user has since left.
                panel.selection_changed.connect(
                    lambda text: self.statusBar().showMessage(
                        text, CHART_SELECTION_TIMEOUT_MS
                    )
                )
                panel.chart_object_clicked.connect(
                    lambda hit, source=panel: self._on_chart_object_clicked(source, hit)
                )
                panel.rows_selected.connect(self._on_rows_selected)
                panel.selected_only_changed.connect(self._on_selected_only_changed)
                panel.functions_trusted.connect(self._reload_function_charts)
                panel.next_step_requested.connect(self._open_next_step_url)
                self._tabs.addTab(panel, name)
        finally:
            self._tabs.blockSignals(False)

        if self._tabs.count() == 0:
            self._sync_chart_navigation()
            self._clear_property_widgets()
            return

        selected = self._tab_index_of_figure(select_figure_id) if select_figure_id is not None else -1
        if selected >= 0:
            self._tabs.setCurrentIndex(selected)
        elif 0 <= current_index < self._tabs.count():
            self._tabs.setCurrentIndex(current_index)
        else:
            self._tabs.setCurrentIndex(0)

        self._sync_chart_navigation()
        self._update_properties_for_current_chart()

    def _on_chart_panel_deleted(self, figure_id: int) -> None:
        """Refresh chart tabs after a chart is deleted from a local panel menu."""
        applogger.info("Chart deleted from panel menu (figure_id=%s)", figure_id)
        self._reload_tabs()
        # Deleting the panel recorded an undo entry (ChartPanel.close); make
        # the menu say so now rather than at a show-time signal the native
        # macOS menu bar never sends.
        self._refresh_undo_item()

    def _on_chart_tab_changed(self, index: int) -> None:
        """Rebind the properties control when the selected chart tab changes."""
        applogger.debug("Chart tab changed to index %s", index)
        self._sync_chart_navigation(rebuild=False)
        self._update_properties_for_current_chart()

    def _sync_chart_navigation(self, *, rebuild: bool = True) -> None:
        """Keep the chart jump bar above the chart in step with the open charts."""
        del rebuild
        jump = getattr(self, "_jump_bar", None)
        if jump is None:
            return
        names = [self._tabs.tabText(i) for i in range(self._tabs.count())]
        jump.set_charts(names, self._tabs.currentIndex())

    def _tab_index_of_figure(self, figure_id: int) -> int:
        """The tab showing figure *figure_id*, or -1."""
        for index in range(self._tabs.count()):
            widget = self._tabs.widget(index)
            if isinstance(widget, ChartPanel) and int(widget.figure_id) == int(figure_id):
                return index
        return -1

    def _step_chart(self, step: int) -> None:
        """Show the chart *step* rows above (-1) or below (+1) the current one."""
        target = self._tabs.currentIndex() + step
        if 0 <= target < self._tabs.count():
            self._tabs.setCurrentIndex(target)

    # Figure payload keys handled outside the generic copy below.
    _FIGURE_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"name", "layout"})

    # Payload keys that are transport, not axis options: they are either
    # addressing (axis_id) or handled explicitly below.
    _AXIS_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"axis_id", "renderer"})

    # ------------------------------------------------------------------
    # Table panel actions
    # ------------------------------------------------------------------
    def _on_table_selected(self, table: str) -> None:
        """Update the preview panel, and the guided analysis, when a table is selected."""
        applogger.debug("Selected table: %s", table)
        self._preview.set_context(self._repo, table)
        guided = getattr(self, "_guided_panel", None)
        if guided is not None:
            guided.set_table(table)

    def _guided_table(self, table: str) -> tuple[Any, dict[str, str]]:
        """A table as the analysis windows read it - visible rows, no application columns - and its SQL types."""
        from app.plugin_api.data_operations.roles_base import APPLICATION_COLUMNS

        frame = self._repo.table_frame(table)
        if "Hide" in frame.columns:
            frame = frame[to_numbers(column_of(frame, "Hide")).fillna(0) == 0]
        frame = frame.drop(columns=[c for c in frame.columns if c in APPLICATION_COLUMNS])
        return frame, {str(row[1]): str(row[2] or "") for row in self._repo.table_info(table)}


    # ------------------------------------------------------------------
    # App menu actions
    # ------------------------------------------------------------------


    def _on_chart_from_columns(self, table: str, columns: list) -> None:
        """New plot on *table*, with the preview's selected columns as x, y, z."""
        self._on_new_plot_tab(table=table, columns=[str(c) for c in columns])

    def _on_histogram_from_column(self, source: str, column: str) -> None:
        """A new figure with *column*'s histogram, its statistics in the notes.

        Straight from the preview, without the New plot window: one column
        has only one sensible chart, and the figure is shown at once.
        """
        if self._repo is None:
            return
        try:
            sql = self._repo.column_series_sql(source, column)
            values = self._repo.query_df(sql)["value"]
        except (DatabaseError, ValueError, KeyError) as exc:
            show_message(self, "chart.histogram_failed", title=_("Histogram"), reason=str(exc))
            return
        if column_summary.numeric_values(values).size == 0:
            show_message(self, "chart.histogram_no_numbers", title=_("Histogram"), column=column)
            return
        try:
            figure_id = int(self._repo.create_figure_descriptor(name=column))
            axis_id = int(self._repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=0, chart_type="Histogram", title=column,
                x_label=column, y_label=_("Count"), options={},
            ))
            self._repo.create_series_descriptor(
                axis_id=axis_id, series_index=0, name=column, sql_query=sql, roles={"value": "value"}, style={},
            )
        except DatabaseError as exc:
            show_message(self, "chart.histogram_failed", title=_("Histogram"), reason=str(exc))
            return
        self._reload_tabs(select_figure_id=figure_id)
        index = self._tab_index_of_figure(figure_id)
        panel = self._tabs.widget(index) if index >= 0 else None
        if isinstance(panel, ChartPanel):
            panel.set_notes_html(column_summary.summary_html(column, values))
        applogger.info("Histogram of %s.%s made as figure %d.", source, column, figure_id)

    def _open_guided_wizard(self, table: str | None = None) -> None:
        """Help me choose (G-02): the wizard on *table* - the selected one by default."""
        from app.dialogs.guided_wizard_dialog import GuidedWizardDialog
        from app.plugin_api.data_operations import table_profile

        table = table or self._table_panel.current
        if not table or self._repo.get_data_source(table) is None:
            show_message(self, "catalog.select_a_table", title=_("Help me choose"))
            return
        frame, declared = self._guided_table(table)
        dialog = GuidedWizardDialog(table, frame, table_profile.profile(table, frame, declared), parent=self)
        dialog.chart_requested.connect(self._create_guided_chart)
        dialog.question_requested.connect(self._open_catalog_entry_with_columns)
        dialog.entry_requested.connect(self.open_catalog_entry)
        self._guided_wizard = dialog
        dialog.exec()

    def _create_guided_chart(self, table: str, idea: Any) -> int | None:
        """The wizard's chart, made at once and shown; its figure id."""
        from app.plugin_api.data_operations import guided_plan

        try:
            figure_id = guided_plan.make_chart(self._repo, table, idea)
        except DatabaseError as exc:
            show_message(self, "chart.create_failed", title=_(idea.chart_type), reason=str(exc))
            return None
        self._reload_tabs(select_figure_id=figure_id)
        self._refresh_undo_item()
        applogger.info("Guided chart %s of %s made as figure %d.", idea.chart_type, table, figure_id)
        return figure_id

    def _on_new_plot_tab(
        self,
        icon: QIcon | None = None,
        *,
        table: str | None = None,
        columns: list[str] | None = None,
        chart_type: str | None = None,
    ) -> None:
        """Create a chart tab, optionally using the operation-list icon.

        ``TableListPanel`` invokes this callback without arguments, while the
        data-operation page supplies the Plot operation icon.
        """
        panel = self._current_chart_panel()
        dialog = NewPlotTabDialog(
            self._repo,
            current_figure_id=(
                int(panel.figure_id) if panel is not None else None
            ),
            current_table=table or self._table_panel.current,
            preferred_columns=columns,
            parent=self,
        )
        if chart_type:
            dialog.select_chart_type(chart_type)

        if icon is None or icon.isNull():
            icon = plugin_icon(plot_operation())

        if not icon.isNull():
            dialog.setWindowIcon(icon)

        if dialog.exec():
            self._table_panel.reload()
            # The chart just made becomes the current one - a new figure, or
            # the figure the new axis went into.
            result = dialog.chart_result
            self._reload_tabs(select_figure_id=result.figure_id if result is not None else None)

    # ------------------------------------------------------------------
    # Search (the field in the rail, Cmd/Ctrl+K)
    # ------------------------------------------------------------------

    def focus_search(self) -> None:
        """Put the cursor in the rail's search field, expanding the rail when it shows icons only."""
        if self._left_panel.compact:
            self._left_panel.set_compact(False)
        self._rail_search.field.setFocus()
        self._rail_search.field.selectAll()

    def _open_catalog_entry_with_columns(self, entry, roles: dict, kinds: dict) -> None:
        """Open what the guided analysis chose with its columns: cast into an analysis, or a chart on them."""
        from app.analysis import catalog

        if entry.kind == catalog.ANALYSIS:
            self._on_table_operation_requested(entry.payload, roles=roles, kinds=kinds)
        elif entry.kind == catalog.SERIES and roles.get("response"):
            self._open_data_operation_on_table(entry.payload, self._table_panel.current, roles)
        elif entry.kind == catalog.CHART:
            columns = [*roles.get("_x", [])[:1], *roles.get("response", [])]
            self._on_new_plot_tab(chart_type=str(entry.payload), columns=columns or None)
        else:
            self.open_catalog_entry(entry)

    def open_catalog_entry(self, entry) -> None:
        """Open what a search found: an analysis, a data operation, a chart type, a table tool."""
        from app.analysis import catalog

        if entry.kind == catalog.ANALYSIS:
            self._on_table_operation_requested(entry.payload)
        elif entry.kind == catalog.SERIES:
            self._on_data_operation_requested(entry.payload)
        elif entry.kind == catalog.CHART:
            self._on_new_plot_tab(chart_type=str(entry.payload))
        elif entry.kind == catalog.TOOL:
            if entry.payload == "edit_table":
                if not self._table_panel.edit_selected_table():
                    show_message(self, "catalog.select_a_table", name=_(entry.name))
                return
            handlers = {
                "doe": self._on_new_table_from_doe,
                "import": self._on_import_data,
                "import_web": self._on_import_web,
                "import_clipboard": self._on_import_clipboard,
                "import_database": self._on_database_table_import,
                "query": self._on_create_query_table,
                "new_table": self._on_new_blank_table,
            }
            handler = handlers.get(str(entry.payload))
            if handler is not None:
                handler()

    def refresh(self, *, reload_preview: bool = True) -> None:
        """Bring the chart, the tables and the properties back in step with the project.

        After a data operation (Preview, Apply, or the rollback of a
        Preview), a table edited or duplicated, an Undo. *reload_preview*
        is False when the data preview asked for this itself, having just
        reloaded.
        """
        panel = self._current_chart_panel()
        if panel is not None:
            panel.reload()
        try:
            # Always: a duplicated or grouped table must appear in the list
            # even when no chart is open.
            self._table_panel.reload()
            if reload_preview and self._table_panel.current:
                self._preview.set_context(self._repo, str(self._table_panel.current))
            guided = getattr(self, "_guided_panel", None)
            if guided is not None:
                guided.set_table(self._table_panel.current)
        except Exception:
            applogger.exception("Failed to refresh the data panes.")
        self._update_properties_for_current_chart()
        self._refresh_undo_item()

    def _on_table_operation_requested(
        self, operation: dict, *, roles: dict | None = None, kinds: dict | None = None
    ) -> None:
        """Open a table operation (Fit Model...) on the table selected in the Tables panel.

        *roles* and *kinds* cast its columns beforehand: a question of the
        guided analysis opens its answer with the columns it asked about.
        """
        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error("Could not load table operation class: %r", operation.get("name"))
            return
        current = self._table_panel.current
        panel = self._current_chart_panel()
        dialog = dialog_class(repo=self._repo, table=str(current) if current else None,
                              figure_id=int(panel.figure_id) if panel is not None else None, parent=self)
        if roles:
            roles = dict(roles)
            chosen_x = roles.pop("_x", None)
            dialog.roles_widget.set_casting(roles, kinds)
            if chosen_x:
                dialog.cast_explanatory(list(chosen_x))
        dialog.applied.connect(self.refresh)
        # Preview draws while the dialog is open: its figure shown at once, dropped when it is rolled back.
        dialog.preview_drawn.connect(self._on_analysis_preview)
        dialog.exec()
        figures = list(dialog.created_figure_ids)
        self._table_panel.reload()
        if figures:
            self._reload_tabs(select_figure_id=figures[0])
        step = getattr(dialog, "next_request", None)
        if step is not None:
            self._open_next_step(step)

    def _on_analysis_preview(self, figure_id: int) -> None:
        """An analysis' Preview drew *figure_id* (shown), or rolled back (-1: the tabs as they were)."""
        if figure_id >= 0:
            self._reload_tabs(select_figure_id=int(figure_id))
        else:
            self._reload_tabs()

    def _open_next_step_url(self, url: str) -> None:
        """A "What next?" link clicked in a chart's notes."""
        from app.plugin_api.data_operations.next_steps import parse_step

        step = parse_step(url)
        if step is not None:
            self._open_next_step(step)

    def _open_next_step(self, step) -> None:
        """Open the analysis a "What next?" link asks for, on its table, the columns cast (G-02).

        A data operation's step (it names a figure) opens on that figure instead.
        """
        from app.scanners.data_operation_scanner import (
            data_operations,
            import_class_from_file,
        )

        if step.figure is not None:
            data = next((entry for entry in data_operations if entry.get("value") == step.operation), None)
            dialog_class = import_class_from_file(data) if data is not None else None
            if dialog_class is None:
                applogger.warning("The next step's analysis %r is not installed.", step.operation)
                return
            self._open_data_operation(dialog_class, plugin_icon(data), figure_id=step.figure)
            return
        from app.scanners.data_operation_scanner import reads_roles

        operation = next((entry for entry in data_operations if entry.get("value") == step.operation), None)
        if operation is None:
            applogger.warning("The next step's analysis %r is not installed.", step.operation)
            return
        if step.table not in set(self._repo.list_user_tables()["Table"].astype(str)):
            applogger.warning("The next step's table %r is not in the project any more.", step.table)
            return
        self._table_panel.select_table(step.table)
        if reads_roles(operation):
            self._on_table_operation_requested(operation, roles=step.roles, kinds=step.kinds)
        else:
            self._open_data_operation_on_table(operation, step.table, step.roles)

    def _open_data_operation_on_table(self, operation: dict, table: str, roles: Mapping[str, list[str]]) -> None:
        """Open a data operation on a table's columns: its response and its time, cast as its Data page reads them."""
        from app.scanners.data_operation_scanner import import_class_from_file

        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error("Could not load data operation class: %r", operation.get("name"))
            return
        self._open_data_operation(dialog_class, plugin_icon(operation), table_roles=(table, roles))

    def _scratch_figure(self) -> int | None:
        """An empty chart to open a data operation on when the project has none.

        The operations are opened on a chart - their series selector reads its
        axes - but they can also read a table's columns, so with only tables
        in the project they still have to open. The figure is deleted again
        if nothing comes of it (_drop_scratch_figure).
        """
        from app.scanners.axis_renderer_scanner import get_renderer, renderers

        chart_type = "Scatter Plot" if get_renderer("Scatter Plot") else (renderers[0]["value"] if renderers else None)
        if chart_type is None:
            applogger.error("No chart type is installed: a data operation cannot be opened without a chart.")
            return None
        figure_id = int(self._repo.create_figure_descriptor(name=_("Chart")))
        self._repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type=str(chart_type),
            title="", x_label="", y_label="", options={},
        )
        return figure_id

    def _drop_scratch_figure(self, figure_id: int) -> bool:
        """Delete the scratch figure unless the operation drew on it; True when deleted."""
        descriptor = self._repo.load_figure_descriptor(figure_id)
        if descriptor is not None and any(axis.series for axis in descriptor.axes):
            return False
        self._repo.delete_figure(figure_id)
        return True

    def _open_data_operation(
        self,
        dialog_class: type,
        icon: QIcon | None = None,
        *,
        figure_id: int | None = None,
        table_roles: tuple[str, Mapping[str, list[str]]] | None = None,
        prepare: Callable[[Any], None] | None = None,
    ) -> None:
        """Open one runtime data-operation dialog on the current chart, or on a scratch one.

        *prepare*, given the dialog before it is shown: what it opens on (Functions on a stored function).

        With no chart in the project the operation opens on an empty chart that
        is kept only if the operation drew something (or reported) on it. *figure_id*
        names the chart instead (a "What next?" link to a data operation).
        """
        panel = self._current_chart_panel() if figure_id is None else None
        scratch: int | None = None
        if figure_id is None:
            if panel is None:
                scratch = self._scratch_figure()
                if scratch is None:
                    return
                figure_id = scratch
            else:
                figure_id = panel.figure_id

        dialog = dialog_class(
            repo=self._repo,
            figure_id=figure_id,
            parent=self,
        )

        if icon is not None and not icon.isNull():
            dialog.setWindowIcon(icon)

        if table_roles is not None:
            self._cast_table_source(dialog, *table_roles)
        if prepare is not None:
            prepare(dialog)
        dialog.applied.connect(self.refresh)
        # A Preview drawn on a new figure: its tab shown while the dialog is open.
        tabs_rebuilt: list[bool] = []
        if hasattr(dialog, "preview_drawn"):  # a dialog of a plugin's own may not have it
            dialog.preview_drawn.connect(lambda figure, rebuilt=tabs_rebuilt: (
                rebuilt.append(True), self._on_analysis_preview(figure)))
        # Bind results to the panel active when the dialog was opened - or, on a
        # scratch chart, to the figure once its tab exists.
        published: list[str] = []
        if panel is not None:
            # Found again by its figure: a Preview on a new figure rebuilds the tabs, and the panel with them.
            dialog.results_published.connect(
                lambda markup, figure=figure_id: self._notes_to_figure(figure, markup)
            )
        else:
            dialog.results_published.connect(published.append)
        figures_before = {figure_id for figure_id, _name in self._repo.load_figures_from_db()}
        dialog.exec()
        if tabs_rebuilt:
            # The tabs were rebuilt under the dialog: the panel it opened on is a new one now.
            self._reload_tabs(select_figure_id=figure_id)
            panel = self._current_chart_panel()
        if panel is not None:
            panel.reload()
        elif scratch is None or published or not self._drop_scratch_figure(figure_id):
            # Kept: it is a chart of the operation's now.
            self._reload_tabs(select_figure_id=figure_id)
            target = self._current_chart_panel()
            for markup in published:
                if target is not None:
                    target.set_notes_html(markup, append=True)
        self._table_panel.reload()

        # An operation may have created a figure of its own, which is a new
        # chart tab rather than a change to this one - reloading only the
        # panel left it invisible until the next restart. But _reload_tabs()
        # rebuilds every tab from scratch - a full ChartPanel, a full render,
        # for every figure in the database - which is what made every Apply
        # redraw every chart regardless of how many the operation actually
        # touched. Only worth paying for when a figure was actually added;
        # the panel.reload() above already covers the ordinary case.
        existing = {figure_id for figure_id, _name in self._repo.load_figures_from_db()}
        created = existing - figures_before
        if created:
            # Shown: the figure the operation made is what its Apply produced.
            self._reload_tabs(select_figure_id=max(created))
        elif scratch is not None and scratch not in existing:
            self._reload_tabs()
        self._update_properties_for_current_chart()
        step = getattr(dialog, "next_request", None)
        if step is not None:
            self._open_next_step(step)

    @staticmethod
    def _cast_table_source(dialog: Any, table: str, roles: Mapping[str, list[str]]) -> None:
        """Data from the table: its response is the Y, its time the X (else the row number)."""
        source = getattr(dialog, "table_source", None)
        if source is None or not roles.get("response"):
            return
        dialog.source_table_radio.setChecked(True)
        source.set_spec(table, (roles.get("time") or [""])[0], [roles["response"][0]])

    def _on_credits(self) -> None:
        """Show who made this and what it is made of."""
        CreditsDialog(parent=self).exec()

    def _on_user_manual(self) -> None:
        """Open the user manual PDF in the system's default viewer."""
        if not USER_MANUAL_PATH.is_file():
            show_message(self, "manual.unavailable", path=str(USER_MANUAL_PATH))
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(USER_MANUAL_PATH)))
        if not opened:
            show_message(self, "manual.unavailable", path=str(USER_MANUAL_PATH))

    def _on_query_builder(self) -> None:
        """Open the query builder and refresh the source lists afterwards."""
        dialog = QueryBuilderDialog(self._repo, parent=self)
        dialog.exec()
        self._table_panel.reload()

    def _on_database_info(self) -> None:
        """Show the database overview, at the top of the Project page (the
        Database menu lands here - see _create_developer_page)."""
        self._select_navigation("developer")

    def _on_copy_chart(self) -> None:
        """Copy the current chart tab's figure to the clipboard.

        A no-op when the current tab is not a chart - decided here, on
        click, rather than by disabling the menu item: the native macOS menu
        bar caches each item's enabled state from the last full rebuild
        (see _refresh_undo_item) and rebuilding the whole bar on every tab
        switch just to keep one item current is not worth it for an action
        that already knows to do nothing when there is nothing to copy.
        """
        panel = self._current_chart_panel()
        if panel is not None:
            panel.copy_chart_to_clipboard()

    def edit_stored_function(self, name: str) -> None:
        """Tables panel > Edit function...: the Functions operation, open on the stored function *name*."""
        from app.scanners.data_operation_scanner import data_operations

        entry = next((op for op in data_operations if op.get("name") == "SeriesFunctionDialog"), None)
        dialog_class = import_class_from_file(entry) if entry is not None else None
        if dialog_class is None:
            applogger.error("The Functions operation is not installed: %r cannot be edited.", name)
            return
        self._open_data_operation(dialog_class, plugin_icon(entry), prepare=lambda dialog: dialog.edit_stored(name))

    def _on_zoom(self) -> None:
        """Toggle the window between its normal and maximized size.

        The nearest cross-platform equivalent to clicking a Mac window's
        green Zoom button, which has no direct Qt API of its own.
        """
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_rows_selected(self, tables: list[str]) -> None:
        """Rows of *tables* were marked Selected (a box on a chart, the preview): every chart and the preview show them (R-07).

        A chart of a table that shows only its Selected rows draws again: the selection is what it shows."""
        for index in range(self._tabs.count()):
            panel = self._tabs.widget(index)
            if isinstance(panel, ChartPanel):
                if set(tables) & set(panel.selected_only_tables()):
                    panel.reload()
                else:
                    panel.show_selected_rows(list(tables))
        self._preview.show_selected_rows(list(tables))

    def _on_selected_only_changed(self, tables: list[str]) -> None:
        """Show only Selected switched for *tables*: every chart of them draws again (R-07)."""
        for index in range(self._tabs.count()):
            panel = self._tabs.widget(index)
            if isinstance(panel, ChartPanel) and set(tables) & panel.row_tables():
                panel.reload()

    def _reload_function_charts(self) -> None:
        """The project's functions trusted: every chart of one draws again (todo ID47)."""
        for index in range(self._tabs.count()):
            panel = self._tabs.widget(index)
            if isinstance(panel, ChartPanel) and panel.drawn_functions():
                panel.reload()

    def _notes_to_figure(self, figure_id: int, markup: str) -> None:
        """Append *markup* to the notes of the chart tab of *figure_id*, if it is open."""
        for index in range(self._tabs.count()):
            widget = self._tabs.widget(index)
            if isinstance(widget, ChartPanel) and widget.figure_id == figure_id:
                widget.set_notes_html(markup, append=True)
                return

    def _current_chart_panel(self) -> ChartPanel | None:
        """Return the currently selected chart panel, if any."""
        widget = self._tabs.currentWidget()
        if isinstance(widget, ChartPanel):
            return widget
        return None

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist the window layout, then collect resources and close."""
        applogger.info("Closing main window")
        try:
            self._remember_layout()
            gc.collect()
        finally:
            super().closeEvent(event)

    # ------------------------------------------------------------------
    # Window layout persistence
    # ------------------------------------------------------------------
    def _remember_layout(self) -> None:
        """Write size, position and splitter sizes to config.json.

        Saved on close rather than on every resize: a splitter drag emits
        hundreds of events and each one would rewrite the file.
        """
        save_window_geometry(self, STATE_KEY)
        set_section(
            SPLITTERS_SECTION,
            {
                "main": self._main_split.sizes(),
                "data_page": self._data_split.sizes(),
            },
        )

    def _restore_layout(self) -> None:
        """Re-apply the saved size, position and splitter sizes.

        A stale entry - a splitter that has gained a pane since, a window saved
        on a monitor that is no longer attached - is ignored by the helpers
        rather than applied blindly.
        """
        restore_window_geometry(self, STATE_KEY)
        sizes = get_section(SPLITTERS_SECTION)
        for splitter, key in ((self._main_split, "main"), (self._data_split, "data_page")):
            saved = sizes.get(key)
            if isinstance(saved, list) and len(saved) == splitter.count():
                splitter.setSizes([int(value) for value in saved])

    # ==================================================================
    # Menus: the menu bar and its actions
    # ==================================================================

    #: Actions moved into the real macOS menu bar via QAction.MenuRole -
    #: Preferences and About are relocated by Cocoa itself into the
    #: application menu next to the apple, wherever they were declared, so
    #: they read here exactly the way they read in the rail's popup menu.
    _MACOS_MENU_ROLES: ClassVar[dict[str, QAction.MenuRole]] = {
        "settings": QAction.MenuRole.PreferencesRole,
        "credits": QAction.MenuRole.AboutRole,
    }

    def _app_menu_items(self) -> list[tuple[str, list[MenuItem | None]]]:
        """Return the app menu's contents, grouped the way macOS expects.

        One list of (title, items) groups, shared by both places the menu
        appears: flattened into one popup for the activity rail's "Menu"
        button and every non-Mac platform (see _flatten_menu_groups), or
        built into one QMenu per group for the real macOS menu bar (see
        _build_macos_menu_bar). One source means the two cannot drift apart
        the way a hand-kept second copy would.
        """
        return [
            (
                _("File"),
                [
                    action_menu_item("new", self._on_new_file),
                    action_menu_item("open", self._on_open_database),
                    self._recent_databases_item(),
                    action_menu_item("import", self._on_import_data),
                    None,
                    action_menu_item("save", self._on_save, shortcut="Ctrl+S"),
                    action_menu_item("save_as", self._on_save_as),
                    None,
                    action_menu_item("project_report", self._on_project_report),
                ],
            ),
            (
                _("Edit"),
                [
                    self._undo_item(),
                    None,
                    action_menu_item("copy", self._on_copy_chart),
                ],
            ),
            (
                _("Database"),
                [
                    action_menu_item("query_builder", self._on_query_builder),
                    action_menu_item("optimize_db", self._on_optimize_db),
                    None,
                    action_menu_item("database_info", self._on_database_info),
                ],
            ),
            (
                _("Help"),
                [
                    # Only on macOS: PreferencesRole (_MACOS_MENU_ROLES) is
                    # what actually pulls this out into the native apple
                    # menu there, so it has to be declared somewhere for
                    # Cocoa to relocate - "Help" is as good a place as any
                    # a user never actually sees it listed under. Everywhere
                    # else it would just be a second, redundant way to reach
                    # what the rail's own dedicated Settings tile already
                    # does one click away.
                    *(
                        [action_menu_item("settings", self._on_settings)]
                        if IS_MACOS else []
                    ),
                    # Elsewhere the rail's Search row (below Workspace) is the way.
                    *(
                        [action_menu_item("search_functions", self.focus_search)]
                        if IS_MACOS else []
                    ),
                    action_menu_item("user_manual", self._on_user_manual),
                    None,
                    action_menu_item("credits", self._on_credits),
                ],
            ),
        ]

    @staticmethod
    def _flatten_menu_groups(
        groups: list[tuple[str, list[MenuItem | None]]]
    ) -> list[MenuItem | None]:
        """One flat list for the popup: every group's items, a separator
        between groups - the shape the popup already had before the real
        macOS menu bar split it into File/Edit/Database/Help."""
        flat: list[MenuItem | None] = []
        for index, (_title, items) in enumerate(groups):
            if index:
                flat.append(None)
            flat.extend(items)
        return flat

    def _undo_item(self) -> MenuItem:
        """The Undo entry, naming the change it will take back.

        "Undo" alone would not say whether it is about to bring back a
        column, a table or a chart - and this undoes one recorded action
        against the database, not the last thing the user touched
        anywhere. Naming it is what keeps the two from being confused.
        """
        entries = self._repo.undo_entries() if self._repo is not None else []
        latest = entries[0] if entries else None
        _icon, label, tooltip = action_presentation("undo")
        text = (
            _("Undo: {what}").format(what=latest.label)
            if latest is not None
            else label
        )
        return MenuItem(
            text=text,
            tooltip=tooltip,
            icon="undo",
            shortcut="Ctrl+Z",
            callback=self._on_undo,
            action_id="undo",
            enabled=latest is not None,
        )

    def _undo_actions(self) -> list[QAction]:
        """Every Undo entry on screen: the rail's popup, and the macOS bar."""
        menus: list[QMenu | None] = [self._app_menu]
        if IS_MACOS:
            menus.extend(
                menu
                for action in self.menuBar().actions()
                if isinstance(menu := action.menu(), QMenu)
            )

        found: list[QAction] = []
        for menu in menus:
            if menu is None:
                continue
            try:
                actions = menu.actions()
            except RuntimeError:
                # Rebuilding the macOS menu bar deletes the QMenu behind the
                # File entry, and a shell of it can outlive that on the Python
                # side. Nothing to refresh in a menu that is already gone.
                continue
            found.extend(
                action for action in actions if str(action.data() or "") == "undo"
            )
        return found

    def _undo_item_state(self) -> tuple[str, bool]:
        """The Undo entry's text and whether it should be enabled, read fresh."""
        entries = self._repo.undo_entries() if self._repo is not None else []
        latest = entries[0] if entries else None
        _icon, label, _tooltip = action_presentation("undo")
        text = (
            _("Undo: {what}").format(what=latest.label)
            if latest is not None
            else label
        )
        return text, latest is not None

    def _sync_undo_item(self) -> None:
        """Push the current Undo state onto the QAction objects on screen.

        Enough on its own for the rail's popup, which is a live QMenu and is
        wired to call this from ``aboutToShow``. Not enough for the native
        macOS menu bar - see :meth:`_refresh_undo_item`.
        """
        text, enabled = self._undo_item_state()
        for action in self._undo_actions():
            action.setText(text)
            action.setEnabled(enabled)

    def _refresh_undo_item(self) -> None:
        """Bring the Undo entry up to date after the stack changed.

        Called straight after anything that records or consumes an undo entry
        - the few UI refresh points every change funnels through
        (_snapshot_descriptors, refresh, _on_chart_panel_deleted,
        ChartPanel.figure_edited) and _on_undo - not every handler.

        The native macOS menu bar makes this more than a property poke: it
        caches each item's text and enabled state from the last time the menu
        was built, never emits ``aboutToShow`` for its items, and so never
        re-reads the QAction - _sync_undo_item's changes simply do not land
        there. A full rebuild does land, and is the same hammer the
        Open-recent list already swings on every change (_build_app_menu).
        """
        self._sync_undo_item()
        if IS_MACOS:
            self._build_app_menu()

    def _recent_databases_item(self) -> MenuItem:
        """The Open recent submenu, built from user.json's own list.

        Every entry names one file, and the tooltip carries the folder it
        is in: two projects called "analysis.dhub" in different places are
        the normal case, and a menu of identical names is a menu of
        guesses. Disabled rather than hidden when the list is empty, so
        the menu keeps its shape between the first and the second launch.
        """
        recent = get_recent_databases()
        items: list[MenuItem | None] = [
            MenuItem(
                text=path.name,
                tooltip=str(path.parent),
                icon="open",
                callback=partial(self._on_open_recent, path),
            )
            for path in recent
        ]
        if items:
            items.append(None)
            items.append(
                MenuItem(text=_("Clear menu"), callback=self._on_clear_recent)
            )

        return MenuItem(
            text=_("Open recent"),
            icon="open",
            submenu=items,
            enabled=bool(recent),
        )

    def _build_app_menu(self) -> None:
        """Create the app menu, and place it where each platform expects it.

        Everywhere else this is the popup the activity rail's "Menu" button
        opens - there is no equivalent slot to move it into. On macOS it is
        a real QMenuBar instead, which is what turns the menu bar next to the
        apple from the bare running-process name into an actual menu: with no
        QMenuBar at all, Cocoa still draws that bar, showing only the
        process's own name (here, "Python", since this is not a signed .app
        bundle) and a default Quit.

        Rebuilt wholesale on every call - after Settings, since language and
        theme are what changes underneath these, and after every undo-stack
        change on macOS, whose native menu bar only reads the item list at
        build time - rather than patched in place, which is simpler and is
        exactly what already happened when this was only ever the popup.
        """
        previous = getattr(self, "_app_menu", None)
        groups = self._app_menu_items()
        # No self._file_menu: it only ever existed for the rail's own File
        # popup button, which nav_file (a page now, not a popup) replaced.
        self._help_menu = create_menu(self, list(groups[-1][1]))
        self._help_menu.setTitle(_("Help & About"))
        # The rail's Help tile holds whichever menu object existed when it
        # was built, so a rebuild has to hand it the new one - otherwise
        # the tile keeps popping up the menu from before the language or
        # the undo stack changed. Guarded because the first build runs
        # before the rail exists.
        self._app_menu = create_menu(self, self._flatten_menu_groups(groups))
        if previous is not None and IS_MACOS:
            # It is parented to this window, so replacing the attribute is not
            # enough to free it - and on macOS this runs on every undo-stack
            # change, often enough for the leak to matter. Only macOS: off it,
            # the activity-rail button still holds this exact object.
            previous.deleteLater()
        # The live popup can just re-read on open; _sync, not _refresh, so it
        # never triggers the macOS menu-bar rebuild from a show handler.
        self._app_menu.aboutToShow.connect(self._sync_undo_item)
        if not IS_MACOS:
            # Off macOS the menu is a popup from the rail, shown only while
            # open: an action that lives only in it has no visible widget to
            # carry its shortcut, so Ctrl+Z, Ctrl+S and the rest did nothing
            # with the popup closed (UI-07). The window carries them, as the
            # macOS hidden menus' actions are carried - the old ones dropped
            # first when the menu is rebuilt.
            for action in getattr(self, "_popup_shortcut_actions", []):
                self.removeAction(action)
            self._popup_shortcut_actions = [
                action for action in self._app_menu.actions() if not action.shortcut().isEmpty()
            ]
            for action in self._popup_shortcut_actions:
                self.addAction(action)

        if IS_MACOS:
            self._build_macos_menu_bar(groups)
            # Cocoa rebuilds its own native menu items - overwriting any
            # rename - at least once more after this call returns, somewhere
            # between menu construction and the window's first activation.
            # Measured, not documented anywhere: a rename applied immediately
            # is gone again within 500ms, while one applied at 2s already
            # survives. Rather than guess the exact moment, this reapplies a
            # few times over the first two seconds and then stops.
            for delay_ms in (0, 150, 400, 800, 1500, 2500):
                QTimer.singleShot(delay_ms, self._rename_macos_native_app_menu_items)

    def _build_macos_menu_bar(
        self, groups: list[tuple[str, list[MenuItem | None]]]
    ) -> None:
        """Populate the real menu bar with one QMenu per group.

        Settings and Credits do not stay wherever this puts them: Cocoa
        pulls any action carrying MenuRole.PreferencesRole/AboutRole out
        into the native application menu - the one already showing next to
        the apple - wherever in the menu bar it was declared. Window is not
        one of the groups: it holds no reusable MenuItem, only two Cocoa
        window operations nothing else needs, so it is built directly here
        instead, between the app's own menus and Help - the usual place on
        a Mac.
        """
        menu_bar = self.menuBar()
        menu_bar.clear()

        # menu_bar.clear() above only lets go of menus it was actually
        # holding - the hidden-but-alive File/Database menus below are
        # never added to it, so without this every rebuild (every
        # undo-stack change; see this method's own docstring) would pile up
        # another orphaned QMenu plus another copy of the same shortcut
        # re-registered on self, instead of replacing the previous one.
        for action in getattr(self, "_macos_hidden_menu_actions", ()):
            self.removeAction(action)
        for menu in getattr(self, "_macos_hidden_menus", ()):
            menu.deleteLater()
        self._macos_hidden_menus: list[QMenu] = []
        self._macos_hidden_menu_actions: list[QAction] = []

        # File and Database are both covered by the nav rail's own pages
        # now (_create_file_page/_create_developer_page) - a second, visible
        # menu for the same actions would just be redundant chrome. Their
        # items are still built (below, into a menu that is never added to
        # the bar) and their shortcuts kept alive on the window itself
        # (self.addAction), so Cmd+S and friends do not stop working just
        # because the menu that used to carry them is gone from view.
        hidden_titles = {_("File"), _("Database")}

        for index, (title, items) in enumerate(groups):
            if index == len(groups) - 1:
                self._build_macos_window_menu(menu_bar)

            hidden = title in hidden_titles
            if hidden:
                menu = QMenu(self)
                menu.setTitle(title)
            else:
                menu = menu_bar.addMenu(title)
            # Cocoa rarely delivers this for a menu-bar menu, and rebuilding
            # the bar from inside a show handler would clear the menu
            # mid-display - so _sync (a plain property poke), never _refresh.
            menu.aboutToShow.connect(self._sync_undo_item)
            for item in items:
                if item is None:
                    menu.addSeparator()
                    continue

                if item.submenu is not None:
                    # The same helper the popup uses, so Open recent is one
                    # list of files rendered twice rather than two lists that
                    # can disagree.
                    child = create_menu(self, item.submenu)
                    child.setTitle(item.text)
                    child.setEnabled(item.enabled)
                    menu.addMenu(child)
                    continue

                create_menu_item(
                    parent=self,
                    menu=menu,
                    icon=item.icon,
                    checkable=item.checkable,
                    text=item.text,
                    tooltip=item.tooltip,
                    key=item.shortcut,
                    action=item.callback,
                    action_id=item.action_id,
                    checked=item.checked,
                    enabled=item.enabled,
                )

            for action in menu.actions():
                role = self._MACOS_MENU_ROLES.get(str(action.data() or ""))
                if role is not None:
                    action.setMenuRole(role)

            if hidden:
                # menu itself keeps every action in it alive regardless
                # (append below); addAction is only for the ones whose
                # shortcut has to keep firing with no visible menu left to
                # carry it - Cmd+S and the rest of File/Database's own.
                self._macos_hidden_menus.append(menu)
                for action in menu.actions():
                    if not action.shortcut().isEmpty():
                        self.addAction(action)
                        self._macos_hidden_menu_actions.append(action)

    def _build_macos_window_menu(self, menu_bar: QMenuBar) -> None:
        """Build the Window menu: Minimize and Zoom.

        Neither is a reusable MenuItem - nothing else in the app ever needs
        "minimize this window" - so, unlike every other menu, this one is
        built directly against the QMenuBar rather than through
        _app_menu_items/create_menu_item.
        """
        menu = menu_bar.addMenu(_("Window"))
        create_menu_item(
            parent=self,
            menu=menu,
            icon=None,
            checkable=False,
            text=_("Minimize"),
            tooltip=_("Minimize"),
            key="Ctrl+M",
            action=self.showMinimized,
        )
        create_menu_item(
            parent=self,
            menu=menu,
            icon=None,
            checkable=False,
            text=_("Zoom"),
            tooltip=_("Zoom"),
            key=None,
            action=self._on_zoom,
        )

    #: Cocoa's own selectors for Hide and Quit - reliable regardless of
    #: title, since neither is backed by a QAction of ours and both keep
    #: these selectors on every Mac, in every language.  About is *not* in
    #: here: it is backed by our own "credits" QAction, so Qt dispatches it
    #: through the same generic "qt_itemFired:" selector as Preferences,
    #: indistinguishable from it by selector alone.  It is matched by
    #: position instead - see _rename_macos_native_app_menu_items.
    _MACOS_APP_MENU_SELECTORS: ClassVar[dict[str, str]] = {
        "hide:": "Hide",
        "terminate:": "Close",
    }

    def _rename_macos_native_app_menu_items(self) -> None:
        """Strip the app name Cocoa insists on adding to About/Hide/Quit.

        `QAction.MenuRole` moves an action into the native application menu,
        but Cocoa then *replaces* its text with its own "About {name}" /
        "Hide {name}" / "Quit {name}" template - not read from the QAction at
        all, so nothing on the Qt side can change it.  The assumption in
        ChartLibre.py that QApplication.setApplicationName() controls "{name}" does
        not hold here: measured directly, Cocoa's template uses
        NSProcessInfo.processName (the running interpreter, "Python" for an
        unsigned script) instead, which no public Qt or Cocoa API renames
        without an actual .app bundle - only reaching past Qt into the
        NSMenuItem itself can.

        So this does not try to make the name correct - it removes it,
        landing on the bare "About" / "Hide" / "Close" the request asked for.
        Hide and Quit are matched by their native Cocoa action selector,
        stable regardless of title or language.  About cannot be: it is
        backed by our own "credits" QAction, so Qt dispatches it through the
        same generic selector as Preferences.  Apple's own Human Interface
        Guidelines fix the application menu's layout on every Mac - About
        first, when present - which is the stable signal used here instead.

        Deferred one event-loop turn past menu construction: Cocoa builds the
        native menu from Qt's QMenuBar lazily, so nothing is there to rename
        yet in the same call that builds it.
        """
        if not _pyobjc_core_is_safe_to_import():
            return

        # Offscreen (tests, scripts) there is no native menu bar to rename.
        if QApplication.platformName() != "cocoa":
            return

        try:
            import AppKit  # type: ignore[import-not-found]
        except ImportError:
            return

        try:
            app = getattr(AppKit, "NSApp", None)
            if app is None:
                ns_application = AppKit.NSApplication
                app = ns_application.sharedApplication()

            app_menu = app.mainMenu().itemAtIndex_(0).submenu()
            if app_menu is None or app_menu.numberOfItems() == 0:
                return

            about_item = app_menu.itemAtIndex_(0)
            if str(about_item.action() or "") == "qt_itemFired:":
                about_item.setTitle_(_("About"))

            for index in range(app_menu.numberOfItems()):
                item = app_menu.itemAtIndex_(index)
                selector = str(item.action()) if item.action() is not None else ""
                bare_text = self._MACOS_APP_MENU_SELECTORS.get(selector)
                if bare_text is not None:
                    item.setTitle_(_(bare_text))
        except Exception:
            applogger.exception("Failed to rename the native macOS app menu items.")

    # ==================================================================
    # Chart properties: the properties panel beside the chart
    # ==================================================================
    #: What a property is called in its Undo step (S-01): "Colour of 'Monthly mean'". Another key: its own
    #: name, in words.
    _PROPERTY_NAMES: ClassVar[dict[str, str]] = {
        "color": "Colour", "linestyle": "Line style", "linewidth": "Line width", "marker": "Marker",
        "markersize": "Marker size", "alpha": "Opacity", "label": "Name", "sql_query": "Query", "title": "Title",
        "x_label": "X label", "y_label": "Y label", "grid": "Grid", "legend": "Legend", "xscale": "X scale",
        "yscale": "Y scale", "width": "Width", "height": "Height", "mpl_style": "Style",
    }

    @classmethod
    def _undo_label(cls, keys: set[str], element: str) -> str:
        """An edit's Undo step, named for what it changes and where (S-01). The same name within a moment is one
        step (_snapshot_descriptors): a slider dragged stays one Undo; another property or element is another."""
        if len(keys) == 1:
            key = next(iter(keys))
            what = _(cls._PROPERTY_NAMES[key]) if key in cls._PROPERTY_NAMES else key.replace("_", " ").capitalize()
        else:
            what = _("Properties")
        return _("{what} of {element}").format(what=what, element=element)

    def _create_properties_control(self) -> QTabWidget:
        """Figure, Axis, Series and Overlays as tabs, each page scrolling on its own.

        Tabs rather than the accordion they were: four headers stacked
        took about 200 px of the panel's height and showed one page at a
        time all the same. Short names, so the four fit a narrow panel;
        when they do not, the tab bar scrolls rather than eliding them.
        """
        self._figure_widget = FigurePropertiesWidget(self)
        self._axis_widget = AxisPropertiesWidget(self)
        self._series_widget = SeriesPropertiesWidget(self)
        self._overlay_widget = OverlayPropertiesWidget(self)

        control = QTabWidget(self)
        control.setObjectName("propertiesTabs")
        control.setDocumentMode(True)
        control.setUsesScrollButtons(True)
        control.setElideMode(Qt.TextElideMode.ElideNone)
        control.tabBar().setExpanding(False)
        # Last: annotations and reference lines are the finishing pass on a
        # chart, done once the data, the axes and the series are right.
        for page, title, tooltip in (
            (self._figure_widget, _("Figure"), _("Figure properties")),
            (self._axis_widget, _("Axis"), _("Axis properties")),
            (self._series_widget, _("Series"), _("Series properties")),
            (self._overlay_widget, _("Overlays"), _("Overlay properties")),
        ):
            # The ground and the room a toolbox page had: the window grey on
            # macOS, white on Windows (the toolboxPage rule in each sheet).
            page.setProperty("toolboxPage", True)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            layout = page.layout()
            if layout is not None and layout.contentsMargins().isNull():
                layout.setContentsMargins(*MARGIN_TOOLBOX_PAGE)
            index = control.addTab(self._scrollable(page), title)
            control.setTabToolTip(index, tooltip)
        control.setCurrentIndex(0)
        self._connect_property_signals()
        self._clear_property_widgets()
        return control

    def _connect_property_signals(self) -> None:
        """Connect property widgets to main-window persistence handlers."""
        self._figure_widget.style_changed.connect(self._on_figure_style_changed)
        self._figure_widget.style_applied_to_all.connect(self._on_style_applied_to_all)
        self._figure_widget.grid_layout_requested.connect(self._on_grid_layout_requested)
        self._figure_widget.layout_preset_requested.connect(self._on_layout_preset_requested)
        self._figure_widget.layout_mosaic_requested.connect(self._on_layout_mosaic_requested)
        self._figure_widget.template_requested.connect(self._on_template_requested)
        self._figure_widget.template_save_requested.connect(self._on_template_save_requested)
        self._figure_widget.figure_options_requested.connect(self._on_figure_options_requested)
        self._axis_widget.axis_selected.connect(self._on_axis_selected)
        # Chosen here, shown there: the chart highlights the axis or series for a moment (S-01).
        self._axis_widget.axis_picked.connect(self._flash_on_chart)
        self._series_widget.series_picked.connect(
            lambda _series_id, name: self._flash_on_chart(self._axis_widget.current_axis_id(), name))
        self._axis_widget.renderer_changed.connect(self._on_axis_renderer_changed)
        self._axis_widget.axis_options_requested.connect(self._on_axis_options_requested)
        self._axis_widget.axis_action_requested.connect(self._on_axis_action_requested)
        self._series_widget.series_options_requested.connect(self._on_series_options_requested)
        self._series_widget.series_order_requested.connect(self._on_series_order_requested)
        self._series_widget.series_delete_requested.connect(self._on_series_delete_requested)
        self._series_widget.function_edit_requested.connect(self._on_function_edit_requested)
        self._overlay_widget.overlay_options_requested.connect(
            self._on_overlay_options_requested
        )

    def _configure_properties_control(self) -> None:
        """Keep the properties control shrink-friendly.

        The properties pane is later wrapped in a scroll area so it can exceed
        the available height without forcing the whole main window taller.
        """
        self._properties_control.setMinimumSize(0, 0)
        self._properties_control.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Ignored,
        )

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        """Wrap *widget* in a QScrollArea with no minimum-height floor of
        its own - see _create_left_stack for why every page that can grow
        without bound needs this."""
        scroll = QScrollArea(self)
        stdSizeAndlayout(scroll)
        scroll.setMinimumSize(0, 0)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _update_properties_for_current_chart(self) -> None:
        """Connect the properties control to the currently selected chart."""
        panel = self._current_chart_panel()
        if panel is None :
            self._clear_property_widgets()
            return

        # A no-op unless _reload_tabs built this panel with defer_render:
        # this is the one place every "the current chart is now this panel"
        # path converges (initial load, tab click, Undo, chart deletion),
        # so it is where a still-unrendered tab's first render belongs.
        panel.ensure_rendered()
        self._set_property_widgets_connected(
            figure_id=int(panel.figure_id),
            figure=panel.figure,
            redraw_callback=panel.reload,
            panel=panel,
        )

    def _set_property_widgets_connected(
        self,
        *,
        figure_id: int,
        figure: Any,
        redraw_callback: Any | None = None,
        panel: ChartPanel | None = None,
    ) -> None:
        """Load the direct QToolBox property pages for one chart.

        Why the property widgets get ``_redraw_properties_chart`` rather than the
        panel's own ``reload``: a full rebuild costs a complete re-render, so
        every redraw request from a spinbox or colour picker has to go through
        the same debounce as the ones raised here.
        """
        self._properties_figure_id = int(figure_id)
        self._properties_figure = figure
        self._properties_redraw_callback = redraw_callback
        self._properties_panel = panel
        for widget in (
            self._figure_widget,
            self._axis_widget,
            self._series_widget,
            self._overlay_widget,
        ):
            widget.set_connected_figure(
                repo=self._repo,
                figure_id=figure_id,
                figure=figure,
                redraw_callback=self._redraw_properties_chart,
            )
        if panel is not None:
            self._figure_widget.set_resize_mode_control(
                panel.resize_mode, panel.set_resize_mode
            )
        current_axis_id = self._axis_widget.current_axis_id()
        self._series_widget.set_current_axis_id(current_axis_id)
        self._overlay_widget.set_axis(current_axis_id)
        self._axis_widget.rebuild_kwargs_editor(current_axis_id)

    def _clear_property_widgets(self) -> None:
        """Clear all direct property pages."""
        # Drop any pending redraw: its target chart is going away.
        self._properties_redraw_timer.stop()
        self._properties_figure_id = None
        self._properties_figure = None
        self._properties_redraw_callback = None
        self._properties_panel = None
        self._figure_widget.clear_connected_figure()
        self._axis_widget.clear_connected_figure()
        self._series_widget.clear_connected_figure()
        self._overlay_widget.clear_connected_figure()
        self._axis_widget.rebuild_kwargs_editor(None)

    def _redraw_properties_chart(self) -> None:
        """Request a reload of the chart connected to the property pages.

        The request is debounced: dragging a spinbox emits a change per step and
        each reload re-renders the whole figure, so without coalescing the UI
        thread stalls for the duration of every intermediate value.  Same
        pattern as the style editor's 300 ms timer, tightened to keep the chart
        feeling live.
        """
        self._properties_redraw_timer.start(PROPERTIES_REDRAW_DEBOUNCE_MS)

    def _flush_properties_chart_redraw(self) -> None:
        """Run the debounced chart reload."""
        if self._properties_redraw_callback is not None:
            self._properties_redraw_callback()

    def _reload_property_widgets(self) -> None:
        """Reload direct property pages while preserving selected axis."""
        if self._properties_figure_id is None or self._properties_figure is None:
            return
        current_axis_id = self._axis_widget.current_axis_id()
        for widget in (
            self._figure_widget,
            self._axis_widget,
            self._series_widget,
            self._overlay_widget,
        ):
            widget.set_connected_figure(
                repo=self._repo,
                figure_id=self._properties_figure_id,
                figure=self._properties_figure,
                redraw_callback=self._properties_redraw_callback,
            )
        if self._properties_panel is not None:
            self._figure_widget.set_resize_mode_control(
                self._properties_panel.resize_mode, self._properties_panel.set_resize_mode
            )
        self._series_widget.set_current_axis_id(current_axis_id)
        self._overlay_widget.set_axis(current_axis_id)
        self._axis_widget.rebuild_kwargs_editor(current_axis_id)

    def _properties_figure_options(self) -> dict[str, Any]:
        """Return mutable figure options for the active property figure."""
        if self._properties_figure_id is None:
            return {}
        desc = self._repo.load_figure_descriptor(self._properties_figure_id)
        if desc is None:
            return {}
        if desc.options is None:
            return {}
        if isinstance(desc.options, dict):
            return dict(desc.options)
        applogger.error("Figure id=%r has invalid options.", desc.id)
        return {}

    def _properties_series_descriptor(self, series_id: int | None) -> Any | None:
        """Return one series descriptor by id from the active figure."""
        if self._properties_figure_id is None or series_id is None:
            return None
        desc = self._repo.load_figure_descriptor(self._properties_figure_id)
        if desc is None or desc.axes is None:
            return None
        for axis_desc in desc.axes:
            for series_desc in list(axis_desc.series or []):
                if int(series_desc.id) == int(series_id):
                    return series_desc
        return None

    def _axis_rows(self) -> list[tuple[int, int, str]]:
        """Return axes for the active property figure."""
        if self._properties_figure_id is None:
            return []
        return [
            (int(axis_id), int(axis_index), str(title or ""))
            for axis_id, axis_index, title in self._repo.list_axes_for_figure(
                int(self._properties_figure_id)
            )
        ]

    def _move_axis_descriptor(self, axis_id: int, delta: int) -> bool:
        """Move one axis through repository-managed index swapping."""
        rows = self._axis_rows()
        ordered_ids = [axis_id_value for axis_id_value, _index, _title in rows]
        if not ordered_ids or self._properties_figure_id is None:
            return False
        try:
            current_index = ordered_ids.index(int(axis_id))
        except ValueError:
            return False
        target_index = current_index + int(delta)
        if target_index < 0 or target_index >= len(ordered_ids):
            return False
        self._repo.swap_axis_indexes(
            figure_id=int(self._properties_figure_id),
            first_axis_id=ordered_ids[current_index],
            second_axis_id=ordered_ids[target_index],
        )
        return True

    def _delete_axis_descriptor(self, axis_id: int) -> bool:
        """Delete one axis through the repository."""
        existing_axis_ids = {axis_id_value for axis_id_value, _index, _title in self._axis_rows()}
        if int(axis_id) not in existing_axis_ids:
            return False
        self._repo.delete_axis(int(axis_id))
        return True

    def _on_figure_style_changed(self, style_text: str) -> None:
        if self._properties_figure_id is None:
            return
        options = self._properties_figure_options()
        style = str(style_text or "")
        if style:
            options["mpl_style"] = style
        else:
            options.pop("mpl_style", None)
        self._repo.set_figure_options(self._properties_figure_id, options)
        self._redraw_properties_chart()

    def _on_style_applied_to_all(self, style_text: str) -> None:
        """Every figure of the project takes *style_text* (or the default, when empty) - after asking; undoable."""
        figures = [int(figure_id) for figure_id, _name in self._repo.load_figures_from_db()]
        if not figures or not ask(self, "figure.confirm_style_to_all", count=len(figures)):
            return
        self._repo.snapshot_for_undo(list(self._repo.DESCRIPTOR_TABLES), label="Style applied to all figures")
        style = str(style_text or "")
        for figure_id in figures:
            options = dict(self._repo.get_figure_options(figure_id) or {})
            if style:
                options["mpl_style"] = style
            else:
                options.pop("mpl_style", None)
            self._repo.set_figure_options(figure_id, options)
        self._reload_tabs(select_figure_id=self._properties_figure_id)

    def _on_grid_layout_requested(self, nrows: int, ncols: int) -> None:
        if self._properties_figure_id is None:
            return
        self._repo.set_figure_grid(
            self._properties_figure_id,
            nrows=int(nrows),
            ncols=int(ncols),
        )
        self._redraw_properties_chart()

    def _on_layout_mosaic_requested(self, text: str) -> None:
        """Arrange every axis of the current figure as the mosaic *text* says; a mistake in it is said, not applied."""
        if self._properties_figure_id is None:
            return
        figure_id = int(self._properties_figure_id)
        axis_ids = [axis_id for axis_id, _axis_index, _title in self._repo.list_axes_for_figure(figure_id)]
        try:
            plan = layout_presets.plan_mosaic(text, axis_ids)
        except ValueError as exc:
            QMessageBox.warning(self, _("Mosaic"), str(exc))
            return
        self._snapshot_descriptors(_("Figure layout"))
        self._repo.apply_axis_layout(
            figure_id=figure_id,
            nrows=plan.nrows,
            ncols=plan.ncols,
            placements=[(p.axis_id, p.axis_index, p.options) for p in plan.axes],
        )
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_template_requested(self, path: str) -> None:
        """Lay the current figure out as the template at *path* says, one undo step; what it could not do, said."""
        from app.data import figure_templates

        if self._properties_figure_id is None:
            return
        try:
            template = figure_templates.load(Path(path))
        except ValueError as exc:
            QMessageBox.warning(self, _("Template"), str(exc))
            return
        self._snapshot_descriptors(_("Figure template"))
        notes = figure_templates.apply_template(self._repo, int(self._properties_figure_id), template)
        for note in notes:
            applogger.info(note)
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_template_save_requested(self, name: str) -> None:
        """The current figure's layout and settings saved as template *name*, in Templates/."""
        from app.data import figure_templates

        if self._properties_figure_id is None:
            return
        descriptor = self._repo.load_figure_descriptor(int(self._properties_figure_id))
        if descriptor is None:
            return
        path = figure_templates.save(figure_templates.to_template(descriptor, name))
        applogger.info(_("Template saved: {path}").format(path=path))

    def _on_layout_preset_requested(self, preset: str) -> None:
        """Arrange every axis of the current figure using *preset*.

        The grid size and every axis's row_span/col_span/sharex/sharey/
        twin_of come entirely from layout_presets.plan_layout - this only
        supplies the one thing it cannot know on its own: which axes the
        figure actually has, in the order the axis panel lists them.
        """
        if self._properties_figure_id is None:
            return
        figure_id = int(self._properties_figure_id)
        axis_ids = [
            axis_id
            for axis_id, _axis_index, _title in self._repo.list_axes_for_figure(figure_id)
        ]
        plan = layout_presets.plan_layout(preset, axis_ids)
        self._repo.apply_axis_layout(
            figure_id=figure_id,
            nrows=plan.nrows,
            ncols=plan.ncols,
            placements=[
                (placement.axis_id, placement.axis_index, placement.options)
                for placement in plan.axes
            ],
        )
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_figure_options_requested(self, payload: dict[str, Any]) -> None:
        """Persist the figure options the properties widget just emitted.

        Same rule as the axis handler: store the whole payload rather than a
        hand-listed subset, so a control added to the widget cannot silently
        fail to persist. None removes the key.
        """
        if self._properties_figure_id is None:
            return

        options = self._properties_figure_options()
        edited = {key for key, value in payload.items()
                  if key not in self._FIGURE_PAYLOAD_NON_OPTIONS and options.get(key) != value}
        self._snapshot_descriptors(self._undo_label(edited, _("the figure")))

        for key, value in payload.items():
            if key in self._FIGURE_PAYLOAD_NON_OPTIONS:
                continue
            if value is None or (key == "mpl_style" and not str(value)):
                options.pop(key, None)
            else:
                options[key] = value

        # "layout" is the older spelling; keep layout_mode authoritative.
        options["layout_mode"] = str(
            payload.get("layout_mode", payload.get("layout", "constrained"))
            or "constrained"
        )

        self._repo.set_figure_options(self._properties_figure_id, options)
        self._rename_figure_if_requested(payload)
        self._redraw_properties_chart()

    def _rename_figure_if_requested(self, payload: dict[str, Any]) -> None:
        """Rename the figure and its chart tab when the payload carries a name."""
        if self._properties_figure_id is None or "name" not in payload:
            return

        name = str(payload.get("name") or "").strip()
        if not name:
            return

        figure_id = int(self._properties_figure_id)
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id)
            if descriptor is None or str(descriptor.name) == name:
                return
            self._repo.set_figure_properties(
                figure_id,
                nrows=int(descriptor.nrows or 1),
                ncols=int(descriptor.ncols or 1),
                name=name,
                options=descriptor.options or {},
            )
        except Exception:
            applogger.exception("Failed to rename figure_id=%s", figure_id)
            return

        self._update_tab_title(figure_id, name)

    def _update_tab_title(self, figure_id: int, name: str) -> None:
        """Retitle the chart tab that shows a given figure."""
        for index in range(self._tabs.count()):
            widget = self._tabs.widget(index)
            if isinstance(widget, ChartPanel) and int(widget.figure_id) == figure_id:
                self._tabs.setTabText(index, name)
                self._tabs.setTabToolTip(index, name)
                self._sync_chart_navigation()
                return

    def _on_chart_object_clicked(self, panel: Any, hit: dict[str, Any]) -> None:
        """A click on the chart opens Chart properties on what it landed on.

        Outside every axis: the Figure page. On an axis: the Axis page on that
        axis - on its X, Y or Z tab for a click on that coordinate's ticks or
        label, on Properties for its title. On a series' line, points or bar:
        the Series page on that series - with Shift, added to the series
        edited together.
        """
        control = self._properties_control
        if panel is not self._current_chart_panel():
            return
        if not control.isVisible():
            # A click on a chart opens Chart properties on what it landed on.
            self._left_panel.select("chart_properties")
        pages = [control.widget(index) for index in range(control.count())]

        def show(widget: QWidget) -> None:
            for index, page in enumerate(pages):
                if page is widget or (isinstance(page, QScrollArea) and page.widget() is widget):
                    control.setCurrentIndex(index)
                    return

        axis_id = hit.get("axis_id")
        if axis_id is None or not self._axis_widget.select_axis(int(axis_id)):
            show(self._figure_widget)
            return
        series = hit.get("series")
        pick = self._series_widget.add_series_named if hit.get("add") else self._series_widget.select_series_named
        if series and pick(str(series)):
            show(self._series_widget)
            return
        part = hit.get("part")
        if part:
            self._axis_widget.show_coordinate(None if part == "title" else str(part))
        show(self._axis_widget)

    def _flash_on_chart(self, axis_id: int | None, series: str | None = None) -> None:
        panel = self._current_chart_panel()
        if panel is not None and axis_id is not None and hasattr(panel, "flash"):
            panel.flash(int(axis_id), series)

    def _on_axis_selected(self, axis_id: int) -> None:
        self._series_widget.set_current_axis_id(axis_id)
        # One axis selector in the application: the overlays panel edits
        # whichever axis this one is on rather than carrying a second combo
        # that could disagree with it.
        self._overlay_widget.set_axis(axis_id)
        self._axis_widget.rebuild_kwargs_editor(axis_id)

    def _on_axis_renderer_changed(self, _renderer_name: str) -> None:
        self._axis_widget.rebuild_kwargs_editor(self._axis_widget.current_axis_id())

    def _on_axis_action_requested(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action", "") or "").strip()
        axis_id_value = payload.get("axis_id")
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)
        self._snapshot_descriptors(_("Axis: {action}").format(action=action))
        changed = False
        if action == "move_up":
            changed = self._move_axis_descriptor(axis_id, -1)
        elif action == "move_down":
            changed = self._move_axis_descriptor(axis_id, 1)
        elif action == "delete":
            changed = self._delete_axis_descriptor(axis_id)
        if changed:
            self._reload_property_widgets()
            self._redraw_properties_chart()

    def _axis_label(self, axis_id: int) -> str:
        """The axis as an Undo step names it: "the axis 'Monthly CO2'", else by its place in the figure."""
        descriptor = self._repo.load_figure_descriptor(self._properties_figure_id) \
            if self._properties_figure_id is not None else None
        for index, axis in enumerate(descriptor.axes if descriptor is not None else []):
            if int(axis.id) == int(axis_id):
                return _("the axis '{name}'").format(name=axis.title) if axis.title else \
                    _("axis {number}").format(number=index + 1)
        return _("the axis")

    def _on_axis_options_requested(self, payload: dict[str, Any]) -> None:
        """Persist the axis options the properties widget just emitted.

        Everything in the payload is stored, rather than a hand-listed subset.
        Why: the previous version copied ~10 named keys, so every control added
        to the widget was silently dropped here - it looked like the setting
        did nothing and reset itself on the next panel switch, because it was
        never written. A whitelist that has to be edited in a second file to
        add a control is a bug waiting to happen twice.

        None means "not configured" and is removed rather than persisted, so a
        disabled control does not overwrite a real value with null.
        """
        axis_id_value = payload.get("axis_id")
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)
        options = self._repo.get_axis_options(axis_id) or {}
        edited = {key for key, value in payload.items()
                  if key not in self._AXIS_PAYLOAD_NON_OPTIONS and options.get(key) != value}
        self._snapshot_descriptors(self._undo_label(edited, self._axis_label(axis_id)))

        for key, value in payload.items():
            if key in self._AXIS_PAYLOAD_NON_OPTIONS:
                continue
            if value is None:
                options.pop(key, None)
            else:
                options[key] = value

        # "hidden" is the legacy spelling the renderer still reads.
        options["hidden"] = bool(payload.get("hide_axis", False))

        renderer_name = str(payload.get("renderer", "") or "").strip()
        if renderer_name:
            options["renderer"] = renderer_name
            options["renderer_name"] = renderer_name
        else:
            options.pop("renderer", None)
            options.pop("renderer_name", None)
        axis_kwargs = self._axis_widget.clean_kwargs()
        if axis_kwargs:
            options["axis_kwargs"] = axis_kwargs
        else:
            options.pop("axis_kwargs", None)
        self._repo.set_axis_options(axis_id, options)
        self._redraw_properties_chart()

    def _on_overlay_options_requested(self, payload: dict[str, Any]) -> None:
        """Persist the annotations, reference lines and measurements of one axis.

        Its own handler rather than the axis one: that payload describes a
        whole axis - renderer, projection, hide_axis - and is written as
        such, so a partial one sent through it would clear what it left
        out. This writes exactly the three keys it owns.
        """
        axis_id_value = payload.get("axis_id")
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)

        self._snapshot_descriptors(_("Overlay properties"))
        options = self._repo.get_axis_options(axis_id) or {}
        for key in ("annotations", "lines", "measurements"):
            value = payload.get(key)
            if not isinstance(value, list):
                continue
            # An empty list means "there are none now", which is a real
            # edit - the user deleted the last row - so the key is removed
            # rather than stored as [].
            if value:
                options[key] = value
            else:
                options.pop(key, None)

        self._repo.set_axis_options(axis_id, options)
        self._redraw_properties_chart()

    def _on_series_order_requested(self, ordered_ids: list[int]) -> None:
        axis_id_value = self._series_widget.current_axis_id()
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)
        self._snapshot_descriptors(_("Reorder series"))
        options = self._repo.get_axis_options(axis_id) or {}
        options["series_order"] = [int(series_id) for series_id in ordered_ids]
        self._repo.set_axis_options(axis_id, options)
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_series_delete_requested(self, series_id: int) -> None:
        self._snapshot_descriptors(_("Delete series"))
        self._repo.delete_series(int(series_id))
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_series_options_requested(self, payload: dict[str, Any]) -> None:
        series_id_value = payload.get("series_id")
        if series_id_value is None:
            return
        series_id = int(series_id_value)
        raw_series_desc = self._properties_series_descriptor(series_id)
        if raw_series_desc is None:
            applogger.error("Series descriptor id=%r not found.", series_id)
            return
        series_desc = cast(Any, raw_series_desc)
        if series_desc.options is None:
            style: dict[str, Any] = {}
        elif isinstance(series_desc.options, dict):
            style = dict(series_desc.options)
        else:
            applogger.error("Series id=%r has invalid style.", series_desc.id)
            return
        edited = {key for key, value in payload.items()
                  if key not in {"series_id", "also_series_ids", "sql_query"} and style.get(key) != value}
        if str(payload.get("sql_query", "") or "").strip() not in ("", str(series_desc.sql_query or "").strip()):
            edited.add("sql_query")
        together = [self._properties_series_descriptor(int(i)) for i in payload.get("also_series_ids") or ()]
        names = ", ".join(f"'{desc.name}'" for desc in (series_desc, *together) if desc is not None)
        self._snapshot_descriptors(self._undo_label(edited, names))
        before = dict(style)
        # Same rule as the axis and figure handlers: everything the widget
        # sends is stored, minus the addressing keys.
        for key, value in payload.items():
            if key in {"series_id", "also_series_ids", "sql_query"}:
                continue
            if value is None:
                style.pop(key, None)
            else:
                style[key] = value

        sql_query = str(payload.get("sql_query", "") or "").strip()
        self._repo.update_series_style(series_id, style)
        try:
            # Typed by hand in the Series properties: the SQL guard's moment.
            if " " in sql_query:
                ensure_read_only_select(sql_query)
        except ValueError as exc:
            QMessageBox.warning(self, _("SQL blocked"), str(exc))
        else:
            self._repo.update_series_sql_query(series_id, sql_query)
        changes = {key: style.get(key) for key in {*before, *style} if before.get(key) != style.get(key)}
        # The series edited together take the same change - all but a name.
        shared = {key: value for key, value in changes.items() if key != "label"}
        others = [self._properties_series_descriptor(int(i)) for i in payload.get("also_series_ids") or ()]
        others = [desc for desc in others if desc is not None and shared]
        for desc in others:
            other_style = dict(desc.options) if isinstance(desc.options, dict) else {}
            for key, value in shared.items():
                if value is None:
                    other_style.pop(key, None)
                else:
                    other_style[key] = value
            self._repo.update_series_style(int(desc.id), other_style)
        self._reload_property_widgets()
        if sql_query == str(series_desc.sql_query or "").strip() and all(
            self._preview_series_style(str(desc.name), changes if desc is series_desc else shared)
            for desc in (series_desc, *others)
        ):
            return
        self._redraw_properties_chart()

    def _on_function_edit_requested(self, payload: dict[str, Any]) -> None:
        """The Function card's Apply: the formula saved for the project, this series' query rewritten; one Undo step.

        The function's stored values and range stay its own: what was edited in the card is this series'.
        Every chart of the function is drawn again - the formula is theirs too.
        """
        from app.data.repo.function_sql import replace_function_sql
        from app.data_operations import sympy_store

        series_id = payload.get("series_id")
        if series_id is not None:
            series_desc = self._properties_series_descriptor(int(series_id))
            edited = payload.get("function") or {}
            stored = sympy_store.get(self._repo, str(edited.get("function_name", "")))
            if series_desc is None or stored is None:
                return
            self._repo.snapshot_for_undo([*self._repo.DESCRIPTOR_TABLES, sympy_store.TABLE],
                                         label=_("Function of '{name}'").format(name=series_desc.name))
            function = {**stored, **{key: edited[key] for key in ("source_type", "symbols", "code_string",
                                                                    "description") if key in edited}}
            sympy_store.save(self._repo, function)
            query = sympy_store.source_sql(function, payload.get("parameters") or {}, payload.get("x"),
                                           payload.get("y"))
            self._repo.update_series_sql_query(int(series_id),
                                               replace_function_sql(str(series_desc.sql_query or ""), query))
            self._refresh_undo_item()
        self._reload_property_widgets()
        self._redraw_properties_chart()
        reload_charts = getattr(self, "_reload_function_charts", None)
        if callable(reload_charts):
            reload_charts()

    def _preview_series_style(self, series: str, changes: dict[str, Any]) -> bool:
        """A colour, line or marker edit shown on the chart's own lines - no render (S-01)."""
        panel = self._properties_panel
        axis_id = self._axis_widget.current_axis_id()
        if panel is None or axis_id is None or not hasattr(panel, "preview_series_style"):
            return False
        if any(value is None for value in changes.values()):
            return False  # back to the style's default: only a render knows it
        return bool(panel.preview_series_style(int(axis_id), series, changes))

    # ==================================================================
    # Project: its info, report and history, undo, recent projects and checks
    # ==================================================================
    #: The project metadata panel is supplied by the composed MainWindow.
    #: Declaring it here keeps the mixin type-checkable without changing the
    #: runtime construction order of the window's widgets.
    _database_info_panel: Any

    #: How many database-check problems the message box lists inline before
    #: it stops and points at Show Details, which holds all of them. Enough
    #: that the usual report is complete on sight, few enough that a database
    #: with hundreds of dangling references does not make a box taller than
    #: the screen.
    MAX_PROBLEMS_SHOWN: int = get_constant("max_problems_shown", 20)

    def _on_project_info(self) -> None:
        """Author, creation date and notes of the open project."""
        if self._repo is None:
            return
        from app.dialogs.project_info_dialog import ProjectInfoDialog

        if ProjectInfoDialog(self._repo, cast(QWidget, self)).exec():
            self._database_info_panel.set_repo(self._repo)
            self.statusBar().showMessage(_("Project info saved."), 4_000)

    def _on_project_report(self) -> None:
        from app.dialogs.project_report_dialog import ProjectReportDialog

        dialog = ProjectReportDialog(self._repo, self)
        if dialog.exec() and dialog.written is not None:
            self.statusBar().showMessage(_("Report written: {path}").format(path=dialog.written), 10_000)

    def _on_project_history(self) -> None:
        from app.dialogs.operation_history_dialog import OperationHistoryDialog

        OperationHistoryDialog(self._repo, None, self).exec()

    def _snapshot_descriptors(self, label: str) -> None:
        """Record the chart settings before an edit changes them.

        Every descriptor table, not the one this edit will touch: they hold
        a few rows each, so working out which is more expensive than
        copying all four - and getting that wrong is an undo that restores
        half of a change (todo.txt P2-11).

        A run of identical labels inside SNAPSHOT_COALESCE_SECONDS is one
        edit: the auto-applying property panels call this on every debounced
        change, and only the first call - the one taken before the edit
        began - has a pre-edit state worth keeping.
        """
        if self._repo is None:
            return

        now = monotonic()
        if (
            label == self._last_snapshot_label
            and now - self._last_snapshot_at < SNAPSHOT_COALESCE_SECONDS
        ):
            self._last_snapshot_at = now
            return
        self._last_snapshot_label = label
        self._last_snapshot_at = now
        self._repo.snapshot_for_undo(self._repo.DESCRIPTOR_TABLES, label=label)
        self._refresh_undo_item()

    def _on_undo(self) -> None:
        """Take back the last recorded change, and show the result."""
        before = self._descriptor_state()
        entry = self._repo.undo_last()
        if entry is None:
            applogger.info("There is nothing to undo.")
            self._refresh_undo_item()
            return

        applogger.info("Undid: %s", entry.describe())
        # A colour, line or marker taken back on the chart shown: back on its lines, as the edit was - no
        # render (S-01). Anything else: everything shown again from the database.
        if set(entry.tables) <= set(self._repo.DESCRIPTOR_TABLES) and self._undo_style_live(before):
            self._reload_property_widgets()
            self._refresh_undo_item()
            return
        # The entry just consumed - the menu then names the next one (or is
        # disabled). On macOS this rebuilds the bar; safe here because Cocoa
        # has already closed the menu before dispatching this.
        self._refresh_after_undo_or_repair()

    def _descriptor_state(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]] | None:
        """The figures and axes as rows, and the series by id - what an Undo of a style is compared on."""
        try:
            figures_and_axes = [self._repo.query_df(f'SELECT * FROM "{table}" ORDER BY id').to_dict("records")
                                for table in ("__figure_descriptors__", "__axis_descriptors__")]
            series = self._repo.query_df('SELECT * FROM "__series_descriptors__"').to_dict("records")
        except Exception:
            return None
        return [*figures_and_axes[0], *figures_and_axes[1]], {int(row["id"]): row for row in series}

    def _undo_style_live(self, before: tuple[list[dict[str, Any]], dict[int, dict[str, Any]]] | None) -> bool:
        """Re-draw an Undo that changed only series styles on the chart shown, on its own lines; False to render."""
        panel = self._current_chart_panel()
        after = self._descriptor_state()
        if before is None or after is None or panel is None or not hasattr(panel, "preview_series_style"):
            return False
        if before[0] != after[0] or set(before[1]) != set(after[1]):
            return False
        shown = {int(row["id"]) for row in after[0]
                 if "figure_id" in row and "axis_index" in row and int(row["figure_id"]) == int(panel.figure_id)}
        restyled = []
        for series_id, old in before[1].items():
            new = after[1][series_id]
            if old == new:
                continue
            if any(old[key] != new[key] for key in old if key != "style_json") or int(new["axis_id"]) not in shown:
                return False
            old_style = json.loads(old["style_json"] or "{}")
            new_style = json.loads(new["style_json"] or "{}")
            changes = {key: new_style.get(key) for key in {*old_style, *new_style} if old_style.get(key) != new_style.get(key)}
            if any(value is None for value in changes.values()):
                return False  # back to the style's default: only a render knows it
            restyled.append((int(new["axis_id"]), str(new["name"]), changes))
        return bool(restyled) and all(panel.preview_series_style(axis_id, name, changes)
                                      for axis_id, name, changes in restyled)

    def _refresh_after_undo_or_repair(self) -> None:
        """Tables, charts and Chart properties shown again from the database, and the Undo item updated."""
        self._table_panel.reload()
        self._preview.clear()
        self._reload_tabs()
        self._update_properties_for_current_chart()
        self._refresh_undo_item()

    def _on_open_recent(self, db_path: Path) -> None:
        """Open one remembered database."""
        if not db_path.exists():
            # Between building the menu and clicking it - or a file on a
            # volume that has since been unmounted.
            applogger.warning(
                "That database is no longer there: %s",
                db_path,
            )
            show_message(self, "database.open_failed", error=db_path)
            self._build_app_menu()
            self._recent_view.refresh()
            return

        applogger.info("Opening recent database: %s", db_path)
        try:
            self._switch_database(db_path)
        except Exception as exc:
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_clear_recent(self) -> None:
        """Forget the list. The database currently open is not affected."""
        clear_recent_databases()
        self._build_app_menu()
        self._recent_view.refresh()

    def _check_project_sql(self) -> None:
        """Check the opened project's saved queries and series SQL, once.

        Anything that would write is blocked (see SqliteRepo.scan_user_sql)
        and listed in a warning shown once the window is up. Nothing is
        changed in the file: fixing or deleting the query is the person's
        call. user.json "sql_guard": false switches this off.
        """
        try:
            found = self._repo.scan_user_sql()
        except Exception:
            applogger.exception("The project SQL check failed.")
            return
        if not found:
            return
        listed = "\n".join(f"\u2022 {line}" for line in found[:12])
        more = len(found) - 12
        if more > 0:
            listed += "\n" + _("...and {count} more.").format(count=more)
        text = _(
            "This project contains SQL that would change the database. It has been "
            "blocked and will not run until it is edited:\n\n{items}"
        ).format(items=listed)
        QTimer.singleShot(0, lambda: QMessageBox.warning(self, _("SQL blocked"), text))

    def _check_project_functions(self) -> None:
        """Charts of stored functions this computer does not trust: asked about once the window is up (ID47).

        Until then they draw empty; trusted, they draw again. Not trusted, each says so over the chart.
        """
        try:
            used = function_names(str(row.get("sql_query") or "") for row in self._repo.list_series_dict())
        except Exception:
            applogger.exception("The project's function check failed.")
            return
        if used:
            QTimer.singleShot(0, self._ask_to_trust_functions)

    def _ask_to_trust_functions(self) -> None:
        from app.data_operations import sympy_store
        from app.dialogs.function_trust_dialog import ask_to_trust

        try:
            if not sympy_store.untrusted(self._repo):
                return
        except Exception:
            return
        if ask_to_trust(self, self._repo):
            self._reload_function_charts()

    def _switch_database(
        self, db_path: Path, prepare: Callable[[], object] | None = None
    ) -> None:
        """Switch the UI to another database and rebind all dependent widgets.

        *prepare* runs once the current database is closed and before
        *db_path* is opened - the moment a file that may be the open one can
        safely be replaced (Load demo, reloading the demo already open). If
        it fails, the database that was open is opened again and the error
        goes to the caller.
        """
        applogger.info("Switching database to %s", db_path)
        # Updates are held on the window's contents, never on the window
        # itself: on macOS, turning a top-level window's updates off and on
        # again around the first switch of a session left it not painting at
        # all - the new project's charts stayed blank (or showed the old
        # ones) until the window was resized.
        contents = self.centralWidget() or self
        contents.setUpdatesEnabled(False)
        try:
            self._tabs.clear()
            self._preview.clear()
            QApplication.processEvents()

            self._repo.close()
            gc.collect()

            QApplication.processEvents()

            if prepare is not None:
                try:
                    prepare()
                except Exception:
                    self._repo = SqliteRepo(db_path=self._db_path)
                    self._table_panel.set_repo(self._repo)
                    self._database_info_panel.set_repo(self._repo)
                    self._table_panel.reload()
                    self._reload_tabs()
                    raise

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            from app.dialogs.project_info_dialog import default_author

            if default_author():
                # A new project carries the author this computer remembers.
                self._repo.set_project_info({"author": default_author()})
            self._table_panel.set_repo(self._repo)
            self._database_info_panel.set_repo(self._repo)
            self._update_window_title()
            set_last_database(db_path)

            self._check_project_sql()
            self._check_project_functions()
            self._table_panel.reload()
            self._reload_tabs()
            self._update_properties_for_current_chart()
            # set_last_database above has just put this file at the top of
            # the recent list; the menu showing that list has to be rebuilt
            # or it goes on showing the order from before the switch - same
            # for the File page's own list.
            self._build_app_menu()
            self._recent_view.refresh()
        finally:
            contents.setUpdatesEnabled(True)
        # A project opened, created or switched to: the Tables page, as the place to start.
        self._select_navigation("tables")

    def _on_new_file(self) -> None:
        """Create a new database, safely replacing an existing file if possible."""
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getSaveFileName(
            self,
            _("New database"),
            base_dir,
            PROJECT_FILE_FILTER,
        )
        if not file_path:
            return

        db_path = SqliteRepo.ensure_dhub_extension(Path(file_path))
        applogger.info("Creating new database: %s", db_path)

        self._tabs.clear()
        self._preview.clear()
        QApplication.processEvents()

        if self._repo:
            self._repo.close()

        gc.collect()
        QApplication.processEvents()

        try:
            if db_path.exists():
                db_path.unlink()

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            self._table_panel.set_repo(self._repo)
            self._database_info_panel.set_repo(self._repo)
            self._update_window_title()
            set_last_database(db_path)
            self._table_panel.reload()
            self._reload_tabs()
            self._select_navigation("tables")
        except Exception as exc:
            applogger.exception("Failed to create database: %s", exc)
            show_message(self, "database.create_failed", error=exc)
            return
        # A new project says who made it and what it is for from the start.
        self._project_info_asked = getattr(self, "_project_info_asked", set()) | {str(db_path)}
        self._on_project_info()

    def _on_load_demo(self, demo: DemoProject) -> None:
        """Copy *demo*, one of the shipped, pre-built demo projects (the File panel's Demo tab), and open it.

        No "where to save it" dialog: that was the one step between picking a
        demo and seeing it, for a file whose name already says what it is and
        that nobody is expected to keep - the same reasoning that lets a first
        run open straight into a fixed, well-known path with no dialog of its
        own (see app.utils.startup.DEFAULT_DATABASE_NAME). The copy lands in
        ``projects/`` beside the application (demos.PROJECTS_DIR), not
        loose in the home directory: the copies are throwaway and one folder
        holds all of them. Loading the same demo again overwrites its previous
        copy in place, which is the point: it puts back the pristine version
        rather than asking what to call a second one. Someone who wants to
        keep a demo under its own name has Save As for that once it is open,
        same as any other project.
        """
        target = SqliteRepo.ensure_dhub_extension(PROJECTS_DIR / demo.path_name)
        applogger.info("Loading demo project %r into %s", demo.file_name, target)
        # The copy is made with the current project closed: it may be this
        # very file - loading the open demo again, to start over - and
        # replacing a database under its own open connection left that
        # connection's pending changes to be replayed onto the fresh copy.
        try:
            self._switch_database(target, prepare=lambda: copy_demo_project(demo, target))
        except Exception as exc:
            applogger.exception("Failed to load demo project: %s", exc)
            show_message(self, "demo.build_failed", error=exc)

    def _on_open_database(self) -> None:
        """Open an existing database and switch the current UI."""
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getOpenFileName(
            self,
            _("Open database"),
            base_dir,
            PROJECT_FILE_FILTER,
        )
        if not file_path:
            return

        db_path = Path(file_path)
        applogger.info("Opening database: %s", db_path)

        try:
            self._switch_database(db_path)
        except Exception as exc:
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_save(self) -> None:
        """Fold the WAL into the .dhub file on disk, right now.

        Not "there is unsaved work": every change already committed through
        SQLite's WAL as it happened. But Cmd+S/Ctrl+S is a reflex, and this
        is the honest thing an already-live database can do under it -
        guarantee the file on disk is not waiting on a WAL checkpoint SQLite
        would otherwise fold in on its own schedule. See SqliteRepo.checkpoint.
        """
        if self._repo is None:
            return
        self._ask_for_missing_project_info()
        self._refresh_project_preview()
        try:
            self._repo.checkpoint()
            self.statusBar().showMessage(_("Database saved."), 4_000)
        except Exception as exc:
            applogger.exception("Checkpoint failed: %s", exc)

    def _ask_for_missing_project_info(self) -> None:
        """Open Project info when the project has no author or no notes.

        Once per project and session: closing the window without filling it
        in means "not now", and the next Save should not ask again.
        """
        if self._repo is None:
            return
        asked = getattr(self, "_project_info_asked", set())
        key = str(self._repo.db_path)
        info = self._repo.project_info()
        if key in asked or (info.get("author") and info.get("notes")):
            return
        asked.add(key)
        self._project_info_asked = asked
        self._on_project_info()

    def _refresh_project_preview(self) -> None:
        """Draw the project's first figure again into its preview picture."""
        if self._repo is None or self._repo.db_path is None:
            return
        try:
            from app.utils.project_preview import write_project_preview

            write_project_preview(self._repo)
        except Exception as exc:
            applogger.warning("Could not draw the project preview: %s", exc)

    def _on_save_as(self) -> None:
        """Save a copy of the current database under a new name, and switch to it.

        ``SqliteRepo.save_as`` uses VACUUM INTO rather than copying the .dhub
        file: with WAL mode active, the file on disk is not the whole
        database until its -wal side file is checkpointed into it, so a
        plain filesystem copy could silently miss recent writes.
        """
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getSaveFileName(
            self,
            _("Save database as"),
            base_dir,
            PROJECT_FILE_FILTER,
        )
        if not file_path:
            return

        target = SqliteRepo.ensure_dhub_extension(Path(file_path))
        if self._db_path is not None and target == self._db_path:
            return

        applogger.info("Saving database as: %s", target)
        self._ask_for_missing_project_info()
        try:
            saved_path = self._repo.save_as(target)
            self._switch_database(saved_path)
        except Exception as exc:
            applogger.exception("Failed to save database as: %s", exc)
            show_message(self, "database.save_as_failed", error=exc)
            return
        # The copy has its own name, so its own preview.
        self._refresh_project_preview()

    def _on_import_clipboard(self) -> None:
        """Import what is on the clipboard: the dialog opens on it, for the preview."""
        self._on_import_data(clipboard=True, transposed=self._clipboard_transposed.isChecked())

    def _on_import_web(self) -> None:
        """Import from a web address or a listed web source: the dialog opens on its Web frame."""
        self._on_import_data(web=True)

    def _on_import_data(
        self, source_path: Path | None = None, *, clipboard: bool = False, transposed: bool = False,
        web: bool = False,
    ) -> None:
        """Open the import dialog and refresh UI if import succeeds.

        ``source_path`` is the file a drop arrived with; the dialog then opens
        already showing it, with its preview and its default table name, which
        is the whole point of dropping it rather than browsing for it.
        """
        dlg = ImportDataDialog(self._repo, parent=self, web=web)
        # Only a path counts: a button's clicked(bool) arrives here as False, which
        # used to be opened as a file called "False".
        if isinstance(source_path, (str, Path)) and str(source_path):
            dlg.load_file(source_path)
        if clipboard and not dlg.load_clipboard(transposed):
            return  # nothing on the clipboard: said, and no empty dialog left behind
        if dlg.exec():
            self._reload_tabs()
            result = getattr(dlg, "import_result", None)
            self._show_table(result.table_name if result is not None else None)

    def _on_database_table_import(self) -> None:
        """Import a table - or what a query returns - from another database, as a linked table."""
        picker = ConnectDatabaseDialog(cast(QWidget, self))
        if not picker.exec() or picker.connection is None:
            return
        if not picker.table and not picker.query:
            return
        connection = picker.connection
        settings = connection.to_link_settings()
        try:
            if picker.query:
                settings["query"] = picker.query
                frame = SERVER_DATABASE_QUERY_READERS[connection.kind](
                    connection, picker.query, skiprows=0, skipfooter=0
                )
            else:
                settings["table"] = picker.table
                _list_tables, read_table = SERVER_DATABASE_READERS[connection.kind]
                frame = read_table(connection, picker.table, skiprows=0, skipfooter=0)
        except Exception as exc:
            applogger.exception("Database import failed")
            show_message(cast(QWidget, self), "import.database_failed", error=exc)
            return
        if frame is None or frame.empty:
            applogger.warning("The database returned no rows to import.")
            return

        # Named for the source table, as Import Data names a file's. A free
        # name - "readings_2" when "readings" exists: import_into_sqlite
        # would replace a table of the same name. "sqlite_" starts SQLite's
        # own names, which it refuses.
        base = re.sub(r"[^0-9A-Za-z_]+", "_", picker.table or f"{connection.display_name()}_query").strip("_")
        if not base or base.lower().startswith("sqlite_") or base[0].isdigit():
            base = f"db_{base}"
        table_name = self._repo.free_table_name(base)
        types = {str(column): ImportDataDialog._guess_sqlite_type(frame[column]) for column in frame.columns}
        self._repo.snapshot_for_undo([table_name], label=f"Import '{table_name}'")
        self._repo.import_into_sqlite(table_name, frame, types)
        applogger.info("Imported %s from %s.", table_name, connection.display_name())
        link_settings = {
            "source": settings,
            "read": {"skiprows": 0, "skip_last": 0, "header": True, "delimiter": None, "encoding": None},
            "destination": {"table": table_name, "normalize_columns": False},
            "columns": {"types": types},
        }
        if not self._repo.upsert_link(
            table_name=table_name,
            source_path=f"{connection.display_name()}#{picker.table or picker.query}",
            settings=link_settings,
        ):
            applogger.warning("Failed to create link for imported table '%s'", table_name)
        self._show_table(table_name)

    def _on_create_query_table(self, *, sql: str | None = None) -> None:
        dialog = QueryBuilderDialog(self._repo, initial_sql=sql, parent=self)
        dialog.exec()
        self._show_table(getattr(dialog, "created_table_name", None))

    def _on_optimize_db(self) -> None:
        """Check the database, report what it found, then compact it.

        The findings go in the box twice on purpose: the problems inline, so
        that what is wrong is readable without clicking anything, and the whole
        grouped report - unreferenced tables included - behind Show Details,
        where it scrolls and can be copied into a bug report.  The status bar
        keeps the count, which is all it has room for.
        """
        # What can be fixed is offered first - stray descriptors, series and
        # links to a table that is gone, tables the app left behind - then
        # the file is checked again and compacted.
        found = self._repo.check_database()
        if found.fixable:
            shown = found.fixable[: self.MAX_PROBLEMS_SHOWN]
            more = len(found.fixable) - len(shown)
            listing = "\n".join([*shown, *([_("...and {count} more.").format(count=more)] if more else [])])
            if ask(self, "database.fix_problems", report=listing, default_yes=True):
                fixed = self._repo.repair_database()
                self.statusBar().showMessage(_("Fixed: {what}").format(what="; ".join(fixed)), 10_000)
                self._refresh_after_undo_or_repair()
        report = self._repo.optimize_db()
        self.statusBar().showMessage(report.summary(), 10_000)

        if report.is_healthy:
            return

        shown = report.problems[: self.MAX_PROBLEMS_SHOWN]
        lines = [report.summary(), "", *shown]
        remaining = len(report.problems) - len(shown)
        if remaining > 0:
            lines.append(
                _("...and {count} more, under Show Details.").format(count=remaining)
            )

        show_message(
            self,
            "database.check_found_problems",
            report="\n".join(lines),
            details=report.details(),
        )

    # ==================================================================
    # Frame: the central area, resizing from the edges, and dropped files
    # ==================================================================
    #: How close to an edge, in pixels, counts as "grab this edge to resize".
    _RESIZE_MARGIN: int = 6

    #: Width of the grab zone at each end of the window's bottom edge, where the corner is.
    _RESIZE_CORNER: int = 14

    #: Cursor shape for each edge/corner combination _resize_edge_at can
    #: return. Absent from here (the interior, or a maximized window) means
    #: the ordinary arrow.
    _RESIZE_CURSORS: ClassVar[dict[Qt.Edge, Qt.CursorShape]] = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.TopEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.TopEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeBDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeBDiagCursor,
    }

    def _create_central_host(self) -> QWidget:
        """Wrap the main splitter in a neutral central widget.

        This prevents the top-level QMainWindow from taking an over-constrained
        size hint directly from the splitter tree.
        """
        host = QWidget(self)
        host.setMinimumSize(0, 0)
        host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(host)
        stdSizeAndlayout(layout)
        if IS_MACOS:
            # Flush to the window's own edge: #leftPanelCard and
            # #activityRail draw their own hairlines, and the NSWindow its
            # own border and corners. The traffic lights sit in the rail's
            # title strip.
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        else:
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)
        if IS_WINDOWS or IS_MACOS:
            # FramelessWindowHint (see __init__) strips every bit of native
            # chrome - border, corner, drop shadow - so without this the
            # window has no visible edge at all against whatever is behind
            # it. Each platform QSS draws a plain 1px outline on this object
            # name (fluent_win11.qss, macos_native.qss); WA_StyledBackground
            # is what makes a QWidget paint a QSS border at all rather than
            # silently ignoring it. Keep this a plain fill with no border
            # radius of its own on macOS: the window's real corners are
            # rounded by its NSWindow layer (apply_native_macos_corner_
            # radius), and a second curve painted here never lands on the
            # same pixels as that one - which is the doubled edge along the
            # top that the strip below used to show.
            #
            # No drop shadow here, unlike the elevated internal cards this
            # replaced: a shadow effect needs room *outside* the widget it
            # is attached to in order to render, and host fills the client
            # area exactly - the OS window edge sits at host's own edge, so
            # the shadow's bleed would just be clipped away. Giving it that
            # room means making the window translucent and inset from its
            # real (now larger, transparent) bounds, which also strands
            # QMainWindow's own statusBar() - built by Qt outside
            # centralWidget entirely - outside the inset border with no
            # background of its own. Worth doing, not safely from here
            # without a way to check it on an actual Windows/Mac build.
            host.setObjectName("windowFrame")
            host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            if IS_MACOS:
                # The NSWindow draws its own edge and corners now; this only
                # paints the fill beneath the translucent rail.
                host.setProperty("nativeRounded", True)

        if IS_WINDOWS:
            # Windows only: a caption strip across the whole window, which
            # is what a Windows 11 window has - icon and title on the left,
            # min/max/close on the right, all of it draggable.
            #
            # macOS deliberately has none. Its traffic lights live at the
            # top of the navigation rail instead (NavigationBar.__init__),
            # so each column keeps its own background all the way to the
            # window's top edge - grey for the rail, white for the panels
            # beside it - the way Finder and System Settings look. A strip
            # here would cut across all of them.
            self._own_title_bar = CustomTitleBar(cast(Any, self), is_macos=False)
            layout.addWidget(self._own_title_bar, 0)
        layout.addWidget(self._main_split, 1)
        return host

    def _resize_edge_at(self, pos: QPoint, watched: QWidget | None = None) -> Qt.Edge:
        """Return which edge(s) of the window *pos* is within the margin of.

        *pos* is in the coordinates of *watched*, the widget eventFilter
        below got the event for: _central_host (the default), or the status
        bar. The status bar is not inside _central_host - QMainWindow puts it
        below - so with it showing, the window's real bottom edge and both
        bottom corners belong to the bar, and the host's own bottom band is
        an interior one with nothing to resize.
        """
        bar = self.statusBar() if hasattr(self, "statusBar") else None
        if bar is not None and bar.isVisible():
            if watched is bar:
                # The bar's top is the host's bottom: not an edge of the window.
                return resize_edges(pos, bar.size(), self._RESIZE_MARGIN, top=False, corner=self._RESIZE_CORNER)
            return resize_edges(pos, self._central_host.size(), self._RESIZE_MARGIN, bottom=False)
        return resize_edges(pos, self._central_host.size(), self._RESIZE_MARGIN)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Give the frameless Windows window edge-drag resizing.

        FramelessWindowHint (see __init__) leaves the OS with no resize
        handles of its own to offer - this is the replacement, the same
        margin-hit-test-plus-startSystemResize technique
        CustomTitleBar.mousePressEvent already uses for the move case.
        A maximized window is left alone: its edges are the screen's own
        edges, and dragging those would resize the display area a user
        grabbing what looks like a window border did not mean to touch.
        """
        if not (
            IS_WINDOWS
            and (watched is self._central_host or watched is self.statusBar())
            and isinstance(event, QMouseEvent)
            and not self.isMaximized()
        ):
            return super().eventFilter(watched, event)

        edges = self._resize_edge_at(event.position().toPoint(), cast(QWidget, watched))
        if event.type() == QEvent.Type.MouseMove:
            cursor = self._RESIZE_CURSORS.get(edges, Qt.CursorShape.ArrowCursor)
            cast(QWidget, watched).setCursor(cursor)
        elif (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and edges
        ):
            handle = self.windowHandle()
            if handle is not None:
                handle.startSystemResize(edges)
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def dropped_paths(event: Any) -> list[Path]:
        """Return the local files carried by a drag, in the order dropped.

        Local files only: a drag from a browser carries a URL that names a
        file on a web server, and ``toLocalFile`` returns an empty string for
        it rather than something ``open()`` would fail on later.
        """
        data = event.mimeData()
        if data is None or not data.hasUrls():
            return []
        return [
            Path(local)
            for url in data.urls()
            if (local := url.toLocalFile())
        ]

    @classmethod
    def accepted_drop(cls, paths: list[Path]) -> tuple[str, list[Path]]:
        """Classify a drop as ``("database"|"import"|"", paths)``.

        A ``.dhub`` is a project to open, anything the import readers know is
        data to import, and everything else is refused - refused *before* the
        cursor changes, so the window says no by not offering to accept it
        rather than by a box after the fact.

        A database wins over data files dropped with it, and only one at a
        time: opening two projects at once has no meaning, and importing into
        a database that is about to be closed has less.
        """
        databases = [path for path in paths if path.suffix.lower() == ".dhub"]
        if len(databases) == 1:
            return "database", databases

        importable = [path for path in paths if is_importable(path)]
        if importable and not databases:
            return "import", importable

        return "", []

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept a drag only when the drop would actually do something."""
        kind, _paths = self.accepted_drop(self.dropped_paths(event))
        if kind:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep the acceptance while the cursor moves over the window."""
        self.dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Open a dropped project, or import dropped data files.

        Several data files are handled one dialog at a time, in the order they
        were dropped, and cancelling one stops the rest: cancel means "not
        this", and the natural reading of it on the second of four files is
        "stop", not "carry on with the next one".
        """
        kind, paths = self.accepted_drop(self.dropped_paths(event))
        if not kind:
            event.ignore()
            return

        event.acceptProposedAction()

        if kind == "database":
            applogger.info("Opening dropped database: %s", paths[0])
            try:
                self._switch_database(paths[0])
            except Exception as exc:
                applogger.exception("Failed to open the dropped database: %s", exc)
                show_message(self, "database.open_failed", error=exc)
            return

        for path in paths:
            applogger.info("Importing dropped file: %s", path)
            if not self._import_one_dropped_file(path):
                break

    def _import_one_dropped_file(self, path: Path) -> bool:
        """Import one dropped file; False when the user cancelled."""
        dialog = ImportDataDialog(self._repo, parent=self)
        dialog.load_file(path)
        if not dialog.exec():
            return False
        self._table_panel.reload()
        self._reload_tabs()
        return True

