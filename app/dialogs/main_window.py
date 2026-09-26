"""Main application window.

Layout: an activity rail on the left switches a QStackedWidget between the data
page (table list plus preview) and the charts page (one ChartPanel per figure in
a tab widget).  A properties QToolBox on the right edits the figure, axis, and
series of whichever chart tab is active.

Chart reloads requested by the property pages are debounced, because each one
re-renders the whole figure.
"""
from __future__ import annotations

import gc
import sys
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, Callable, cast

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QCursor,
    QDesktopServices,
    QIcon,
    QMouseEvent,
    QResizeEvent,
    QShowEvent,
)
from app import APP_ICON, APP_NAME
from app.charts import layout_presets
from app.widgets.custom_title_bar import CustomTitleBar
from app.dialogs.log_viewer_dialog import LogViewerDialog
from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel
from app.widgets.nav_bar import NavigationBar
from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.dialogs.import_data_dialog import ImportDataDialog, is_importable
from app.data.demo_project import PROJECTS_DIR, copy_demo_project
from app.dialogs.load_demo_dialog import LoadDemoDialog
from app.dialogs.credits_dialog import CreditsDialog
from app.widgets.database_info_panel import DatabaseInfoPanel
from app.dialogs.renderer_helper_dialog import RendererHelperDialog
from app.dialogs.series_operation_builder_dialog import SeriesOperationBuilderDialog
from app.dialogs.function_creator_dialog import FunctionCreatorDialog
from app.dialogs.edit_localization_dialog import EditLocalizationDialog
from app.dialogs.query_builder_dialog import QueryBuilderDialog
from app.widgets.axis_properties import AxisPropertiesWidget
from app.widgets.overlay_properties import OverlayPropertiesWidget
from app.widgets.figure_properties import FigurePropertiesWidget
from app.widgets.series_properties import SeriesPropertiesWidget
from app.widgets.series_operation import SeriesOperationWidget
from app.scanners.series_operation_scanner import import_class_from_file
from app.styles.style import (
    IS_MACOS,
    MenuItem,
    PANEL_MIN_WIDTH,
    action_menu_item,
    action_presentation,
    SPACING_DEFAULT,
    SPLITTER_HANDLE_WIDTH,
    apply_native_macos_corner_radius,
    apply_rounded_window_mask,
    apply_toolbox_header_metrics,
    apply_toolbox_page_metrics,
    CardFrame,
    TitledCard,
    create_action_button,
    create_menu,
    create_menu_item,
    create_section_title,
    icon_from_svg_source,
    mark_destructive_button,
    relax_minimum_width,
    repolish_widget,
    stdSizeAndlayout,
    _pyobjc_core_is_safe_to_import,
)
from app.widgets.table_list import TableListPanel
from app.widgets.table_preview import TablePreviewPanel
from app.utils.config import (
    clear_recent_databases,
    get_constant,
    get_recent_databases,
    get_section,
    set_last_database,
    set_section,
)
from app.utils.dialog_state import restore_window_geometry, save_window_geometry
from app.utils.startup import PROJECT_FILE_FILTER
from app.utils.messages import show_message
from app.logs.logger import applogger
from app.utils.i18n import _
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMenu,
    QMenuBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Coalescing window for property-driven chart reloads, in milliseconds.
# Long enough to swallow a spinbox drag, short enough to feel immediate.
PROPERTIES_REDRAW_DEBOUNCE_MS: int = get_constant("properties_redraw_debounce_ms", 120)

# A run of same-label descriptor snapshots inside this many seconds is
# treated as one edit and recorded once. The auto-applying property panels
# fire _snapshot_descriptors on every debounced change; without this a
# single slider drag would leave a stack of identical "Figure properties"
# undo entries in front of the state worth going back to.
SNAPSHOT_COALESCE_SECONDS: float = get_constant("snapshot_coalesce_seconds", 2.0)

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
# list, the property pages, the series-operation panels. Added to the rail's
# own width rather than used as the whole left pane, since the rail's width
# differs per platform; see _create_main_split.
PANEL_DEFAULT_WIDTH: int = get_constant("panel_default_width", 340)

# How long a picked-point readout stays in the status bar, in milliseconds.
# Long enough to read and write down, short enough that it is gone before it
# can be mistaken for a description of some later chart.
CHART_SELECTION_TIMEOUT_MS: int = get_constant("chart_selection_timeout_ms", 15_000)

IS_WINDOWS: bool = sys.platform == "win32"


class MainWindow(QMainWindow):
    """Main window with custom activity rail and chart tabs.

    The top-level window must remain shrinkable. To avoid child widgets
    propagating a large minimum height upward, the central layout explicitly
    uses zero minimum sizes and a scrollable properties page.
    """

    #: How many database-check problems the message box lists inline before
    #: it stops and points at Show Details, which holds all of them.  Enough
    #: that the usual report is complete on sight, few enough that a database
    #: with hundreds of dangling references does not make a box taller than
    #: the screen.
    MAX_PROBLEMS_SHOWN: int = get_constant("max_problems_shown", 20)

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
        #: Set on the first showEvent, whether or not the attempt actually
        #: succeeds - apply_native_macos_corner_radius needs a native window
        #: handle that does not exist yet at __init__ time, and the
        #: environment it depends on (pyobjc installed or not) cannot change
        #: mid-run, so one attempt is enough.
        self._native_corner_radius_attempted: bool = False
        #: True only once that attempt has actually succeeded - resizeEvent
        #: below uses this (not the flag above) to decide whether its own
        #: QRegion fallback mask is still needed.
        self._native_corner_radius_active: bool = False
        if IS_WINDOWS or IS_MACOS:
            # Frameless everywhere but Linux (whose window managers already
            # draw a native title bar this app has no reason to fight) -
            # macOS's own native chrome was the same plain title-bar-plus-
            # traffic-lights strip as any other window, worth removing for
            # the same reason it was worth removing on Windows: it is 40px
            # of screen fully given over to nothing but the window title,
            # which CustomTitleBar (below) replaces one-for-one on both.
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.resize(1200, 800)

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
        self._preview.refresh.connect(self.refresh)
        # Left-side pages.
        self._data_page = self._create_data_page()
        self._properties_control = self._create_properties_control()
        self._configure_properties_control()

        # Build VS Code-like rail + stacked pages.
        self._build_app_menu()
        self._left_stack = self._create_left_stack()
        self._left_rail: NavigationBar = self._create_activity_rail()
        self._left_panel = self._create_left_panel()
        self._configure_left_panel()

        # Main split: left panel + chart tabs.
        self._main_split = self._create_main_split()

        # Wrap the splitter in a plain central widget with a zero-minimum layout.
        self._central_host = self._create_central_host()
        self.setCentralWidget(self._central_host)
        if IS_WINDOWS or IS_MACOS:
            # Frameless (see setWindowFlag above), so the OS gives us no edge
            # resize handles at all - _resize_edge_at/_update_resize_cursor
            # below fill that in. Watching _central_host, not self: it fills
            # the entire client area (setCentralWidget), margin included, so
            # self - the QMainWindow - never actually sees a mouse event of
            # its own to filter. The 8px contentsMargins in
            # _create_central_host is exactly the band this hit-tests.
            self._central_host.setMouseTracking(True)
            self._central_host.installEventFilter(self)

        # Default page: the tables, not whatever happens to be first in the
        # rail. File sits above them now, and opening onto an empty file
        # page would hide the data the window was just opened on.
        self._set_nav_index(self._left_rail.action_ids.index("nav_data"))
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

    _STATUS_BAR_COLORS = {
        "normal": "#007ACC",
        "success": "#16825D",
        "busy": "#B35C00",
        "warning": "#9A6700",
        "error": "#C42B1C",
    }

    def _configure_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setObjectName("vscodeStatusBar")
        bar.setSizeGripEnabled(False)
        bar.setFixedHeight(24)
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
        bar.addWidget(self._status_project, 0)
        bar.addWidget(self._status_context, 1)
        bar.addPermanentWidget(self._status_state, 0)
        self._set_status_state("normal")

    def _set_status_state(self, state: str = "normal", message: str | None = None, timeout_ms: int = 0) -> None:
        color = self._STATUS_BAR_COLORS.get(state, self._STATUS_BAR_COLORS["normal"])
        stylesheet = (
            f"QStatusBar#vscodeStatusBar {{ background: {color}; color: white; border: none; }} "
            "QStatusBar#vscodeStatusBar::item { border: none; } "
            "QStatusBar#vscodeStatusBar QLabel { color: white; background: transparent; padding: 0 4px; }"
        )
        self.statusBar().setStyleSheet(stylesheet)
        if message is not None:
            self._status_state.setText(message)
            if timeout_ms > 0:
                QTimer.singleShot(timeout_ms, lambda: self._set_status_state("normal", _("Ready")))

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
        self.setWindowTitle(f"{APP_NAME} | {project}")
        if hasattr(self, "_status_project"):
            self._status_project.setText(project)
            self._status_project.setToolTip(str(self._db_path) if self._db_path else "")

    def _toggle_workspace(self) -> None:
        """Collapse left content to the navigation rail, or restore it."""
        hiding = self._left_stack.isVisible()
        sizes = self._main_split.sizes()
        if hiding:
            self._left_panel_restore_width = max(sizes[0], 360)
            self._left_stack.hide()
            rail_width = self._left_rail.width()
            self._left_panel.setMinimumWidth(rail_width)
            self._left_panel.setMaximumWidth(rail_width)
            self._main_split.setSizes([rail_width, max(sizes[1], 1)])
        else:
            self._left_panel.setMaximumWidth(16777215)
            self._left_panel.setMinimumWidth(PANEL_MIN_WIDTH + self._left_rail.width())
            self._left_stack.show()
            restore_width = int(getattr(self, "_left_panel_restore_width", 420))
            total = max(sum(sizes), restore_width + CHART_PANE_MIN_WIDTH)
            self._main_split.setSizes([restore_width, max(total - restore_width, 1)])
        self._left_rail.set_workspace_hidden(hiding)

    #: config.json key remembering whether the rail is collapsed to icons.
    NAV_COMPACT_KEY: str = "navigation_compact"

    def set_navigation_compact(self, compact: bool) -> None:
        """Collapse the navigation rail to its icons, or restore its labels.

        Different from _toggle_workspace, which hides the *panel* beside the
        rail: this keeps every page reachable and only drops the words, so
        the rail narrows from 200px to 48 and hands that width to whatever
        is next to it. The width the splitter gives the left pane moves by
        the same amount, or collapsing the rail would only widen the empty
        gap inside it.
        """
        compact = bool(compact)
        was = self._left_rail.width()
        self._left_rail.set_compact(compact)
        moved = self._left_rail.width() - was

        if self._left_stack.isVisible():
            self._left_panel.setMinimumWidth(
                PANEL_MIN_WIDTH + self._left_rail.width()
            )
            sizes = self._main_split.sizes()
            if len(sizes) == 2:
                self._main_split.setSizes(
                    [max(sizes[0] + moved, 1), max(sizes[1] - moved, 1)]
                )

        set_section(STATE_KEY, {**get_section(STATE_KEY), self.NAV_COMPACT_KEY: compact})

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

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _create_properties_control(self) -> QToolBox:
        """Create the properties QToolBox directly in the main window."""
        self._figure_widget = FigurePropertiesWidget(self)
        self._axis_widget = AxisPropertiesWidget(self)
        self._series_widget = SeriesPropertiesWidget(self)
        self._overlay_widget = OverlayPropertiesWidget(self)

        control = QToolBox(self)
        control.setObjectName("propertiesToolBox")
        self._figure_properties_index = control.addItem(
            self._figure_widget,
            _("Figure properties"),
        )
        self._axis_properties_index = control.addItem(
            self._axis_widget,
            _("Axis properties"),
        )
        self._series_properties_index = control.addItem(
            self._series_widget,
            _("Series properties"),
        )
        # Last: annotations and reference lines are the finishing pass on a
        # chart, done once the data, the axes and the series are right.
        self._overlay_properties_index = control.addItem(
            self._overlay_widget,
            _("Overlay properties"),
        )
        control.setCurrentIndex(self._figure_properties_index)
        # Section headers are sized from font metrics; QSS padding alone leaves
        # the labels clipped (see apply_toolbox_header_metrics).
        apply_toolbox_header_metrics(control)
        apply_toolbox_page_metrics(control)
        self._connect_property_signals()
        self._clear_property_widgets()
        return control

    def _connect_property_signals(self) -> None:
        """Connect property widgets to main-window persistence handlers."""
        self._figure_widget.style_changed.connect(self._on_figure_style_changed)
        self._figure_widget.grid_layout_requested.connect(self._on_grid_layout_requested)
        self._figure_widget.layout_preset_requested.connect(self._on_layout_preset_requested)
        self._figure_widget.figure_options_requested.connect(self._on_figure_options_requested)
        self._axis_widget.axis_selected.connect(self._on_axis_selected)
        self._axis_widget.renderer_changed.connect(self._on_axis_renderer_changed)
        self._axis_widget.axis_options_requested.connect(self._on_axis_options_requested)
        self._axis_widget.axis_action_requested.connect(self._on_axis_action_requested)
        self._series_widget.series_options_requested.connect(self._on_series_options_requested)
        self._series_widget.series_order_requested.connect(self._on_series_order_requested)
        self._series_widget.series_delete_requested.connect(self._on_series_delete_requested)
        self._overlay_widget.overlay_options_requested.connect(
            self._on_overlay_options_requested
        )

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
        #tab_bar.setExpanding(False)
        
        # Without this the bar still asks for the full width of every title.
        tab_bar.setMinimumWidth(300)
        tab_bar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

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

    def _configure_left_panel(self) -> None:
        """Configure the composite left panel to avoid height lock-up."""
        self._left_panel.setMinimumSize(0, 0)
        self._left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

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
            # Flush to the window frame's own edge, same as the style demo
            # (app/tests/manual_macos_style_demo.py, which uses this exact
            # zero margin): #leftPanelCard and #activityRail already draw
            # their own hairline edges, and #windowFrame its own 1px
            # outline below - a contentsMargins here on top of those was
            # just extra white gutter around the whole window that the
            # demo never had. The traffic lights sit inside the title bar
            # strip added below, which insets them from the corner on its
            # own.
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
            self._own_title_bar = CustomTitleBar(self, is_macos=False)
            layout.addWidget(self._own_title_bar, 0)
        layout.addWidget(self._main_split, 1)
        return host

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
        rail = getattr(self, "_left_rail", None)
        return None if rail is None else rail.title_bar

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
    @staticmethod
    def _apply_surface_shadow(widget: QWidget, *, blur: int = 24, y_offset: int = 5) -> None:
        """Add a restrained Fluent elevation shadow to a top-level surface."""
        effect = QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(float(blur))
        effect.setOffset(0.0, float(y_offset))
        effect.setColor(QColor(0, 0, 0, 42))
        widget.setGraphicsEffect(effect)

    def _create_data_page(self) -> QWidget:
        """Create the Data page with table list and preview splitter."""
        page = CardFrame(self, "dataPageCard", margins=(0, 0, 0, 0))
        page.setProperty("elevated", True)
        page.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        raw_layout = page.layout()
        if isinstance(raw_layout, QBoxLayout):
            layout = raw_layout
        else:
            layout = QVBoxLayout(page)
        split = self._data_split = QSplitter(Qt.Orientation.Vertical, page)
        split.setChildrenCollapsible(True)
        split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        split.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._table_panel.setMinimumSize(0, 0)
        self._table_panel.setContentsMargins(5,5,5,5)
        self._table_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._preview.setContentsMargins(5,5,5,5)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        split.addWidget(self._table_panel)
        split.addWidget(self._preview)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)

        layout.addWidget(split, 1)
        return page

    def _create_series_operations_page(self) -> QWidget:
        """Create the series operations page."""
        widget = SeriesOperationWidget(self)
        widget.operation_requested.connect(self._on_series_operation_requested)
        return widget

    def _on_series_operation_requested(self, operation: dict) -> None:
        """Handle a built-in or runtime series-operation request."""
        icon = SeriesOperationWidget.plugin_icon(operation)

        if operation.get("name") == "NewPlotTabDialog":
            self._on_new_plot_tab(icon)
            return

        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error(
                "Could not load series operation class: %r",
                operation.get("name"),
            )
            return

        self._open_series_operation(dialog_class, icon)

    def _create_left_stack(self) -> QStackedWidget:
        """Create the stacked pages shown next to the activity rail.

        The properties, Database and File pages are each wrapped in a
        QScrollArea so they scroll vertically instead of forcing the whole
        main window to keep a large minimum height - a QStackedWidget's own
        minimumSizeHint is the max over *every* page it holds, current or
        not, so an unbounded page (many tables, many recent projects) would
        otherwise inflate the window's floor even while some other, shorter
        page is the one actually showing.
        """
        stack = QStackedWidget(self)
        # Top padding here only, not on #activityRail beside it: the rail's
        # own nav rows have their own icon+label affordance to read as
        # "away from the edge" below the title bar strip (see
        # main_window._create_central_host), but a page's own content
        # starting flush against that same edge read as cramped.
        stack.setContentsMargins(0, 12, 0, 0)
        stack.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding,)
        # Page order follows NavigationBar.action_ids exactly - the rail
        # hands back the index of the tile that was clicked, nothing
        # richer, so the two lists are one ordering split across two
        # files. File comes first, directly under Workspace: it is where
        # a session starts (new, open, import, the recent list).
        stack.addWidget(self._scrollable(self._create_file_page()))
        stack.addWidget(self._data_page)
        stack.addWidget(self._scrollable(self._properties_control))
        stack.addWidget(self._create_series_operations_page())
        stack.addWidget(self._scrollable(self._create_database_page()))
        stack.addWidget(self._scrollable(self._create_developer_page()))
        return stack

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

    def _create_file_page(self) -> QWidget:
        """Workspace (New/Open/Import/Load demo), Save, and Open Recent.

        Page index 4, matching NavigationBar.action_ids' "nav_file" - the
        rail's own popup-menu File button (off macOS only) used to be the
        only way to reach these; a page reachable on every platform, same
        as Database's, replaces it. Load demo lives here, not under Help
        (see _app_menu_items) - starting from a demo is a way of starting a
        workspace, not a piece of documentation.
        """
        page = QWidget(self)
        # Same "white page the cards float on" convention the properties
        # toolbox pages get from apply_toolbox_page_metrics - a plain
        # QWidget paints no stylesheet background at all without
        # WA_StyledBackground, so without both of these this page stayed
        # transparent next to Tables (a card) and Chart properties
        # (already toolboxPage), the one grey gap in an otherwise white
        # left panel.
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(page)
        stdSizeAndlayout(layout)

        layout.addWidget(
            self._titled_card(page, _("Workspace"), self._fill_workspace_card, object_name="fileWorkspaceCard")
        )
        layout.addWidget(
            self._titled_card(page, _("Save"), self._fill_save_card, object_name="fileSaveCard")
        )
        layout.addWidget(
            self._titled_card(page, _("Open recent"), self._fill_recent_card, object_name="fileRecentCard")
        )
        layout.addStretch(1)
        return page

    def _titled_card(
        self,
        parent: QWidget,
        title: str,
        fill: Callable[[CardFrame], None],
        *,
        object_name: str | None = None,
    ) -> QWidget:
        """A section title *above* its card - see style.TitledCard."""
        titled = TitledCard(parent, title, object_name)
        fill(titled.card)
        return titled

    @staticmethod
    def _card_layout(card: CardFrame) -> QBoxLayout:
        """Return CardFrame's box layout, logging and recovering if absent."""
        layout = card.layout()
        if isinstance(layout, QBoxLayout):
            return layout
        applogger.error("CardFrame did not create a box layout")
        return QVBoxLayout(card)

    def _fill_workspace_card(self, card: CardFrame) -> None:
        layout = self._card_layout(card)
        new_open_row = QHBoxLayout()
        stdSizeAndlayout(new_open_row)
        create_action_button(parent=card, action_id="new", action=self._on_new_file, layout=new_open_row)
        create_action_button(parent=card, action_id="open", action=self._on_open_database, layout=new_open_row)
        create_action_button(parent=card, action_id="import", action=self._on_import_data, layout=new_open_row)
        new_open_row.addStretch(1)
        layout.addLayout(new_open_row)

        demo_row = QHBoxLayout()
        stdSizeAndlayout(demo_row)
        create_action_button(parent=card, action_id="load_demo", action=self._on_load_demo, layout=demo_row)
        demo_row.addStretch(1)
        layout.addLayout(demo_row)

    def _fill_save_card(self, card: CardFrame) -> None:
        layout = self._card_layout(card)
        save_row = QHBoxLayout()
        stdSizeAndlayout(save_row)
        create_action_button(parent=card, action_id="save", action=self._on_save, layout=save_row)
        create_action_button(parent=card, action_id="save_as", action=self._on_save_as, layout=save_row)
        save_row.addStretch(1)
        layout.addLayout(save_row)

    def _fill_recent_card(self, card: CardFrame) -> None:
        self._file_page = card
        self._recent_list_layout = self._card_layout(card)
        self._refresh_recent_list()

    def _refresh_recent_list(self) -> None:
        """(Re)populate the File page's Open Recent list from user.json.

        A list of one button per line, not the dropdown this used to be -
        every entry visible at once, the way the rest of this page already
        reads. Rebuilt on every call rather than cached: user.json's recent
        list changes between visits to this page (opening or saving a
        project adds to it), the same reason the native File menu's own
        Open Recent submenu rebuilds itself on every show.
        """
        layout = self._recent_list_layout
        self._clear_layout(layout)

        recent = get_recent_databases()
        if not recent:
            placeholder = QLabel(_("No recent projects"), self._file_page)
            placeholder.setProperty("muted", True)
            layout.addWidget(placeholder)
            return

        open_icon, _open_text, _open_tooltip = action_presentation("open")
        for path in recent:
            row = QHBoxLayout()
            stdSizeAndlayout(row)
            create_action_button(
                parent=self._file_page,
                action_id="open",
                action=partial(self._on_open_recent, path),
                layout=row,
                presentation=(open_icon, path.name, str(path.parent)),
            )
            row.addStretch(1)
            layout.addLayout(row)

        clear_row = QHBoxLayout()
        stdSizeAndlayout(clear_row)
        clear_icon, _clear_text, _clear_tooltip = action_presentation("clear")
        clear_button = create_action_button(
            parent=self._file_page,
            action_id="clear",
            action=self._on_clear_recent,
            layout=clear_row,
            presentation=(
                clear_icon,
                _("Clear list"),
                _("Forget the list of recently opened projects"),
            ),
        )
        # Red, not just another left-aligned button in the same column as
        # every recent-project row above it - the whole list otherwise reads
        # as one undifferentiated stack of rows, with nothing marking this
        # one as the one a misclick cannot undo.
        mark_destructive_button(clear_button)
        clear_row.addStretch(1)
        layout.addLayout(clear_row)

    @staticmethod
    def _clear_layout(layout: QLayout) -> None:
        """Empty *layout*, deleting every widget it holds - nested
        sub-layouts (one QHBoxLayout row per recent entry) included.
        ``takeAt`` alone only detaches an item; the widget underneath
        stays alive and parented, so without this each refresh would
        leave the previous run's buttons sitting invisibly on top of
        the new ones."""
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                continue
            sub_layout = item.layout()
            if sub_layout is not None:
                MainWindow._clear_layout(sub_layout)

    def _create_database_page(self) -> QWidget:
        """Query Builder, plus the database overview (info/tables/Optimize
        DB/links) - each its own section.

        Page index 3, matching NavigationBar.action_ids' "nav_database" -
        the two have to stay in step, since _set_nav_index addresses this
        stack by the same index the rail reports. DatabaseInfoPanel used to
        be its own modal dialog (see git history); embedded here it sits
        beside the action it always belonged next to, rather than in a
        window of its own. Optimize DB lives inside DatabaseInfoPanel's own
        header (see its optimize_action parameter), not here beside Query
        Builder: it operates on the size/page stats that card shows, not on
        the query workflow.
        """
        page = QWidget(self)
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(page)
        stdSizeAndlayout(layout)

        layout.addWidget(self._create_query_builder_section(page))

        self._database_info_panel = DatabaseInfoPanel(
            self._repo, page, optimize_action=self._on_optimize_db,
        )
        layout.addWidget(self._database_info_panel, 1)
        return page

    def _create_query_builder_section(self, parent: QWidget) -> QWidget:
        """Query Builder, as its own card: one button, with the catalogue's
        own description shown underneath rather than only in the tooltip -
        this page's first section, so it reads without hovering."""
        return self._titled_card(
            parent, _("Query Builder"), self._fill_query_builder_card,
            object_name="queryBuilderCard",
        )

    def _fill_query_builder_card(self, card: CardFrame) -> None:
        card_layout = self._card_layout(card)
        button_row = QHBoxLayout()
        stdSizeAndlayout(button_row)
        create_action_button(
            parent=card,
            action_id="query_builder",
            action=self._on_query_builder,
            layout=button_row,
        )
        button_row.addStretch(1)
        card_layout.addLayout(button_row)

        _icon, _text, tooltip = action_presentation("query_builder")
        description = QLabel(tooltip, card)
        description.setWordWrap(True)
        description.setProperty("muted", True)
        card_layout.addWidget(description)

    #: (action_id, handler) - the Developer menu's old group, one section
    #: each. See _create_developer_page for why they live here now instead.
    _DEV_TOOLS: tuple[tuple[str, str], ...] = (
        ("edit_localization", "_on_edit_localization"),
        ("series_operation_builder", "_on_series_operation_builder"),
        ("function_creator", "_on_function_creator"),
        ("renderer_helper", "_on_renderer_helper"),
    )

    def _create_developer_page(self) -> QWidget:
        """Scaffolding tools and the translation catalogue editor.

        Page index 5, matching NavigationBar.action_ids' "nav_developer" -
        the old "Developer" menu group, moved here rather than merely
        hidden the way File/Database's own groups still are (see
        _app_menu_items): none of these four actions carries a shortcut
        worth preserving, and off macOS there was no menu bar to begin
        with - only the rail's own flattened popup, which itself no
        longer has a button anywhere pointing at it, making this group
        unreachable there before this page existed.
        """
        page = QWidget(self)
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(page)
        stdSizeAndlayout(layout)

        for action_id, handler_name in self._DEV_TOOLS:
            layout.addWidget(
                self._create_dev_tool_section(page, action_id, handler_name)
            )

        layout.addStretch(1)
        return page

    def _create_dev_tool_section(
        self, parent: QWidget, action_id: str, handler_name: str
    ) -> QWidget:
        """One scaffolding tool as its own card - button plus the
        catalogue's own description shown underneath, same pattern as
        Query Builder's own section on the Database page."""
        _icon, text, tooltip = action_presentation(action_id)

        def fill(card: CardFrame) -> None:
            card_layout = self._card_layout(card)
            button_row = QHBoxLayout()
            stdSizeAndlayout(button_row)
            create_action_button(
                parent=card,
                action_id=action_id,
                action=getattr(self, handler_name),
                layout=button_row,
            )
            button_row.addStretch(1)
            card_layout.addLayout(button_row)

            description = QLabel(tooltip, card)
            description.setWordWrap(True)
            description.setProperty("muted", True)
            card_layout.addWidget(description)

        return self._titled_card(parent, text, fill, object_name=f"{action_id}Card")

    # ------------------------------------------------------------------
    # Activity rail
    # ------------------------------------------------------------------
    #: Actions moved into the real macOS menu bar via QAction.MenuRole -
    #: Preferences and About are relocated by Cocoa itself into the
    #: application menu next to the apple, wherever they were declared, so
    #: they read here exactly the way they read in the rail's popup menu.
    _MACOS_MENU_ROLES: dict[str, QAction.MenuRole] = {
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
                    action_menu_item("log_viewer", self._show_log_viewer),
                    None,
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
        (_snapshot_descriptors, refresh, refresh2, _on_chart_panel_deleted,
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
        entry = self._repo.undo_last()
        if entry is None:
            applogger.info("There is nothing to undo.")
            self._refresh_undo_item()
            return

        applogger.info("Undid: %s", entry.describe())
        self._table_panel.reload()
        self._preview.clear()
        self._reload_tabs()
        self._update_properties_for_current_chart()
        # The entry just consumed - update the menu to name the next one (or
        # disable it). On macOS this rebuilds the bar; safe here because Cocoa
        # has already closed the menu before dispatching this.
        self._refresh_undo_item()

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

    def _on_open_recent(self, db_path: Path) -> None:
        """Open one remembered database."""
        if not db_path.exists():
            # Between building the menu and clicking it - or a file on a
            # volume that has since been unmounted.
            applogger.warning(
                "That database is no longer there: %s",
                db_path,
                show_dialog=False,
                raise_error=False,
            )
            show_message(self, "database.open_failed", error=db_path)
            self._build_app_menu()
            self._refresh_recent_list()
            return

        applogger.info("Opening recent database: %s", db_path)
        try:
            self._switch_database(db_path)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_clear_recent(self) -> None:
        """Forget the list. The database currently open is not affected."""
        clear_recent_databases()
        self._build_app_menu()
        self._refresh_recent_list()

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
        rail = getattr(self, "_left_rail", None)
        if rail is not None:
            rail.set_help_menu(self._help_menu)
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
        # now (_create_file_page/_create_database_page) - a second, visible
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
    _MACOS_APP_MENU_SELECTORS: dict[str, str] = {
        "hide:": "Hide",
        "terminate:": "Close",
    }

    def _rename_macos_native_app_menu_items(self) -> None:
        """Strip the app name Cocoa insists on adding to About/Hide/Quit.

        `QAction.MenuRole` moves an action into the native application menu,
        but Cocoa then *replaces* its text with its own "About {name}" /
        "Hide {name}" / "Quit {name}" template - not read from the QAction at
        all, so nothing on the Qt side can change it.  The assumption in
        main.py that QApplication.setApplicationName() controls "{name}" does
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

        try:
            import AppKit  # type: ignore[import-not-found]
        except Exception:
            return

        try:
            app = getattr(AppKit, "NSApp", None)
            if app is None:
                ns_application = getattr(AppKit, "NSApplication")
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


    def _create_activity_rail(self) -> NavigationBar:
        """Create the navigation rail and say what its tiles do here.

        The rail itself is host-agnostic (see nav_bar's own docstring): it
        reports clicks and this is where they are given meaning, rather
        than the rail reaching into this window for a stack to switch and
        a settings dialog to open.
        """
        rail = NavigationBar(self, is_macos=IS_MACOS)
        rail.page_selected.connect(self._set_nav_index)
        rail.workspace_toggled.connect(self._on_workspace_tile_clicked)
        rail.action_triggered.connect(self._on_rail_action)
        rail.set_help_menu(self._help_menu)
        self._nav_group = rail.button_group
        self._nav_buttons = rail.buttons
        self._nav_action_ids = rail.action_ids
        self._settings_button = rail.settings_button
        self._help_button = rail.help_button
        return rail

    def _on_workspace_tile_clicked(self) -> None:
        """Collapse the panel beside the rail, but never re-open it here.

        The tile is a one-way "hide" - it is checked while hidden, and
        clicking a page tile is what brings the panel back - so a click
        arriving while the panel is already hidden must do nothing rather
        than toggle it open again.
        """
        if self._left_stack.isVisible():
            self._toggle_workspace()

    def _on_rail_action(self, key: str) -> None:
        """Run a footer tile's action, named by key."""
        if key == "settings":
            self._on_settings()

    def _create_left_panel(self) -> QWidget:
        """Create the left-side area: activity rail + stacked content."""
        # Flush, not padded: #leftPanelCard's own QSS draws it as a plain
        # white panel with a hairline on its own right edge (Mail/Finder's
        # sidebar-plus-list arrangement - see macos_native.qss), not a
        # floating rounded card. The default MARGIN_CARD (10px all round)
        # left a white gutter framing the grey activity rail on every
        # side - most visible above and below it, where nothing else
        # explains a gap.
        panel = CardFrame(
            self, "leftPanelCard", orientation=Qt.Orientation.Horizontal,
            margins=(0, 0, 0, 0),
        )
        panel.setProperty("elevated", True)
        panel.setMinimumSize(0, 0)
        panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        raw_layout = panel.layout()
        if not isinstance(raw_layout, QBoxLayout):
            raise RuntimeError("CardFrame did not create a box layout")
        layout = raw_layout
        layout.addWidget(self._left_rail, 0)
        layout.addWidget(self._left_stack, 1)

        return panel

    def _set_nav_index(self, index: int) -> None:
        """Select one content page and restore the pane when necessary."""
        if not 0 <= index < self._left_stack.count():
            return
        if not self._left_stack.isVisible():
            self._toggle_workspace()
        self._left_stack.setCurrentIndex(index)
        self._left_rail.select_page(index)
        if self._left_rail.action_ids[index] == "nav_file":
            # Same reason the native menu's Open Recent rebuilds on every
            # show: user.json's list can have changed since this page was
            # last visited.
            self._refresh_recent_list()
        applogger.debug("Left navigation page changed to index %s", index)

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
        split.setChildrenCollapsible(False)
        split.setOpaqueResize(True)
        split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        split.setMinimumSize(0, 0)
        split.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        relax_minimum_width(self._left_panel)
        # The rail is fixed-width and always visible, so the floor applies to
        # the content next to it, not to the panel as a whole. Read from the
        # rail itself, not the NAV_BAR_WIDTH constant: NavigationBar picks
        # its own width per platform (a macOS sidebar is wider than a
        # Windows Fluent tile rail), so the constant alone is only ever
        # right for one of them.
        self._left_panel.setMinimumWidth(PANEL_MIN_WIDTH + self._left_rail.width())

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
        self._chart_surface = CardFrame(
            self, "chartSurfaceCard", margins=(0, 0, 0, 0),
        )
        self._chart_surface.setProperty("elevated", True)
        self._chart_surface.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        chart_layout = self._chart_surface.layout()
        if not isinstance(chart_layout, QBoxLayout):
            raise RuntimeError("CardFrame did not create a box layout")
        chart_layout.addWidget(self._tabs, 1)

        split.addWidget(self._left_panel)
        split.addWidget(self._chart_surface)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        # Measured from the rail outwards, not a flat 360: the left pane
        # holds the navigation rail *and* the panel beside it, and the rail
        # is 200px wide on macOS against 132 on Windows. A fixed 360 left
        # the macOS panel 160px of usable width - the table list and the
        # series-operation panels were unreadably narrow in it - while
        # giving Windows 228 for the same panel.
        split.setSizes([self._left_rail.width() + PANEL_DEFAULT_WIDTH, 840])

        return split

    # ------------------------------------------------------------------
    # Database switching / reload
    # ------------------------------------------------------------------
    def _switch_database(self, db_path: Path) -> None:
        """Switch the UI to another database and rebind all dependent widgets."""
        applogger.info("Switching database to %s", db_path)
        self.setUpdatesEnabled(False)
        try:
            self._tabs.clear()
            self._preview.clear()
            QApplication.processEvents()

            self._repo.close()
            gc.collect()

            QApplication.processEvents()

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            self._table_panel.set_repo(self._repo)
            self._database_info_panel.set_repo(self._repo)
            self._update_window_title()
            set_last_database(db_path)

            self._table_panel.reload()
            self._reload_tabs()
            self._update_properties_for_current_chart()
            # set_last_database above has just put this file at the top of
            # the recent list; the menu showing that list has to be rebuilt
            # or it goes on showing the order from before the switch - same
            # for the File page's own list.
            self._build_app_menu()
            self._refresh_recent_list()
        finally:
            self.setUpdatesEnabled(True)


    def _reload_tabs(self) -> None:
        """Reload chart tabs from the current repository."""
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
                self._tabs.addTab(panel, name)
        finally:
            self._tabs.blockSignals(False)

        if self._tabs.count() == 0:
            self._clear_property_widgets()
            return

        if 0 <= current_index < self._tabs.count():
            self._tabs.setCurrentIndex(current_index)
        else:
            self._tabs.setCurrentIndex(0)

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
        self._update_properties_for_current_chart()

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

    def _on_grid_layout_requested(self, nrows: int, ncols: int) -> None:
        if self._properties_figure_id is None:
            return
        self._repo.set_figure_grid(
            self._properties_figure_id,
            nrows=int(nrows),
            ncols=int(ncols),
        )
        self._redraw_properties_chart()

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

    # Figure payload keys handled outside the generic copy below.
    _FIGURE_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"name", "layout"})

    def _on_figure_options_requested(self, payload: dict[str, Any]) -> None:
        """Persist the figure options the properties widget just emitted.

        Same rule as the axis handler: store the whole payload rather than a
        hand-listed subset, so a control added to the widget cannot silently
        fail to persist. None removes the key.
        """
        if self._properties_figure_id is None:
            return

        self._snapshot_descriptors(_("Figure properties"))
        options = self._properties_figure_options()

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
                return

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

    # Payload keys that are transport, not axis options: they are either
    # addressing (axis_id) or handled explicitly below.
    _AXIS_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"axis_id", "renderer"})

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
        self._snapshot_descriptors(_("Axis properties"))
        options = self._repo.get_axis_options(axis_id) or {}

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
        if series_desc.style is None:
            style: dict[str, Any] = {}
        elif isinstance(series_desc.style, dict):
            style = dict(series_desc.style)
        else:
            applogger.error("Series id=%r has invalid style.", series_desc.id)
            return
        self._snapshot_descriptors(_("Series properties"))
        # Same rule as the axis and figure handlers: everything the widget
        # sends is stored, minus the addressing keys.
        for key, value in payload.items():
            if key in {"series_id", "sql_query"}:
                continue
            if value is None:
                style.pop(key, None)
            else:
                style[key] = value

        sql_query = str(payload.get("sql_query", "") or "").strip()
        self._repo.update_series_style(series_id, style)
        self._repo.update_series_sql_query(series_id, sql_query)
        self._reload_property_widgets()
        self._redraw_properties_chart()

    # ------------------------------------------------------------------
    # Table panel actions
    # ------------------------------------------------------------------
    def _on_table_selected(self, table: str) -> None:
        """Update the preview panel when a table is selected."""
        applogger.debug("Selected table: %s", table)
        self._preview.set_context(self._repo, table)


    # ------------------------------------------------------------------
    # App menu actions
    # ------------------------------------------------------------------
        
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
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to create database: %s", exc)
            show_message(self, "database.create_failed", error=exc)

    def _on_load_demo(self) -> None:
        """Copy one of the shipped, pre-built demo projects, and open it.

        No "where to save it" dialog: that was the one step between picking a
        demo and seeing it, for a file whose name already says what it is and
        that nobody is expected to keep - the same reasoning that lets a first
        run open straight into a fixed, well-known path with no dialog of its
        own (see app.utils.startup.DEFAULT_DATABASE_NAME). The copy lands in
        ``projects/`` beside the application (demo_project.PROJECTS_DIR), not
        loose in the home directory: the copies are throwaway and one folder
        holds all of them. Loading the same demo again overwrites its previous
        copy in place, which is the point: it puts back the pristine version
        rather than asking what to call a second one. Someone who wants to
        keep a demo under its own name has Save As for that once it is open,
        same as any other project.
        """
        picker = LoadDemoDialog(self)
        if not picker.exec() or picker.chosen is None:
            return
        demo = picker.chosen

        target = PROJECTS_DIR / demo.path_name
        applogger.info("Loading demo project %r into %s", demo.file_name, target)
        try:
            copy_demo_project(demo, target)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to load demo project: %s", exc)
            show_message(self, "demo.build_failed", error=exc)
            return

        self._switch_database(target)

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
        except Exception as exc:  # noqa: BLE001
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
        try:
            self._repo.checkpoint()
            self.statusBar().showMessage(_("Database saved."), 4_000)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Checkpoint failed: %s", exc)

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
        try:
            saved_path = self._repo.save_as(target)
            self._switch_database(saved_path)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to save database as: %s", exc)
            show_message(self, "database.save_as_failed", error=exc)

    def _on_import_data(self, source_path: Path | None = None) -> None:
        """Open the import dialog and refresh UI if import succeeds.

        ``source_path`` is the file a drop arrived with; the dialog then opens
        already showing it, with its preview and its default table name, which
        is the whole point of dropping it rather than browsing for it.
        """
        dlg = ImportDataDialog(self._repo, parent=self)
        if source_path is not None:
            dlg.load_file(source_path)
        if dlg.exec():
            self._table_panel.reload()
            self._reload_tabs()


    def _on_new_plot_tab(self, icon: QIcon | None = None) -> None:
        """Create a chart tab, optionally using the operation-list icon.

        ``TableListPanel`` invokes this callback without arguments, while the
        series-operation page supplies the Plot operation icon.
        """
        panel = self._current_chart_panel()
        dialog = NewPlotTabDialog(
            self._repo,
            current_figure_id=(
                int(panel.figure_id) if panel is not None else None
            ),
            current_table=self._table_panel.current,
            parent=self,
        )

        if icon is None or icon.isNull():
            icon = SeriesOperationWidget.plugin_icon(
                {
                    "name": "NewPlotTabDialog",
                    "value": "Plot",
                    "icon": NewPlotTabDialog.Icon,
                    "builtin": True,
                }
            )

        if not icon.isNull():
            dialog.setWindowIcon(icon)

        if dialog.exec():
            self._table_panel.reload()
            self._reload_tabs()

    def _get_current_figure_id(self) -> tuple[ChartPanel | None, int | None]:
        """Return the figure ID of the currently selected chart tab, if any."""
        panel = self._current_chart_panel()
        if panel is None:
            return None, None
        id=panel.figure_id
        return (panel, id)
    
    def refresh(self):
        panel, id = self._get_current_figure_id()
        if id is not None and panel is not None:
            panel.reload()
            self._table_panel.reload()
        self._refresh_undo_item()

    def refresh2(self):
        """Refresh active chart and data panes after series-operation Preview/Apply.

        Series operation dialogs emit ``applied`` for three cases:
        - Preview wrote temporary chart/data changes.
        - Apply committed final changes.
        - Close/Cancel rolled preview changes back.

        The previous implementation only rebound the properties pane, so the
        chart canvas and selected dataset preview could remain stale.
        """
        panel, _figure_id = self._get_current_figure_id()
        if panel is not None:
            panel.reload()

        try:
            self._table_panel.reload()
            if self._table_panel.current:
                self._preview.set_context(self._repo, str(self._table_panel.current))
        except Exception:
            applogger.exception("Failed to refresh data panes after chart operation.")

        self._update_properties_for_current_chart()
        self._refresh_undo_item()

    def _open_series_operation(
        self,
        dialog_class: type,
        icon: QIcon | None = None,
    ) -> None:
        """Open one runtime series-operation dialog on the current chart."""
        panel, figure_id = self._get_current_figure_id()
        if figure_id is None or panel is None:
            return

        dialog = dialog_class(
            repo=self._repo,
            figure_id=figure_id,
            parent=self,
        )

        if icon is not None and not icon.isNull():
            dialog.setWindowIcon(icon)

        dialog.applied.connect(self.refresh2)
        # Bind results to the panel active when the dialog was opened.
        dialog.results_published.connect(
            lambda markup, target=panel: target.set_notes_html(
                markup,
                append=True,
            )
        )
        figure_count_before = self._tabs.count()
        dialog.exec()
        panel.reload()
        self._table_panel.reload()

        # An operation may have created a figure of its own, which is a new
        # chart tab rather than a change to this one - reloading only the
        # panel left it invisible until the next restart. But _reload_tabs()
        # rebuilds every tab from scratch - a full ChartPanel, a full render,
        # for every figure in the database - which is what made every Apply
        # redraw every chart regardless of how many the operation actually
        # touched. Only worth paying for when a figure was actually added;
        # the panel.reload() above already covers the ordinary case.
        if len(self._repo.load_figures_from_db()) != figure_count_before:
            self._reload_tabs()
        self._update_properties_for_current_chart()
   

    def _on_optimize_db(self) -> None:
        """Check the database, report what it found, then compact it.

        The findings go in the box twice on purpose: the problems inline, so
        that what is wrong is readable without clicking anything, and the whole
        grouped report - unreferenced tables included - behind Show Details,
        where it scrolls and can be copied into a bug report.  The status bar
        keeps the count, which is all it has room for.
        """
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
        """Switch to the embedded Database page (nav rail and Database menu
        both land here - see _create_database_page)."""
        self._set_nav_index(self._left_rail.action_ids.index("nav_database"))

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

    def _on_edit_localization(self) -> None:
        """Open Edit Localization: browse/edit a translation catalogue."""
        EditLocalizationDialog(self).exec()

    def _on_series_operation_builder(self) -> None:
        """Open the Series Operation Builder: scaffold a new operation file."""
        SeriesOperationBuilderDialog(self).exec()

    def _on_function_creator(self) -> None:
        """Open the Function Creator: scaffold a new fit-function file."""
        FunctionCreatorDialog(self).exec()

    def _on_renderer_helper(self) -> None:
        """Open the Renderer Helper: scaffold a new chart-type file."""
        RendererHelperDialog(self).exec()

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

    def _current_chart_panel(self) -> ChartPanel | None:
        """Return the currently selected chart panel, if any."""
        widget = self._tabs.currentWidget()
        if isinstance(widget, ChartPanel):
            return widget
        return None

    # ------------------------------------------------------------------
    # Frameless window resize (Windows and macOS)
    # ------------------------------------------------------------------
    #: How close to an edge, in pixels, counts as "grab this edge to resize".
    _RESIZE_MARGIN: int = 6

    #: Cursor shape for each edge/corner combination _resize_edge_at can
    #: return. Absent from here (the interior, or a maximized window) means
    #: the ordinary arrow.
    _RESIZE_CURSORS: dict[Qt.Edge, Qt.CursorShape] = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.TopEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.TopEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeBDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeBDiagCursor,
    }

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Round the window's real corners once it has a native handle.

        apply_native_macos_corner_radius needs window.winId() to resolve to
        an actual NSWindow, which is only meaningful once the widget is
        mapped - __init__ is too early. Tried once: unlike the QRegion mask
        in resizeEvent below, a CALayer's corner radius does not need
        redoing on every resize (see that function's own docstring).
        """
        super().showEvent(event)
        if self._native_corner_radius_attempted or not IS_MACOS:
            return
        self._native_corner_radius_attempted = True
        if apply_native_macos_corner_radius(self):
            self._native_corner_radius_active = True
            # The mask deliberately stays: see resizeEvent on why a
            # reported success is not proof the native clip is visible.
            # What did visibly double against the native edge was
            # #windowFrame's 1px hairline, not its fill: two independent
            # curves at the same radius do not land on the same pixels, so
            # the border traced one of them just inside the other. The
            # "nativeRounded" property drops that border (macos_native.qss)
            # while keeping the radius, so the fill still rounds.
            self._central_host.setProperty("nativeRounded", True)
            repolish_widget(self._central_host)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Keep the window's rounded-corner mask sized to the window.

        macOS only. The mask runs whether or not
        apply_native_macos_corner_radius (see showEvent) reported success:
        it used to be skipped in that case, on the grounds that the native
        CALayer clip anti-aliases the curve better than a QRegion can and
        the mask would only put the staircase back. That holds when the
        native clip actually takes - and when it silently does not, which
        is what was reported, skipping the mask leaves a square window
        with nothing rounding it. A slightly harder curve is worth far
        more than no curve, and the two agree on the same radius, so the
        mask is simply always applied and whichever clip is tighter wins.
        The mask is also what rounds the children filling the frame edge
        to edge (#leftPanelCard, the chart tabs), whose own square corners
        would otherwise poke past #windowFrame's painted curve - see
        apply_rounded_window_mask's own docstring. Cleared while
        maximized: a maximized window's edges are the screen's own, and a
        real macOS window is square-cornered there too.
        """
        super().resizeEvent(event)
        if not IS_MACOS:
            return
        if self.isMaximized():
            self.clearMask()
        else:
            apply_rounded_window_mask(self)

    def _resize_edge_at(self, pos: QPoint) -> Qt.Edge:
        """Return which edge(s) of _central_host *pos* is within the margin of.

        *pos* is in _central_host's own coordinates - the widget
        eventFilter below watches, not the window's (self never sees a
        mouse event of its own to filter; see __init__'s comment on why).
        """
        rect = self._central_host.rect()
        margin = self._RESIZE_MARGIN
        edges = Qt.Edge(0)
        if pos.x() <= margin:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= rect.width() - margin:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= margin:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= rect.height() - margin:
            edges |= Qt.Edge.BottomEdge
        return edges

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Give the frameless window (Windows and macOS) edge-drag resizing.

        FramelessWindowHint (see __init__) leaves the OS with no resize
        handles of its own to offer - this is the replacement, the same
        margin-hit-test-plus-startSystemResize technique
        CustomTitleBar.mousePressEvent already uses for the move case.
        A maximized window is left alone: its edges are the screen's own
        edges, and dragging those would resize the display area a user
        grabbing what looks like a window border did not mean to touch.
        """
        if not (
            (IS_WINDOWS or IS_MACOS)
            and watched is self._central_host
            and isinstance(event, QMouseEvent)
            and not self.isMaximized()
        ):
            return super().eventFilter(watched, event)

        edges = self._resize_edge_at(event.position().toPoint())
        if event.type() == QEvent.Type.MouseMove:
            cursor = self._RESIZE_CURSORS.get(edges, Qt.CursorShape.ArrowCursor)
            self._central_host.setCursor(cursor)
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

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------
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

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        """Accept a drag only when the drop would actually do something."""
        kind, _paths = self.accepted_drop(self.dropped_paths(event))
        if kind:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:  # noqa: N802
        """Keep the acceptance while the cursor moves over the window."""
        self.dragEnterEvent(event)

    def dropEvent(self, event: Any) -> None:  # noqa: N802
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
            except Exception as exc:  # noqa: BLE001
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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
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