"""Main application window.

Layout: a section-driven NavigationBar owns the left bar and its associated
panel stack. Chart tabs remain in the main surface. A properties QToolBox edits
the figure, axis, and series of whichever chart tab is active.

Chart reloads requested by the property pages are debounced, because each one
re-renders the whole figure.
"""
from __future__ import annotations

import gc
import re
import sys
from pathlib import Path
from time import monotonic
import warnings
from typing import Any, Callable, cast

import pandas as pd

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QShortcut,
)
from app import APP_ICON, APP_NAME
from app.charts import layout_presets
from app.widgets.chart_jump_bar import ChartJumpBar
from app.widgets.custom_title_bar import CustomTitleBar
from app.dialogs.log_viewer_dialog import LogViewerDialog
from app.dialogs.main_menus import MainWindowMenus
from app.data.repo._common import ensure_read_only_select
from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.widgets.chart_panel import ChartPanel
from app.widgets.nav_bar import NavBarItem, NavButton, NavigationBar, NavPanel
from app.widgets.recent_projects import RecentProjectsView
from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.dialogs.import_data_dialog import ImportDataDialog, is_importable
from app.data.demos import PROJECTS_DIR, copy_demo_project
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
    EXPANDED_CLIENT_AREA_HINT,
    IS_MACOS,
    PANEL_MIN_WIDTH,
    action_presentation,
    SPACING_DEFAULT,
    SPLITTER_HANDLE_WIDTH,
    apply_toolbox_header_metrics,
    apply_toolbox_page_metrics,
    CardFrame,
    icon_from_svg_source,
    relax_minimum_width,
    stdSizeAndlayout,
    symbol_icon,
)
from app.widgets.table_list import TableListPanel
from app.widgets.table_preview import TablePreviewPanel
from app.utils.config import (
    clear_recent_databases,
    get_constant,
    get_section,
    set_last_database,
    set_section,
)
from app.utils.dialog_state import restore_window_geometry, save_window_geometry
from app.utils.startup import PROJECT_FILE_FILTER
from app.utils.screen_fit import fit_on_show
from app.utils.messages import show_message
from app.utils import column_summary
from app.logs.logger import applogger
from app.utils.i18n import _
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QFileDialog,
    QLabel,
    QInputDialog,
    QMessageBox,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBox,
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


class MainWindow(MainWindowMenus, QMainWindow):
    _tabs: QTabWidget
    _properties_control: QToolBox
    _figure_widget: FigurePropertiesWidget
    _axis_widget: AxisPropertiesWidget
    _series_widget: SeriesPropertiesWidget
    _overlay_widget: OverlayPropertiesWidget
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
        # The scientific libraries the series operations need, imported in
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
        self._preview.refresh.connect(self.refresh)
        self._preview.chart_requested.connect(self._on_chart_from_columns)
        self._preview.histogram_requested.connect(self._on_histogram_from_column)
        # Left-side pages.
        self._data_page = self._create_data_page()
        self._properties_control = self._create_properties_control()
        self._configure_properties_control()

        # One navigation component owns both the bar and its panel stack.
        # MainWindow supplies section definitions and handles action-only rows.
        self._build_app_menu()
        self._left_panel: NavigationBar = self._create_navigation()
        # Compatibility aliases for splitter/title-bar helpers. They no longer
        # participate in page selection; NavigationBar selects widgets by key.
        self._left_rail = self._left_panel
        self._left_stack = self._left_panel.panels
        # Cmd+[ / Cmd+] on macOS (Qt maps Ctrl to Command), as in Xcode.
        for keys, step in (("Ctrl+[", -1), ("Ctrl+]", +1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(lambda s=step: self._step_chart(s))
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

        # Default page: the tables, not whatever happens to be first in the
        # rail. File sits above them now, and opening onto an empty file
        # page would hide the data the window was just opened on.
        self._select_navigation("tables")
        self._check_project_sql()
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
        # macOS: no window title at all. The title bar is transparent and
        # AppKit would draw the text across the rail; the status bar below
        # already names the project.
        self.setWindowTitle("" if IS_MACOS else f"{APP_NAME} | {project}")
        if hasattr(self, "_status_project"):
            self._status_project.setText(project)
            self._status_project.setToolTip(str(self._db_path) if self._db_path else "")

    def _toggle_workspace(self) -> None:
        """Hide the panel beside the bar, or bring it back (the Workspace row)."""
        self._left_panel.toggle_panels()

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

    def _create_data_page(self) -> NavPanel:
        """Tables above, the selected table's data below, the handle between them dragged."""
        page = NavPanel(self, resizable=True)
        page.add_frame(_("Tables"), self._table_panel, object_name="tablesFrame", stretch=1, margins=(0, 0, 0, 0))
        page.add_frame(_("Data"), self._preview, object_name="dataFrame", stretch=2, margins=(0, 0, 0, 0))
        assert page.splitter is not None
        self._data_split = page.splitter
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
        if operation.get("name") == "QueryBuilderDialog":
            self._on_query_builder()
            return

        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error(
                "Could not load series operation class: %r",
                operation.get("name"),
            )
            return

        self._open_series_operation(dialog_class, icon)

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
        control.addItem(
            self._axis_widget,
            _("Axis properties"),
        )
        control.addItem(
            self._series_widget,
            _("Series properties"),
        )
        # Last: annotations and reference lines are the finishing pass on a
        # chart, done once the data, the axes and the series are right.
        control.addItem(
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
        # The charts are chosen from the sidebar's Charts section and the
        # jump bar above the chart now (see _sync_chart_navigation); the
        # QTabWidget stays as the page stack, its own bar hidden.
        tab_bar.hide()

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

    def _create_new_table_page(self) -> NavPanel:
        """The ways to make a source table, each with its sentence."""
        page = NavPanel(self)
        page.add_buttons(_("Create a new table"), [
            NavButton("new", self._on_new_table_from_doe, _("From DOE"),
                      _("Generate an experiment matrix with factors and responses."), described=True),
            NavButton("new", self._on_new_blank_table, _("New Blank Table"),
                      _("Create an empty table with named columns."), described=True),
            NavButton("import", self._on_import_data, _("Import Data"),
                      _("Create a table from a file, clipboard, database, or web source."), described=True),
        ], object_name="newTableCard")
        return page

    def _on_new_table_from_doe(self) -> None:
        try:
            # Runtime import keeps this optional operation out of MainWindow's
            # static dependency graph. If the module is not installed, the
            # existing exception handler displays a useful message.
            from importlib import import_module

            doe_module = import_module("app.dialogs.doe_dialog")
            dialog_type = cast(Any, getattr(doe_module, "DOEExperimentDialog"))
            _panel, figure_id = self._get_current_figure_id()
            dialog = dialog_type(
                repo=self._repo,
                figure_id=int(figure_id or 0),
                parent=self,
            )
            dialog.applied.connect(self.refresh)
            dialog.exec()
            self._table_panel.reload()
            table_name = getattr(dialog, "created_table_name", None)
            if table_name:
                self._preview.set_context(self._repo, str(table_name))
                self._select_navigation("tables")
        except Exception as exc:  # noqa: BLE001
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
            self._table_panel.reload()
            self._preview.set_context(self._repo, name)
            self._select_navigation("tables")
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to create blank table: %s", exc)
            QMessageBox.warning(self, _("New Blank Table"), str(exc))

    def _create_file_page(self) -> NavPanel:
        """Workspace, Demo, Import, Save, Export and Open recent.

        Load demo has a frame of its own, not a place under Help: starting
        from a demo is a way of starting, not a piece of documentation.
        Import adds tables to the open project, where New and Open replace
        it - a different kind of step, so a frame of its own too.
        """
        page = NavPanel(self)
        page.add_buttons(_("Workspace"), [
            NavButton("new", self._on_new_file),
            NavButton("open", self._on_open_database),
            NavButton("project_info", self._on_project_info),
        ], object_name="fileWorkspaceCard")
        page.add_buttons(_("Demo"), [NavButton("load_demo", self._on_load_demo, described=True)],
                         object_name="fileDemoCard")
        page.add_buttons(_("Import"), [NavButton("import", self._on_import_data)], object_name="fileImportCard")
        page.add_buttons(_("Save"), [NavButton("save", self._on_save), NavButton("save_as", self._on_save_as)],
                         object_name="fileSaveCard")
        page.add_buttons(_("Export"), [NavButton("project_report", self._on_project_report)],
                         object_name="fileExportCard")
        self._recent_view = RecentProjectsView(page)
        self._recent_view.open_requested.connect(self._on_open_recent)
        self._recent_view.clear_requested.connect(self._on_clear_recent)
        # The recent list takes whatever height the page has left.
        page.add_frame(_("Open recent"), self._recent_view, object_name="fileRecentCard", stretch=1)
        return page

    def _on_project_info(self) -> None:
        """Author, creation date and notes of the open project."""
        if self._repo is None:
            return
        from app.dialogs.project_info_dialog import ProjectInfoDialog

        if ProjectInfoDialog(self._repo, self).exec():
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

    def _create_developer_page(self) -> NavPanel:
        """The project's history and database, then the scaffolding tools.

        The Database page's content lives here since that page went; Query
        Builder moved to the Series Operations list. Each tool has a frame
        of its own with its catalogue sentence under the button.
        """
        page = NavPanel(self)
        page.add_buttons(_("History"), [NavButton("project_history", self._on_project_history)],
                         object_name="projectHistoryCard")
        self._database_info_panel = DatabaseInfoPanel(self._repo, page, optimize_action=self._on_optimize_db)
        page.add_widget(self._database_info_panel)
        for action_id, handler in (
            ("edit_localization", self._on_edit_localization),
            ("series_operation_builder", self._on_series_operation_builder),
            ("function_creator", self._on_function_creator),
            ("renderer_helper", self._on_renderer_helper),
            ("log_viewer", self._show_log_viewer),
        ):
            page.add_buttons(action_presentation(action_id)[1], [NavButton(action_id, handler, described=True)],
                             object_name=f"{action_id}Card")
        return page

    # ------------------------------------------------------------------
    # Undo and recent projects (the menus that offer them: main_menus.py)
    # ------------------------------------------------------------------
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
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_clear_recent(self) -> None:
        """Forget the list. The database currently open is not affected."""
        clear_recent_databases()
        self._build_app_menu()
        self._recent_view.refresh()

    @staticmethod
    def _navigation_icon(action_id: str) -> QIcon:
        icon, _text, _tooltip = action_presentation(action_id)
        return icon

    def _create_navigation(self) -> NavigationBar:
        """Create the bar and panel host from one section dictionary."""
        file_panel = self._create_file_page()
        tables_panel = self._data_page
        new_table_panel = self._create_new_table_page()
        chart_properties_panel = self._scrollable(self._properties_control)
        series_operations_panel = self._create_series_operations_page()
        developer_panel = self._create_developer_page()

        sections: dict[str, dict[str, NavBarItem]] = {
            "Tools": {
                "workspace": NavBarItem(
                    _("Workspace"), self._navigation_icon("nav_workspace"), None,
                    _("Hide the panel"),
                ),
                "file": NavBarItem(
                    _("File"), self._navigation_icon("nav_file"), file_panel,
                    _("New, open, import and save"),
                ),
                "tables": NavBarItem(
                    _("Tables"), self._navigation_icon("nav_data"), tables_panel,
                    _("Show data tables"),
                ),
                "new_table": NavBarItem(
                    _("New table"), self._navigation_icon("nav_new_table"), new_table_panel,
                    _("Create a new table"),
                ),
                "chart_properties": NavBarItem(
                    _("Chart properties"), self._navigation_icon("nav_chart_options"),
                    chart_properties_panel, _("Edit the current chart"),
                ),
                "series_operations": NavBarItem(
                    _("Series operations"), self._navigation_icon("nav_series_operations"),
                    series_operations_panel, _("Transform or analyse chart series"),
                ),
                "developer": NavBarItem(
                    _("Developer"), self._navigation_icon("nav_developer"), developer_panel,
                    _("Developer tools"),
                ),
            },
            "Charts": {},
        }
        if IS_WINDOWS:
            sections["Settings"] = {
                "settings": NavBarItem(
                    _("Settings"), self._navigation_icon("settings"), None,
                    _("Open application settings"),
                )
            }

        header = CustomTitleBar(cast(Any, self), is_macos=True) if IS_MACOS else None
        # The pages keep the same gap from the rail as from the panel's right
        # and bottom edges - flush, every card ran into the rail and the
        # window's edge, which showed on Windows' white page (todo W-01).
        navigation = NavigationBar(
            sections, parent=self, header=header,
            panel_margins=(SPACING_DEFAULT, 12, SPACING_DEFAULT, SPACING_DEFAULT),
            panel_min_width=PANEL_MIN_WIDTH,
        )
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
        if key.startswith("chart:"):
            try:
                figure_id = int(key.partition(":")[2])
            except ValueError:
                return
            index = self._tab_index_of_figure(figure_id)
            if index >= 0:
                self._tabs.setCurrentIndex(index)

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
        # series-operation panels were unreadably narrow in it - while
        # giving Windows 228 for the same panel.
        split.setSizes([self._left_panel.visible_bar_width + PANEL_DEFAULT_WIDTH, 840])

        return split

    # ------------------------------------------------------------------
    # Database switching / reload
    # ------------------------------------------------------------------
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
        QTimer.singleShot(0, self, lambda: QMessageBox.warning(self, _("SQL blocked"), text))

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

    #: Chart type -> (SF Symbol, theme icon) for the sidebar's Charts list.
    _CHART_TYPE_SYMBOLS: tuple[tuple[str, str, str], ...] = (
        ("Pie", "chart.pie", "office-chart-pie"),
        ("Fishbone", "arrow.triangle.branch", "office-chart-line"),
        ("Pareto", "chart.bar.xaxis", "office-chart-bar"),
        ("Histogram", "chart.bar", "office-chart-bar"),
        ("Bar", "chart.bar", "office-chart-bar"),
        ("3D", "cube", "office-chart-area"),
        ("Surface", "cube", "office-chart-area"),
        ("Heatmap", "square.grid.3x3", "office-chart-area"),
        ("Contour", "circle.dotted.circle", "office-chart-area"),
        ("Hexbin", "hexagon", "office-chart-area"),
        ("Table", "tablecells", "x-office-spreadsheet"),
        ("Text", "textformat", "text-x-generic"),
        ("Box", "square.split.1x2", "office-chart-bar"),
        ("Violin", "waveform.path", "office-chart-area"),
        ("Scatter", "chart.dots.scatter", "office-chart-scatter"),
        ("Time", "clock", "office-chart-line"),
        ("Timeline", "calendar.day.timeline.left", "office-chart-line"),
    )

    def _chart_icon(self, chart_type: str) -> QIcon:
        for key, sf_name, theme in self._CHART_TYPE_SYMBOLS:
            if key.lower() in chart_type.lower():
                return symbol_icon(sf_name, theme)
        return symbol_icon("chart.xyaxis.line", "office-chart-line")

    def _sync_chart_navigation(self, *, rebuild: bool = True) -> None:
        """Rebuild chart actions and keep the chart jump bar synchronized."""
        navigation = getattr(self, "_left_panel", None)
        jump = getattr(self, "_jump_bar", None)
        if not isinstance(navigation, NavigationBar) or jump is None:
            return
        names = [self._tabs.tabText(i) for i in range(self._tabs.count())]
        current = self._tabs.currentIndex()
        if rebuild:
            try:
                chart_types = self._repo.figure_chart_types()
            except Exception:
                applogger.debug(
                    "Chart types for the sidebar could not be read.", exc_info=True
                )
                chart_types = {}
            chart_items: dict[str, NavBarItem] = {}
            for index, name in enumerate(names):
                widget = self._tabs.widget(index)
                if not isinstance(widget, ChartPanel):
                    continue
                figure_id = int(widget.figure_id)
                chart_items[f"chart:{figure_id}"] = NavBarItem(
                    name,
                    self._chart_icon(chart_types.get(figure_id, "")),
                    None,
                    name,
                )
            navigation.set_section("Charts", chart_items)
        jump.set_charts(names, current)

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
                self._sync_chart_navigation()
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
        if series_desc.options is None:
            style: dict[str, Any] = {}
        elif isinstance(series_desc.options, dict):
            style = dict(series_desc.options)
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
        try:
            # Typed by hand in the Series properties: the SQL guard's moment.
            if " " in sql_query:
                ensure_read_only_select(sql_query)
        except ValueError as exc:
            QMessageBox.warning(self, _("SQL blocked"), str(exc))
        else:
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
            return
        # A new project says who made it and what it is for from the start.
        self._project_info_asked = getattr(self, "_project_info_asked", set()) | {str(db_path)}
        self._on_project_info()

    def _on_load_demo(self) -> None:
        """Copy one of the shipped, pre-built demo projects, and open it.

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
        picker = LoadDemoDialog(self)
        if not picker.exec() or picker.chosen is None:
            return
        demo = picker.chosen

        target = SqliteRepo.ensure_dhub_extension(PROJECTS_DIR / demo.path_name)
        applogger.info("Loading demo project %r into %s", demo.file_name, target)
        # The copy is made with the current project closed: it may be this
        # very file - loading the open demo again, to start over - and
        # replacing a database under its own open connection left that
        # connection's pending changes to be replayed onto the fresh copy.
        try:
            self._switch_database(target, prepare=lambda: copy_demo_project(demo, target))
        except Exception as exc:  # noqa: BLE001
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
        self._ask_for_missing_project_info()
        self._refresh_project_preview()
        try:
            self._repo.checkpoint()
            self.statusBar().showMessage(_("Database saved."), 4_000)
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001 - a preview must never stop a save
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
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to save database as: %s", exc)
            show_message(self, "database.save_as_failed", error=exc)
            return
        # The copy has its own name, so its own preview.
        self._refresh_project_preview()

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

    def _on_new_plot_tab(
        self,
        icon: QIcon | None = None,
        *,
        table: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
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
            current_table=table or self._table_panel.current,
            preferred_columns=columns,
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
            # The chart just made becomes the current one - a new figure, or
            # the figure the new axis went into.
            result = dialog.chart_result
            self._reload_tabs(select_figure_id=result.figure_id if result is not None else None)

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
        # Always: a duplicated or grouped table must appear in the list
        # even when no chart is open.
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
        figures_before = {figure_id for figure_id, _name in self._repo.load_figures_from_db()}
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
        created = {figure_id for figure_id, _name in self._repo.load_figures_from_db()} - figures_before
        if created:
            # Shown: the figure the operation made is what its Apply produced.
            self._reload_tabs(select_figure_id=max(created))
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
        """Show the database overview, at the top of the Developer page (the
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
    # Frameless window resize (Windows)
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