"""Main application window: the navigation bar and its panels on the left, the charts on the right.

The window is assembled here - the panels, the bar, the splitter, the chart
tabs - and its larger concerns live in mixins of their own beside it:

    main_menus.py             the menu bar, the app menu, Undo and Open recent
    main_project.py           the project: new, open, save, demo, import, undo
    main_chart_properties.py  the Chart properties tabs and what they ask for
    main_frame.py             the frameless window's edges and dropped files
"""
from __future__ import annotations

import gc
import re
from pathlib import Path
import warnings
from typing import Any, cast

import pandas as pd

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QShortcut,
)
from app import APP_ICON, APP_NAME
from app.widgets.chart_jump_bar import ChartJumpBar
from app.main_window.custom_title_bar import CustomTitleBar
from app.dialogs.log_viewer_dialog import LogViewerDialog
from app.dialogs.main_chart_properties import MainWindowChartProperties
from app.main_window.main_frame import MainWindowFrame
from app.main_window.main_menus import MainWindowMenus
from app.main_window.main_project import MainWindowProject
from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.widgets.chart_panel import ChartPanel
from app.widgets.nav_bar import NavBarItem, NavButton, NavigationBar, NavPanel
from app.widgets.analysis_panel import AnalysisPanel
from app.scanners.table_operation_scanner import import_class_from_file as import_table_operation
from app.widgets.recent_projects import RecentProjectsView
from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.dialogs.credits_dialog import CreditsDialog
from app.widgets.database_info_panel import DatabaseInfoPanel
from app.developer_helpers.renderer_helper import RendererHelperDialog
from app.developer_helpers.series_operation_helper import SeriesOperationBuilderDialog
from app.developer_helpers.function_creator_helper import FunctionCreatorDialog
from app.developer_helpers.localization_helper import EditLocalizationDialog
from app.dialogs.query_builder_dialog import QueryBuilderDialog
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget
from app.widgets.chart_properties.overlay_properties import OverlayPropertiesWidget
from app.widgets.chart_properties.figure_properties import FigurePropertiesWidget
from app.widgets.chart_properties.series_properties import SeriesPropertiesWidget
from app.widgets.series_operation import SeriesOperationWidget
from app.scanners.series_operation_scanner import import_class_from_file
from app.styles.style import (
    EXPANDED_CLIENT_AREA_HINT,
    IS_MACOS,
    IS_WINDOWS,
    PANEL_MIN_WIDTH,
    action_presentation,
    SPACING_DEFAULT,
    SPLITTER_HANDLE_WIDTH,
    CardFrame,
    icon_from_svg_source,
    relax_minimum_width,
    symbol_icon,
)
from app.widgets.table_list import TableListPanel
from app.widgets.table_preview import TablePreviewPanel
from app.utils.config import (
    get_constant,
    get_section,
    set_section,
)
from app.utils.dialog_state import restore_window_geometry, save_window_geometry
from app.utils.screen_fit import fit_on_show
from app.utils.messages import show_message
from app.utils import column_summary
from app.logs.logger import applogger
from app.utils.i18n import _
from PySide6.QtWidgets import (
    QBoxLayout,
    QLabel,
    QInputDialog,
    QMessageBox,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QWidget,
)

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



class MainWindow(MainWindowMenus, MainWindowChartProperties, MainWindowProject, MainWindowFrame, QMainWindow):
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
        self._preview.refresh.connect(lambda: self.refresh(reload_preview=False))
        self._table_panel.changed.connect(self.refresh)
        self._preview.chart_requested.connect(self._on_chart_from_columns)
        self._preview.histogram_requested.connect(self._on_histogram_from_column)
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
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

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

        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error(
                "Could not load series operation class: %r",
                operation.get("name"),
            )
            return

        self._open_series_operation(dialog_class, icon)

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

    def _create_new_table_page(self) -> NavPanel:
        """The ways to make a source table, each with its sentence."""
        page = NavPanel(self)
        page.add_buttons(_("Create a new table"), [
            NavButton("new", self._on_new_blank_table, _("New Blank Table"),
                      _("Create an empty table with named columns."), described=True),
            NavButton("new", self._on_new_table_from_doe, _("Create a table from a DOE"),
                      _("Generate an experiment matrix with factors and responses."), described=True),
            NavButton("import", self._on_import_data, _("Import Data"),
                      _("Create a table from a file, clipboard or web source."), described=True),
            NavButton("import", self._on_database_table_import, _("Import Database Table"),
                      _("Create a table from a database table."), described=True),
            NavButton("query", self._on_create_query_table, _("Create Query Table"),
                      _("Create a table from a database query."), described=True),
        ], object_name="newTableCard")
        return page

    def _on_new_table_from_doe(self) -> None:
        from app.dialogs.doe_dialog import DOEExperimentDialog

        try:
            dialog = DOEExperimentDialog(self._repo, parent=self)
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
    # Navigation bar
    # ------------------------------------------------------------------
    @staticmethod
    def _navigation_icon(action_id: str) -> QIcon:
        icon, _text, _tooltip = action_presentation(action_id)
        return icon

    def _create_navigation(self) -> NavigationBar:
        """Create the bar and panel host from one section dictionary."""
        file_panel = self._create_file_page()
        tables_panel = self._data_page
        new_table_panel = self._create_new_table_page()
        chart_properties_panel = self._properties_control
        series_operations_panel = self._create_series_operations_page()
        analysis_panel = AnalysisPanel(self)
        analysis_panel.operation_requested.connect(self._on_table_operation_requested)
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
                "analysis": NavBarItem(
                    _("Analysis"), self._navigation_icon("nav_analysis"),
                    analysis_panel, _("Models of a table's columns"),
                ),
                "developer": NavBarItem(
                    _("Developer"), self._navigation_icon("nav_developer"), developer_panel,
                    _("Developer tools"),
                ),
            },
            "Charts": {},
        }
        if not IS_MACOS:
            # macOS has Help in its menu bar; elsewhere the rail is the menu.
            sections["Settings"] = {
                "help": NavBarItem(
                    _("Help"), self._navigation_icon("nav_help"), None,
                    _("User manual and credits"),
                ),
            }
        if IS_WINDOWS:
            sections.setdefault("Settings", {})["settings"] = NavBarItem(
                _("Settings"), self._navigation_icon("settings"), None,
                _("Open application settings"),
            )

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
        if key == "help":
            self._popup_help_menu()
            return
        if key.startswith("chart:"):
            try:
                figure_id = int(key.partition(":")[2])
            except ValueError:
                return
            index = self._tab_index_of_figure(figure_id)
            if index >= 0:
                self._tabs.setCurrentIndex(index)

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

    # Figure payload keys handled outside the generic copy below.
    _FIGURE_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"name", "layout"})

    # Payload keys that are transport, not axis options: they are either
    # addressing (axis_id) or handled explicitly below.
    _AXIS_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"axis_id", "renderer"})

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

    def refresh(self, *, reload_preview: bool = True) -> None:
        """Bring the chart, the tables and the properties back in step with the project.

        After a series operation (Preview, Apply, or the rollback of a
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
        except Exception:
            applogger.exception("Failed to refresh the data panes.")
        self._update_properties_for_current_chart()
        self._refresh_undo_item()

    def _on_table_operation_requested(self, operation: dict) -> None:
        """Open a table operation (Fit Model...) on the table selected in the Tables panel."""
        dialog_class = import_table_operation(operation)
        if dialog_class is None:
            applogger.error("Could not load table operation class: %r", operation.get("name"))
            return
        current = self._table_panel.current
        dialog = dialog_class(repo=self._repo, table=str(current) if current else None, parent=self)
        dialog.applied.connect(self.refresh)
        dialog.exec()
        figures = list(getattr(dialog, "created_figure_ids", []) or [])
        self._table_panel.reload()
        if figures:
            self._reload_tabs(select_figure_id=figures[0])

    def _open_series_operation(
        self,
        dialog_class: type,
        icon: QIcon | None = None,
    ) -> None:
        """Open one runtime series-operation dialog on the current chart."""
        panel = self._current_chart_panel()
        if panel is None:
            return
        figure_id = panel.figure_id

        dialog = dialog_class(
            repo=self._repo,
            figure_id=figure_id,
            parent=self,
        )

        if icon is not None and not icon.isNull():
            dialog.setWindowIcon(icon)

        dialog.applied.connect(self.refresh)
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
    # Qt events
    # ------------------------------------------------------------------
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

