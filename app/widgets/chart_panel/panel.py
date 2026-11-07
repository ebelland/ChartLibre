"""ChartPanel with direct Matplotlib canvas sizing.

This version deliberately avoids an intermediate host widget for FIT and
FIT_PROPORTIONAL modes.  The QScrollArea owns the visible chart viewport and the
FigureCanvasQTAgg is the scroll area's widget directly.
"""
from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
import PySide6.QtGui
from matplotlib import rcParams
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch, Rectangle, StepPatch, Wedge
from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.plugin_api.charts.render_figure import render_figure_from_descriptor
from app.styles.style import (
    CONTEXT_MENU_ICONS,
    SPLITTER_HANDLE_WIDTH,
    colored_icon,
    load_icon,
    MenuItem,
    action_presentation,
    create_menu,
    create_menu_item,
    create_toolbar_button,
    theme_is_dark,
)
from app.styles.style import action as catalogue_action
from app.utils.config import get_section, update_section
from app.utils.i18n import _
from app.widgets.chart_panel.canvas import ChartCanvas
from app.widgets.chart_panel.constants import (
    BACKGROUND_AUTO,
    CHART_AREA_MIN_HEIGHT,
    CONFIG_BACKGROUND_COLOR,
    CONFIG_INITIAL_ZOOM,
    CONFIG_MAX_ZOOM,
    CONFIG_MIN_ZOOM,
    CONFIG_RESIZE_MODE,
    CONFIG_SECTION,
    FIGURE_VIEW_OPTIONS_KEY,
    FIXED_MODE_SCREEN_DPI,
    LIVE_REFRESH_INTERVAL_MS,
    NOTES_FIRST_OPEN_MIN_HEIGHT,
    NOTES_INITIAL_FRACTION,
    NOTES_MIN_HEIGHT,
    PICK_TOLERANCE_POINTS,
    RESIZE_MODES,
    TOOLBAR_ACTIONS_TO_REMOVE,
    axis_text,
)
from app.widgets.chart_panel.tools import ChartToolsMixin
from app.widgets.html_results import HtmlResultsView

#: The Matplotlib navigation actions the chart's own buttons stand for, with the catalogue entry of each.
#: None is a separator.
NAVIGATION_BUTTONS: Final[tuple[tuple[str, str] | None, ...]] = (
    ("home", "chart_home"), ("back", "chart_back"), ("forward", "chart_forward"), None,
    ("pan", "chart_pan"), ("zoom", "chart_zoom"),
)


def _symbol_ink() -> str:
    """The grey the chart's fallback icons are drawn in: dark on a light theme, light on a dark one."""
    return "#c9ced6" if theme_is_dark() else "#4a5263"


def _inked(icon: QIcon, colour: str) -> QIcon:
    """*icon* with its ink in *colour*, at the size and padding it has.

    Not ``colored_icon``: that one scales the glyph's ink to fill the square,
    which for Matplotlib's icons - drawn with a wide margin - made them
    smaller than the platform glyphs beside them on a Retina screen.
    """
    inked = QIcon()
    for ratio in (1.0, 2.0):
        pixmap = icon.pixmap(QSize(18, 18), ratio)
        if pixmap.isNull():
            continue
        pixmap = pixmap.copy()
        painter = QPainter(pixmap)
        try:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(pixmap.rect(), QColor(colour))
        finally:
            painter.end()
        inked.addPixmap(pixmap)
    return inked if not inked.isNull() else icon


class ChartNavigation(NavigationToolbar2QT):
    """Matplotlib's navigation - its view history, pan and zoom - without its bar.

    The bar itself stays hidden: its bold black icons and native buttons were
    the one part of the window that looked like no platform. Its actions are
    shown instead by the chart's own buttons (``QToolButton.setDefaultAction``
    shares the action, so the checked Pan or Zoom and a greyed Back follow
    Matplotlib's state with no code of ours), and the cursor's coordinates it
    would write in its own label go to *on_message*. Pan, zoom and their
    history are Matplotlib's, unchanged: nothing here reimplements them.
    """

    _on_message: Any = None

    def set_message(self, s: str) -> None:
        if self._on_message is not None:
            self._on_message(s)


#: How long a selection made in Chart properties stays highlighted on the chart.
FLASH_MS = 900


#: The highlight's colour: the application's accent.
FLASH_COLOR = "#2563EB"


#: A series' style keys the chart can show by changing its lines and points, with no new render.
PREVIEW_STYLE_SETTERS = frozenset({"color", "linestyle", "marker", "markersize"})


#: A Stem Plot's style: one Matplotlib format string for each of its three parts.
STEM_FMT_KEYS = frozenset({"linefmt", "markerfmt", "basefmt"})


#: The values that mean "no line" or "no marker".
_NONE_STYLES = frozenset({"", " ", "None", "none"})


def _follows(artist: Any) -> frozenset[str]:
    """The series' style keys *artist* takes (SeriesArtistTags.follow): all of them unless said."""
    return getattr(artist, "_chartlibre_follows", PREVIEW_STYLE_SETTERS)


def _style_steps(
    artist: Any, changes: dict[str, Any], *, has_points: bool, sizes: bool, has_lines: bool = True
) -> tuple[bool, list[Any]]:
    """(whether *artist* can show *changes* as it is, the calls that do it).

    A line keeps being a line and its markers markers; on a series drawn as
    a line and points, the marker belongs to the points. A series drawn as
    points only has no line to restyle: a line style that would draw one is for
    the render to show (*has_lines* False), where it used to be ignored while the
    edit was reported as shown.
    """
    steps: list[Any] = []
    for key, value in changes.items():
        if key not in _follows(artist):
            continue
        if isinstance(artist, Patch) and not isinstance(artist, Wedge):
            # A step line (Stairs) and its legend swatch: an outline, and a fill if it has one.
            if key == "color":
                to_rgba(value)  # ValueError for a colour Matplotlib does not know
                filled = artist.get_fill()
                steps.append(lambda v=value, fill=filled: (artist.set_edgecolor(v), fill and artist.set_facecolor(v)))
            elif key == "linestyle":
                if str(value) in _NONE_STYLES or str(artist.get_linestyle()) in _NONE_STYLES:
                    return False, []
                steps.append(lambda v=value: artist.set_linestyle(v))
        elif isinstance(artist, Line2D):
            if key == "color":
                to_rgba(value)  # ValueError for a colour Matplotlib does not know
                steps.append(lambda v=value: artist.set_color(v))
            elif key == "linestyle":
                if str(value) in _NONE_STYLES or artist.get_linestyle() in _NONE_STYLES:
                    return False, []
                steps.append(lambda v=value: artist.set_linestyle(v))
            elif not has_points:
                if str(artist.get_marker()) in _NONE_STYLES or (key == "marker" and str(value) in _NONE_STYLES):
                    return False, []
                setter = artist.set_marker if key == "marker" else artist.set_markersize
                steps.append(lambda v=value, set_=setter: set_(v))
        else:  # the points
            if key == "linestyle":
                if not has_lines and str(value) not in _NONE_STYLES:
                    return False, []  # it would add a line: only a render can
                continue
            if key == "color":
                if artist.get_array() is not None:
                    return False, []
                to_rgba(value)  # ValueError for a colour Matplotlib does not know
                follows = np.array_equal(artist.get_edgecolor(), artist.get_facecolor())
                steps.append(lambda v=value, edge=follows: (artist.set_facecolor(v), edge and artist.set_edgecolor(v)))
            elif key == "marker":
                if str(value) in _NONE_STYLES:
                    return False, []
                style = MarkerStyle(value)
                path = style.get_path().transformed(style.get_transform())
                steps.append(lambda p=path: artist.set_paths([p]))
            elif key == "markersize" and sizes:
                if len(artist.get_sizes()) > 1:
                    return False, []  # sized by a column
                steps.append(lambda v=value: artist.set_sizes([float(v) ** 2]))
    return True, steps


class ChartPanel(ChartToolsMixin, QFrame):
    """Embed a repository-backed Matplotlib chart.

    Important sizing rule:
        FIT and FIT_PROPORTIONAL do not use any host panel. The Matplotlib
        canvas is installed directly into the QScrollArea.
    """

    delete_requested = Signal(int)

    #: Emitted after the panel itself writes an undo-recordable change to the
    #: repository (a reference line dropped from the right-click menu). The
    #: window listens so it can bring the Undo entry up to date - the native
    #: macOS menu bar never asks on its own.
    figure_edited = Signal()

    #: Emitted with a one-line description of whatever the user just clicked on
    #: in the chart. The panel has no status bar of its own, and deliberately
    #: does not go looking for the window's: it says what was selected and lets
    #: whoever owns the window decide where that belongs.
    selection_changed = Signal(str)
    #: A plain click on the chart: {"axis_id": int | None, "series": str | None} (see the Picking section).
    chart_object_clicked = Signal(dict)
    #: The tables whose Selected rows a box selection on this chart just set (todo R-07).
    rows_selected = Signal(list)
    #: The tables whose Show only Selected this chart just switched (R-07): every chart of them draws again.
    selected_only_changed = Signal(list)
    #: A "What next?" link in the notes (a report kept with the chart) was clicked: its URL.
    next_step_requested = Signal(str)
    #: The project's stored functions were trusted from this chart's note (todo ID47): the charts of them draw again.
    functions_trusted = Signal()

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
        *,
        defer_render: bool = False,
    ) -> None:
        """Initialise the panel and render its first chart state.

        ``defer_render=True`` skips that first render - the caller must
        call :meth:`ensure_rendered` once the panel actually needs to show
        something. For a project opened with many figures (one ChartPanel
        per figure, all built up front by MainWindow._reload_tabs), only
        the one tab actually shown needs a render before the window becomes
        interactive; the rest can wait until the user clicks their tab.
        """
        super().__init__(parent)

        self._repo = repo
        self._figure_id = int(figure_id)
        self._needs_initial_render = bool(defer_render)
        self._deleted = False
        self._pending_canvas_sync = False
        self._pending_canvas_redraw = False
        self._last_canvas_size = QSize()
        #: Bumped whenever the figure's contents are rebuilt (reload); with
        #: the figure's size and dpi it is what a draw shows. _drawn_state is
        #: the same triple at the last draw, so a geometry sync can tell a
        #: canvas that is already up to date from one that needs drawing.
        self._content_revision = 0
        self._drawn_state: tuple[Any, ...] | None = None

        self._canvas_sync_timer = QTimer(self)
        self._canvas_sync_timer.setSingleShot(True)
        self._canvas_sync_timer.timeout.connect(self._run_pending_canvas_geometry_sync)

        # A short second pass catches tab/page activation and final layout
        # geometry without changing the old gwXG resize architecture.
        self._canvas_late_sync_timer = QTimer(self)
        self._canvas_late_sync_timer.setSingleShot(True)
        self._canvas_late_sync_timer.timeout.connect(self._run_pending_canvas_geometry_sync)

        self._config = get_section(CONFIG_SECTION)
        chart_config: dict[str, Any] = self._config

        self._min_zoom_percent:int = int(
            chart_config.get(CONFIG_MIN_ZOOM, 25)
        )
        self._max_zoom_percent:int = int(
            chart_config.get(CONFIG_MAX_ZOOM, 250)
        )
        if self._max_zoom_percent < self._min_zoom_percent:
            self._max_zoom_percent = self._min_zoom_percent

        self._zoom_percent:int = self._clamp_zoom(
            int(chart_config.get(CONFIG_INITIAL_ZOOM, 64))
        )
        self._resize_mode = self._normalized_resize_mode(
            # Fixed by default: the figure at its own size, zoom and scroll bars.
            chart_config.get(CONFIG_RESIZE_MODE, "FIXED")
        )
        # "auto": a desk colour from the theme, so the figure reads as a sheet
        # of paper on it. White was the default written back by every panel,
        # never a choice anyone made - nothing in the interface sets it - so
        # a saved white is read as the default too.
        saved_background = str(chart_config.get(CONFIG_BACKGROUND_COLOR, BACKGROUND_AUTO) or BACKGROUND_AUTO)
        self._background_color: str = (
            BACKGROUND_AUTO if saved_background.lower() in ("#ffffff", "white") else saved_background
        )

        # The config value is only the default for newly-created figures.
        # If this figure has its own view state, it wins.
        self._apply_persisted_view_state()

        self._apply_persisted_figure_metrics_to_rcparams()

        self._figure = Figure()
        self._reset_figure_metrics_from_rcparams_for_reload()
        self._fixed_figure_size_inches = tuple(
            float(v) for v in self._figure.get_size_inches()
        )
        # The configured dpi, never the figure's: see
        # _capture_fixed_metrics_from_rendered_figure.
        self._fixed_figure_dpi = float(rcParams.get("figure.dpi", 100.0) or 100.0)

        # Legend handle -> the artists it stands for. Rebuilt by every render,
        # bound here so a pick arriving before the first one cannot fail.
        self._legend_targets: dict[Any, list[Any]] = {}

        self._canvas = ChartCanvas(self._figure)
        self._canvas.resize_settled.connect(lambda: self._schedule_canvas_geometry_sync(redraw=True))
        self._init_hover_readout()
        self._init_ruler()
        self._init_crosshair()
        self._init_annotation_drag()
        self._init_box_selection()
        self._init_live_updates()
        self._toolbar = ChartNavigation(self._canvas, parent=self, coordinates=False)

        self._init_ui()
        self.set_resize_mode(self._resize_mode, persist=False, redraw=False)
        if self._needs_initial_render:
            self._capture_fixed_metrics_from_rendered_figure()
        else:
            self.reload()

    def reapply_view_geometry(self) -> None:
        """Put this panel's own view sizing back after someone else set metrics.

        The Figure properties panel writes the configured size and dpi onto
        the figure it is connected to; in FIXED mode that dpi has the zoom
        folded in, so without this the zoom was lost and the figure drawn at
        a fraction of its canvas.
        """
        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()
        else:
            # The figure's own dpi is for export; on screen it is drawn at the
            # screen's density (see _reset_figure_metrics_from_rcparams_for_reload).
            self._set_configured_dpi(FIXED_MODE_SCREEN_DPI)
        # Now, not on the next turn of the event loop: changing the dpi has
        # already queued a draw_idle, and it ran first - drawing the figure
        # at its export size (6.4 x 4.8 in) into the canvas, which then
        # showed that buffer corner-anchored and stale after a tab switch.
        self._sync_canvas_geometry(redraw=True)
        self._schedule_canvas_geometry_sync(redraw=True)

    def ensure_rendered(self) -> None:
        """Run the deferred first render, once, if ``defer_render`` skipped it.

        Idempotent, so callers do not need to track whether this panel's
        turn has already come - the properties panel's own "connect to the
        current chart" path calls this on every tab switch regardless.
        """
        if not self._needs_initial_render:
            return
        self._needs_initial_render = False
        self.reload()

    # ------------------------------------------------------------------
    # UI initialisation
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        """Build widgets with direct-canvas FIT modes and scrollable FIXED mode.

        Display policy:
            * FIT: the Matplotlib canvas is placed directly in ChartPanel's
              main layout below the toolbar row and fills that client area.
            * FIT_PROPORTIONAL: the same direct canvas placement is used,
              but the canvas width/height are fixed to preserve the natural
              figure aspect ratio.
            * FIXED: the canvas is moved into a QScrollArea.  The scroll-area
              viewport shows the configured chart panel background color.
        """
        self.setObjectName("ChartPanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(0, 0)

        self._configure_canvas()
        self._configure_toolbar()
        self._top_row = self._build_top_row()

        self._fixed_scroll_area = QScrollArea(self)
        self._fixed_scroll_area.setObjectName("chartScrollArea")
        self._fixed_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._fixed_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fixed_scroll_area.setWidgetResizable(False)
        self._fixed_scroll_area.viewport().installEventFilter(self)
        self._configure_fixed_scroll_area()

        # Keep the old gwXG chart layout intact, but host it in a stable
        # splitter child so the HTML results pane can be resized and collapsed
        # independently from the chart display.
        self._chart_area = QWidget(self)
        self._chart_area.setObjectName("chartArea")
        self._chart_area.setMinimumHeight(CHART_AREA_MIN_HEIGHT)
        self._chart_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # The chart area changes size on its own whenever the results pane
        # below it opens, closes or has its divider dragged - the panel
        # itself stays the same size then, so its resizeEvent never fires.
        # FIT follows the canvas's own resize, but FIT_PROPORTIONAL gives
        # the canvas a fixed size and would keep the stale one.
        self._chart_area.installEventFilter(self)

        self._main_layout = QVBoxLayout(self._chart_area)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        # Show only Selected on one of the chart's tables: said over the chart, with the way back (R-07).
        self._filter_note = QLabel(self._chart_area)
        self._filter_note.setProperty("muted", True)
        self._filter_note.setWordWrap(True)
        self._filter_note.setTextFormat(Qt.TextFormat.RichText)
        self._filter_note.setContentsMargins(8, 4, 8, 4)
        self._filter_note.linkActivated.connect(self._on_filter_note_link)
        self._filter_note.hide()
        self._main_layout.addWidget(self._top_row, 0)
        self._main_layout.addWidget(self._filter_note, 0)
        self._main_layout.addWidget(self._canvas, 1)

        self._notes_html = ""
        self._notes_view = HtmlResultsView(self)
        self._notes_view.link_requested.connect(self.next_step_requested)
        self._notes_view.setObjectName("chartNotesView")
        self._notes_view.setMinimumHeight(NOTES_MIN_HEIGHT)
        self._notes_view.clear_requested.connect(self.clear_notes)

        self._content_split = QSplitter(Qt.Orientation.Vertical, self)
        self._content_split.setObjectName("chartContentSplit")
        self._content_split.setChildrenCollapsible(True)
        self._content_split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        self._content_split.addWidget(self._chart_area)
        self._content_split.addWidget(self._notes_view)
        self._content_split.setCollapsible(0, False)
        self._content_split.setCollapsible(1, True)
        self._content_split.setStretchFactor(0, 1)
        self._content_split.setStretchFactor(1, 0)

        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(self._content_split, 1)

        # Empty until an operation publishes something.
        self._notes_view.setVisible(False)
        self._restore_notes_state()

        self._apply_background_color_to_widgets()

    def _configure_fixed_scroll_area(self) -> None:
        """Configure the scroll area used only by FIXED mode."""
        self._fixed_scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self._fixed_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._fixed_scroll_area.setMinimumSize(0, 0)
        self._fixed_scroll_area.setMaximumSize(QSize(16777215, 16777215))
        self._fixed_scroll_area.viewport().setMinimumSize(0, 0)

    #: How much one wheel notch changes the visible range.  0.8 zooms in by a
    #: fifth per notch: small enough to stop where you meant to, large enough
    #: that crossing two orders of magnitude does not need forty notches.
    ZOOM_STEP: Final[float] = 0.8

    def _configure_canvas(self) -> None:
        """Configure the Matplotlib canvas for direct scroll-area use."""
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Connected through Matplotlib rather than through a Qt event filter.
        # The handler needs the cursor position in *data* coordinates to zoom
        # about the point under it, and mpl_connect delivers exactly that;
        # a Qt wheel event carries pixels, which would have to be transformed
        # back by hand for each axis.
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        # Picking is likewise Matplotlib's own: it already knows which artist
        # owns each pixel and which of its points are under the cursor, and
        # redoing that hit test against Qt coordinates would mean reimplementing
        # marker sizes, transforms and axis scales for every renderer.
        self._canvas.mpl_connect("pick_event", self._on_pick)
        # A plain click also selects what it landed on in Chart properties.
        self._canvas.mpl_connect("button_press_event", self._on_object_click_press)
        self._canvas.mpl_connect("button_release_event", self._on_object_click_release)
        self._canvas.mpl_connect("key_press_event", self._on_escape_to_figure)
        # Ctrl + left click creates an axis-level annotation at the clicked
        # data coordinates.  This is deliberately a mouse-button event, not
        # a pick event, so it works even when there is no marker exactly
        # under the cursor.
        # Hover readout. Throttled and blitted; see _on_motion.
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        # Any of these repaints the canvas, so the cached background that
        # blitting restores is no longer what is underneath.
        self._canvas.mpl_connect("draw_event", self._on_canvas_drawn)
        self._canvas.mpl_connect("resize_event", lambda _e: self._invalidate_hover_background())
        self._canvas.mpl_connect("axes_leave_event", lambda _e: self._hide_hover_annotation())
        self._canvas.mpl_connect("axes_leave_event", lambda _e: self._hide_crosshair())
        # Dragging an axis annotation to reposition it: press picks it up
        # (or does nothing, if the click missed one), motion follows the
        # pointer, release writes the new position back and persists it.
        self._canvas.mpl_connect("button_press_event", self._on_annotation_drag_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_annotation_drag_motion)
        self._canvas.mpl_connect("button_release_event", self._on_annotation_drag_release)
        # Box selection: an alternative tool, toggled on from the actions
        # menu - see _on_annotation_drag_press's own guard for why the two
        # never fire on the same drag.
        self._canvas.mpl_connect("button_press_event", self._on_box_select_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_box_select_motion)
        self._canvas.mpl_connect("button_release_event", self._on_box_select_release)
        self._canvas.setMinimumSize(0, 0)
        self._canvas.setMaximumSize(QSize(16777215, 16777215))
        self._canvas.setAutoFillBackground(False)
        # Deliberately *not* translucent.  FigureCanvasQTAgg sets
        # WA_OpaquePaintEvent - "I paint every pixel, do not erase me" - and
        # WA_TranslucentBackground is the opposite promise, that the background
        # shows through.  With both set nothing ever clears the canvas rect
        # while Agg blits an RGBA buffer over it: Windows happens to hand back
        # a clean surface, macOS keeps the previous frame, and successive draws
        # pile up as ghost axes and thickening text.
        #
        # There is nothing to see through it in any case: paintEvent fills the
        # panel and the viewport filter fills the strip the canvas does not
        # cover, so the transparency only ever exposed stale pixels.
        #
        # _testChart.py, beside ChartLibre.py, shows the difference on a Mac.
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._canvas.installEventFilter(self)
        self._canvas.customContextMenuRequested.connect(self._show_context_menu)

    def _configure_toolbar(self) -> None:
        """Keep Matplotlib's navigation, hide its bar, and give its actions the app's icons.

        An action whose catalogue icon is missing on this machine (a Linux
        desktop without an icon theme) keeps Matplotlib's own, so no button is
        ever blank.
        """
        self._toolbar.setObjectName("chartToolbar")
        self._remove_toolbar_actions(TOOLBAR_ACTIONS_TO_REMOVE)
        self._toolbar.hide()
        actions = getattr(self._toolbar, "_actions", {})
        for entry in NAVIGATION_BUTTONS:
            if entry is None or entry[0] not in actions:
                continue
            name, action_id = entry
            icon, text, tooltip = action_presentation(action_id)
            action = actions[name]
            # The platform's glyph; failing that Matplotlib's own, lightened
            # from its bold black to the ink every other icon is drawn in.
            action.setIcon(icon if not icon.isNull() else _inked(action.icon(), _symbol_ink()))
            if text:
                action.setText(text)
            if tooltip:
                action.setToolTip(tooltip)
        self._toolbar._on_message = self._show_navigation_message

    def _show_navigation_message(self, text: str) -> None:
        """Show what Matplotlib's navigation says - the cursor's x and y, the active tool - beside the buttons."""
        label = getattr(self, "_coordinates_label", None)
        if label is not None:
            label.setText(" ".join(str(text or "").split()))

    def _build_navigation_pill(self, parent: QWidget) -> QFrame:
        """The chart's navigation buttons in one rounded group: reset, back, forward | pan, zoom."""
        pill = QFrame(parent)
        pill.setObjectName("chartNavPill")
        pill.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(pill)
        layout.setContentsMargins(3, 2, 3, 2)
        layout.setSpacing(1)
        actions = getattr(self._toolbar, "_actions", {})
        for entry in NAVIGATION_BUTTONS:
            if entry is None:
                separator = QFrame(pill)
                separator.setObjectName("chartNavSeparator")
                separator.setFixedSize(1, 16)
                layout.addSpacing(3)
                layout.addWidget(separator, 0, Qt.AlignmentFlag.AlignVCenter)
                layout.addSpacing(3)
                continue
            action = actions.get(entry[0])
            if action is None:
                continue
            button = QToolButton(pill)
            button.setObjectName("chartToolButton")
            button.setDefaultAction(action)
            button.setAutoRaise(True)
            button.setIconSize(QSize(18, 18))
            # Fixed, not left to the sheet: a style without the pill's rule
            # (or the generic QToolButton padding) would stretch or squeeze it.
            button.setFixedSize(QSize(28, 28))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            layout.addWidget(button)
        # Delete the chart: at the pill's end, apart and in red, as a destructive action (out of the menu).
        separator = QFrame(pill)
        separator.setObjectName("chartNavSeparator")
        separator.setFixedSize(1, 16)
        layout.addSpacing(3)
        layout.addWidget(separator, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addSpacing(3)
        delete = QToolButton(pill)
        delete.setObjectName("chartToolButton")
        delete.setIcon(colored_icon(load_icon("delete"), "#dc2626", 18))
        delete.setIconSize(QSize(18, 18))
        delete.setFixedSize(QSize(28, 28))
        delete.setAutoRaise(True)
        delete.setToolTip(_("Delete this figure"))
        delete.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        delete.clicked.connect(lambda _checked=False: self.delete_chart())
        layout.addWidget(delete)
        return pill

    def _build_top_row(self) -> QWidget:
        """Create a fixed-height top control row."""
        top_row = QWidget(self)
        top_row.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )

        layout = QHBoxLayout(top_row)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)
        layout.addWidget(self._build_navigation_pill(top_row), 0)
        # Where the cursor is, in the data's units - what Matplotlib's bar
        # wrote at its right end. Ignored width: a long reading is cut, it
        # never pushes the zoom controls out of the row.
        self._coordinates_label = QLabel("", top_row)
        self._coordinates_label.setObjectName("chartCoordLabel")
        self._coordinates_label.setProperty("muted", True)
        self._coordinates_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._coordinates_label.setMinimumWidth(0)
        layout.addWidget(self._coordinates_label, 1)

        self._zoom_label = QLabel(f"{self._zoom_percent}%", top_row)
        self._zoom_label.setObjectName("chartZoomLabel")

        self._zoom_fit_button = create_toolbar_button(self, "zoom_fit", self._zoom_best_fit, layout)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal, top_row)
        self._zoom_slider.setObjectName("chartZoomSlider")
        self._zoom_slider.setMinimum(self._min_zoom_percent)
        self._zoom_slider.setMaximum(self._max_zoom_percent)
        self._zoom_slider.setValue(self._zoom_percent)
        self._zoom_slider.setSingleStep(5)
        self._zoom_slider.setPageStep(25)
        self._zoom_slider.setMinimumWidth(60)
        self._zoom_slider.setMaximumWidth(140)
        self._zoom_slider.setTracking(False)
        self._zoom_slider.valueChanged.connect(self.set_zoom_percent)


        layout.addWidget(self._zoom_slider, 0)
        layout.addWidget(self._zoom_label, 0)

        return top_row

    def _zoom_best_fit(self) -> None:
        """Fit the whole figure in the visible area - width *and* height.

        It used to fill the width only, which was fine while the chart had
        the panel to itself; with the HTML results pane open below, the
        height left over is much smaller and a width fit pushed the bottom
        of the figure out of sight. The size used is the scroll area's
        without scrollbars, since a fit leaves none to show.
        """
        if self._resize_mode != "FIXED":
            return

        available = self._fixed_scroll_area.maximumViewportSize()
        if available.width() <= 0 or available.height() <= 0:
            return

        base_size = self._current_figure_pixel_size(apply_zoom=False)
        if base_size.width() <= 0 or base_size.height() <= 0:
            return

        # Floor, not round: rounding up by one percent is enough to bring
        # back the scrollbars this is meant to remove.
        zoom_percent = int(
            min(
                available.width() * 100.0 / float(base_size.width()),
                available.height() * 100.0 / float(base_size.height()),
            )
        )

        self.set_zoom_percent(
            self._clamp_zoom(zoom_percent)
        )

    def _remove_toolbar_actions(self, action_names: set[str]) -> None:
        """Remove selected built-in Matplotlib toolbar actions by label."""
        for action in list(self._toolbar.actions()):
            text = (action.text() or "").strip()
            if text in action_names:
                self._toolbar.removeAction(action)
                action.deleteLater()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def figure_id(self) -> int:
        """Return the repository figure id rendered by this panel."""
        return self._figure_id

    @property
    def figure(self) -> Figure:
        """Return the Matplotlib figure object."""
        return self._figure

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        """Return the Matplotlib canvas widget."""
        return self._canvas

    @property
    def resize_mode(self) -> str:
        """Return this panel's current fit mode, for the properties combo."""
        return self._resize_mode

    def close(self) -> bool:
        """Delete the figure from the repository and close the panel widget."""
        if self._deleted:
            return super().close()

        self._canvas_sync_timer.stop()
        self._canvas_late_sync_timer.stop()
        self._live_timer.stop()

        try:
            # Before the delete opens its transaction: the undo store
            # attaches its own database, which SQLite forbids mid-transaction.
            # Every descriptor table, since the cascade takes the figure, its
            # axes and their series together and undo has to bring back all
            # three (todo.txt P2-11). Done here rather than in delete_figure
            # so the operation dialogs' cleanup of an unapplied result figure
            # does not leave a spurious entry behind. The label is a plain
            # string, not translated - the same as delete_table's, so the two
            # undo entries read the same way until both are localised.
            # While a dialog's preview is open its savepoint holds the transaction,
            # and the figure is only there for the preview: the dialog's close rolls
            # it back, so there is nothing for the undo store to take back.
            if not self._repo.preview_savepoint_active:
                title = self._repo.get_figure_title(self._figure_id)
                self._repo.snapshot_for_undo(
                    self._repo.DESCRIPTOR_TABLES,
                    label=f"Delete figure '{title}'" if title else "Delete figure",
                )
            self._repo.delete_figure(figure_id=self._figure_id)
            self._deleted = True
            self.delete_requested.emit(self._figure_id)
        except Exception:
            applogger.exception(
                "Failed to delete figure (figure_id=%s)",
                self._figure_id,
            )
            return False

        return super().close()

    def closeEvent(self, event: PySide6.QtGui.QCloseEvent) -> None:
        """Keep default Qt close handling explicit for strict typing tools."""
        super().closeEvent(event)

    def reload(self) -> None:
        """Reload the latest descriptor from the repository and re-render it."""
        self._figure.clear()
        self._discard_hover_annotation()
        self._discard_ruler()
        self._discard_crosshair()
        self._discard_annotation_drag()
        self._discard_box_selection()
        self._refresh_config_from_source()
        self._apply_persisted_figure_metrics_to_rcparams()
        self._reset_figure_metrics_from_rcparams_for_reload()

        descriptor = self._repo.load_figure_descriptor(self._figure_id)
        if descriptor is None:
            applogger.warning("No descriptor found for figure_id=%s", self._figure_id)
            self._content_revision += 1
            self._capture_fixed_metrics_from_rendered_figure()
            self._schedule_canvas_geometry_sync(redraw=True)
            return

        try:
            self._prepare_canvas_geometry_before_render()
            original_descriptor = descriptor
            descriptor = self._descriptor_prepared_for_render(original_descriptor)
            render_figure_from_descriptor(
                figure=self._figure,
                descriptor=descriptor,
                repo=self._repo,
            )
            self._make_data_artists_pickable()
            self._draw_selected_rows()
            self._update_filter_note()
            self._connect_x_sync_callbacks()

            # Do not normalize again from len(self._figure.axes). Matplotlib can
            # contain extra Axes created by a stale grid, colorbars, twinx axes,
            # or other renderer-side helpers. The descriptor is the authoritative
            # source for real chart-axis count, and it has already been
            # normalized before rendering. Re-normalizing from rendered axes is
            # what kept a 2x1 layout after a preview axis was deleted.
        except Exception:
            applogger.exception("Failed to render figure_id=%s", self._figure_id)
            self._figure.clear()

        # Only now, with the figure complete: a renderer may draw while it
        # builds (the Text chart's adjustText measures its labels that way),
        # and that half-built draw must not pass for this revision's.
        self._content_revision += 1

        self._capture_fixed_metrics_from_rendered_figure()
        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()

        self._schedule_canvas_geometry_sync(redraw=True, late=True)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_resize_mode(value: object) -> str:
        """Return a valid resize mode, falling back to FIXED."""
        mode = str(value or "").strip().upper()
        return mode if mode in RESIZE_MODES else "FIXED"

    def _apply_persisted_view_state(self) -> None:
        """Restore this figure's own resize mode and zoom when present."""
        try:
            options = self._repo.get_figure_options(self._figure_id)
        except Exception:
            applogger.exception(
                "Failed to read view state for figure_id=%s", self._figure_id
            )
            return

        view = options.get(FIGURE_VIEW_OPTIONS_KEY)
        if not isinstance(view, dict):
            return

        if CONFIG_RESIZE_MODE in view:
            self._resize_mode = self._normalized_resize_mode(view[CONFIG_RESIZE_MODE])

        self._pending_notes_html = str(view.get("notes_html", "") or "")
        sizes = view.get("notes_split")
        self._pending_notes_sizes = (
            [int(value) for value in sizes]
            if isinstance(sizes, list) and len(sizes) == 2
            else None
        )

        try:
            if "zoom_percent" in view:
                self._zoom_percent = self._clamp_zoom(int(view["zoom_percent"]))
        except (TypeError, ValueError):
            applogger.warning(
                "Ignoring invalid persisted zoom for figure_id=%s: %r",
                self._figure_id,
                view.get("zoom_percent"),
            )

    def _persist_view_state(self) -> None:
        """Write this figure's resize mode and zoom into descriptor options."""
        if self._deleted:
            return
        try:
            options = self._repo.get_figure_options(self._figure_id)
            old_view = options.get(FIGURE_VIEW_OPTIONS_KEY)
            view = dict(old_view) if isinstance(old_view, dict) else {}
            new_view = {
                CONFIG_RESIZE_MODE: self._resize_mode,
                "zoom_percent": int(self._zoom_percent),
            }
            if self._notes_html:
                new_view["notes_html"] = self._notes_html
                new_view["notes_split"] = [int(size) for size in self._content_split.sizes()]
            if view == new_view:
                return
            options[FIGURE_VIEW_OPTIONS_KEY] = new_view
            self._repo.set_figure_options(self._figure_id, options)
        except Exception:
            applogger.exception(
                "Failed to persist view state for figure_id=%s", self._figure_id
            )

    def _persist_chart_panel_config(self) -> None:
        """Persist global defaults for newly-created figures.

        Three keys through ``update_section``, not the whole document through
        ``save_config``.  This panel used to hold a copy of the entire
        configuration from the moment it was built and write all of it back
        here - so a settings change made in another window between those two
        moments was reverted by whoever moved a zoom slider last.
        """
        update_section(
            CONFIG_SECTION,
            **{
                CONFIG_RESIZE_MODE: self._resize_mode,
                CONFIG_INITIAL_ZOOM: int(self._zoom_percent),
                CONFIG_BACKGROUND_COLOR: self._background_color,
            },
        )

    # ==================================================================
    # Notes: The chart panel's notes: the HTML results pane under the chart.
    # ==================================================================
    # ------------------------------------------------------------------
    # Notes / HTML results pane
    # ------------------------------------------------------------------
    def set_notes_html(self, markup: str, *, append: bool = False) -> None:
        """Show HTML results below the chart, revealing the pane if needed."""
        markup = str(markup or "")
        if not markup.strip():
            return

        if append and self._notes_html.strip():
            self._notes_html = f"{self._notes_html}<hr>{markup}"
        else:
            self._notes_html = markup

        self._notes_view.setHtml(self._notes_html)
        self._set_notes_visible(True)
        self._persist_view_state()

    def notes_html(self) -> str:
        """Return the markup currently shown below the chart."""
        return self._notes_html

    def clear_notes(self) -> None:
        """Empty and fully collapse the notes pane."""
        self._notes_html = ""
        self._notes_view.clear()
        self._set_notes_visible(False)
        self._persist_view_state()

    def _restore_notes_state(self) -> None:
        """Re-show persisted notes and splitter position for this figure."""
        markup = getattr(self, "_pending_notes_html", "")
        if not markup:
            return

        self._notes_html = markup
        self._notes_view.setHtml(markup)
        self._notes_view.setVisible(True)

        sizes = getattr(self, "_pending_notes_sizes", None)
        if sizes:
            self._content_split.setSizes(sizes)

    def _set_notes_visible(self, visible: bool) -> None:
        """Show/hide notes and let users collapse the pane completely."""
        was_visible = self._notes_view.isVisible()
        self._notes_view.setVisible(visible)
        if not visible:
            sizes = self._content_split.sizes()
            chart_height = max(CHART_AREA_MIN_HEIGHT, sum(sizes) if sizes else self.height())
            self._content_split.setSizes([chart_height, 0])
            self._schedule_canvas_geometry_sync(redraw=True, late=True)
            return
        if was_visible:
            return

        total = max(self.height(), CHART_AREA_MIN_HEIGHT + NOTES_FIRST_OPEN_MIN_HEIGHT)
        notes_height = max(NOTES_FIRST_OPEN_MIN_HEIGHT, int(total * NOTES_INITIAL_FRACTION))
        self._content_split.setSizes([total - notes_height, notes_height])
        self._schedule_canvas_geometry_sync(redraw=True, late=True)

    # ==================================================================
    # Picking: Clicks on the chart: picking a point, a bar, a legend entry; scroll zoom.
    # ==================================================================
    def _on_scroll_zoom(self, event: Any) -> None:
        """Zoom about the cursor on Ctrl/Cmd + wheel.

        Modifier-gated deliberately.  In FIXED mode the canvas lives in a
        QScrollArea and a bare wheel scrolls it, which is what a bare wheel
        should do; taking that over would make the chart behave differently
        from every other scrollable thing in the window.  Requiring the
        modifier also means the two gestures never fight, in any mode.

        Zooming about the cursor rather than about the centre is the whole
        point: it is what lets you walk into a feature by pointing at it,
        instead of alternating zoom and pan.
        """
        if event.inaxes is None or not event.step:
            return
        # Matplotlib reports held modifiers in event.key; on macOS Qt maps the
        # Command key to Control, so this is Cmd there without special-casing.
        if "control" not in str(event.key or ""):
            return

        # One notch in is a smaller window, one notch out a larger one.
        factor = self.ZOOM_STEP if event.step > 0 else 1.0 / self.ZOOM_STEP

        axes = event.inaxes
        for get_limits, set_limits, position in (
            (axes.get_xlim, axes.set_xlim, event.xdata),
            (axes.get_ylim, axes.set_ylim, event.ydata),
        ):
            low, high = get_limits()
            if position is None or not math.isfinite(position):
                continue
            # Each side is scaled by its own distance to the cursor, which is
            # what keeps the point under the pointer fixed while the rest of
            # the range closes in on it.
            set_limits(
                position - (position - low) * factor,
                position + (high - position) * factor,
            )

        self._canvas.draw_idle()

    #: A press and a release this close (pixels) are a click, not a drag.
    CLICK_SLOP_PX = 4

    def _on_object_click_press(self, event: Any) -> None:
        self._object_click_at = (event.x, event.y) if event.button == 1 else None

    def _on_object_click_release(self, event: Any) -> None:
        """A plain left click on the chart: say which figure, axis and series it landed on.

        Emitted as ``chart_object_clicked`` with ``axis_id`` (None outside every
        axis: the figure), ``series`` (the series' name, or None) and ``part``
        ("x", "y" or "z" for a click on that coordinate's ticks or label,
        "title" on the axis' title, None inside the plotting area) and
        ``add`` (Shift held: add the series to those edited together); the main
        window selects that object in Chart properties when the panel is
        shown. Not after a drag, nor while a navigation tool, box selection
        or an annotation drag owns the mouse.
        """
        start = getattr(self, "_object_click_at", None)
        self._object_click_at = None
        if start is None or event.button != 1:
            return
        if abs(event.x - start[0]) > self.CLICK_SLOP_PX or abs(event.y - start[1]) > self.CLICK_SLOP_PX:
            return
        if str(getattr(self._toolbar, "mode", "")) != "" or getattr(self, "_select_mode_enabled", False):
            return
        axes, part = event.inaxes, None
        if axes is None:
            # Outside every plotting area may still be on an axis: its ticks,
            # tick labels or label, or its title - not the figure.
            axes, part = self._axis_part_at(event)
        axis_id = getattr(axes, "_chartlibre_axis_id", None) if axes is not None else None
        series = self._series_name_at(axes, event) if axis_id is not None and part is None else None
        add = "shift" in str(event.key or "")
        self.chart_object_clicked.emit({"axis_id": axis_id, "series": series, "part": part, "add": add})

    def _on_escape_to_figure(self, event: Any) -> None:
        """Esc on the chart goes back to the figure, as a click outside every axis would."""
        if event.key != "escape":
            return
        if str(getattr(self._toolbar, "mode", "")) != "" or getattr(self, "_select_mode_enabled", False):
            return
        self.chart_object_clicked.emit({"axis_id": None, "series": None, "part": None})

    def _axis_part_at(self, event: Any) -> tuple[Any, str | None]:
        """(the axes, "x"/"y"/"z") for a click on an axis' ticks or labels, (the axes, "title")
        on its title, (None, None) elsewhere."""
        renderer = self._canvas.get_renderer()
        # The canvas' figure: the one drawn, and so the one clicked.
        for axes in self._canvas.figure.axes:
            if getattr(axes, "_chartlibre_axis_id", None) is None:
                continue
            title = axes.title
            if title.get_visible() and title.get_text() and title.get_window_extent(renderer).contains(event.x, event.y):
                return axes, "title"
            for coordinate in ("x", "y", "z"):
                axis = getattr(axes, f"{coordinate}axis", None)
                if axis is None or not axis.get_visible():
                    continue
                box = axis.get_tightbbox(renderer)
                if box is not None and box.contains(event.x, event.y):
                    return axes, coordinate
        return None, None

    @staticmethod
    def _series_name_at(axes: Any, event: Any) -> str | None:
        """The name of the series drawn under the click, from its artist's label."""
        def named(label: Any) -> str | None:
            text = str(label or "").strip()
            return None if not text or text.startswith("_") else text

        # Bars first: their patches carry no label, their container does.
        for container in getattr(axes, "containers", []):
            if any(getattr(patch, "contains", None) and patch.contains(event)[0] for patch in container):
                name = named(container.get_label())
                if name:
                    return name
        for artist in (*axes.lines, *axes.collections, *axes.patches):
            try:
                hit = artist.get_visible() and artist.contains(event)[0]
            except Exception:
                hit = False
            name = named(artist.get_label()) if hit else None
            if name:
                return name
        return None

    def _make_data_artists_pickable(self) -> None:
        """Arm picking on the lines and marker collections a render produced.

        Done here rather than by the renderers because it is a *viewer*
        decision, not a chart property: every renderer would otherwise have to
        remember to opt in, and a figure saved by one version would come back
        unpickable in another. Setting it after the render also covers the
        renderers that draw through Matplotlib helpers we do not construct
        ourselves.

        Lines and collections report a point. Patches - bars and wedges - are
        armed too but read out differently: a patch has no per-point index, so
        the old code left them unpickable entirely and clicking a bar chart did
        nothing at all. The fix is not to force them through the point path,
        which would say "1 point" and mean the whole shape, but to report what
        is actually meaningful about a bar: its category and its value. See
        _describe_patch.

        The legend is armed last, and for a different purpose: clicking a
        legend entry toggles its series rather than reading anything out.
        """
        for axes in self._figure.axes:
            for artist in (*axes.lines, *axes.collections):
                artist.set_picker(PICK_TOLERANCE_POINTS)

            # True, not a tolerance in points: a patch is a filled area, so
            # "inside the shape" is the hit test, and a distance from its edge
            # would make the middle of a tall bar unclickable.
            for patch in axes.patches:
                patch.set_picker(True)

        self._make_legend_pickable()

    def _make_legend_pickable(self) -> None:
        """Let a click on a legend entry hide and show its series.

        Wired here rather than at render time for the same reason as the data
        artists: the legend is rebuilt by every render, so the connection has
        to be re-established against the new handles each time.
        """
        self._legend_targets = {}

        for axes in self._figure.axes:
            legend = axes.get_legend()
            if legend is None:
                continue

            handles = list(getattr(legend, "legend_handles", None) or [])
            labels = [text.get_text() for text in legend.get_texts()]

            # Map each legend handle back to the artists it stands for. Matched
            # by label rather than by position: a renderer that draws a series
            # as several artists - a line plus its error bars - produces one
            # legend entry for the group, and toggling only the first would
            # leave the error bars floating on their own.
            by_label: dict[str, list[Any]] = {}
            for artist in (*axes.lines, *axes.collections, *axes.patches):
                label = str(artist.get_label() or "")
                if label and not label.startswith("_"):
                    by_label.setdefault(label, []).append(artist)

            for handle, label in zip(handles, labels):
                targets = by_label.get(str(label), [])
                if not targets:
                    continue
                handle.set_picker(PICK_TOLERANCE_POINTS)
                self._legend_targets[handle] = targets

            for text in legend.get_texts():
                # The label text is the larger target and the one people
                # actually aim at; the swatch alone is a few pixels.
                targets = by_label.get(str(text.get_text()), [])
                if targets:
                    text.set_picker(True)
                    self._legend_targets[text] = targets

    def _on_pick(self, event: Any) -> None:
        """Describe the clicked point, or the clicked group of points.

        Matplotlib reports a pick as an artist plus ``ind``, a list of the
        indices under the cursor - one entry for a marker clicked on its own,
        several where markers overlap or where a rubber-band selection covers
        them. That single mechanism covers both cases the readout has to
        distinguish, so there is no separate "multiple selection" path here:
        one index reads out the value, several read out the summary.
        """
        artist = event.artist

        # Legend first: a legend handle is also a line or a patch, so testing
        # it after the data cases would read out the swatch's own coordinates
        # instead of toggling the series.
        if artist in getattr(self, "_legend_targets", {}):
            self._toggle_series_visibility(artist)
            return

        # A patch carries no index, so it never reaches the point path below.
        if isinstance(artist, Patch):
            self._describe_patch(artist)
            return

        # ``or []`` would be the obvious way to default this, and it raises:
        # ind is a NumPy array, and asking an array of more than one element
        # whether it is true is ambiguous - so clicking a *group* of markers,
        # the case this readout exists for, was the case that crashed. A
        # single marker has one element, is unambiguous, and worked, which is
        # why it survived. None is the only "no indices" ind can be.
        raw_indices = getattr(event, "ind", None)
        if raw_indices is None:
            return
        indices = [int(index) for index in np.atleast_1d(raw_indices)]
        if not indices:
            return

        # A scatter is a PathCollection and carries its points as Nx2 offsets;
        # a line carries two parallel sequences. Nothing else is made pickable.
        if hasattr(artist, "get_offsets"):
            points = artist.get_offsets()
            pairs = [(float(points[i][0]), float(points[i][1])) for i in indices]
        else:
            x_data, y_data = artist.get_data()
            pairs = [(float(x_data[i]), float(y_data[i])) for i in indices]

        axes = artist.axes
        # Matplotlib invents a label for an artist that was not given one -
        # "_child0", "_collection2" - and marks decoration as "_nolegend_".
        # Both start with an underscore by that convention, and neither is
        # worth showing, so an underscore means "no name" here.
        series = str(artist.get_label() or "").strip()
        if not series or series.startswith("_"):
            series = _("series")

        if len(pairs) == 1:
            x_value, y_value = pairs[0]
            self.selection_changed.emit(
                _("{series} — x: {x}, y: {y}").format(
                    series=series,
                    x=axis_text(axes.xaxis, x_value),
                    y=axis_text(axes.yaxis, y_value),
                )
            )
            return

        self.selection_changed.emit(
            _("{series} — {count} points, mean x: {x}, mean y: {y}").format(
                series=series,
                count=len(pairs),
                x=axis_text(axes.xaxis, sum(x for x, _unused in pairs) / len(pairs)),
                y=axis_text(axes.yaxis, sum(y for _unused, y in pairs) / len(pairs)),
            )
        )

    def _toggle_series_visibility(self, handle: Any) -> None:
        """Show or hide the series a legend entry stands for.

        The legend entry itself is dimmed rather than removed, so a hidden
        series still has something to click to bring it back - a legend that
        deleted its own entry would be a one-way door.
        """
        targets = self._legend_targets.get(handle) or []
        if not targets:
            return

        visible = not targets[0].get_visible()
        for artist in targets:
            artist.set_visible(visible)

        handle.set_alpha(1.0 if visible else 0.25)

        label = str(targets[0].get_label() or "").strip() or _("series")
        self.selection_changed.emit(
            _("{series} shown").format(series=label)
            if visible
            else _("{series} hidden").format(series=label)
        )
        self._canvas.draw_idle()

    def _describe_patch(self, patch: Any) -> None:
        """Read out a bar or a wedge by what it means, not by index.

        A patch has no per-point index, which is why these were left
        unpickable and clicking a bar chart did nothing. What is meaningful
        about a bar is its category and its height, and both can be recovered
        from the geometry: the tick label under its centre names it, and the
        rectangle's height is the value.

        A wedge carries its share in its angles rather than in a height, so it
        reports a percentage instead.
        """
        axes = patch.axes
        if axes is None:
            return

        series = str(patch.get_label() or "").strip()
        if not series or series.startswith("_"):
            series = ""

        if isinstance(patch, Wedge):
            share = abs(patch.theta2 - patch.theta1) / 360.0  # pyright: ignore[reportAttributeAccessIssue]
            name = series or _("slice")
            self.selection_changed.emit(
                _("{series} — {percent:.1f}% of the total").format(
                    series=name, percent=share * 100.0
                )
            )
            return

        if isinstance(patch, Rectangle):
            width = float(patch.get_width())
            height = float(patch.get_height())
            if self._bar_is_horizontal(axes, patch):
                value, value_axis = width, axes.xaxis
                centre, category_axis = patch.get_y() + height / 2.0, axes.yaxis
            else:
                value, value_axis = height, axes.yaxis
                centre, category_axis = patch.get_x() + width / 2.0, axes.xaxis

            category = self._category_at(category_axis, centre)
            label = " — ".join(part for part in (series, category) if part)
            self.selection_changed.emit(
                _("{label}: {value}").format(
                    label=label or _("bar"),
                    value=axis_text(value_axis, value),
                )
            )
            return

        if series:
            self.selection_changed.emit(series)

    @staticmethod
    def _bar_is_horizontal(axes: Any, patch: Any) -> bool:
        """Say whether a bar grows along x rather than along y.

        Read from the BarContainer that owns the patch, which records the
        orientation ax.bar/ax.barh gave it. Guessing from the rectangle's own
        geometry does not work: a tall thin horizontal bar and a tall thin
        vertical bar have the same shape, and only the container knows which
        dimension is the value.

        The geometric fallback is for a patch drawn directly rather than
        through the bar helpers, where there is no container to ask.
        """
        for container in getattr(axes, "containers", []) or []:
            if patch in list(getattr(container, "patches", []) or []):
                return str(getattr(container, "orientation", "vertical")) == "horizontal"

        # No container: a bar sits on its baseline, so the coordinate that is
        # pinned to zero tells us which way it grows.
        return float(patch.get_x()) == 0.0 and float(patch.get_y()) != 0.0

    @staticmethod
    def _category_at(axis: Any, position: float) -> str:
        """Return the tick label at a position, or the position itself.

        A categorical bar chart puts the category names in the tick labels and
        the bars at integer positions, so the label under the bar's centre is
        its name. A numeric bar chart has no such names, and there the position
        formatted by the axis is the honest answer.
        """
        try:
            locations = list(axis.get_ticklocs())
            labels = [text.get_text() for text in axis.get_ticklabels()]
            for location, label in zip(locations, labels):
                if abs(float(location) - position) < 0.5 and label:
                    return str(label)
        except Exception:
            pass
        return axis_text(axis, position)

    # ------------------------------------------------------------------
    # Chart properties -> chart: show what is selected there (S-01)
    # ------------------------------------------------------------------
    def flash(self, axis_id: int, series: str | None = None) -> bool:
        """Highlight an axis (a frame round it) or one of its series (a halo under it), for a moment.

        Drawn as extra artists over the real ones and removed after
        FLASH_MS, so nothing of the chart itself changes. False when the
        axis is not on this chart.
        """
        from PySide6.QtCore import QTimer

        axes = next((a for a in self._canvas.figure.axes if getattr(a, "_chartlibre_axis_id", None) == int(axis_id)), None)
        if axes is None:
            return False
        added: list[Any] = []
        if series:
            for line in axes.lines:
                if str(line.get_label()) == series or str(line.get_label()).startswith(f"{series} "):
                    halo, = axes.plot(line.get_xdata(), line.get_ydata(), color=FLASH_COLOR, alpha=0.35,
                                      linewidth=float(line.get_linewidth()) + 6, solid_capstyle="round",
                                      zorder=line.get_zorder() - 0.01, label="_flash")
                    added.append(halo)
            for collection in axes.collections:
                offsets = getattr(collection, "get_offsets", lambda: [])()
                if str(collection.get_label()) == series and len(offsets):
                    sizes = collection.get_sizes()
                    size = float(sizes[0]) if len(sizes) else 36.0
                    added.append(axes.scatter(offsets[:, 0], offsets[:, 1], s=size * 4, color=FLASH_COLOR, alpha=0.3,
                                              linewidths=0, zorder=collection.get_zorder() - 0.01, label="_flash"))
        if not added:
            added.append(axes.add_patch(Rectangle((0, 0), 1, 1, transform=axes.transAxes, fill=False,
                                                  edgecolor=FLASH_COLOR, linewidth=2.5, clip_on=False,
                                                  zorder=1000, label="_flash")))
        self._canvas.draw_idle()

        def clear() -> None:
            for artist in added:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):  # already gone with a redraw of the figure
                    pass
            self._canvas.draw_idle()

        QTimer.singleShot(FLASH_MS, clear)
        return True

    def _preview_stem_style(self, axis_id: int, series: str, changes: dict[str, Any]) -> bool:
        """A Stem Plot's linefmt / markerfmt / basefmt on the drawn stems, baseline and marker.

        False - for a render - for an empty format (it means "follow the
        style"), one Matplotlib refuses, a part the format string does not
        say (a colour alone on a marker would need the cycle), or a chart
        without the tagged artists.
        """
        from matplotlib.axes._base import _process_plot_format

        axes = next((a for a in self._canvas.figure.axes if getattr(a, "_chartlibre_axis_id", None) == int(axis_id)), None)
        if axes is None:
            return False
        parts = {
            a._chartlibre_stem: a
            for a in axes.get_children()
            if getattr(a, "_chartlibre_series", None) == series and hasattr(a, "_chartlibre_stem")
        }
        plan = []
        for key, fmt in changes.items():
            artist = parts.get(key)
            if artist is None or not str(fmt).strip():
                return False
            try:
                linestyle, marker, color = _process_plot_format(str(fmt))
            except ValueError:
                return False
            if key == "linefmt":
                if linestyle in (None, "None") or color is None:
                    return False
                plan.append(lambda a=artist, ls=linestyle, c=color: (a.set_color(c), a.set_linestyle(ls)))
            else:
                if marker in (None, "None") and key == "markerfmt":
                    return False
                if key == "markerfmt":
                    plan.append(lambda a=artist, m=marker, c=color: (a.set_marker(m), c and a.set_color(c)))
                else:
                    if color is None:
                        return False
                    plan.append(lambda a=artist, c=color, ls=linestyle: (a.set_color(c), ls and a.set_linestyle(ls)))
        for step in plan:
            step()
        self._canvas.draw_idle()
        return True

    def preview_series_style(self, axis_id: int, series: str, changes: dict[str, Any]) -> bool:
        """Show a series' new colour, line style, marker or marker size on the chart as drawn - no render.

        On the artists its renderer tagged with its name (SeriesArtistTags)
        and its legend entry. False - for the caller to render the chart -
        for any other key, a renderer that does not tag, a series drawn with
        more than lines and points (error bars, bars, a band), a line or
        points the edit would add or remove, colour-mapped points, or a value
        Matplotlib refuses.
        """
        if changes and set(changes) <= STEM_FMT_KEYS:
            return self._preview_stem_style(axis_id, series, changes)
        if not changes or not set(changes) <= set(PREVIEW_STYLE_SETTERS):
            return False
        axes = next((a for a in self._canvas.figure.axes if getattr(a, "_chartlibre_axis_id", None) == int(axis_id)), None)
        if axes is None:
            return False
        mine = [a for a in axes.get_children() if getattr(a, "_chartlibre_series", None) == series]
        lines = [a for a in mine if isinstance(a, Line2D)]
        points = [a for a in mine if isinstance(a, PathCollection)]
        steps = [a for a in mine if isinstance(a, StepPatch)]
        if not mine or len(lines) + len(points) + len(steps) != len(mine):
            return False
        if any(not any(key in _follows(a) for a in mine) for key in changes):
            return False  # nothing drawn takes it: only a render can show it
        labelled = {str(a.get_label()): a for a in mine if not str(a.get_label()).startswith("_")}
        handles = []
        for legend in (axes.get_legend(), *self._canvas.figure.legends):
            for handle, text in zip(legend.legend_handles, legend.get_texts()) if legend is not None else ():
                if text.get_text() in labelled:
                    if not isinstance(handle, (Line2D, PathCollection, Patch)):
                        return False
                    handle._chartlibre_follows = _follows(labelled[text.get_text()])
                    handles.append(handle)
        has_points = any("marker" in _follows(a) for a in points)
        try:
            plan = [(a, *_style_steps(a, changes, has_points=has_points, sizes=a not in handles, has_lines=bool(lines)))
                    for a in (*lines, *points, *steps, *handles)]
        except ValueError:
            return False
        if any(not ok for _artist, ok, _steps in plan):
            return False
        try:
            for _artist, _ok, steps in plan:
                for step in steps:
                    step()
        except (TypeError, ValueError):
            return False
        self._canvas.draw_idle()
        return True

    # ==================================================================
    # Hover: The readout that follows the pointer over the data.
    # ==================================================================
    #: How long the pointer must be still before the readout is recomputed.
    #: Mouse motion arrives far faster than a hit test plus a blit can be done,
    #: so handling every event makes the chart lag behind the cursor. 40ms is
    #: below the ~100ms at which a response stops feeling immediate, and it
    #: collapses a fast sweep across the plot into a handful of tests instead
    #: of hundreds.
    HOVER_INTERVAL_MS: Final[int] = 40

    #: Pixels. Wider than the click tolerance: hovering is a coarse gesture and
    #: nobody positions the pointer precisely before reading a value.
    HOVER_TOLERANCE_PIXELS: Final[float] = 12.0

    def _init_hover_readout(self) -> None:
        """Prepare the throttle timer and the blitting state."""
        self._hover_event: Any | None = None
        self._hover_annotation: Any | None = None
        self._hover_background: Any | None = None
        self._hover_axes: Any | None = None

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(self.HOVER_INTERVAL_MS)
        self._hover_timer.timeout.connect(self._update_hover_readout)

    def _on_motion(self, event: Any) -> None:
        """Record the pointer and let the timer decide when to act.

        Deliberately does no work: this runs on every mouse move, and a hit
        test here would be the thing that makes the chart feel slow.
        """
        self._hover_event = event
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _draw_state(self) -> tuple[Any, ...]:
        """What a draw of the figure right now would show: contents, pixels, dpi.

        The size is the figure's in whole device pixels - the size of the
        buffer a draw fills, which is what matters: a draw at one size is no
        draw at another. (The figure follows its canvas to the pixel, FIXED
        mode included - see _apply_figure_size_from_canvas.)
        """
        width, height = self._figure.bbox.size
        return (self._content_revision, int(width), int(height), float(self._figure.dpi))

    def _on_canvas_drawn(self, _event: Any) -> None:
        """After every full draw: note what is on screen, and keep it for hovering.

        The pixels just drawn are the background the hover readout blits over,
        unless the readout or the crosshair is part of them. Taking them now -
        a copy of the buffer - spares the first hover after every redraw a
        second full draw of its own.
        """
        self._drawn_state = self._draw_state()
        annotation = self._hover_annotation
        if self._crosshair_enabled or (annotation is not None and annotation.get_visible()):
            self._hover_background = None
            return
        try:
            self._hover_background = self._canvas.copy_from_bbox(self._figure.bbox)  # pyright: ignore[reportAttributeAccessIssue]
        except (AttributeError, RuntimeError, ValueError):
            self._hover_background = None

    def _invalidate_hover_background(self) -> None:
        """Drop the cached pixels after anything that repaints the canvas.

        The background is a snapshot of the chart without the annotation on it.
        Zooming, panning and resizing all invalidate it, and blitting a stale
        one paints the old chart back over the new one.

        Only the bitmap is dropped. The annotation artist itself survives a
        repaint, and clearing _hover_axes here would make every hover rebuild
        it and force the full draw that blitting exists to avoid - the blit
        path would then never be reached at all.
        """
        self._hover_background = None

    def _discard_hover_annotation(self) -> None:
        """Forget the annotation artist itself, after the figure is cleared.

        figure.clear() destroys every artist including this one, so the
        reference left behind belongs to no axes and cannot be drawn.
        """
        self._hover_annotation = None
        self._hover_axes = None
        self._hover_background = None

    def _update_hover_readout(self) -> None:
        """Find the point under the pointer and draw its value beside it."""
        event = self._hover_event
        inaxes = getattr(event, "inaxes", None) if event is not None else None

        if self._crosshair_enabled:
            # Tracks the raw pointer position regardless of whether a data
            # point is nearby - unlike the annotation below, which only
            # shows for an actual hit. Always a full draw_idle() rather
            # than the blit path: see _update_crosshair for why, and why
            # that is safe to mix with the hover annotation's own blitting.
            self._update_crosshair(event if inaxes is not None else None)

        if inaxes is None:
            self._hide_hover_annotation()
            self._show_axis_part_hint(event)
            return
        self._show_axis_part_hint(None)

        hit = self._nearest_point_to(event)
        if hit is None:
            self._hide_hover_annotation()
            return

        artist, x_value, y_value = hit
        axes = inaxes

        series = str(artist.get_label() or "").strip()
        if not series or series.startswith("_"):
            series = _("series")

        text = _("{series}\nx: {x}\ny: {y}").format(
            series=series,
            x=axis_text(axes.xaxis, x_value),
            y=axis_text(axes.yaxis, y_value),
        )

        annotation = self._hover_annotation_for(axes)
        annotation.xy = (x_value, y_value)
        annotation.set_text(text)
        annotation.set_visible(True)
        if self._crosshair_enabled:
            # _update_crosshair already issued a draw_idle() this tick;
            # blitting on top of it would restore a background captured
            # before the crosshair moved - see the comment there.
            self._canvas.draw_idle()
        else:
            self._blit_hover(axes, annotation)

    def _show_axis_part_hint(self, event: Any) -> None:
        """Over an axis' ticks, label or title: a pointing hand and what a click will edit (S-01).

        Nothing is drawn - the cursor and the canvas' tooltip only - so
        hovering stays as cheap as before. Left alone while a navigation tool
        owns the cursor.
        """
        part = None
        if event is not None and str(getattr(self._toolbar, "mode", "")) == "":
            _axes, part = self._axis_part_at(event)
        hints = {"x": _("X axis: click to edit it"), "y": _("Y axis: click to edit it"),
                 "z": _("Z axis: click to edit it"), "title": _("Axis title: click to edit the axis")}
        if part in hints:
            self._canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            self._canvas.setToolTip(hints[part])
            self._axis_hint_shown = True
        elif getattr(self, "_axis_hint_shown", False):
            self._canvas.unsetCursor()
            self._canvas.setToolTip("")
            self._axis_hint_shown = False

    def _nearest_point_to(self, event: Any) -> tuple[Any, float, float] | None:
        """Return (artist, x, y) for the closest data point, or None.

        Distance is measured in display pixels, not in data units: the two axes
        rarely share a scale, and a chart of millivolts against seconds would
        otherwise treat a step along x as thousands of times nearer than a step
        along y.
        """
        axes = event.inaxes
        best: tuple[float, Any, float, float] | None = None

        for artist in (*axes.lines, *axes.collections):
            if not artist.get_visible():
                continue

            try:
                if hasattr(artist, "get_offsets"):
                    points = np.asarray(artist.get_offsets(), dtype=float)
                    if points.size == 0:
                        continue
                    xs, ys = points[:, 0], points[:, 1]
                else:
                    xs, ys = (np.asarray(a, dtype=float) for a in artist.get_data())
                if xs.size == 0:
                    continue

                pixels = axes.transData.transform(np.column_stack([xs, ys]))
            except Exception:
                continue

            offsets = pixels - np.array([event.x, event.y], dtype=float)
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
            index = int(np.nanargmin(distances))
            distance = float(distances[index])

            if distance <= self.HOVER_TOLERANCE_PIXELS and (
                best is None or distance < best[0]
            ):
                best = (distance, artist, float(xs[index]), float(ys[index]))

        return None if best is None else (best[1], best[2], best[3])

    def _hover_annotation_for(self, axes: Any) -> Any:
        """Return the annotation artist, moving it to *axes* when needed.

        One artist reused rather than one per axes: it is only ever visible in
        the axes under the pointer, and a per-axes cache would have to be
        invalidated on every re-render alongside everything else.
        """
        if self._hover_annotation is not None and self._hover_axes is axes:
            return self._hover_annotation

        if self._hover_annotation is not None:
            try:
                self._hover_annotation.remove()
            except Exception:
                pass
            # The annotation was part of the old axes' contents, so the cached
            # background no longer matches what is on screen.
            self._hover_background = None

        annotation = axes.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=10_000,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffe0", "ec": "#808080", "alpha": 0.95},
            arrowprops={"arrowstyle": "-", "color": "#808080", "linewidth": 0.8},
            annotation_clip=False,
        )
        annotation.set_visible(False)
        self._hover_annotation = annotation
        self._hover_axes = axes
        return annotation

    def _blit_hover(self, axes: Any, annotation: Any) -> None:
        """Repaint only the annotation, over a cached copy of the chart.

        This is the whole reason hovering is affordable. A full draw() re-runs
        every renderer for every series; blitting restores a bitmap and draws
        one text box, so the cost does not grow with the size of the data.
        """
        canvas = self._canvas
        try:
            if self._hover_background is None:
                # Capture without the annotation, or it would be baked into
                # the background and smear across the plot as the pointer moves.
                annotation.set_visible(False)
                canvas.draw()
                self._hover_background = canvas.copy_from_bbox(self._figure.bbox)  # pyright: ignore[reportAttributeAccessIssue]
                annotation.set_visible(True)

            canvas.restore_region(self._hover_background)
            axes.draw_artist(annotation)
            canvas.blit(self._figure.bbox)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            # Any backend that cannot blit still gets a correct, slower chart.
            self._hover_background = None
            canvas.draw_idle()

    def _hide_hover_annotation(self) -> None:
        """Take the readout away once the pointer leaves every point."""
        annotation = self._hover_annotation
        if annotation is None or not annotation.get_visible():
            return

        annotation.set_visible(False)
        try:
            if self._hover_background is not None:
                self._canvas.restore_region(self._hover_background)
                self._canvas.blit(self._figure.bbox)  # pyright: ignore[reportAttributeAccessIssue]
                return
        except Exception:
            self._hover_background = None
        self._canvas.draw_idle()

    # ==================================================================
    # Menus: The chart's menus: actions, right-click, reference lines and annotations added from them.
    # ==================================================================
    def _build_actions_menu(self) -> QMenu:
        """Create the chart actions menu shared by button and canvas.

        Fit / Fit proportional / Fixed used to live here as three checkable
        entries kept in sync by comparing an action's (translated) label
        against the resize mode - which meant the checkmarks stopped
        matching anything the moment the interface language changed. The
        choice is made from the Figure Properties panel now, as a combo
        box next to the figure's other display settings; see
        FigurePropertiesWidget.set_resize_mode_control.
        """
        menu = create_menu(
            self,
            [
                MenuItem(_("Reload"), _("Reload chart"), PySide6.QtGui.QKeySequence.StandardKey.Refresh, self.reload,False,"reload"),
                MenuItem(_("Copy"),_("Copy figure"),PySide6.QtGui.QKeySequence.StandardKey.Copy,self.copy_chart_to_clipboard,False,"copy"),
                MenuItem(_("Save figure"),_("Save as picture"), PySide6.QtGui.QKeySequence.StandardKey.SaveAs,self.save_chart_as,False,"save"),
                MenuItem(_("Export report…"), _("This chart and its notes as an HTML or PDF report, like File > Export for the whole project"), None, self.export_report, False, "project_report"),
                MenuItem(_("Export view as CSV…"),_("Save the rows currently on screen as a CSV file"),None,self.export_view_as_csv,False,"export_csv"),
                MenuItem(
                    text=catalogue_action("graph_template").translated_text(),
                    tooltip=catalogue_action("graph_template").translated_description(),
                    icon="graph_template",
                    submenu=[
                        MenuItem(_("Save this look as a template…"), _("Keep this figure's style, axes and series look under a name"), None, self.save_as_template),
                        MenuItem(_("Apply a template…"), _("Give this figure the look of a saved template; its data, titles and labels stay"), None, self.apply_saved_template),
                        None,
                        MenuItem(_("Delete a template…"), _("Forget a saved template"), None, self.delete_saved_template),
                    ],
                ),
                None,
                MenuItem(
                    text=_("Crosshair"),
                    tooltip=_("A guide line under the pointer, shared across every axes of this figure."),
                    callback=self._on_toggle_crosshair,
                    checkable=True,
                    checked=self._crosshair_enabled,
                    icon="crosshair",
                ),
                MenuItem(
                    text=_("Link zoom/pan across axes"),
                    tooltip=_("Zooming or panning one axes' x range applies it to every other axes of this figure too."),
                    callback=self._on_toggle_link_x_zoom,
                    checkable=True,
                    checked=self._link_x_zoom_enabled,
                    icon="zoom_fit",
                ),
                MenuItem(
                    text=_("Select points"),
                    tooltip=_("Drag a rectangle over the chart to select points, for hiding or splitting into a new series."),
                    callback=self._on_toggle_select_mode,
                    checkable=True,
                    checked=self._select_mode_enabled,
                    icon="select_points",
                ),
                # Show only Selected, for each table this chart reads rows of (R-07).
                *(MenuItem(
                    text=_("Show only the Selected rows of '{table}'").format(table=table),
                    tooltip=_("Every chart of '{table}' draws only its rows marked Selected; the others stay, "
                              "out of view - nothing is hidden, and analyses read every row.").format(table=table),
                    callback=lambda checked, table=table: self._set_selected_only(table, checked),
                    checkable=True,
                    checked=self._repo.selected_only(table),
                    icon="select_points",
                ) for table in sorted(self.row_tables())),
                MenuItem(
                    text=_("Live updates"),
                    tooltip=_("Redraw this chart every {seconds:g}s, for a source table a rig or an import keeps filling in.").format(
                        seconds=LIVE_REFRESH_INTERVAL_MS / 1000
                    ),
                    callback=self._on_toggle_live_updates,
                    checkable=True,
                    checked=self._live_updates_enabled,
                    icon="live_updates",
                ),
            ],
            # No icons on the Mac, as its own menus; the shortcuts shown as it shows them.
            icons=CONTEXT_MENU_ICONS,
        )
        return menu

    def _show_context_menu(self, pos: QPoint) -> None:
        """Show the actions menu, plus what can be done at the point clicked.

        The cached menu covers the whole figure - reload, copy, save,
        delete - and is the same wherever the click landed. The reference
        lines are not: they are added *at* a value on one axis, so they
        only appear when the click is inside an axes, and they name the
        value they would use.
        """
        self.context_menu_for(pos).exec(self._canvas.mapToGlobal(pos))

    def context_menu_for(self, pos: QPoint) -> QMenu:
        """Return the menu a right-click at *pos* should open.

        Built fresh every time, inside the axes or out: Crosshair and Link
        zoom/pan are checkable, and a cached QMenu built once at __init__
        would keep showing whatever checked state was true the first time
        it was ever shown.
        """
        target = self._axis_at(pos)
        if target is None:
            return self._build_actions_menu()

        axis_id, axes, x_value, y_value = target
        menu = self._build_actions_menu()
        menu.addSeparator()
        create_menu_item(
            icons=CONTEXT_MENU_ICONS,
            parent=self,
            menu=menu,
            icon="add.vertical",
            checkable=False,
            text=_("Add vertical line at x = {value}").format(
                value=axis_text(axes.xaxis, x_value)
            ),
            tooltip=_("A line across the whole axes, at this x."),
            key=None,
            action=lambda: self._add_reference_line(axis_id, "vertical", x_value),
        )
        create_menu_item(
            icons=CONTEXT_MENU_ICONS,
            parent=self,
            menu=menu,
            icon="add.horizontal",
            checkable=False,
            text=_("Add horizontal line at y = {value}").format(
                value=axis_text(axes.yaxis, y_value)
            ),
            tooltip=_("A line along the whole axes, at this y."),
            key=None,
            action=lambda: self._add_reference_line(axis_id, "horizontal", y_value),
        )
        create_menu_item(
            icons=CONTEXT_MENU_ICONS,
            parent=self,
            menu=menu,
            icon="add.annotation",
            checkable=False,
            text=_("Add annotation here…"),
            tooltip=_("A text note pinned to this point: x = {x}, y = {y}.").format(
                x=axis_text(axes.xaxis, x_value),
                y=axis_text(axes.yaxis, y_value),
            ),
            key=None,
            action=lambda: self._add_annotation(axis_id, x_value, y_value),
        )
        menu.addSeparator()
        self._add_measure_menu_items(menu, axes, axis_id, x_value, y_value)
        self._add_selection_menu_items(menu, axis_id)
        return menu

    def _add_selection_menu_items(self, menu: QMenu, axis_id: int) -> None:
        """Add "Hide"/"New series" for the pending box selection, if any.

        Only shown for the axis the rectangle was drawn on - a selection
        does not carry over to a right-click on a different axes.
        """
        selection = self._last_selection
        if not selection or selection["axis_id"] != axis_id:
            return

        menu.addSeparator()
        point_count = sum(len(info["x"]) for info in selection["series"].values())
        create_menu_item(
            parent=self,
            menu=menu,
            icon="hide_points",
            checkable=False,
            text=_("Hide {count} selected point(s)").format(count=point_count),
            tooltip=_("Hide these points from this series without changing its data."),
            key=None,
            action=self._hide_selection,
        )
        create_menu_item(
            parent=self,
            menu=menu,
            icon="add_series",
            checkable=False,
            text=_("New series from selection"),
            tooltip=_("Add a new series holding only the selected points."),
            key=None,
            action=self._create_series_from_selection,
        )

    def _add_measure_menu_items(
        self, menu: QMenu, axes: Any, axis_id: int, x_value: float, y_value: float
    ) -> None:
        """Add the ruler's two-click flow to a context menu.

        "Measure to here" only appears once a start point is pending on
        this same axes - measuring across two axes that may not even share
        units would report a slope neither one means.
        """
        pending_here = (
            self._ruler_start is not None and self._ruler_start[0] == axis_id
        )
        create_menu_item(
            parent=self,
            menu=menu,
            icon="measure",
            checkable=False,
            text=_("Measure from here"),
            tooltip=_("Start measuring the distance and slope to another point."),
            key=None,
            action=lambda: self._measure_from_here(axes, axis_id, x_value, y_value),
        )
        if pending_here:
            create_menu_item(
                parent=self,
                menu=menu,
                icon="measure",
                checkable=False,
                text=_("Measure to here"),
                tooltip=_("Finish the measurement at this point."),
                key=None,
                action=lambda: self._measure_to_here(axes, axis_id, x_value, y_value),
            )
        if self._ruler_start is not None:
            # A completed measurement is not cleared from here any more -
            # it is stored on the axis (apply_measurements) and deleted
            # from the Overlay panel's Measurements tab, the same way a
            # reference line is. This only cancels a still-pending point.
            create_menu_item(
                parent=self,
                menu=menu,
                icon="clear",
                checkable=False,
                text=_("Cancel measurement"),
                tooltip=_("Forget the pending point without finishing the measurement."),
                key=None,
                action=self._clear_ruler,
            )

    def _axis_at(self, pos: QPoint) -> tuple[int, Any, float, float] | None:
        """Return (axis id, axes, x, y) for the axes under *pos*, or None.

        The axis id comes off the axes itself: render_figure tags every one
        it creates with ``_chartlibre_axis_id``, so a click maps back to the
        descriptor without this panel keeping a second mapping that could
        fall out of step with the figure.

        Qt counts pixels from the top-left and Matplotlib from the
        bottom-left; ``mouseEventCoords`` is Matplotlib's own conversion,
        device pixel ratio included, and using it is what makes this agree
        with the toolbar's cursor readout on a Retina screen.
        """
        try:
            x_pixels, y_pixels = self._canvas.mouseEventCoords(pos)
        except Exception:
            return None

        # Last first: a twin axes is created after the axes it overlays and
        # is the one on top, so it is the one a click belongs to.
        for axes in reversed(list(self._figure.axes)):
            axis_id = getattr(axes, "_chartlibre_axis_id", None)
            if axis_id is None or not axes.bbox.contains(x_pixels, y_pixels):  # pyright: ignore[reportAttributeAccessIssue]
                continue
            try:
                x_value, y_value = axes.transData.inverted().transform(
                    (x_pixels, y_pixels)
                )
            except Exception:
                return None
            return int(axis_id), axes, float(x_value), float(y_value)
        return None

    def _add_reference_line(
        self, axis_id: int, orientation: str, value: float
    ) -> None:
        """Store one reference line on the axis, and redraw.

        The same "lines" option the Overlay properties panel edits, so a
        line dropped here can be restyled, moved or deleted there - see
        BaseAxisRenderer.apply_reference_lines. Nothing about the style is
        decided here: at the moment of the click the only thing known is
        where.
        """
        try:
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES,
                label=_("Add {orientation} line").format(orientation=orientation),
            )
            options = dict(self._repo.get_axis_options(int(axis_id)) or {})
            lines = list(options.get("lines") or [])
            lines.append(
                {"orientation": orientation, "value": float(value), "kwargs": {}}
            )
            options["lines"] = lines
            self._repo.set_axis_options(int(axis_id), options)
        except Exception:
            applogger.exception("Could not add a reference line.")
            return

        applogger.info(
            "Added a %s reference line at %g on axis %s", orientation, value, axis_id
        )
        self.reload()
        self.figure_edited.emit()

    def _add_annotation(self, axis_id: int, x_value: float, y_value: float) -> None:
        """Prompt for text and pin a note to the clicked point, then redraw.

        Writes the same "annotations" key the Overlay properties panel edits,
        so a note dropped here can be restyled, moved or deleted there - see
        BaseAxisRenderer.apply_annotations. Only the position and the text are
        decided here; the type is a plain text note.
        """
        text, accepted = QInputDialog.getText(
            self, _("Add annotation"), _("Annotation text:")
        )
        if not accepted or not text.strip():
            return

        try:
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES,
                label=_("Add annotation"),
            )
            options = dict(self._repo.get_axis_options(int(axis_id)) or {})
            annotations = list(options.get("annotations") or [])
            annotations.append(
                {
                    "x": float(x_value),
                    "y": float(y_value),
                    "type": "text",
                    "text": text.strip(),
                    "kwargs": {},
                }
            )
            options["annotations"] = annotations
            self._repo.set_axis_options(int(axis_id), options)
        except Exception:
            applogger.exception("Could not add an annotation.")
            return

        applogger.info(
            "Added an annotation at (%g, %g) on axis %s", x_value, y_value, axis_id
        )
        self.reload()
        self.figure_edited.emit()

