"""ChartPanel with direct Matplotlib canvas sizing.

This version deliberately avoids an intermediate host widget for FIT and
FIT_PROPORTIONAL modes.  The QScrollArea owns the visible chart viewport and the
FigureCanvasQTAgg is the scroll area's widget directly.
"""
from __future__ import annotations

from typing import Any, Final

from PySide6.QtCore import QSize, Qt, QTimer, Signal
import PySide6.QtGui
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib import rcParams
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from app.plugin_api.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import SPLITTER_HANDLE_WIDTH, create_toolbar_button
from app.utils.config import get_section, update_section
from app.widgets.html_results import HtmlResultsView

from app.widgets.chart_panel.canvas import ChartCanvas
from app.widgets.chart_panel.constants import (
    TOOLBAR_ICON_SIZE,
    TOOLBAR_ACTIONS_TO_REMOVE,
    CONFIG_SECTION,
    CONFIG_RESIZE_MODE,
    CONFIG_MIN_ZOOM,
    CONFIG_MAX_ZOOM,
    CONFIG_INITIAL_ZOOM,
    CONFIG_BACKGROUND_COLOR,
    FIGURE_VIEW_OPTIONS_KEY,
    RESIZE_MODES,
    CHART_AREA_MIN_HEIGHT,
    NOTES_MIN_HEIGHT,
    FIXED_MODE_SCREEN_DPI,
)
from app.widgets.chart_panel.notes import NotesMixin
from app.widgets.chart_panel.picking import PickingMixin
from app.widgets.chart_panel.hover import HoverMixin
from app.widgets.chart_panel.tools import ToolsMixin
from app.widgets.chart_panel.menus import MenusMixin
from app.widgets.chart_panel.geometry import GeometryMixin
from app.widgets.chart_panel.actions import ActionsMixin


class ChartPanel(NotesMixin, PickingMixin, HoverMixin, ToolsMixin, MenusMixin, GeometryMixin, ActionsMixin, QFrame):
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
    #: A plain click on the chart: {"axis_id": int | None, "series": str | None} (see PickingMixin).
    chart_object_clicked = Signal(dict)
    #: The tables whose Selected rows a box selection on this chart just set (todo R-07).
    rows_selected = Signal(list)
    #: The tables whose Show only Selected this chart just switched (R-07): every chart of them draws again.
    selected_only_changed = Signal(list)
    #: A "What next?" link in the notes (a report kept with the chart) was clicked: its URL.
    next_step_requested = Signal(str)

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
        self._background_color: str = chart_config.get(CONFIG_BACKGROUND_COLOR, "#ffffff")

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
        self._toolbar = NavigationToolbar2QT(
            self._canvas,
            parent=self,
            coordinates=True,
        )

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
        # _testChart.py, beside main.py, shows the difference on a Mac.
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._canvas.installEventFilter(self)
        self._canvas.customContextMenuRequested.connect(self._show_context_menu)

    def _configure_toolbar(self) -> None:
        """Make the Matplotlib toolbar compact and QSS-addressable."""
        self._toolbar.setObjectName("chartToolbar")
        self._toolbar.setMovable(False)
        self._toolbar.setFloatable(False)
        self._toolbar.setIconSize(TOOLBAR_ICON_SIZE)
        self._toolbar.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self._toolbar.setMinimumWidth(0)
        self._toolbar.setFixedHeight(self._toolbar.sizeHint().height())
        self._remove_toolbar_actions(TOOLBAR_ACTIONS_TO_REMOVE)

        for button in self._toolbar.findChildren(QToolButton):
            button.setAutoRaise(True)
            button.setObjectName("chartToolButton")

    def _build_top_row(self) -> QWidget:
        """Create a fixed-height top control row."""
        top_row = QWidget(self)
        top_row.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )

        layout = QHBoxLayout(top_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toolbar, 0)
        layout.addStretch(1)

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
            title = self._repo.get_figure_title(self._figure_id)
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES,
                label=f"Delete figure '{title}'" if title else "Delete figure",
            )
            self._repo.delete_figure(figure_id=self._figure_id)
            self._deleted = True
            self.delete_requested.emit(self._figure_id)
        except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
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
        except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
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
        except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
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
        except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
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
