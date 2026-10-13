"""Resize modes, zoom, background, and the figure's geometry and metrics."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, QSize, Qt
import PySide6.QtGui
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from matplotlib import rcParams

from app.charts.descriptor_grid import descriptor_axis_count, descriptor_prepared_for_render
from app.logs.logger import applogger
from app.utils.config import get_section
from app.utils.figure_metrics import (
    CM_PER_INCH,
    DEFAULT_FIGURE_DPI,
    DEFAULT_FIGURE_SIZE_IN,
    figure_metrics_from_options,
)
from app.utils.hidpi import apply_configured_dpi, inches_to_logical, logical_to_inches

from app.widgets.chart_panel.constants import DEFAULT_ASPECT_RATIO, CONFIG_SECTION, FIXED_MODE_SCREEN_DPI


class GeometryMixin:
    """Resize modes, zoom, background, and the figure's geometry and metrics. A part of ChartPanel (see panel.py)."""

    # ------------------------------------------------------------------
    # Resize / zoom / background helpers
    # ------------------------------------------------------------------
    def minimumSizeHint(self) -> QSize:
        """Keep parent layouts from using the chart content as minimum size."""
        return QSize(80, 60)

    def sizeHint(self) -> QSize:
        """Return a moderate preferred size, independent from canvas content."""
        return QSize(800, 450)

    def showEvent(self, event: PySide6.QtGui.QShowEvent) -> None:
        """Refresh geometry when a hidden chart tab becomes visible.

        Also a backstop for the deferred first render (see __init__'s
        ``defer_render``): MainWindow always calls ensure_rendered() itself
        before connecting the properties panel to a newly-current tab, but
        this covers becoming visible any other way too.
        """
        super().showEvent(event)
        self.ensure_rendered()
        self._schedule_canvas_geometry_sync(redraw=True, late=True)

    def event(self, event: QEvent) -> bool:
        """Catch tab/page layout events that are not resize events."""
        handled = super().event(event)
        try:
            event_type = event.type()
        except Exception:
            return handled
        if event_type == QEvent.Type.Show:
            self._schedule_canvas_geometry_sync(redraw=True, late=True)
        elif event_type in (QEvent.Type.LayoutRequest, QEvent.Type.PolishRequest):
            # A check, not a redraw: Qt posts LayoutRequest whenever a child's
            # size hint changes - the toolbar's "x=…, y=…" readout does on
            # every mouse move - and redrawing the whole figure for each one
            # made hovering cost a full draw per move. A real size change
            # still redraws, in _sync_canvas_geometry.
            self._schedule_canvas_geometry_sync(redraw=False, late=True)
        return handled

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Re-sync the canvas geometry after resize events.

        Paint events are deliberately *not* handled here.

        They used to be: the viewport's Paint was intercepted, the background
        filled by hand, and True returned.  Two things were wrong with that.
        It painted a widget from outside its own paintEvent, and returning True
        cancelled the paint Qt was about to perform - so the viewport was never
        marked clean, and the compositor was free to leave stale content
        anywhere in the window.  Tiles of unrelated widgets repeating down the
        window are what that looks like on macOS.

        Nothing had to replace it.  ``_apply_background_color_to_widgets``
        already gives the viewport ``autoFillBackground`` and the chart
        background colour, so Qt erases it on every paint through the normal
        path and the canvas draws on top.  The filter was suppressing the very
        paint that would have cleared the area it was trying to clear.
        """
        try:
            event_type = event.type()
        except Exception:
            return False

        if event_type != QEvent.Type.Resize:
            return super().eventFilter(watched, event)

        if watched is self._canvas:
            self._schedule_canvas_geometry_sync(redraw=True)
            return super().eventFilter(watched, event)

        if watched is self._fixed_scroll_area.viewport() or watched is self._chart_area:
            self._schedule_canvas_geometry_sync(redraw=True)

        return super().eventFilter(watched, event)

    def _fill_background(self, widget: QWidget) -> None:
        """Fill a widget's whole rect with the configured background colour."""
        color = PySide6.QtGui.QColor(self._background_color)
        if not color.isValid():
            color = PySide6.QtGui.QColor("#ffffff")

        painter = PySide6.QtGui.QPainter(widget)
        try:
            painter.fillRect(widget.rect(), color)
        finally:
            painter.end()

    def paintEvent(self, event: PySide6.QtGui.QPaintEvent) -> None:
        """Paint the panel background before the frame and children.

        Why this is not left to the stylesheet: in FIT_PROPORTIONAL the canvas
        is centred and smaller than the panel, so a strip of the panel is left
        uncovered.  With an opaque-painting child on top, macOS does not erase
        that strip and it keeps the previous frame's pixels.  Filling it
        explicitly costs one rect fill and removes the dependency on
        platform-specific erase behaviour.
        """
        self._fill_background(self)
        super().paintEvent(event)

    def resizeEvent(self, event: PySide6.QtGui.QResizeEvent) -> None:
        """Update canvas geometry when ChartPanel is resized."""
        super().resizeEvent(event)
        # The strip the canvas vacates belongs to the panel; ask for a repaint
        # explicitly rather than trusting the platform to invalidate the newly
        # exposed area (see paintEvent).
        self.update()
        self._schedule_canvas_geometry_sync(redraw=True)


    def set_resize_mode(
        self,
        resize_mode: str,
        *,
        persist: bool = True,
        redraw: bool = True,
    ) -> None:
        """Set resize behaviour and persist it for this figure by default."""
        self._resize_mode = self._normalized_resize_mode(resize_mode)
        is_fixed = self._resize_mode == "FIXED"

        self._zoom_fit_button.setVisible(is_fixed)
        self._zoom_slider.setVisible(is_fixed)
        self._zoom_label.setVisible(is_fixed)

        if is_fixed:
            self._move_canvas_to_fixed_page()
            self._apply_fixed_mode_pixel_size()
        else:
            self._move_canvas_to_direct_layout(
                centered=self._resize_mode == "FIT_PROPORTIONAL"
            )
            self._restore_unzoomed_figure_metrics()

        if persist:
            self._persist_chart_panel_config()
            self._persist_view_state()
        if redraw:
            self._schedule_canvas_geometry_sync(redraw=True)

    def _remove_active_chart_widget_from_layout(self) -> None:
        """Detach chart display widgets from the main layout.

        Qt layouts tolerate removeWidget() for widgets that are not currently
        managed by the layout.  Removing both possible chart widgets makes mode
        switching deterministic and prevents a hidden scroll area from remaining
        as a layout item.
        """
        self._main_layout.removeWidget(self._canvas)
        self._main_layout.removeWidget(self._fixed_scroll_area)

    def _move_canvas_to_direct_layout(self, *, centered: bool) -> None:
        """Put the canvas directly in the ChartPanel layout for FIT modes.

        FIT and FIT_PROPORTIONAL intentionally share the same parent/layout
        path.  FIT lets the layout expand the canvas.  FIT_PROPORTIONAL keeps
        the same layout path, but gives the canvas a fixed proportional size
        and lets the layout center it in the chart client area.
        """
        self._remove_active_chart_widget_from_layout()

        if self._canvas.parent() is self._fixed_scroll_area.viewport():
            self._fixed_scroll_area.takeWidget()

        self._fixed_scroll_area.setVisible(False)
        self._fixed_scroll_area.lower()
        self._canvas.setParent(self._chart_area)
        self._canvas.show()
        self._canvas.raise_()

        if centered:
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            self._main_layout.addWidget(
                self._canvas,
                1,
                Qt.AlignmentFlag.AlignCenter,
            )
        else:
            # Clear any fixed-size constraint left by FIT_PROPORTIONAL/FIXED.
            self._canvas.setMinimumSize(0, 0)
            self._canvas.setMaximumSize(QSize(16777215, 16777215))
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self._main_layout.addWidget(self._canvas, 1)

        self._active_chart_widget = self._canvas

    def _move_canvas_to_fixed_page(self) -> None:
        """Put the canvas into the scroll area for FIXED mode."""
        self._remove_active_chart_widget_from_layout()

        if self._canvas.parent() is not self._fixed_scroll_area.viewport():
            self._fixed_scroll_area.takeWidget()
            self._fixed_scroll_area.setWidget(self._canvas)

        self._main_layout.addWidget(self._fixed_scroll_area, 1)
        self._fixed_scroll_area.setVisible(True)
        self._fixed_scroll_area.raise_()
        self._active_chart_widget = self._fixed_scroll_area

    def set_zoom_percent(self, percent: int) -> None:
        """Set FIXED-mode zoom and persist it to config.json."""
        new_value = self._clamp_zoom(percent)
        if new_value == self._zoom_percent:
            return

        self._zoom_percent = new_value
        self._zoom_label.setText(f"{self._zoom_percent}%")

        if self._zoom_slider.value() != self._zoom_percent:
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(self._zoom_percent)
            self._zoom_slider.blockSignals(False)

        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()
            self._schedule_canvas_geometry_sync(redraw=True)

        self._persist_chart_panel_config()
        self._persist_view_state()

    def background_color(self) -> str:
        """Return the background colour used around the chart canvas."""
        return self._background_color

    def _apply_background_color_to_widgets(self) -> None:
        """Apply the panel background without changing the Matplotlib figure.

        The background belongs to the Qt client area around the canvas.  It must
        be visible in FIT_PROPORTIONAL margins and in the FIXED scroll viewport,
        but it must not alter Matplotlib figure or axes face colors.

        ``autoFillBackground`` here is what erases the scroll viewport before
        each paint, which matters because the canvas sets WA_OpaquePaintEvent
        and so paints only its own rect.  Do not add a Paint handler for the
        viewport on top of this: one existed, returned True, and cancelled this
        erase - see eventFilter.
        """
        qcolor = PySide6.QtGui.QColor(self._background_color) 
        if not qcolor.isValid():
            qcolor = PySide6.QtGui.QColor("#ffffff")
            self._background_color = qcolor.name()
            applogger.warning("Invalid background color")

        color_name = qcolor.name()

        for widget in self._background_painted_widgets():
            widget.setAutoFillBackground(True)
            widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            palette = widget.palette()
            palette.setColor(PySide6.QtGui.QPalette.ColorRole.Window, qcolor)
            palette.setColor(PySide6.QtGui.QPalette.ColorRole.Base, qcolor)
            widget.setPalette(palette)

        # Style sheets make the background deterministic even when the app-wide
        # QSS disables palette painting for QFrame/QScrollArea descendants.
        self.setStyleSheet(f"QFrame#ChartPanel {{ background-color: {color_name}; }}")
        self._fixed_scroll_area.setStyleSheet("QScrollArea#chartScrollArea { "
            f"background-color: {color_name}; border: none; "
            "}"
        )
        self._fixed_scroll_area.viewport().setStyleSheet(f"background-color: {color_name};")


    def _background_painted_widgets(self) -> tuple[QWidget, ...]:
        """Return widgets that should show the chosen background colour."""
        return (
            self,
            self._chart_area,
            self._notes_view,
            self._fixed_scroll_area,
            self._fixed_scroll_area.viewport(),
        )

    def _schedule_canvas_geometry_sync(self, *, redraw: bool, late: bool = False) -> None:
        """Coalesce resize-driven canvas updates using owned timers.

        The old gwXG sizing path is kept intact.  The only change here is that
        the callback is owned by the widget, so it cannot fire after close, and
        a late pass is available for tab activation/final layout geometry.
        """
        self._pending_canvas_redraw = self._pending_canvas_redraw or bool(redraw)

        if not self._pending_canvas_sync:
            self._pending_canvas_sync = True
            self._canvas_sync_timer.start(0)

        if late:
            self._canvas_late_sync_timer.start(80)

    def _run_pending_canvas_geometry_sync(self) -> None:
        """Run the pending geometry sync with the merged redraw flag."""
        if self._deleted:
            return
        redraw = self._pending_canvas_redraw
        self._pending_canvas_redraw = False
        self._sync_canvas_geometry(redraw=redraw)

    def _sync_canvas_geometry(self, *, redraw: bool) -> None:
        """Synchronize canvas and Matplotlib figure geometry.

        FIT is pure Qt layout management.  The canvas receives its size from
        the ChartPanel layout and this method only redraws when that size
        changes.

        FIT_PROPORTIONAL stays as close as possible to FIT: the canvas is still
        a direct widget in the ChartPanel layout.  The only difference from FIT
        is that the canvas width and height are fixed to the largest size that
        preserves the natural figure aspect ratio inside the same chart client
        area used by FIT.

        FIXED moves the canvas into _fixed_scroll_area and gives it the zoomed,
        fixed figure pixel size.
        """
        self._pending_canvas_sync = False
        figure_before = (tuple(self._figure.get_size_inches()), self._figure.dpi)

        if self._resize_mode == "FIXED":
            target_size = self._current_figure_pixel_size(apply_zoom=True)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            if self._canvas.size() != target_size:
                self._canvas.setFixedSize(target_size)
                self._fixed_scroll_area.viewport().update()
            self._apply_figure_size_from_canvas(target_size)

        elif self._resize_mode == "FIT_PROPORTIONAL":
            area_size = self._direct_chart_client_size()
            if area_size.width() <= 0 or area_size.height() <= 0:
                return

            target_size = self._proportional_size_for_viewport(area_size)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            if self._canvas.size() != target_size:
                self._canvas.setFixedSize(target_size)
                self._main_layout.invalidate()

            self._apply_figure_size_from_canvas(target_size)

        else:
            # FIT: the layout owns the widget geometry.  Do not call
            # setFixedSize(), resize(), or updateGeometry() here, otherwise Qt
            # and the manual synchronizer can enter a resize feedback loop.
            # Still update Matplotlib's logical figure size from the canvas that
            # Qt has assigned.  Without this, tab switches and dialog closes can
            # leave the drawing using the previous panel's inches/DPI even
            # though the QWidget has a new pixel size.
            target_size = QSize(self._canvas.size())
            if target_size.width() <= 0 or target_size.height() <= 0:
                return
            self._apply_figure_size_from_canvas(target_size)

        figure_changed = (tuple(self._figure.get_size_inches()), self._figure.dpi) != figure_before
        if redraw or figure_changed or target_size != self._last_canvas_size:
            self._last_canvas_size = QSize(target_size)
            # Not when the canvas already shows exactly this figure at this
            # size - a resize Matplotlib redrew for itself, a tab shown again:
            # each such second draw cost a whole render of the figure.
            # Nor for a tab nobody can see: showEvent asks again when it is.
            if self._drawn_state != self._draw_state() and self.isVisible():
                self._canvas.draw_idle()

    def _direct_chart_client_size(self) -> QSize:
        """Return the chart area available below the toolbar row."""
        contents = self._chart_area.contentsRect()
        top_row_height = self._top_row.height() if self._top_row.isVisible() else 0
        layout_spacing = self._main_layout.spacing() if top_row_height > 0 else 0
        width = max(0, contents.width())
        height = max(0, contents.height() - top_row_height - layout_spacing)
        return QSize(width, height)

    def _apply_figure_size_from_canvas(self, canvas_size: QSize) -> None:
        """Make Matplotlib's figure size match the canvas.

        ``canvas_size`` is in logical pixels; ``figure.dpi`` counts device
        pixels, because ``FigureCanvasQT._update_pixel_ratio`` multiplied it by
        the display's ratio.  The two have to be reconciled, and this is the
        conversion Matplotlib itself performs in ``FigureCanvasQT.resizeEvent``:

            w = event.size().width() * self.device_pixel_ratio
            winch = w / self.figure.dpi

        Dropping the ratio - which this method used to do - makes the figure
        half the size of the widget on a 2x screen.  Agg then paints a quarter
        of the widget's area and the remainder keeps whatever the backing store
        held, which is the field of artefacts reported on macOS and never on
        Windows, where the ratio is 1 and the mistake cancels out.
        """
        if canvas_size.width() <= 0 or canvas_size.height() <= 0:
            return

        # FIXED too: the canvas is a whole number of pixels, and the figure
        # takes exactly those. Putting the configured inches back (6.4 where
        # 781 pixels at 122% make 6.4016) left the figure a pixel off its own
        # canvas - redrawn twice on every show, or, once that was suppressed,
        # left with a fresh blank buffer of the new size. The configured
        # inches stay in _fixed_figure_size_inches, for export.
        width_in = logical_to_inches(canvas_size.width(), self._figure, self._canvas)
        height_in = logical_to_inches(canvas_size.height(), self._figure, self._canvas)
        self._figure.set_size_inches(width_in, height_in, forward=False)

    def _proportional_size_for_viewport(self, viewport_size: QSize) -> QSize:
        """Return largest size inside viewport preserving current aspect ratio."""
        viewport_width = max(1, viewport_size.width())
        viewport_height = max(1, viewport_size.height())
        ratio = self._current_figure_aspect_ratio()

        width_from_height = int(round(float(viewport_height) * ratio))
        if width_from_height <= viewport_width:
            return QSize(max(1, width_from_height), viewport_height)

        height_from_width = int(round(float(viewport_width) / ratio))
        return QSize(viewport_width, max(1, height_from_width))

    def _clamp_zoom(self, percent: int) -> int:
        return max(self._min_zoom_percent, min(self._max_zoom_percent, int(percent)))

    def _current_figure_aspect_ratio(self) -> float:
        """Return the unzoomed rendered figure aspect ratio.

        FIT_PROPORTIONAL must preserve the chart's natural rendered aspect, not
        the current canvas aspect.  In FIT mode the Matplotlib figure is resized
        to the available client rectangle, so using figure.get_size_inches()
        here would make proportional mode inherit the last FIT rectangle and
        look identical to FIT.  The captured fixed metrics are the stable source
        of truth for proportional sizing.
        """
        try:
            width_in, height_in = self._fixed_figure_size_inches
            width_value = float(width_in)
            height_value = float(height_in)
        except Exception:
            return DEFAULT_ASPECT_RATIO

        if width_value <= 0.0 or height_value <= 0.0:
            return DEFAULT_ASPECT_RATIO

        return width_value / height_value

    # ------------------------------------------------------------------
    # Descriptor and pre-render geometry normalization
    # ------------------------------------------------------------------
    def _descriptor_prepared_for_render(
        self,
        descriptor: Any,
        *,
        axis_count_override: int | None = None,
    ) -> Any:
        """Return a render descriptor with oversized/inconsistent grids fixed.

        The grid-normalization logic itself lives in ``app.charts.
        descriptor_grid`` - it is pure descriptor-tree manipulation with
        nothing Qt about it, and is tested there directly.
        """
        prepared, changed = descriptor_prepared_for_render(
            descriptor, axis_count_override=axis_count_override
        )
        if changed:
            applogger.debug(
                "Normalized chart grid before render (figure_id=%s, axes=%s)",
                self._figure_id,
                axis_count_override
                if axis_count_override is not None
                else descriptor_axis_count(descriptor),
            )
        return prepared

    def _prepare_canvas_geometry_before_render(self) -> None:
        """Synchronize Qt and Matplotlib sizes before rendering.

        ``reload()`` may be called immediately after a modal dialog closes, while
        Qt still has stale splitter/tab geometry cached.  For FIT modes the
        Matplotlib figure size is derived from the *current* chart client area,
        so force the owning window layouts to settle, read the visible size, and
        apply that size before the descriptor is rendered.
        """
        self._activate_current_layouts()

        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()
            self._apply_figure_size_from_canvas(self._current_figure_pixel_size(apply_zoom=True))
            return

        area_size = self._current_fit_client_size()
        if area_size.width() <= 0 or area_size.height() <= 0:
            return

        if self._resize_mode == "FIT_PROPORTIONAL":
            target_size = self._proportional_size_for_viewport(area_size)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            self._canvas.setFixedSize(target_size)
            self._main_layout.invalidate()
        else:
            target_size = area_size
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self._canvas.setMinimumSize(0, 0)
            self._canvas.setMaximumSize(16777215, 16777215)
            self._canvas.resize(target_size.width(), target_size.height())

        self._apply_figure_size_from_canvas(target_size)
        self._last_canvas_size = QSize(target_size)

    def _current_fit_client_size(self) -> QSize:
        """Return the best current canvas area for FIT-style reloads.

        The direct chart client rectangle is authoritative once layouts are
        settled.  If Qt still reports an empty size, fall back to the visible
        window dimensions so reload can still render a sensible fitted figure.
        """
        area_size = self._direct_chart_client_size()
        if area_size.width() > 0 and area_size.height() > 0:
            return area_size
        return self._fallback_chart_client_size_from_window()

    def _activate_current_layouts(self) -> None:
        """Ask Qt layouts to resolve the current window/widget geometry now."""
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        for widget in (self.window(), self, self._content_split, self._chart_area):
            if isinstance(widget, QWidget):
                widget.updateGeometry()
            layout = widget.layout() if isinstance(widget, QWidget) else None
            if layout is not None:
                layout.activate()

        self._fixed_scroll_area.updateGeometry()
        self._fixed_scroll_area.viewport().updateGeometry()

    def _fallback_chart_client_size_from_window(self) -> QSize:
        """Estimate chart client size from the visible window when layout is stale."""
        window = self.window()
        width = max(self._chart_area.width(), self.width())
        height = max(self._chart_area.height(), self.height())

        if isinstance(window, QWidget):
            window_size = window.size()
            width = max(width, window_size.width())
            height = max(height, window_size.height())

        top_row_height = self._top_row.sizeHint().height() if self._top_row.isVisible() else 0
        notes_height = self._notes_view.height() if self._notes_view.isVisible() else 0
        height = max(0, height - top_row_height - notes_height)
        return QSize(max(0, width), max(0, height))

    # ------------------------------------------------------------------
    # Config and figure metrics
    # ------------------------------------------------------------------
    def _chart_config(self) -> dict[str, Any]:
        """Return this panel's own section as a real dict."""
        return get_section(CONFIG_SECTION)

    def _refresh_config_from_source(self) -> None:
        """Reload this panel's section through the app.utils.config API."""
        self._config = self._chart_config()

    def _apply_persisted_figure_metrics_to_rcparams(self) -> None:
        """Apply this figure's own width, height and DPI to rcParams.

        Read from the figure descriptor rather than from config.json: the
        metrics belong to the figure, so every panel has to set them before it
        renders instead of inheriting whatever the previously-opened figure
        left in this process-wide rcParams.

        A figure with no metrics of its own falls back to Matplotlib's own
        defaults, *not* to whatever the last figure left in rcParams. Leaving
        rcParams untouched meant a plain figure inherited the width, height
        and dpi of the last figure that did carry metrics - so it rendered
        oversized, and FIXED-mode "fit to width" then fitted that wrong size.
        """
        try:
            descriptor = self._repo.load_figure_descriptor(self._figure_id)
            options = getattr(descriptor, "options", None) if descriptor else None
            metrics = figure_metrics_from_options(
                options if isinstance(options, dict) else None
            )
            if metrics is None:
                width_in, height_in = DEFAULT_FIGURE_SIZE_IN
                dpi = DEFAULT_FIGURE_DPI
            else:
                width_cm, height_cm, dpi = metrics
                width_in = width_cm / CM_PER_INCH
                height_in = height_cm / CM_PER_INCH

            rcParams["figure.figsize"] = [width_in, height_in]
            rcParams["figure.dpi"] = dpi

            # And the FIXED baseline, which is the same statement: this is the
            # size the figure is configured to be.
            #
            # Without this the baseline was only ever *captured from the last
            # render* - and in FIXED mode what gets rendered is the baseline
            # itself, so it fed on its own output. Whatever the first render
            # produced became permanent, and the width and height typed into
            # the figure properties were applied to rcParams, applied to the
            # figure, and then overwritten by the stale baseline before the
            # descriptor was drawn. Which is exactly how they "were ignored".
            self._fixed_figure_size_inches = (width_in, height_in)
            self._fixed_figure_dpi = float(dpi)
        except Exception:
            applogger.exception(
                "Failed to apply figure metrics (figure_id=%s)", self._figure_id
            )

    def _reset_figure_metrics_from_rcparams_for_reload(self) -> None:
        """Reset logical figure size and base DPI from current rcParams."""
        try:
            figsize = rcParams.get("figure.figsize", None)
            dpi = rcParams.get("figure.dpi", None)

            if figsize is not None and len(figsize) >= 2:
                width_in = float(figsize[0])
                height_in = float(figsize[1])
                if width_in > 0.0 and height_in > 0.0:
                    self._figure.set_size_inches(width_in, height_in, forward=False)

            if dpi is not None and float(dpi) > 0.0:
                # On screen a figure is drawn at the screen's density in every
                # mode, as FIXED already did: its own dpi is an export
                # resolution (Save figure, reports, the project preview), and
                # drawing at it scaled the text with it - a 200 dpi figure in
                # a FIT panel came out with letters twice the size.
                self._set_configured_dpi(FIXED_MODE_SCREEN_DPI)
        except Exception:
            applogger.exception(
                "Failed to reset figure metrics from rcParams (figure_id=%s)",
                self._figure_id,
            )

    def _set_configured_dpi(self, dpi: float) -> None:
        """Set the figure's dpi from a value the user configured.

        ``figure.dpi`` is expected to be in device pixels everywhere else in
        this class, because that is the invariant the Qt backend maintains:
        ``_set_device_pixel_ratio`` multiplies the dpi by the display's ratio
        and only revisits it when the ratio itself changes.

        rcParams and config.json hold the dpi the user asked for, with no
        ratio in it.  Writing that straight onto the figure - which this used
        to do on every reload - silently breaks the invariant for the rest of
        the session, and the figure comes out at the wrong size against the
        canvas again.  Reloading a chart was enough to bring the artefacts back
        on a Retina screen even after the resize path was fixed.

        ``_canvas`` may not exist yet: the constructor prepares the figure from
        rcParams before building the canvas around it.  That is not a problem
        to work around, because ``FigureCanvasQT.__init__`` applies the ratio to
        whatever dpi it finds - so the figure ends up correctly scaled either
        way, and the only thing needed here is not to raise.
        """
        apply_configured_dpi(self._figure, getattr(self, "_canvas", None), dpi)

    def _capture_fixed_metrics_from_rendered_figure(self) -> None:
        """Capture rendered figure metrics for FIXED mode."""
        try:
            width_in, height_in = self._figure.get_size_inches()

            if float(width_in) > 0.0 and float(height_in) > 0.0:
                self._fixed_figure_size_inches = (float(width_in), float(height_in))

            # The dpi is *not* read back from the figure.
            #
            # Two things have been done to it by the time this runs: the Qt
            # backend multiplied it by the display's pixel ratio, and FIXED
            # mode multiplied it again by the zoom.  Capturing that as the base
            # dpi folds the zoom into the baseline, so the next zoom multiplies
            # an already-zoomed value and the chart grows without bound - which
            # is exactly how FIXED zoom stopped being correct.
            #
            # rcParams holds the dpi the user configured, with neither factor
            # in it.  That is the only stable baseline.
            self._fixed_figure_dpi = float(rcParams.get("figure.dpi", 100.0) or 100.0)
        except Exception:
            applogger.exception(
                "Failed to capture rendered figure metrics (figure_id=%s)",
                self._figure_id,
            )

    def _current_figure_pixel_size(self, *, apply_zoom: bool) -> QSize:
        """Return the FIXED-mode canvas size in *logical* pixels.

        The result goes to ``QWidget.setFixedSize``, which speaks logical
        pixels.  Sized from ``FIXED_MODE_SCREEN_DPI``, a fixed on-screen
        reference, rather than from the figure's own configured dpi - dpi is
        a print/export resolution and must not also change how big the
        figure looks on screen.  A 16 inch figure occupies 1600 logical
        pixels at 100% zoom on every display and at every configured dpi;
        only the number of device pixels behind them differs, and that is
        the backend's business rather than this method's.
        """
        try:
            width_in, height_in = self._fixed_figure_size_inches
        except Exception:
            width_in, height_in = 16.0, 9.0

        zoom = self._zoom_percent / 100.0 if apply_zoom else 1.0
        return QSize(
            int(round(inches_to_logical(width_in, FIXED_MODE_SCREEN_DPI, zoom))),
            int(round(inches_to_logical(height_in, FIXED_MODE_SCREEN_DPI, zoom))),
        )

    def _restore_unzoomed_figure_metrics(self) -> None:
        """Restore captured logical size and base DPI after leaving FIXED mode."""
        try:
            width_in, height_in = self._fixed_figure_size_inches
            # The screen's density in, device dpi onto the figure; the
            # figure's own dpi stays in _fixed_figure_dpi, for saving.
            self._set_configured_dpi(FIXED_MODE_SCREEN_DPI)
            self._figure.set_size_inches(width_in, height_in, forward=False)
        except Exception:
            applogger.exception(
                "Failed to restore unzoomed figure metrics (figure_id=%s)",
                self._figure_id,
            )

    def _apply_fixed_mode_pixel_size(self) -> None:
        """Size the canvas for FIXED mode at the current zoom.

        Zoom is applied to ``FIXED_MODE_SCREEN_DPI``, the same on-screen
        reference ``_current_figure_pixel_size`` sizes the widget from - not
        to the figure's own configured dpi, which must stay a pure
        print/export setting.  The figure is rendered at exactly the dpi the
        widget's logical size needs (times the display's own pixel ratio, via
        ``_set_configured_dpi``); using the configured dpi here instead was
        what made the on-screen figure grow or shrink with it.  Width, height
        and the physical print/export size stay in inches throughout - only
        the number of on-screen pixels changes with zoom.

        Neither factor is stored: ``zoom`` from the slider, the display's
        ratio by ``_set_configured_dpi``.  Storing either one is what made
        zoom compound.
        """
        if self._resize_mode != "FIXED":
            return

        try:
            zoom = self._zoom_percent / 100.0
            width_in, height_in = self._fixed_figure_size_inches
            self._set_configured_dpi(FIXED_MODE_SCREEN_DPI * zoom)
            self._figure.set_size_inches(width_in, height_in, forward=False)
            pixels = self._current_figure_pixel_size(apply_zoom=True)
            self._canvas.setFixedSize(pixels)
            # The whole pixels the canvas got, not the inches asked for: see
            # _apply_figure_size_from_canvas.
            self._figure.set_size_inches(
                logical_to_inches(pixels.width(), self._figure, self._canvas),
                logical_to_inches(pixels.height(), self._figure, self._canvas),
                forward=False,
            )
        except Exception:
            applogger.exception(
                "Failed to apply fixed-mode pixel size (figure_id=%s)",
                self._figure_id,
            )
