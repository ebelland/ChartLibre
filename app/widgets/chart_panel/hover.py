"""The readout that follows the pointer over the data."""
from __future__ import annotations

from typing import Any, Final

from PySide6.QtCore import QTimer

import numpy as np

from app.utils.i18n import _

from app.widgets.chart_panel.constants import axis_text


class HoverMixin:
    """The readout that follows the pointer over the data. A part of ChartPanel (see panel.py)."""

    # ------------------------------------------------------------------
    # Hover readout
    # ------------------------------------------------------------------

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
            return

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
