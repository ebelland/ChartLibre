"""Interactive tools: crosshair, dragged annotations, box selection, live updates, ruler."""
from __future__ import annotations

import json
from typing import Any, Final

from PySide6.QtCore import QTimer

import numpy as np
from matplotlib.patches import Rectangle

from app.data.data_source import quote_identifier
from app.logs.logger import applogger
from app.plugin_api.data_operations.dialog_base import generated_table_name
from app.utils.i18n import _

from app.widgets.chart_panel.constants import LIVE_REFRESH_INTERVAL_MS


class ToolsMixin:
    """Interactive tools: crosshair, dragged annotations, box selection, live updates, ruler. A part of ChartPanel (see panel.py)."""

    # ------------------------------------------------------------------
    # Crosshair
    # ------------------------------------------------------------------
    #: Colour for the crosshair lines - grey rather than the hover
    #: annotation's own colours, so it reads as a ruled guide under the
    #: data rather than as a second thing competing for attention.
    CROSSHAIR_COLOR: Final[str] = "#808080"

    def _init_crosshair(self) -> None:
        """Reset crosshair state to "off, nothing drawn"."""
        self._crosshair_enabled = False
        #: axes -> its vertical line, shared x across every axes.
        self._crosshair_vlines: dict[Any, Any] = {}
        #: axes -> its horizontal line, only ever visible on the one axes
        #: under the pointer - y scales are not shared, so a horizontal
        #: line on every axes would compare unrelated numbers.
        self._crosshair_hlines: dict[Any, Any] = {}
        #: The linked-zoom/pan toggle lives here too: both are "act across
        #: every axes of this figure" settings a person turns on together
        #: when comparing stacked panels.
        self._link_x_zoom_enabled = False
        self._syncing_x_limits = False

    def _discard_crosshair(self) -> None:
        """Forget the crosshair's artists after figure.clear() destroyed them.

        Mirrors _discard_hover_annotation/_discard_ruler: the enabled flags
        survive a reload (a person's choice to see a crosshair should not
        reset itself every time the chart redraws), only the per-axes
        artists and the pending link-zoom guard do not.
        """
        self._crosshair_vlines = {}
        self._crosshair_hlines = {}
        self._syncing_x_limits = False

    def _on_toggle_crosshair(self, checked: bool) -> None:
        self._crosshair_enabled = bool(checked)
        if not self._crosshair_enabled:
            self._hide_crosshair()

    def _on_toggle_link_x_zoom(self, checked: bool) -> None:
        self._link_x_zoom_enabled = bool(checked)

    def _update_crosshair(self, event: Any | None) -> None:
        """Move the crosshair to the pointer, across every chart axes.

        Always a full draw_idle() rather than blitting: the hover
        annotation's blit restores a *cached* background snapshot, and
        that snapshot is only ever valid with the crosshair at the
        position it was captured at. Keeping the crosshair on the blit
        path too would mean re-capturing the background on every single
        pointer move - the crosshair's whole reason to exist - which
        defeats blitting's purpose anyway. draw_idle() here also drops
        _hover_background through the existing draw_event connection
        (_invalidate_hover_background), so the *next* blit-only hover
        update - crosshair off, or pointer over a bare point - always
        recaptures a fresh background rather than reusing a stale one.
        """
        if event is None or event.xdata is None:
            self._hide_crosshair()
            return

        active_axes = event.inaxes
        for axes in self._figure.axes:
            if not hasattr(axes, "_dhub_axis_id"):
                continue  # a colorbar or other helper axes, not a chart one

            vline = self._crosshair_vlines.get(axes)
            if vline is None:
                vline = axes.axvline(
                    event.xdata,
                    color=self.CROSSHAIR_COLOR,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=9_000,
                )
                self._crosshair_vlines[axes] = vline
            else:
                vline.set_xdata([event.xdata, event.xdata])
            vline.set_visible(True)

            hline = self._crosshair_hlines.get(axes)
            if axes is active_axes and event.ydata is not None:
                if hline is None:
                    hline = axes.axhline(
                        event.ydata,
                        color=self.CROSSHAIR_COLOR,
                        linewidth=0.8,
                        linestyle="--",
                        zorder=9_000,
                    )
                    self._crosshair_hlines[axes] = hline
                else:
                    hline.set_ydata([event.ydata, event.ydata])
                hline.set_visible(True)
            elif hline is not None:
                hline.set_visible(False)

        self._canvas.draw_idle()

    def _hide_crosshair(self) -> None:
        """Hide every crosshair line without discarding the artists."""
        changed = False
        for line in (*self._crosshair_vlines.values(), *self._crosshair_hlines.values()):
            if line.get_visible():
                line.set_visible(False)
                changed = True
        if changed:
            self._canvas.draw_idle()

    def _connect_x_sync_callbacks(self) -> None:
        """Wire every chart axes' xlim_changed to the link-zoom sync.

        xlim_changed fires however the limits changed - scroll zoom, the
        Matplotlib toolbar's pan/zoom tools, a rectangle zoom - so this one
        connection covers all of them rather than patching each entry
        point that can move an axes' x range.
        """
        for axes in self._figure.axes:
            if hasattr(axes, "_dhub_axis_id"):
                axes.callbacks.connect("xlim_changed", self._on_axes_xlim_changed)  # pyright: ignore[reportAttributeAccessIssue]

    def _on_axes_xlim_changed(self, changed_axes: Any) -> None:
        """Apply one axes' new x range to every other chart axes.

        Guarded by _syncing_x_limits: setting another axes' xlim fires its
        own xlim_changed right back at this same callback, and without the
        guard two linked axes would bounce the change back and forth.
        """
        if not self._link_x_zoom_enabled or self._syncing_x_limits:
            return

        xlim = changed_axes.get_xlim()
        self._syncing_x_limits = True
        try:
            for axes in self._figure.axes:
                if axes is changed_axes or not hasattr(axes, "_dhub_axis_id"):
                    continue
                if axes.get_xlim() != xlim:
                    axes.set_xlim(xlim)
        finally:
            self._syncing_x_limits = False
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Draggable annotations
    # ------------------------------------------------------------------
    def _init_annotation_drag(self) -> None:
        """Reset drag state to "nothing being dragged"."""
        #: (axis_id, annotation index, artist, axes) of the annotation
        #: currently being dragged, or None. The axes is kept alongside the
        #: axis id so a drop can be refused if the pointer left it - x/y
        #: from a different axes' coordinate system would be meaningless
        #: written onto this one.
        self._dragging_annotation: tuple[int, int, Any, Any] | None = None

    def _discard_annotation_drag(self) -> None:
        """Forget an in-progress drag after figure.clear() destroyed its artist."""
        self._dragging_annotation = None

    def _annotation_artist_at(self, axes: Any, event: Any) -> tuple[int, Any] | None:
        """Return (index, artist) for the axis annotation under *event*.

        Hit-tests only artists BaseAxisRenderer.apply_annotation tagged
        with ``_dhub_annotation_index`` - the hover readout, a
        measurement's label, and any other Text this axes happens to carry
        are left alone; none of them are ever tagged.
        """
        for artist in axes.texts:
            index = getattr(artist, "_dhub_annotation_index", None)
            if index is None:
                continue
            try:
                hit, _details = artist.contains(event)
            except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
                continue
            if hit:
                return int(index), artist
        return None

    def _on_annotation_drag_press(self, event: Any) -> None:
        """Start dragging an axis annotation, if the click landed on one.

        Left button only, and only while no Matplotlib navigation tool
        (pan/zoom) is active - those clicks belong to the toolbar, not to
        this. Also skipped while box selection is the active tool, so a
        drag never has to guess whether it started a selection or moved
        an annotation.
        """
        if self._select_mode_enabled:
            return
        if event.button != 1 or event.inaxes is None:
            return
        if str(getattr(self._toolbar, "mode", "")) != "":
            return

        axis_id = getattr(event.inaxes, "_dhub_axis_id", None)
        if axis_id is None:
            return

        hit = self._annotation_artist_at(event.inaxes, event)
        if hit is None:
            return

        index, artist = hit
        self._dragging_annotation = (int(axis_id), index, artist, event.inaxes)

    def _on_annotation_drag_motion(self, event: Any) -> None:
        """Follow the pointer while an annotation is being dragged."""
        if self._dragging_annotation is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        _axis_id, _index, artist, axes = self._dragging_annotation
        if event.inaxes is not axes:
            return

        if hasattr(artist, "xy"):
            # An Annotation (the "arrow" type): xy is the anchor point the
            # arrow points to and the stored x/y - xyann (the text/arrow
            # tip) stays at its own offset from it, unaffected.
            artist.xy = (event.xdata, event.ydata)
        else:
            artist.set_position((event.xdata, event.ydata))
        self._canvas.draw_idle()

    def _on_annotation_drag_release(self, event: Any) -> None:
        """Finish an annotation drag: write the new position back, or snap
        back to the stored one if the drop is not usable."""
        if self._dragging_annotation is None:
            return
        axis_id, index, _artist, axes = self._dragging_annotation
        self._dragging_annotation = None

        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            self.reload()
            return

        try:
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES, label=_("Move annotation")
            )
            options = dict(self._repo.get_axis_options(int(axis_id)) or {})
            annotations = list(options.get("annotations") or [])
            if index < 0 or index >= len(annotations):
                self.reload()
                return
            entry = dict(annotations[index])
            entry["x"] = float(event.xdata)
            entry["y"] = float(event.ydata)
            annotations[index] = entry
            options["annotations"] = annotations
            self._repo.set_axis_options(int(axis_id), options)
        except Exception:  # noqa: BLE001
            applogger.exception("Could not move the annotation.")
            self.reload()
            return

        self.reload()
        self.figure_edited.emit()

    # ------------------------------------------------------------------
    # Box selection
    # ------------------------------------------------------------------
    #: Colour for the selection rectangle - matches the crosshair rather
    #: than the ruler/hover colours, since like the crosshair this is a
    #: guide the user draws rather than a value read off the data.
    SELECT_COLOR: Final[str] = "#1f77b4"

    def _init_box_selection(self) -> None:
        """Reset box selection to "off, nothing selected"."""
        self._select_mode_enabled = False
        #: (axes, x0, y0) of the corner a drag started from, or None.
        self._selection_start: tuple[Any, float, float] | None = None
        #: The rubber-band rectangle artist while a drag is in progress.
        self._selection_rect_artist: Any | None = None
        #: The last completed selection - axis id, its bounds, and per
        #: series (keyed by the series' own name, the same identity the
        #: pick/legend code already matches artists by) the descriptor
        #: fields a "Hide"/"New series" action needs. None once acted on,
        #: cleared, or the pointer draws a new one.
        self._last_selection: dict[str, Any] | None = None

    def _discard_box_selection(self) -> None:
        """Forget an in-progress drag after figure.clear() destroyed its artist.

        The completed selection survives a reload deliberately: the axes it
        refers to are gone, but its bounds and series metadata are not tied
        to any artist, so "Hide selected points" still works after e.g. a
        crosshair toggle triggers a redraw in between selecting and acting.
        """
        self._selection_start = None
        self._selection_rect_artist = None

    def _on_toggle_select_mode(self, checked: bool) -> None:
        self._select_mode_enabled = bool(checked)
        if not self._select_mode_enabled:
            self._discard_box_selection()
            self._canvas.draw_idle()

    def _on_box_select_press(self, event: Any) -> None:
        """Start a rubber-band selection, if the tool is on."""
        if not self._select_mode_enabled or event.button != 1:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if not hasattr(event.inaxes, "_dhub_axis_id"):
            return

        self._selection_start = (event.inaxes, float(event.xdata), float(event.ydata))
        rect = Rectangle(
            (event.xdata, event.ydata),
            0,
            0,
            facecolor=self.SELECT_COLOR,
            edgecolor=self.SELECT_COLOR,
            alpha=0.15,
            linewidth=1.0,
            linestyle="--",
            zorder=9_500,
        )
        event.inaxes.add_patch(rect)
        self._selection_rect_artist = rect
        self._canvas.draw_idle()

    def _on_box_select_motion(self, event: Any) -> None:
        """Grow the rubber-band rectangle to follow the pointer."""
        if self._selection_start is None or self._selection_rect_artist is None:
            return
        axes, x0, y0 = self._selection_start
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            return

        x1, y1 = float(event.xdata), float(event.ydata)
        self._selection_rect_artist.set_bounds(
            min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        )
        self._canvas.draw_idle()

    def _on_box_select_release(self, event: Any) -> None:
        """Finish the drag: collect the points it covers, or drop it."""
        if self._selection_start is None:
            return
        axes, x0, y0 = self._selection_start
        self._selection_start = None
        rect_artist = self._selection_rect_artist
        self._selection_rect_artist = None
        if rect_artist is not None:
            rect_artist.remove()

        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            self._last_selection = None
            self._canvas.draw_idle()
            return

        x1, y1 = float(event.xdata), float(event.ydata)
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        if x_min == x_max or y_min == y_max:
            # A click, not a drag - nothing to select.
            self._last_selection = None
            self._canvas.draw_idle()
            return

        axis_id = int(getattr(axes, "_dhub_axis_id"))
        self._last_selection = self._collect_selection(axis_id, axes, x_min, x_max, y_min, y_max)
        self._canvas.draw_idle()

        if self._last_selection is None:
            self.selection_changed.emit(_("No points in the selection."))
            return
        point_count = sum(len(info["x"]) for info in self._last_selection["series"].values())
        self.selection_changed.emit(
            _("{count} point(s) selected across {series} series - right-click to act on them.").format(
                count=point_count, series=len(self._last_selection["series"])
            )
        )

    def _collect_selection(
        self, axis_id: int, axes: Any, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> dict[str, Any] | None:
        """Return every plotted point inside the rectangle, grouped by series.

        Matched to a series descriptor by artist label, the same identity
        _on_pick and the legend toggle already use - a renderer draws a
        series' Line2D/PathCollection labelled with the series' own name.
        """
        series_rows = {str(row["name"]): row for row in self._repo.get_series(axis_id) or []}
        matches: dict[str, Any] = {}

        for artist in (*axes.lines, *axes.collections):
            label = str(artist.get_label() or "").strip()
            if not label or label.startswith("_") or label not in series_rows:
                continue

            if hasattr(artist, "get_offsets"):
                points = np.asarray(artist.get_offsets())
                if points.size == 0:
                    continue
                xs, ys = points[:, 0].astype(float), points[:, 1].astype(float)
            else:
                x_data, y_data = artist.get_data()
                try:
                    xs = np.asarray(x_data, dtype=float)
                    ys = np.asarray(y_data, dtype=float)
                except (TypeError, ValueError):
                    continue

            mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
            if not mask.any():
                continue

            row = series_rows[label]
            matches[label] = {
                "series_id": int(row["id"]),
                "sql_query": str(row["sql_query"] or ""),
                "roles": json.loads(row["roles"]) if row["roles"] else {},
                "style": json.loads(row["style_json"]) if row["style_json"] else {},
                "x": xs[mask],
                "y": ys[mask],
            }

        if not matches:
            return None
        return {
            "axis_id": axis_id,
            "x0": x_min,
            "x1": x_max,
            "y0": y_min,
            "y1": y_max,
            "series": matches,
        }

    def _hide_selection(self) -> None:
        """Hide every selected point by rewriting its series' own query.

        Not a rowid-based Hide flag (that only ever worked for a series
        reading a plain table, and even then turned out unreliable in
        practice - see todo.txt P-02). Wrapping the series' existing SQL in
        a WHERE clause that keeps only the points OUTSIDE the rectangle
        works for any series, table-backed or not - the same
        "read the series' own query as a subquery" idea
        _create_series_from_selection already uses, just with the
        condition negated (De Morgan: NOT(xmin<=x<=xmax AND ymin<=y<=ymax)
        is x<xmin OR x>xmax OR y<ymin OR y>ymax) and updating the existing
        series in place instead of adding a new one.
        """
        selection = self._last_selection
        if not selection:
            return
        self._last_selection = None

        try:
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES, label=_("Hide selected points")
            )
            hid_any = False
            for info in selection["series"].values():
                roles = info["roles"]
                if "x" not in roles or "y" not in roles:
                    continue
                new_sql = (
                    f"SELECT * FROM ({info['sql_query']}) "
                    f"WHERE (x < {selection['x0']!r} OR x > {selection['x1']!r} "
                    f"OR y < {selection['y0']!r} OR y > {selection['y1']!r})"
                )
                self._repo.update_series_sql_query(int(info["series_id"]), new_sql)
                hid_any = True
        except Exception:  # noqa: BLE001
            applogger.exception("Could not hide the selected points.")
            return

        if not hid_any:
            return
        self.reload()
        self.figure_edited.emit()

    def _create_series_from_selection(self) -> None:
        """Add one new series per selected series, holding only its selected rows.

        Works for any series query, not only a plain table: the rectangle is
        just a WHERE clause wrapped around whatever the source series already
        selects, the same "read the series' own query as a subquery" idea
        the rest of the app already leans on for a series-over-a-series.
        """
        selection = self._last_selection
        if not selection:
            return
        self._last_selection = None

        axis_id = selection["axis_id"]
        try:
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES, label=_("New series from selection")
            )
            next_index = len(self._repo.get_series(axis_id) or [])
            created_any = False
            for label, info in selection["series"].items():
                sql = (
                    f"SELECT * FROM ({info['sql_query']}) "
                    f"WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ?"
                )
                frame = self._repo.query_df(
                    sql, (selection["x0"], selection["x1"], selection["y0"], selection["y1"])
                )
                if frame.empty:
                    continue
                table_name = generated_table_name(f"{label}_selection")
                self._repo.import_dataframe(frame, table_name=table_name, normalize_columns=False)
                style = dict(info["style"])
                style.pop("generated_selection", None)
                style["generated_selection"] = True
                self._repo.create_series_descriptor(
                    axis_id=axis_id,
                    series_index=next_index,
                    name=_("{name} (selection)").format(name=label),
                    sql_query=f"SELECT * FROM {quote_identifier(table_name)}",
                    roles=info["roles"],
                    style=style,
                )
                next_index += 1
                created_any = True
        except Exception:  # noqa: BLE001
            applogger.exception("Could not create a series from the selection.")
            return

        if not created_any:
            return
        self.reload()
        self.figure_edited.emit()

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------
    def _init_live_updates(self) -> None:
        """Turn live updates off and create its timer.

        Called once from __init__, not on every reload: like the crosshair
        and box-selection toggles, a person's choice to watch this chart
        live is a viewer setting that should survive the chart's own
        redraws, not something reload() resets every time it runs.
        """
        self._live_updates_enabled = False
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(LIVE_REFRESH_INTERVAL_MS)
        self._live_timer.timeout.connect(self._on_live_timer_tick)

    def _on_toggle_live_updates(self, checked: bool) -> None:
        self._live_updates_enabled = bool(checked)
        if self._live_updates_enabled:
            self._live_timer.start()
        else:
            self._live_timer.stop()

    def _on_live_timer_tick(self) -> None:
        """Redraw on the timer, unless a gesture on this chart is mid-flight.

        reload() clears the figure and rebuilds it from the descriptor, so
        a redraw arriving mid-drag would pull the artist a drag or a
        pending ruler point refers to out from under it. A completed box
        selection is not guarded here: it already survives reload() by
        design (see _discard_box_selection), so a timer tick between
        selecting and right-clicking "Hide selected" does not lose it.
        """
        if (
            self._dragging_annotation is not None
            or self._selection_start is not None
            or self._ruler_start is not None
        ):
            return
        self.reload()

    # ------------------------------------------------------------------
    # Ruler / measure mode
    # ------------------------------------------------------------------
    #: Colour for the pending point's marker. A completed measurement's own
    #: colour lives in its stored kwargs instead (BaseAxisRenderer.
    #: apply_measurement), defaulting to this same value.
    RULER_COLOR: Final[str] = "#d62728"

    def _init_ruler(self) -> None:
        """Reset the ruler to "no pending point"."""
        #: (axis id, x, y) of the point "Measure from here" was clicked at,
        #: or None while nothing is pending. A *finished* measurement is
        #: not tracked here at all - see _measure_to_here - it is stored on
        #: the axis and drawn back by BaseAxisRenderer.apply_measurements
        #: like a reference line, not kept as a transient artist.
        self._ruler_start: tuple[int, float, float] | None = None
        #: The pending point's own marker artist, if one is currently shown.
        self._ruler_artists: list[Any] = []

    def _discard_ruler(self) -> None:
        """Forget the pending point after the figure it was drawn on is gone.

        Mirrors _discard_hover_annotation: figure.clear() (in reload())
        already destroyed the marker artist, so the reference is simply
        dropped rather than removed a second time.
        """
        self._ruler_start = None
        self._ruler_artists = []

    def _clear_ruler(self) -> None:
        """Remove the pending point's marker, and forget it.

        Unlike _discard_ruler, the figure here is still alive - this is
        either "Cancel measurement", or the first step of starting a new
        one over an old pending point - so the marker is actually removed
        rather than merely forgotten.
        """
        for artist in self._ruler_artists:
            try:
                artist.remove()
            except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
                pass
        self._ruler_artists = []
        self._ruler_start = None
        self._canvas.draw_idle()

    def _measure_from_here(
        self, axes: Any, axis_id: int, x_value: float, y_value: float
    ) -> None:
        """Start a measurement at the clicked point.

        Replaces any point already pending - only one measurement can be
        in progress at a time, so starting a new one always reads as "the
        current one", never as an old pending click left behind by mistake.
        """
        self._clear_ruler()
        self._ruler_start = (axis_id, x_value, y_value)
        (marker,) = axes.plot(
            [x_value],
            [y_value],
            marker="+",
            markersize=11,
            markeredgewidth=1.6,
            color=self.RULER_COLOR,
            linestyle="none",
            zorder=10_000,
        )
        self._ruler_artists.append(marker)
        self._canvas.draw_idle()

    def _measure_to_here(
        self, axes: Any, axis_id: int, x_value: float, y_value: float
    ) -> None:
        """Finish the pending measurement at the clicked point, and store it.

        A no-op if nothing is pending, or if the second click landed on a
        different axes than the first - dx/dy/slope mean nothing between
        two axes that may not even share units.

        Written to the axis' own "measurements" - the same storage a
        reference line or an annotation gets (BaseAxisRenderer.
        apply_measurements draws it back) - rather than drawn here as a
        transient artist: a measurement worth taking is worth it still
        being there next time the chart opens, and worth deleting from the
        Overlay panel's Measurements tab the same way a line is.
        """
        if self._ruler_start is None or self._ruler_start[0] != axis_id:
            return

        _axis_id, x0, y0 = self._ruler_start
        self._clear_ruler()

        try:
            self._repo.snapshot_for_undo(
                self._repo.DESCRIPTOR_TABLES, label=_("Add measurement")
            )
            options = dict(self._repo.get_axis_options(int(axis_id)) or {})
            measurements = list(options.get("measurements") or [])
            measurements.append(
                {"x0": x0, "y0": y0, "x1": x_value, "y1": y_value, "kwargs": {}}
            )
            options["measurements"] = measurements
            self._repo.set_axis_options(int(axis_id), options)
        except Exception:  # noqa: BLE001
            applogger.exception("Could not add a measurement.")
            return

        applogger.info(
            "Added a measurement (%g, %g) -> (%g, %g) on axis %s",
            x0, y0, x_value, y_value, axis_id,
        )
        self.reload()
        self.figure_edited.emit()
