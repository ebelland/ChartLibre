"""Clicks on the chart: picking a point, a bar, a legend entry; scroll zoom."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.patches import Patch, Rectangle, Wedge

from app.utils.i18n import _

from app.widgets.chart_panel.constants import PICK_TOLERANCE_POINTS, axis_text


class PickingMixin:
    """Clicks on the chart: picking a point, a bar, a legend entry; scroll zoom. A part of ChartPanel (see panel.py)."""

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
        axis: the figure) and ``series`` (the series' name, or None); the main
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
        axes = event.inaxes
        axis_id = getattr(axes, "_dhub_axis_id", None) if axes is not None else None
        series = self._series_name_at(axes, event) if axis_id is not None else None
        self.chart_object_clicked.emit({"axis_id": axis_id, "series": series})

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
            except Exception:  # noqa: BLE001 - an artist that cannot hit-test is simply not it
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

    # Ctrl + left click used to append an annotation to the axis options here
    # and reload. It is gone: the text was the fixed word "Annotation", there
    # was no way to edit it, and nothing on the chart deleted it - the only
    # way back was the annotations table in the axis properties. A dialog that
    # asks for the text at the point clicked replaces it (see N-7 in
    # todo.txt); the axes-to-descriptor mapping it will need is in this file's
    # history.

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
