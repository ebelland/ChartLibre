"""The chart's menus: actions, right-click, reference lines and annotations added from them."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint
import PySide6.QtGui
from PySide6.QtWidgets import QInputDialog, QMenu

from app.logs.logger import applogger
from app.styles.style import MenuItem, action as catalogue_action, create_menu, create_menu_item
from app.utils.i18n import _

from app.widgets.chart_panel.constants import LIVE_REFRESH_INTERVAL_MS, axis_text


class MenusMixin:
    """The chart's menus: actions, right-click, reference lines and annotations added from them. A part of ChartPanel (see panel.py)."""

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
                MenuItem(_("Reload"), _("Reload chart"), None, self.reload,False,"reload"),
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
                None,
                MenuItem(_("Delete"),_("Delete this figure"),PySide6.QtGui.QKeySequence.StandardKey.Delete,self.delete_chart,False,"delete"),
            ],
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
        it creates with ``_dhub_axis_id``, so a click maps back to the
        descriptor without this panel keeping a second mapping that could
        fall out of step with the figure.

        Qt counts pixels from the top-left and Matplotlib from the
        bottom-left; ``mouseEventCoords`` is Matplotlib's own conversion,
        device pixel ratio included, and using it is what makes this agree
        with the toolbar's cursor readout on a Retina screen.
        """
        try:
            x_pixels, y_pixels = self._canvas.mouseEventCoords(pos)
        except Exception:  # noqa: BLE001 - a backend without the helper
            return None

        # Last first: a twin axes is created after the axes it overlays and
        # is the one on top, so it is the one a click belongs to.
        for axes in reversed(list(self._figure.axes)):
            axis_id = getattr(axes, "_dhub_axis_id", None)
            if axis_id is None or not axes.bbox.contains(x_pixels, y_pixels):  # pyright: ignore[reportAttributeAccessIssue]
                continue
            try:
                x_value, y_value = axes.transData.inverted().transform(
                    (x_pixels, y_pixels)
                )
            except Exception:  # noqa: BLE001 - a 3D or polar axes may refuse
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            applogger.exception("Could not add an annotation.")
            return

        applogger.info(
            "Added an annotation at (%g, %g) on axis %s", x_value, y_value, axis_id
        )
        self.reload()
        self.figure_edited.emit()
