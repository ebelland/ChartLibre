"""The Chart properties panel's side of the main window.

The four editors - Figure, Axis, Series, Overlays - are widgets of their
own (figure_properties.py and its siblings); what lives here is how the
window wires them to the chart shown: building the tabs, connecting each
editor to the current figure, and carrying out what an editor asks for -
a grid, a layout preset, an axis moved or deleted, a series reordered,
options written back to the project. A mixin of MainWindow.
"""
from __future__ import annotations

from typing import Any, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QWidget

from app.charts import layout_presets
from app.data.repo._common import ensure_read_only_select
from app.logs.logger import applogger
from app.styles.style import MARGIN_TOOLBOX_PAGE, stdSizeAndlayout
from app.utils.config import get_constant
from app.utils.i18n import _
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget
from app.widgets.chart_panel import ChartPanel
from app.widgets.chart_properties.figure_properties import FigurePropertiesWidget
from app.widgets.chart_properties.overlay_properties import OverlayPropertiesWidget
from app.widgets.chart_properties.series_properties import SeriesPropertiesWidget

# Coalescing window for property-driven chart reloads, in milliseconds.
# Long enough to swallow a spinbox drag, short enough to feel immediate.
PROPERTIES_REDRAW_DEBOUNCE_MS: int = get_constant("properties_redraw_debounce_ms", 120)


class MainWindowChartProperties:
    """A part of MainWindow; ``self`` is the window."""

    def _create_properties_control(self) -> QTabWidget:
        """Figure, Axis, Series and Overlays as tabs, each page scrolling on its own.

        Tabs rather than the accordion they were: four headers stacked
        took about 200 px of the panel's height and showed one page at a
        time all the same. Short names, so the four fit a narrow panel;
        when they do not, the tab bar scrolls rather than eliding them.
        """
        self._figure_widget = FigurePropertiesWidget(self)
        self._axis_widget = AxisPropertiesWidget(self)
        self._series_widget = SeriesPropertiesWidget(self)
        self._overlay_widget = OverlayPropertiesWidget(self)

        control = QTabWidget(self)
        control.setObjectName("propertiesTabs")
        control.setDocumentMode(True)
        control.setUsesScrollButtons(True)
        control.setElideMode(Qt.TextElideMode.ElideNone)
        control.tabBar().setExpanding(False)
        # Last: annotations and reference lines are the finishing pass on a
        # chart, done once the data, the axes and the series are right.
        for page, title, tooltip in (
            (self._figure_widget, _("Figure"), _("Figure properties")),
            (self._axis_widget, _("Axis"), _("Axis properties")),
            (self._series_widget, _("Series"), _("Series properties")),
            (self._overlay_widget, _("Overlays"), _("Overlay properties")),
        ):
            # The ground and the room a toolbox page had: the window grey on
            # macOS, white on Windows (see apply_toolbox_page_metrics).
            page.setProperty("toolboxPage", True)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            layout = page.layout()
            if layout is not None and layout.contentsMargins().isNull():
                layout.setContentsMargins(*MARGIN_TOOLBOX_PAGE)
            index = control.addTab(self._scrollable(page), title)
            control.setTabToolTip(index, tooltip)
        control.setCurrentIndex(0)
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
