"""Strict-typed, responsive axis properties editor for ChartLibre.

Edits one axis: its title and labels, its scale and limits, its grid and
legend, and the selected renderer's own Kwargs/Options, which are built from
the renderer's declarations rather than laid out by hand (see
``app/charts/base.py``).

Reloads route every control through ``_FormSignalBlocker`` so that
repopulating the form does not read back as the user editing it - a cascade
that used to apply an edit per widget on every axis change.
"""
from __future__ import annotations

from typing import Any, Final, TypeAlias, cast

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.widgets.font_scale_control import FontScaleControl
from app.logs.logger import applogger
from app.charts import axis_options
from app.charts.render_figure import SUPPORTED_AXIS_SCALES
from app.styles.style import (
    MARGIN_PANEL,
    CardFrame,
    TitledCard,
    create_action_button,
    mark_icon_only,
    stdSizeAndlayout,
    configure_combo_width,
)
from app.utils.messages import ask
from app.widgets.chart_properties.base_properties import BaseProperties
from app.widgets.dictionary_editor import DictEditorPanel

AxisDescriptorLike: TypeAlias = Any
RendererConfig: TypeAlias = dict[str, Any]
AxisPayload: TypeAlias = dict[str, Any]

from app.scanners.axis_renderer_scanner import get_renderer,import_class_from_file
from app.utils.i18n import _, tr
MAX_QT_HEIGHT: int = 16_777_215

# The editor offers exactly what the renderer knows how to apply, so the two
# cannot drift: these come straight from render_figure.
AXIS_SCALES: Final[tuple[str, ...]] = SUPPORTED_AXIS_SCALES
#: Kept for the figure options elsewhere in this module; the axis grid and
#: tick controls now come from app.charts.axis_options, which is the same
#: vocabulary the renderer applies.
# The empty entry means "leave the Matplotlib default alone".

def _plain_options(value: object) -> dict[str, Any]:
    """Return a copy of a descriptor/options mapping with strict typing."""
    if isinstance(value, dict):
        return dict(cast(dict[str, Any], value))
    return {}

class AxisPropertiesWidget(BaseProperties):
    """Compact editor for axis descriptor properties.

    The widget intentionally works with the current descriptor schema. Missing
    descriptor fields are logged instead of guessed through legacy fallbacks.
    """

    axis_selected = Signal(int)
    axis_options_requested = Signal(dict)
    axis_action_requested = Signal(dict)
    renderer_changed = Signal(str)

    KWARGS_PANEL_MIN_HEIGHT: int = 120

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # _repo/_figure_id/_figure/_redraw_callback come from BaseProperties.
        self._current_axis_id: int | None = None
        self._axis_map: dict[int, AxisDescriptorLike] = {}
        self._kwargs_editor: DictEditorPanel | None = None
        # (axis_id, renderer_key, resolved kwargs values) of the editor
        # currently built - see rebuild_kwargs_editor's no-op guard.
        self._kwargs_build_signature: tuple[Any, ...] | None = None
        self._limit_spins: dict[tuple[str, str], QDoubleSpinBox] = {}
        self._limit_auto_checks: dict[tuple[str, str], QCheckBox] = {}
        # Rebuilt with the editor it sits on - see rebuild_kwargs_editor.
        # create_action_button() and mark_icon_only() return a generic button
        # instance rather than a QPushButton specifically, so the attribute must
        # match that broader Qt type.
        self._btn_reset_kwargs: QAbstractButton | None = None
        # Expanding/Expanding already set by BaseProperties.
        self.setMinimumHeight(0)
        self._build_ui()
        self._install_auto_apply(self._emit_axis_options_requested)
        self._connect_auto_apply()
        self.clear_connected_figure()

    def _connect_auto_apply(self) -> None:
        """Apply an axis edit a short moment after the last control change.

        ``_axis_combo`` is left out - it selects which axis is shown, not an
        edit to one. Every form control is populated inside
        ``_form_signal_blocker`` on a reload, so the change signals only ever
        reach the timer when the user is the one turning the control.
        """
        widgets = (
            self._axis_label_edit,
            self._x_label_edit,
            self._y_label_edit,
            self._z_label_edit,
            self._projection_combo,
            self._sharex_check,
            self._sharey_check,
            self._sharez_check,
            self._hide_axis_check,
        ) + self._extended_option_widgets()

        for widget in widgets:
            if isinstance(widget, QLineEdit):
                self._apply_on_commit(widget)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._queue_auto_apply)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._queue_auto_apply)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._queue_auto_apply)
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self._queue_auto_apply)
        self._font_scale.value_changed.connect(self._queue_auto_apply)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
 
    def _configure_combo_width(
        self,
        combo: QComboBox,
        minimum_contents_length: int = 0,
    ) -> None:
        """Make combo boxes use the available panel width.

        Thin wrapper kept so the call sites read the same as before; the rule
        itself is shared with the figure panel in ``style``.
        """
        configure_combo_width(combo, minimum_contents_length)

    def _build_ui(self) -> None:
        """Build X/Y/Z pages that host the shared Properties and Kwargs cards."""
        root = QVBoxLayout(self)
        root.setContentsMargins(*MARGIN_PANEL)
        root.setSpacing(12)
        root.addWidget(self._build_axis_selector_section())

        self._build_coordinate_controls()
        self._shared_properties_section = self._build_shared_properties_frame()
        self._shared_kwargs_section = self._build_kwargs_section()

        self._tabs = QTabWidget(self, tabShape=QTabWidget.TabShape.Rounded)
        self._tabs.setObjectName("axisPropertiesTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setMinimumHeight(0)
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._coordinate_shared_hosts: dict[str, QVBoxLayout] = {}
        for axis in axis_options.AXES:
            self._tabs.addTab(self._build_coordinate_tab(axis), axis.upper())
        self._tabs.currentChanged.connect(self._mount_shared_sections)
        root.addWidget(self._tabs, 1)
        self._mount_shared_sections(0)

    def _mount_shared_sections(self, index: int) -> None:
        """Move the single shared editors into the active coordinate page."""
        if index < 0 or index >= len(axis_options.AXES):
            return
        axis = axis_options.AXES[index]
        host = self._coordinate_shared_hosts.get(axis)
        if host is None:
            return
        host.addWidget(self._shared_properties_section, 0)
        host.addWidget(self._shared_kwargs_section, 1)
        self._shared_properties_section.show()
        self._shared_kwargs_section.show()

    def _build_shared_properties_frame(self) -> QWidget:
        """Build compact axis-wide properties mounted in the active XYZ page."""
        section = TitledCard(self, _("Properties"), "axisPropertiesCard")
        raw_layout = section.card.layout()
        if not isinstance(raw_layout, QVBoxLayout):
            raise RuntimeError("TitledCard did not create a vertical box layout")

        self._axis_label_edit = QLineEdit(section.card)
        self._projection_combo = QComboBox(section.card)
        for label, value in (
            ("rectilinear", "rectilinear"),
            ("polar", "polar"),
            ("3d", "3d"),
        ):
            self._projection_combo.addItem(label, value)
        self._hide_axis_check = QCheckBox(_("Hide"), section.card)
        self._font_scale = FontScaleControl(section.card)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def add_field(row: int, column: int, caption: str, widget: QWidget) -> None:
            grid.addWidget(QLabel(caption, section.card), row, column)
            grid.addWidget(widget, row, column + 1)

        add_field(0, 0, _("Title"), self._axis_label_edit)
        add_field(0, 2, _("Font"), self._font_scale)
        add_field(1, 0, _("Proj."), self._projection_combo)
        add_field(1, 2, _("Visible"), self._hide_axis_check)

        span = self._build_span_row(section.card)
        self._row_span_spin.setMaximumWidth(72)
        self._col_span_spin.setMaximumWidth(72)
        grid.addWidget(QLabel(_("Span"), section.card), 2, 0)
        grid.addWidget(span, 2, 1, 1, 3)

        grid.addWidget(QLabel(_("Symlog"), section.card), 3, 0)
        grid.addWidget(self._linthresh_spin, 3, 1, 1, 3)

        spine_row = QWidget(section.card)
        spine_layout = QHBoxLayout(spine_row)
        stdSizeAndlayout(spine_layout)
        names = {"top": _("Top"), "right": _("Right"), "bottom": _("Bottom"), "left": _("Left")}
        short = {"top": "T", "right": "R", "bottom": "B", "left": "L"}
        for name, check in self._spine_checks().items():
            check.setText(short[name])
            check.setToolTip(names[name])
            spine_layout.addWidget(check)
        spine_layout.addStretch(1)
        grid.addWidget(QLabel(_("Spines"), section.card), 4, 0)
        grid.addWidget(spine_row, 4, 1, 1, 3)
        raw_layout.addLayout(grid)
        return section

    def _build_coordinate_controls(self) -> None:
        self._coordinate_labels = {}
        self._share_checks = {}
        self._scale_combos = {}
        self._scale_base_spins = {}
        self._invert_checks = {}
        self._tick_length_spins = {}
        self._tick_rotation_spins = {}
        self._grid_combos = {}
        self._tick_combos = {}
        for axis in axis_options.AXES:
            label = QLineEdit(self)
            share = QCheckBox(_("Share {axis}").format(axis=axis.upper()), self)
            scale = QComboBox(self)
            for value in AXIS_SCALES:
                scale.addItem(value, value)
            scale.currentIndexChanged.connect(self._update_scale_control_state)
            base = QDoubleSpinBox(self)
            base.setRange(1.1, 1000.0); base.setDecimals(2); base.setValue(10.0)
            invert = QCheckBox(_("Invert direction"), self)
            tick_length = QDoubleSpinBox(self)
            tick_length.setRange(0.0, 50.0)
            tick_length.setDecimals(1)
            tick_length.setSingleStep(0.5)
            tick_rotation = QDoubleSpinBox(self)
            tick_rotation.setRange(-180.0, 180.0)
            tick_rotation.setDecimals(0)
            tick_rotation.setSingleStep(15.0)
            self._coordinate_labels[axis] = label
            self._share_checks[axis] = share
            self._scale_combos[axis] = scale
            self._scale_base_spins[axis] = base
            self._invert_checks[axis] = invert
            self._tick_length_spins[axis] = tick_length
            self._tick_rotation_spins[axis] = tick_rotation
            for which in axis_options.WHICH:
                grid = QComboBox(self); tick = QComboBox(self)
                for value, text in axis_options.GRID_CHOICES: grid.addItem(_(text), value)
                for value, text in axis_options.TICK_CHOICES: tick.addItem(_(text), value)
                self._grid_combos[(axis, which)] = grid
                self._tick_combos[(axis, which)] = tick
            for edge in ("min", "max"):
                spin = QDoubleSpinBox(self)
                spin.setRange(-1.0e15, 1.0e15); spin.setDecimals(6)
                auto = QCheckBox(_("Auto"), self); auto.setChecked(True)
                auto.toggled.connect(lambda checked, target=spin: target.setDisabled(checked))
                self._limit_spins[(axis, edge)] = spin
                self._limit_auto_checks[(axis, edge)] = auto
        self._x_label_edit = self._coordinate_labels["x"]
        self._y_label_edit = self._coordinate_labels["y"]
        self._z_label_edit = self._coordinate_labels["z"]
        self._sharex_check, self._sharey_check, self._sharez_check = (self._share_checks[a] for a in "xyz")
        self._x_scale_combo, self._y_scale_combo, self._z_scale_combo = (self._scale_combos[a] for a in "xyz")
        self._x_scale_base_spin, self._y_scale_base_spin, self._z_scale_base_spin = (self._scale_base_spins[a] for a in "xyz")
        self._invert_x_check, self._invert_y_check, self._invert_z_check = (self._invert_checks[a] for a in "xyz")
        self._linthresh_spin = QDoubleSpinBox(self)
        self._linthresh_spin.setRange(1e-9, 1e9)
        self._linthresh_spin.setDecimals(6)
        self._linthresh_spin.setValue(1.0)
        # Legacy aliases retained for code that still references the X controls.
        self._tick_length_spin = self._tick_length_spins["x"]
        self._x_tick_rotation_spin = self._tick_rotation_spins["x"]
        self._hide_spine_top_check = QCheckBox(_("Top"), self); self._hide_spine_right_check = QCheckBox(_("Right"), self)
        self._hide_spine_bottom_check = QCheckBox(_("Bottom"), self); self._hide_spine_left_check = QCheckBox(_("Left"), self)

    def _build_coordinate_tab(self, axis: str) -> QWidget:
        """Build one XYZ page with coordinate controls and a shared-editor host."""
        page = QWidget(self._tabs)
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        section = CardFrame(page, f"axis{axis.upper()}Card")
        raw_layout = section.layout()
        if not isinstance(raw_layout, QVBoxLayout):
            raise RuntimeError("CardFrame did not create a vertical box layout")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def add_field(row: int, column: int, caption: str, widget: QWidget) -> None:
            grid.addWidget(QLabel(caption, section), row, column)
            grid.addWidget(widget, row, column + 1)

        grid.addWidget(QLabel(_("Label"), section), 0, 0)
        grid.addWidget(self._coordinate_labels[axis], 0, 1, 1, 4)
        add_field(1, 0, _("Scale"), self._scale_combos[axis])
        add_field(1, 2, _("Base"), self._scale_base_spins[axis])
        add_field(2, 0, _("Tick length"), self._tick_length_spins[axis])
        add_field(2, 2, _("Rotation"), self._tick_rotation_spins[axis])

        for row_index, (edge, caption) in enumerate(
            (("min", _("Range min")), ("max", _("Range max"))), start=3
        ):
            spin = self._limit_spins[(axis, edge)]
            spin.setMaximumWidth(110)
            range_row = QWidget(section)
            range_layout = QHBoxLayout(range_row)
            stdSizeAndlayout(range_layout)
            range_layout.addWidget(spin)
            range_layout.addWidget(self._limit_auto_checks[(axis, edge)])
            range_layout.addStretch(1)
            grid.addWidget(QLabel(caption, section), row_index, 0)
            grid.addWidget(range_row, row_index, 1, 1, 4)

        grid.addWidget(QLabel(_("Grid"), section), 5, 0)
        grid.addWidget(QLabel(_("Major"), section), 5, 1)
        grid.addWidget(self._grid_combos[(axis, "major")], 5, 2)
        grid.addWidget(QLabel(_("Minor"), section), 5, 3)
        grid.addWidget(self._grid_combos[(axis, "minor")], 5, 4)
        grid.addWidget(QLabel(_("Ticks"), section), 6, 0)
        grid.addWidget(QLabel(_("Major"), section), 6, 1)
        grid.addWidget(self._tick_combos[(axis, "major")], 6, 2)
        grid.addWidget(QLabel(_("Minor"), section), 6, 3)
        grid.addWidget(self._tick_combos[(axis, "minor")], 6, 4)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(4, 1)

        share = self._share_checks[axis]
        share.setText(_("Share {axis}").format(axis=axis.upper()))
        invert = self._invert_checks[axis]
        invert.setText(_("Invert direction"))
        checks = QWidget(section)
        checks_layout = QHBoxLayout(checks)
        stdSizeAndlayout(checks_layout)
        checks_layout.addWidget(share)
        checks_layout.addWidget(invert)
        checks_layout.addStretch(1)
        grid.addWidget(checks, 7, 0, 1, 5)

        raw_layout.addLayout(grid)
        page_layout.addWidget(section, 0)
        shared_host = QVBoxLayout()
        shared_host.setContentsMargins(0, 0, 0, 0)
        shared_host.setSpacing(10)
        page_layout.addLayout(shared_host, 1)
        self._coordinate_shared_hosts[axis] = shared_host
        return page

    def _build_axis_selector_section(self) -> QWidget:
        """Create axis selector, axis actions and renderer display."""
        section = CardFrame(self, "axisSelectorCard")
        raw_layout = section.layout()
        if not isinstance(raw_layout, QVBoxLayout):
            raise RuntimeError("CardFrame did not create a vertical box layout")
        layout = raw_layout

        self._axis_combo = QComboBox(section)
        self._configure_combo_width(self._axis_combo, minimum_contents_length=24)
        # The tooltip that keeps the elided name readable is wired by
        # configure_combo_width, so it is not repeated here.
        self._axis_combo.currentIndexChanged.connect(self._on_axis_combo_changed)
        layout.addWidget(self._axis_combo)

        action_row = QWidget(section)
        action_layout = QHBoxLayout(action_row)
        stdSizeAndlayout(action_layout)
        
        self._btn_move_up = create_action_button(
                                parent=action_row,
                                action_id="up",
                                action=self._on_move_up_clicked,
                                layout=action_layout,
                            )
        self._btn_move_down = create_action_button(
                                  parent=action_row,
                                  action_id="down",
                                  action=self._on_move_down_clicked,
                                  layout=action_layout,
                              )
        self._btn_delete = create_action_button(
                               parent=action_row,
                               action_id="delete",
                               action=self._on_delete_clicked,
                               layout=action_layout,
                           )
        # No Apply button: every field applies itself a short moment after
        # it changes (see _connect_auto_apply). Up / Down / Delete stay -
        # they reorder and remove axes, not commit the form.
        action_layout.addStretch(1)
        layout.addWidget(action_row)

        self._renderer_value = QLabel("", section)
        self._renderer_value.setObjectName("axisRendererValue")
        self._renderer_value.setProperty("rendererLabel", True)
        self._renderer_value.setContentsMargins(0, 0, 0, 0)
        self._renderer_value.setWordWrap(True)
        self._renderer_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._renderer_value.hide()
        return section

    def _build_option_page(self, object_name: str) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        """Return an option page: its scroll area, its card, and the layout.

        The axis properties panel itself must not be hosted by an outer
        QScrollArea. Each option page carries its own instead, so the
        QTabWidget and the top-level AxisPropertiesWidget expand to their
        parent rather than advertising the tallest page's sizeHint - and a
        page longer than the panel scrolls on its own without dragging the
        selector card above it out of view.
        """
        scroll = QScrollArea(self._tabs)
        scroll.setObjectName("axisOptionsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        section = CardFrame(scroll, object_name)
        section.setMinimumHeight(0)
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        raw_layout = section.layout()
        if not isinstance(raw_layout, QVBoxLayout):
            raise RuntimeError("CardFrame did not create a vertical box layout")
        layout = raw_layout
        scroll.setWidget(section)
        return scroll, section, layout

    # Matches the 1..6 range the Figure panel offers for rows/cols, so a span
    # can never claim more of the grid than the grid itself can have.
    MAX_GRID_SPAN: int = 6

    def _build_span_row(self, section: QWidget) -> QWidget:
        """Create the row/column span controls for non-uniform layouts.

        An axis normally fills one grid cell at its position. Raising either
        spin box lets it fill a rectangle of cells instead - e.g. a wide axis
        across the top row of a 2x2 grid with two narrower axes below it.
        Overlapping another axis's cells falls back to a plain compact grid
        on render rather than drawing on top of it.

        Returned as a row widget rather than as a form of its own: it joins
        the Position form above it, whose label column it has to share or
        else "Span" sits a few pixels off from "Sharing".
        """
        self._row_span_spin = QSpinBox(section)
        self._col_span_spin = QSpinBox(section)
        for spin in (self._row_span_spin, self._col_span_spin):
            stdSizeAndlayout(spin)
            spin.setRange(1, self.MAX_GRID_SPAN)
            spin.setValue(1)
            spin.setToolTip(
                _(
                    "Grid cells this axis spans from its position, so it can "
                    "take up more room than the other axes in the figure."
                )
            )

        span_row = QWidget(section)
        span_layout = QHBoxLayout(span_row)
        span_layout.setContentsMargins(0, 0, 0, 0)
        span_layout.setSpacing(8)
        span_layout.addWidget(QLabel(_("Rows"), span_row))
        span_layout.addWidget(self._row_span_spin, 1)
        span_layout.addSpacing(8)
        span_layout.addWidget(QLabel(_("Cols"), span_row))
        span_layout.addWidget(self._col_span_spin, 1)
        return span_row

    def _update_limit_control_state(self) -> None:
        for key, spin in self._limit_spins.items():
            spin.setEnabled(not self._limit_auto_checks[key].isChecked())

    def _update_scale_control_state(self) -> None:
        """Enable only the scale parameters the selected scales actually use."""
        x_scale, y_scale, z_scale = self._selected_scales()

        self._x_scale_base_spin.setEnabled(x_scale in {"log", "symlog"})
        self._y_scale_base_spin.setEnabled(y_scale in {"log", "symlog"})
        self._z_scale_base_spin.setEnabled(z_scale in {"log", "symlog"})
        self._linthresh_spin.setEnabled(
            "symlog" in {x_scale, y_scale, z_scale}
        )

    def _selected_scales(self) -> tuple[str, str, str]:
        """Return the chosen x, y and z scales, defaulting to linear."""
        return tuple(  # type: ignore[return-value]
            str(combo.currentData() or "linear")
            for combo in (
                self._x_scale_combo,
                self._y_scale_combo,
                self._z_scale_combo,
            )
        )

    # ------------------------------------------------------------------
    # Scale / tick / grid / spine options
    # ------------------------------------------------------------------
    def _extended_option_widgets(self) -> tuple[QWidget, ...]:
        """Return the controls added by the scale and decoration sections."""
        return (
            self._row_span_spin,
            self._col_span_spin,
            self._x_scale_combo,
            self._y_scale_combo,
            self._z_scale_combo,
            self._x_scale_base_spin,
            self._y_scale_base_spin,
            self._z_scale_base_spin,
            self._linthresh_spin,
            self._invert_x_check,
            self._invert_y_check,
            self._invert_z_check,
            *self._tick_length_spins.values(),
            *self._tick_rotation_spins.values(),
            self._hide_spine_top_check,
            self._hide_spine_right_check,
            self._hide_spine_bottom_check,
            self._hide_spine_left_check,
            *self._limit_auto_checks.values(),
            *self._grid_combos.values(),
            *self._tick_combos.values(),
            *self._limit_spins.values(),
        )

    @staticmethod
    def _select_combo_value(combo: QComboBox, value: object, fallback: str) -> None:
        """Select a combo entry by data, falling back when it is unknown."""
        index = combo.findData(str(value if value is not None else fallback))
        combo.setCurrentIndex(index if index >= 0 else max(0, combo.findData(fallback)))

    @staticmethod
    def _float_option(options: dict[str, Any], key: str, default: float) -> float:
        """Read a float option, keeping the default when it is unusable."""
        try:
            return float(options.get(key, default))
        except (TypeError, ValueError):
            applogger.warning("Invalid axis option %s=%r", key, options.get(key))
            return default

    def _int_option(self, options: dict[str, Any], key: str, default: int) -> int:
        """Read a clamped int option, keeping the default when it is unusable."""
        try:
            value = int(options.get(key, default))
        except (TypeError, ValueError):
            applogger.warning("Invalid axis option %s=%r", key, options.get(key))
            return default
        return max(1, min(value, self.MAX_GRID_SPAN))

    def _load_extended_axis_options(self, options: dict[str, Any]) -> None:
        """Populate the scale, tick, grid and spine controls from options."""
        self._row_span_spin.setValue(self._int_option(options, "row_span", 1))
        self._col_span_spin.setValue(self._int_option(options, "col_span", 1))
        self._select_combo_value(self._x_scale_combo, options.get("x_scale"), "linear")
        self._select_combo_value(self._y_scale_combo, options.get("y_scale"), "linear")
        self._select_combo_value(self._z_scale_combo, options.get("z_scale"), "linear")
        self._x_scale_base_spin.setValue(self._float_option(options, "x_scale_base", 10.0))
        self._y_scale_base_spin.setValue(self._float_option(options, "y_scale_base", 10.0))
        self._z_scale_base_spin.setValue(self._float_option(options, "z_scale_base", 10.0))
        self._linthresh_spin.setValue(self._float_option(options, "x_linthresh", 1.0))

        self._invert_x_check.setChecked(bool(options.get("invert_x", False)))
        self._invert_y_check.setChecked(bool(options.get("invert_y", False)))
        self._invert_z_check.setChecked(bool(options.get("invert_z", False)))

        # Through axis_options so a figure saved before these existed opens
        # with the settings it actually had, rather than with four Autos.
        for key, combo in self._grid_combos.items():
            self._select_combo_value(
                combo, axis_options.grid_setting(options, *key), axis_options.AUTO
            )
        for key, combo in self._tick_combos.items():
            self._select_combo_value(
                combo, axis_options.tick_setting(options, *key), axis_options.AUTO
            )

        for key, spin in self._limit_spins.items():
            axis, edge = key
            value = axis_options.manual_limit(options, axis, edge)
            automatic = value is None
            self._limit_auto_checks[key].setChecked(automatic)
            spin.setValue(spin.minimum() if automatic else value)
            spin.setEnabled(not automatic)
        self._update_limit_control_state()

        legacy_tick_length = self._float_option(options, "tick_length", 0.0)
        for axis in axis_options.AXES:
            self._tick_length_spins[axis].setValue(
                self._float_option(options, f"{axis}_tick_length", legacy_tick_length)
            )
            self._tick_rotation_spins[axis].setValue(
                self._float_option(options, f"{axis}_tick_rotation", 0.0)
            )

        for name, check in self._spine_checks().items():
            check.setChecked(bool(options.get(f"hide_spine_{name}", False)))

    def _clear_extended_axis_options(self) -> None:
        """Reset the scale, tick, grid and spine controls to their defaults."""
        self._load_extended_axis_options({})
        self._update_scale_control_state()

    def _spine_checks(self) -> dict[str, QCheckBox]:
        """Return the spine visibility checkboxes keyed by spine name."""
        return {
            "top": self._hide_spine_top_check,
            "right": self._hide_spine_right_check,
            "bottom": self._hide_spine_bottom_check,
            "left": self._hide_spine_left_check,
        }

    def _extended_axis_options_payload(self) -> dict[str, Any]:
        """Return the scale, tick, grid and spine part of the axis payload.

        Optional numeric settings are emitted as None when they are zero or the
        control is disabled, so that "not configured" stays distinguishable from
        "configured to zero" once the payload reaches the renderer.
        """
        x_scale, y_scale, z_scale = self._selected_scales()
        tick_lengths = {
            axis: float(spin.value())
            for axis, spin in self._tick_length_spins.items()
        }

        payload: dict[str, Any] = {
            "row_span": int(self._row_span_spin.value()),
            "col_span": int(self._col_span_spin.value()),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "z_scale": z_scale,
            "x_scale_base": (
                float(self._x_scale_base_spin.value())
                if x_scale in {"log", "symlog"}
                else None
            ),
            "y_scale_base": (
                float(self._y_scale_base_spin.value())
                if y_scale in {"log", "symlog"}
                else None
            ),
            "z_scale_base": (
                float(self._z_scale_base_spin.value())
                if z_scale in {"log", "symlog"}
                else None
            ),
            "x_linthresh": (
                float(self._linthresh_spin.value())
                if "symlog" in {x_scale, y_scale, z_scale}
                else None
            ),
            "invert_x": bool(self._invert_x_check.isChecked()),
            "invert_y": bool(self._invert_y_check.isChecked()),
            "invert_z": bool(self._invert_z_check.isChecked()),
            # Legacy global length follows X for older renderers/projects.
            "tick_length": tick_lengths["x"] if tick_lengths["x"] > 0.0 else None,
            "x_tick_rotation": float(self._tick_rotation_spins["x"].value()),
        }
        # One threshold control for all three: symlog's linear region is a
        # property of the data's units, and an axis whose scale is not symlog
        # ignores the key entirely.
        payload["y_linthresh"] = payload["x_linthresh"]
        payload["z_linthresh"] = payload["x_linthresh"]
        for axis in axis_options.AXES:
            length = tick_lengths[axis]
            payload[f"{axis}_tick_length"] = length if length > 0.0 else None
            payload[f"{axis}_tick_rotation"] = float(
                self._tick_rotation_spins[axis].value()
            )

        for (axis, which), combo in self._grid_combos.items():
            payload[axis_options.grid_key(axis, which)] = str(
                combo.currentData() or axis_options.AUTO
            )
        for (axis, which), combo in self._tick_combos.items():
            payload[axis_options.tick_key(axis, which)] = str(
                combo.currentData() or axis_options.AUTO
            )

        # The legacy keys are written as the four settings imply, so a figure
        # edited here still renders in a build that predates them - and so
        # nothing downstream reads a stale "grid: True" that the user has
        # since turned off.
        payload["grid"] = False
        payload["grid_which"] = "major"
        payload["grid_axis"] = "both"
        payload["minor_ticks"] = any(
            str(self._tick_combos[(axis, "minor")].currentData() or axis_options.AUTO)
            not in (axis_options.AUTO, axis_options.OFF)
            for axis in axis_options.AXES
        )

        payload["limits_mode"] = getattr(axis_options, "LIMITS_MANUAL", "manual")
        for key, spin in self._limit_spins.items():
            axis, edge = key
            payload[axis_options.limit_key(axis, edge)] = (
                None if self._limit_auto_checks[key].isChecked() else float(spin.value())
            )

        for name, check in self._spine_checks().items():
            payload[f"hide_spine_{name}"] = bool(check.isChecked())

        return payload

    def _build_kwargs_section(self) -> QWidget:
        """Create the host section for DictEditorPanel."""
        section = TitledCard(self, _("Kwargs"), "axisKwargsCard")
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        raw_layout = section.card.layout()
        if not isinstance(raw_layout, QVBoxLayout):
            raise RuntimeError("TitledCard did not create a vertical box layout")
        layout = raw_layout
        # The reset button is not built here: it belongs on the editor's own
        # search row (see rebuild_kwargs_editor), and the editor is rebuilt
        # from scratch for every axis, so the button is too.
        self._kwargs_host = QFrame(section)
        self._kwargs_host.setObjectName("axisKwargsPanel")
        self._kwargs_host.setFrameShape(QFrame.Shape.NoFrame)
        self._kwargs_host.setMinimumHeight(self.KWARGS_PANEL_MIN_HEIGHT)
        self._kwargs_host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._kwargs_layout = QVBoxLayout(self._kwargs_host)
        self._kwargs_layout.setContentsMargins(0, 0, 0, 0)
        self._kwargs_layout.setSpacing(0)

        placeholder = QLabel(
            _("Additional kwargs editor can be inserted here by the host."),
            self._kwargs_host,
        )
        placeholder.setWordWrap(True)
        placeholder.setContentsMargins(0, 0, 0, 0)
        self._kwargs_layout.addWidget(placeholder, 0)
        layout.addWidget(self._kwargs_host, 1)
        return section

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # set_connected_figure is BaseProperties': attach the four attributes,
    # then call _reload_from_descriptor() below, which already ends by
    # enabling or disabling this editor's own controls itself.

    def clear_connected_figure(self) -> None:
        """Reset UI and disable editing."""
        self.setUpdatesEnabled(False)
        try:
            self._current_axis_id = None
            self._axis_map.clear()
            with QSignalBlocker(self._axis_combo):
                self._axis_combo.clear()
            self._clear_axis_fields()
            super().clear_connected_figure()
        finally:
            self.setUpdatesEnabled(True)

    def current_axis_id(self) -> int | None:
        """Return the currently selected axis id."""
        return self._current_axis_id

    def set_kwargs_widget(self, widget: QWidget) -> None:
        """Replace kwargs host content."""
        self._replace_layout_widget(self._kwargs_layout, widget)

    def _abandon_kwargs_editor(self) -> None:
        """Drop the live editor before showing a note widget in its place."""
        self._kwargs_build_signature = None
        if self._kwargs_editor is not None:
            self._kwargs_editor.commit_pending_edits()
            self._kwargs_editor = None

    def rebuild_kwargs_editor(self, axis_id: int | None) -> None:
        """Rebuild the renderer kwargs editor for ``axis_id``.

        A no-op when the axis, its renderer and its resolved kwargs values
        are all identical to the last build. ``_reload_property_widgets``
        calls this on every series reorder, axis move and layout-preset
        apply in ``main_window.py`` even though none of those touch the
        selected axis's own renderer or kwargs, and a full rebuild replaces
        the whole ``DictEditorPanel`` tree - one row per kwarg - for no
        visible change. The comparison is on the *resolved values*
        (``get_kwargs(current_options)``), not on object identity or on
        ``axis_kwargs`` alone, so a real change is still picked up however
        it happened - Undo included - even with the axis_id and renderer
        unchanged: any option that feeds a kwarg's resolution changes what
        this computes, not just the axis_kwargs sub-dict (see
        ``BaseAxisRenderer._sources``).
        """
        axis_desc = self._axis_map.get(int(axis_id)) if axis_id is not None else None
        if axis_desc is None:
            self._abandon_kwargs_editor()
            self.set_kwargs_widget(self._build_note_widget(_("Select an axis to edit kwargs.")))
            return

        renderer_key = self._axis_renderer(axis_desc, self._axis_options(axis_desc))
        if not renderer_key:
            self._abandon_kwargs_editor()
            self.set_kwargs_widget(self._build_note_widget(_("Renderer not found.")))
            return

        renderer: RendererConfig | None = get_renderer(renderer_key)
        if renderer is None:
            self._abandon_kwargs_editor()
            self.set_kwargs_widget(self._build_note_widget(_("Renderer not found.")))
            return

        try:
            renderer_class = import_class_from_file(renderer)
            if renderer_class is None:
                raise RuntimeError("Renderer class not found")
            renderer_instance = renderer_class()
        except Exception:
            applogger.exception("Failed to load renderer kwargs schema")
            self._abandon_kwargs_editor()
            self.set_kwargs_widget(
                self._build_note_widget(_("Failed to load renderer kwargs schema."))
            )
            return

        schema = getattr(renderer_instance, "Kwargs", None)
        if not schema:
            self._abandon_kwargs_editor()
            self.set_kwargs_widget(
                self._build_note_widget(_("No kwargs available for this renderer."))
            )
            return
        if self._repo is None:
            return

        axis_id_int = int(axis_id) if axis_id is not None else 0
        current_options: dict[str, Any] = _plain_options(
            self._repo.get_axis_options(axis_id_int) or {}
        )
        values = renderer_instance.get_kwargs(current_options)

        signature = (axis_id_int, renderer_key, tuple(sorted(values.items())))
        if self._kwargs_editor is not None and signature == self._kwargs_build_signature:
            return

        if self._kwargs_editor is not None:
            self._kwargs_editor.commit_pending_edits()
            self._kwargs_editor = None

        editor = DictEditorPanel(schema, self)
        self._configure_kwargs_editor(editor)
        if values:
            editor.set_values(values)
            self._configure_kwargs_editor(editor)
        # Auto-apply dropped the Apply button (see _connect_auto_apply); this
        # editor is rebuilt from scratch per axis/renderer rather than wired
        # up front like the static form controls, so it has to queue its own
        # apply here or a kwargs edit is never persisted until some other
        # field happens to be touched too.
        editor.valuesChanged.connect(self._queue_auto_apply)

        # DictEditorPanel.reset_to_defaults() already existed - every kwarg's
        # schema default *is* the "leave this to the style sheet" sentinel
        # (see kwarg_spec.DEFAULT) - it just had no button anywhere calling
        # it, so the only way back to "stop overriding this" was retyping
        # "default" into each row by hand. Resets the live editor only, the
        # same as any other edit here: auto-apply persists it a moment later.
        #
        # Built per editor, and on the editor's own search row: one strip of
        # chrome over the tree instead of two, and no button left behind on
        # a tab showing "select an axis" with nothing to reset. It is owned
        # by the editor from here on, so it dies with it on the next rebuild
        # rather than being a stale pointer into a deleted panel.
        reset_button = mark_icon_only(
            create_action_button(
                parent=editor,
                action_id="reset_kwargs_to_defaults",
                action=self._reset_kwargs_to_defaults,
                layout=None,
            )
        )
        if reset_button is not None:
            self._btn_reset_kwargs = reset_button
            editor.add_search_row_widget(reset_button)

        self._kwargs_editor = editor
        self._kwargs_build_signature = signature
        self.set_kwargs_widget(editor)

    def clean_kwargs(self) -> dict[str, Any]:
        """Commit and return non-empty kwargs values."""
        if self._kwargs_editor is None:
            return {}
        self._kwargs_editor.commit_pending_edits()
        values = cast(dict[str, Any], self._kwargs_editor.get_values())
        return {key: value for key, value in values.items() if value != ""}

    def _reset_kwargs_to_defaults(self) -> None:
        """Reset every row of the kwargs editor to "leave it to the style".

        Only the live editor changes here - exactly like typing a new value
        into any other row, this is not applied to the axis until Apply is
        pressed, so a reset can still be backed out of.
        """
        if self._kwargs_editor is None:
            return
        self._kwargs_editor.reset_to_defaults()

    def _build_note_widget(self, text: str) -> QWidget:
        note = QLabel(text, self)
        note.setWordWrap(True)
        note.setContentsMargins(0, 0, 0, 0)
        return note

    def _configure_kwargs_editor(self, editor: QWidget) -> QWidget:
        """Make renderer kwargs content expand without forcing panel growth."""
        if isinstance(editor, QFrame):
            editor.setFrameShape(QFrame.Shape.NoFrame)
            editor.setLineWidth(0)
            editor.setMidLineWidth(0)

        set_resizable = getattr(editor, "setWidgetResizable", None)
        if callable(set_resizable):
            set_resizable(True)
        horizontal_policy = getattr(editor, "setHorizontalScrollBarPolicy", None)
        if callable(horizontal_policy):
            horizontal_policy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        vertical_policy = getattr(editor, "setVerticalScrollBarPolicy", None)
        if callable(vertical_policy):
            vertical_policy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        editor.setMinimumHeight(min(editor.sizeHint().height(), self.KWARGS_PANEL_MIN_HEIGHT))
        editor.setMaximumHeight(MAX_QT_HEIGHT)
        editor.setContentsMargins(0, 0, 0, 0)
        editor.updateGeometry()
        return editor

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _set_enabled_state(self, enabled: bool) -> None:
        """Enable or disable interactive controls."""
        widgets: tuple[QWidget, ...] = (
            self._axis_combo,
            self._btn_move_up,
            self._btn_move_down,
            self._btn_delete,
            self._axis_label_edit,
            self._x_label_edit,
            self._y_label_edit,
            self._z_label_edit,
            self._projection_combo,
            self._sharex_check,
            self._sharey_check,
            self._sharez_check,
            self._hide_axis_check,
            self._renderer_value,
            self._tabs,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        self._update_axis_action_buttons()

    def _reload_from_descriptor(self) -> None:
        """Rebuild the axis selector from the current descriptor."""
        if self._repo is None or self._figure_id is None:
            self.clear_connected_figure()
            return

        self.setUpdatesEnabled(False)
        try:
            desc = self._repo.load_figure_descriptor(self._figure_id)
            previous_axis_id = self._current_axis_id
            self._axis_map.clear()
            # Not nulled here: every caller of set_connected_figure/
            # reload_controls follows it with rebuild_kwargs_editor, and
            # that method now owns the editor's lifecycle - it commits and
            # replaces it when the axis/renderer/kwargs actually changed,
            # and leaves it alone (skipping a full QTreeWidget rebuild)
            # when they did not. Nulling it here unconditionally would
            # defeat that: rebuild_kwargs_editor only skips when an editor
            # is still there to reuse.

            with QSignalBlocker(self._axis_combo):
                self._axis_combo.clear()
                if desc is None:
                    applogger.warning(
                        "No figure descriptor found for figure_id=%s. Axis editing disabled.",
                        self._figure_id,
                    )
                    return
                for axis_desc in self._axes_from_descriptor(desc):
                    axis_id = int(axis_desc.id)
                    self._axis_map[axis_id] = axis_desc
                    self._axis_combo.addItem(
                        self._axis_display_label(axis_desc, axis_id),
                        axis_id,
                    )
        finally:
            self.setUpdatesEnabled(True)

        if self._axis_combo.count() == 0:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            self._set_enabled_state(False)
            return
        self._set_enabled_state(True)
        self._select_axis(previous_axis_id)

    def _axes_from_descriptor(self, desc: Any) -> list[AxisDescriptorLike]:
        """Return axes from a validated descriptor."""
        axes = getattr(desc, "axes", None)
        if axes is None:
            applogger.error(
                "Figure descriptor id=%r has no axes list. Axis editing stopped.",
                getattr(desc, "id", self._figure_id),
            )
            return []
        return list(axes)

    def _select_axis(self, preferred_axis_id: int | None) -> None:
        """Select the previous axis when possible, otherwise select the first."""
        index_to_select = 0
        if preferred_axis_id is not None:
            for index in range(self._axis_combo.count()):
                axis_id_data = self._axis_combo.itemData(index)
                if axis_id_data is not None and int(axis_id_data) == preferred_axis_id:
                    index_to_select = index
                    break
        self._axis_combo.setCurrentIndex(index_to_select)
        self._on_axis_combo_changed(index_to_select)

    def _update_axis_action_buttons(self) -> None:
        """Enable axis structure buttons for the current combo row."""
        has_axes = self._axis_combo.count() > 0
        current_index = self._axis_combo.currentIndex()
        is_enabled = self._axis_combo.isEnabled() and has_axes
        self._btn_move_up.setEnabled(is_enabled and current_index > 0)
        self._btn_move_down.setEnabled(
            is_enabled and current_index < self._axis_combo.count() - 1
        )
        self._btn_delete.setEnabled(is_enabled)

    # ------------------------------------------------------------------
    # Descriptor binding
    # ------------------------------------------------------------------
    def _axis_display_label(
        self,
        axis_desc: AxisDescriptorLike,
        axis_id: int,
    ) -> str:
        """Build combo text for one axis."""
        options = self._axis_options(axis_desc)
        title = str(
            getattr(axis_desc, "title", None)
            or options.get("label")
            or f"Axis {axis_id}"
        ).strip()
        renderer = self._axis_renderer(axis_desc, options)
        return f"{title} [{renderer}]" if renderer else title

    def _load_axis_descriptor(self, axis_desc: AxisDescriptorLike | None) -> None:
        """Load one axis descriptor into the form."""
        if axis_desc is None:
            self._clear_axis_fields()
            return

        options = self._axis_options(axis_desc)
        projection = str(options.get("projection") or "rectilinear").strip()
        renderer = self._axis_renderer(axis_desc, options)
        self._current_axis_id = int(axis_desc.id)

        with self._form_signal_blocker():
            self._axis_label_edit.setText(
                str(
                    options.get("label")
                    or options.get("title")
                    or getattr(axis_desc, "title", "")
                    or ""
                ).strip()
            )
            self._x_label_edit.setText(str(options.get("x_label", "")).strip())
            self._y_label_edit.setText(str(options.get("y_label", "")).strip())
            self._z_label_edit.setText(str(options.get("z_label", "")).strip())
            self._projection_combo.setCurrentIndex(
                max(0, self._projection_combo.findData(projection))
            )
            self._sharex_check.setChecked(bool(options.get("sharex", False)))
            self._sharey_check.setChecked(bool(options.get("sharey", False)))
            self._sharez_check.setChecked(bool(options.get("sharez", False)))
            self._hide_axis_check.setChecked(
                bool(options.get("hide_axis", options.get("hidden", False)))
            )
            self._font_scale.set_factor(options.get("font_scale", 1.0))
            self._load_extended_axis_options(options)

        # Translated for reading; the raw chart_type is what the signal
        # carries, since every listener looks the renderer up by that name.
        self._renderer_value.setText(tr(renderer))
        self.renderer_changed.emit(renderer)

    class _FormSignalBlocker:
        """Small context manager that owns QSignalBlocker instances."""

        def __init__(self, widgets: tuple[QWidget, ...]) -> None:
            self._widgets = widgets
            self._blockers: list[QSignalBlocker] = []

        def __enter__(self) -> None:
            self._blockers = [QSignalBlocker(widget) for widget in self._widgets]

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            self._blockers.clear()

    def _form_signal_blocker(self) -> _FormSignalBlocker:
        return self._FormSignalBlocker(
            (
                self._axis_label_edit,
                self._x_label_edit,
                self._y_label_edit,
                self._z_label_edit,
                self._projection_combo,
                self._sharex_check,
                self._sharey_check,
                self._sharez_check,
                self._hide_axis_check,
            )
            + self._extended_option_widgets()
        )

    def _clear_axis_fields(self) -> None:
        """Clear axis-specific form fields."""
        self._current_axis_id = None
        with self._form_signal_blocker():
            self._axis_label_edit.clear()
            self._x_label_edit.clear()
            self._y_label_edit.clear()
            self._z_label_edit.clear()
            self._projection_combo.setCurrentIndex(0)
            self._sharex_check.setChecked(False)
            self._sharey_check.setChecked(False)
            self._sharez_check.setChecked(False)
            self._hide_axis_check.setChecked(False)
            self._font_scale.set_factor(1.0)
            self._clear_extended_axis_options()
        self._renderer_value.clear()
        self.renderer_changed.emit("")

    def _axis_options(self, axis_desc: AxisDescriptorLike) -> dict[str, Any]:
        """Return axis options as a mutable plain dictionary."""
        options = getattr(axis_desc, "options", None)
        if options is None:
            return {}
        if isinstance(options, dict):
            return dict(cast(dict[str, Any], options))
        applogger.error(
            "Axis id=%r has invalid options. Editing stopped.",
            getattr(axis_desc, "id", None),
        )
        return {}

    def _axis_renderer(
        self,
        axis_desc: AxisDescriptorLike,
        options: dict[str, Any],
    ) -> str:
        """Resolve the renderer name from options or descriptor chart_type."""
        renderer = (
            options.get("renderer")
            or options.get("renderer_name")
            or getattr(axis_desc, "name", "")
            or ""
        )
        return str(renderer).strip()

    # ------------------------------------------------------------------
    # UI events
    # ------------------------------------------------------------------
    def _on_axis_combo_changed(self, index: int) -> None:
        """Handle axis selector changes."""
        if index < 0:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            return
        axis_id_data = self._axis_combo.itemData(index)
        if axis_id_data is None:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            return

        axis_id = int(axis_id_data)
        axis_desc = self._axis_map.get(axis_id)
        if axis_desc is None:
            applogger.error(
                "Axis descriptor id=%r not found in axis map. Editing stopped.",
                axis_id,
            )
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            return

        self._load_axis_descriptor(axis_desc)
        self._update_axis_action_buttons()
        self.axis_selected.emit(axis_id)

    def _emit_axis_action(
        self,
        action: str,
        axis_id: int | None = None,
        index: int | None = None,
    ) -> None:
        """Emit a structural axis action for host-side persistence."""
        selected_axis_id = self._axis_combo.currentData() if axis_id is None else axis_id
        selected_index = self._axis_combo.currentIndex() if index is None else index
        payload: AxisPayload = {
            "action": action,
            "axis_id": selected_axis_id,
            "figure_id": self._figure_id,
            "index": selected_index,
        }
        self.axis_action_requested.emit(payload)

    def _swap_axis_rows(self, first_row: int, second_row: int) -> None:
        """Swap two combo rows while preserving user-data axis ids."""
        first_text = self._axis_combo.itemText(first_row)
        first_data = self._axis_combo.itemData(first_row)
        second_text = self._axis_combo.itemText(second_row)
        second_data = self._axis_combo.itemData(second_row)
        with QSignalBlocker(self._axis_combo):
            self._axis_combo.setItemText(first_row, second_text)
            self._axis_combo.setItemData(first_row, second_data)
            self._axis_combo.setItemText(second_row, first_text)
            self._axis_combo.setItemData(second_row, first_data)
            self._axis_combo.setCurrentIndex(second_row)
        self._on_axis_combo_changed(second_row)

    def _move_selected_axis(self, delta: int) -> None:
        """Move the selected axis row locally and notify the host."""
        current_row = self._axis_combo.currentIndex()
        target_row = current_row + delta
        if current_row < 0 or target_row < 0 or target_row >= self._axis_combo.count():
            return
        axis_id = self._axis_combo.currentData()
        if axis_id is None:
            return
        action = "move_up" if delta < 0 else "move_down"
        self._swap_axis_rows(current_row, target_row)
        self._emit_axis_action(action, axis_id=int(axis_id), index=target_row)

    def _delete_selected_axis(self) -> None:
        """Delete selected axis row locally and notify the host."""
        current_row = self._axis_combo.currentIndex()
        axis_id = self._axis_combo.currentData()
        if current_row < 0 or axis_id is None:
            return
        axis_id_int = int(axis_id)
        if not ask(self, "axis.confirm_delete", axis=self._axis_combo.itemText(current_row)):
            return
        self._emit_axis_action("delete", axis_id=axis_id_int, index=current_row)
        self._axis_map.pop(axis_id_int, None)
        with QSignalBlocker(self._axis_combo):
            self._axis_combo.removeItem(current_row)
            next_row = min(current_row, self._axis_combo.count() - 1)
            if next_row >= 0:
                self._axis_combo.setCurrentIndex(next_row)
        if self._axis_combo.count() == 0:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
        else:
            self._on_axis_combo_changed(self._axis_combo.currentIndex())

    def _on_move_up_clicked(self) -> None:
        self._move_selected_axis(-1)

    def _on_move_down_clicked(self) -> None:
        self._move_selected_axis(1)

    def _on_delete_clicked(self) -> None:
        self._delete_selected_axis()

    def _emit_axis_options_requested(self) -> None:
        """Emit the current axis options payload."""
        label_text = self._axis_label_edit.text().strip()
        payload: AxisPayload = {
            "axis_id": self._current_axis_id,
            "label": label_text,
            "title": label_text,
            "x_label": self._x_label_edit.text().strip(),
            "y_label": self._y_label_edit.text().strip(),
            "z_label": self._z_label_edit.text().strip(),
            "projection": str(self._projection_combo.currentData() or "rectilinear"),
            "sharex": bool(self._sharex_check.isChecked()),
            "sharey": bool(self._sharey_check.isChecked()),
            "sharez": bool(self._sharez_check.isChecked()),
            "hide_axis": bool(self._hide_axis_check.isChecked()),
            "font_scale": float(self._font_scale.factor()),
            "renderer": self._renderer_value.text().strip(),
        }
        payload.update(self._extended_axis_options_payload())
        # No "annotations" key: they moved to OverlayPropertiesWidget, and
        # the window merges a payload into the stored options rather than
        # replacing them - so leaving the key out leaves them alone, which
        # is exactly right for a panel that no longer edits them.
        self.axis_options_requested.emit(payload)

    # ------------------------------------------------------------------
    # Kwargs editor host helpers
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Default event filter retained for compatibility."""
        return super().eventFilter(watched, event)

    def _replace_layout_widget(self, layout: QVBoxLayout, widget: QWidget) -> None:
        """Replace kwargs content with a plain panel widget.

        The outgoing widget is hidden and scheduled for deletion, and
        deliberately *not* unparented on the way out.  ``setParent(None)``
        does not simply detach a widget - it promotes it to a *top-level
        window*, and the kwargs editor rebuilt here on every panel switch
        was left as exactly that: a stray, fully-populated window Qt could
        show on its own afterwards, floating over the application with its
        own title bar.

        ``takeAt`` has already removed it from the layout, so keeping the
        parent costs nothing and means the widget can never become a window
        in the first place - not even for the one event-loop pass between
        ``deleteLater`` and the deletion actually happening, which is a real
        window on screen if Qt shows it in the meantime.
        """
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget() if item is not None else None
            if child is not None:
                child.hide()
                child.deleteLater()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget.setContentsMargins(0, 0, 0, 0)
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        widget.setMinimumHeight(self.KWARGS_PANEL_MIN_HEIGHT)
        widget.setMaximumHeight(MAX_QT_HEIGHT)
        layout.addWidget(widget, 0)
