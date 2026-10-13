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
    QSizePolicy,
    QSpinBox,
    QPushButton,
    QStackedWidget,
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

    KWARGS_PANEL_MIN_HEIGHT: int = 220

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # _repo/_figure_id/_figure/_redraw_callback come from BaseProperties.
        self._current_axis_id: int | None = None
        #: The selected axis's renderer name, untranslated (the label shows it translated).
        self._renderer_key = ""
        self._axis_map: dict[int, AxisDescriptorLike] = {}
        self._kwargs_editor: DictEditorPanel | None = None
        # (axis_id, renderer_key, resolved kwargs values) of the editor
        # currently built - see rebuild_kwargs_editor's no-op guard.
        self._kwargs_build_signature: tuple[Any, ...] | None = None
        self._limit_spins: dict[str, QDoubleSpinBox] = {}
        self._limit_auto_checks: dict[str, QCheckBox] = {}
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
            self._coordinate_label,
            self._projection_combo,
            *self._share_checks.values(),
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
        """The axis selector, then tabs: Properties, X, Y, Z (one coordinate page, moved between them), Kwargs."""
        root = QVBoxLayout(self)
        root.setContentsMargins(*MARGIN_PANEL)
        root.setSpacing(12)
        root.addWidget(self._build_axis_selector_section())

        self._build_coordinate_controls()
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("axisPropertiesTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tabs.addTab(self._tab_page(self._build_shared_properties_frame()), _("Properties"))
        self._coordinate_page = self._build_coordinate_page()
        # Shown by the first X, Y or Z tab that hosts it; until then it would
        # be a stray widget in the panel's corner.
        self._coordinate_page.hide()
        self._coordinate_hosts: dict[str, QVBoxLayout] = {}
        for axis in axis_options.AXES:
            page = self._tab_page()
            self._coordinate_hosts[axis] = cast(QVBoxLayout, page.layout())
            self._tabs.addTab(page, axis.upper())
        self._tabs.addTab(self._tab_page(self._build_kwargs_section(), stretch=1), _("Kwargs"))
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)
        self._on_tab_changed(self._tabs.currentIndex())

    def _tab_page(self, content: QWidget | None = None, *, stretch: int = 0) -> QWidget:
        """A tab's page on the standard ground, *content* at its top."""
        page = QWidget(self._tabs if hasattr(self, "_tabs") else self)
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)
        if content is not None:
            layout.addWidget(content, stretch)
        if not stretch:
            layout.addStretch(1)
        return page

    def _on_tab_changed(self, index: int) -> None:
        """An X, Y or Z tab shows the one coordinate page, filled with that coordinate's settings."""
        axes = axis_options.AXES
        if not 1 <= index <= len(axes):
            return
        self._coordinate_axis = axes[index - 1]
        host = self._coordinate_hosts[self._coordinate_axis]
        host.insertWidget(0, self._coordinate_page)
        self._coordinate_page.show()
        self._show_coordinates()

    def _build_shared_properties_frame(self) -> QWidget:
        """Build compact axis-wide properties mounted in the active XYZ page."""
        section = TitledCard(self, _("Axis"), "axisPropertiesCard")
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

        grid.addWidget(QLabel(_("Title"), section.card), 0, 0)
        grid.addWidget(self._axis_label_edit, 0, 1, 1, 3)
        add_field(1, 0, _("Proj."), self._projection_combo)
        add_field(1, 2, _("Font"), self._font_scale)
        grid.addWidget(QLabel(_("Visible"), section.card), 2, 0)
        grid.addWidget(self._hide_axis_check, 2, 1)

        span = self._build_span_row(section.card)
        self._row_span_spin.setMaximumWidth(72)
        self._col_span_spin.setMaximumWidth(72)
        grid.addWidget(QLabel(_("Span"), section.card), 3, 0)
        grid.addWidget(span, 3, 1, 1, 3)

        def check_row(checks: list[QCheckBox]) -> QWidget:
            holder = QWidget(section.card)
            layout = QHBoxLayout(holder)
            stdSizeAndlayout(layout)
            for check in checks:
                layout.addWidget(check)
            layout.addStretch(1)
            return holder

        names = {"top": _("Top"), "right": _("Right"), "bottom": _("Bottom"), "left": _("Left")}
        short = {"top": "T", "right": "R", "bottom": "B", "left": "L"}
        for name, check in self._spine_checks().items():
            check.setText(short[name])
            check.setToolTip(names[name])
        grid.addWidget(QLabel(_("Hide spines"), section.card), 4, 0)
        grid.addWidget(check_row(list(self._spine_checks().values())), 4, 1, 1, 3)
        grid.addWidget(QLabel(_("Share"), section.card), 5, 0)
        grid.addWidget(check_row(list(self._share_checks.values())), 5, 1, 1, 3)
        raw_layout.addLayout(grid)
        return section

    # ------------------------------------------------------------------
    # Coordinates: one set of controls, applied to the X, Y, Z ticked
    # ------------------------------------------------------------------
    #: A coordinate's settings as the card shows them; min/max None is Auto.
    _COORDINATE_DEFAULTS: dict[str, Any] = {
        "label": "", "scale": "linear", "base": 10.0, "invert": False,
        "tick_length": 0.0, "tick_rotation": 0.0, "tick_length_minor": 0.0, "tick_rotation_minor": 0.0,
        "min": None, "max": None,
        **{f"{kind}_{which}": axis_options.AUTO for kind in ("grid", "ticks") for which in axis_options.WHICH},
    }

    def _build_coordinate_controls(self) -> None:
        """One control per setting; the values live per coordinate in ``_coordinates``."""
        self._coordinates: dict[str, dict[str, Any]] = {
            axis: dict(self._COORDINATE_DEFAULTS) for axis in axis_options.AXES
        }
        #: The coordinate the X, Y, Z tabs show now.
        self._coordinate_axis = "x"
        self._share_checks: dict[str, QCheckBox] = {}
        for axis in axis_options.AXES:
            check = QCheckBox(axis.upper(), self)
            check.setToolTip(_("Share the {axis} range with the other axes that share it.").format(axis=axis.upper()))
            self._share_checks[axis] = check
        self._coordinate_label = QLineEdit(self)
        self._scale_combo = QComboBox(self)
        for value in AXIS_SCALES:
            self._scale_combo.addItem(value, value)
        self._scale_combo.currentIndexChanged.connect(self._update_scale_control_state)
        self._scale_base_spin = QDoubleSpinBox(self)
        self._scale_base_spin.setRange(1.1, 1000.0); self._scale_base_spin.setDecimals(2); self._scale_base_spin.setValue(10.0)
        self._invert_check = QCheckBox(_("Invert"), self)
        self._invert_check.setToolTip(_("Invert direction"))
        self._tick_length_spin = QDoubleSpinBox(self)
        self._tick_length_spin.setRange(0.0, 50.0); self._tick_length_spin.setDecimals(1); self._tick_length_spin.setSingleStep(0.5)
        self._tick_rotation_spin = QDoubleSpinBox(self)
        self._tick_rotation_spin.setRange(-180.0, 180.0); self._tick_rotation_spin.setDecimals(0); self._tick_rotation_spin.setSingleStep(15.0)
        self._minor_tick_length_spin = QDoubleSpinBox(self)
        self._minor_tick_length_spin.setRange(0.0, 50.0); self._minor_tick_length_spin.setDecimals(1); self._minor_tick_length_spin.setSingleStep(0.5)
        self._minor_tick_rotation_spin = QDoubleSpinBox(self)
        self._minor_tick_rotation_spin.setRange(-180.0, 180.0); self._minor_tick_rotation_spin.setDecimals(0); self._minor_tick_rotation_spin.setSingleStep(15.0)
        self._grid_combos: dict[str, QComboBox] = {}
        self._tick_combos: dict[str, QComboBox] = {}
        for which in axis_options.WHICH:
            grid = QComboBox(self); tick = QComboBox(self)
            for value, text in axis_options.GRID_CHOICES: grid.addItem(_(text), value)
            for value, text in axis_options.TICK_CHOICES: tick.addItem(_(text), value)
            self._grid_combos[which] = grid
            self._tick_combos[which] = tick
        for edge in ("min", "max"):
            spin = QDoubleSpinBox(self)
            spin.setRange(-1.0e15, 1.0e15); spin.setDecimals(6)
            # No text: the spin box says "Auto" itself while the box is ticked.
            auto = QCheckBox(self); auto.setChecked(True)
            auto.setToolTip(_("Automatic range"))
            auto.toggled.connect(lambda checked, target=spin: target.setDisabled(checked))
            self._limit_spins[edge] = spin
            self._limit_auto_checks[edge] = auto
        self._linthresh_spin = QDoubleSpinBox(self)
        self._linthresh_spin.setRange(1e-9, 1e9)
        self._linthresh_spin.setDecimals(6)
        self._linthresh_spin.setValue(1.0)
        self._hide_spine_top_check = QCheckBox(_("Top"), self); self._hide_spine_right_check = QCheckBox(_("Right"), self)
        self._hide_spine_bottom_check = QCheckBox(_("Bottom"), self); self._hide_spine_left_check = QCheckBox(_("Left"), self)
        # A change goes into every coordinate ticked - before the auto-apply
        # timer reads them for the payload.
        for field, signal in self._coordinate_signals().items():
            for one in signal:
                one.connect(lambda *_args, f=field: self._store_field(f))

    def _coordinate_signals(self) -> dict[str, tuple[Any, ...]]:
        """Each coordinate setting and the signals that say it was changed."""
        signals: dict[str, tuple[Any, ...]] = {
            "label": (self._coordinate_label.textChanged,),
            "scale": (self._scale_combo.currentIndexChanged,),
            "base": (self._scale_base_spin.valueChanged,),
            "invert": (self._invert_check.toggled,),
            "tick_length": (self._tick_length_spin.valueChanged,),
            "tick_rotation": (self._tick_rotation_spin.valueChanged,),
            "tick_length_minor": (self._minor_tick_length_spin.valueChanged,),
            "tick_rotation_minor": (self._minor_tick_rotation_spin.valueChanged,),
        }
        for edge in ("min", "max"):
            signals[edge] = (self._limit_spins[edge].valueChanged, self._limit_auto_checks[edge].toggled)
        for which in axis_options.WHICH:
            signals[f"grid_{which}"] = (self._grid_combos[which].currentIndexChanged,)
            signals[f"ticks_{which}"] = (self._tick_combos[which].currentIndexChanged,)
        return signals

    def _read_field(self, field: str) -> Any:
        """The value the card shows for *field*."""
        if field == "label":
            return self._coordinate_label.text().strip()
        if field == "invert":
            return bool(self._invert_check.isChecked())
        if field == "scale":
            return str(self._scale_combo.currentData() or "linear")
        if field in ("min", "max"):
            return None if self._limit_auto_checks[field].isChecked() else float(self._limit_spins[field].value())
        if field.startswith(("grid_", "ticks_")):
            kind, which = field.split("_", 1)
            combo = (self._grid_combos if kind == "grid" else self._tick_combos)[which]
            return str(combo.currentData() or axis_options.AUTO)
        return float(self._value_spins()[field].value())

    def _write_field(self, field: str, value: Any) -> None:
        """Show *value* for *field* in the card."""
        if field == "label":
            self._coordinate_label.setText(str(value or ""))
        elif field == "invert":
            self._invert_check.setChecked(bool(value))
        elif field == "scale":
            self._select_combo_value(self._scale_combo, value, "linear")
        elif field in ("min", "max"):
            spin, auto = self._limit_spins[field], self._limit_auto_checks[field]
            auto.setChecked(value is None)
            spin.setValue(spin.minimum() if value is None else float(value))
            spin.setEnabled(value is not None)
        elif field.startswith(("grid_", "ticks_")):
            kind, which = field.split("_", 1)
            combo = (self._grid_combos if kind == "grid" else self._tick_combos)[which]
            self._select_combo_value(combo, value, axis_options.AUTO)
        else:
            self._value_spins()[field].setValue(float(value))

    def _value_spins(self) -> dict[str, QDoubleSpinBox]:
        """The settings shown in a plain number box."""
        return {"base": self._scale_base_spin, "tick_length": self._tick_length_spin,
                "tick_rotation": self._tick_rotation_spin, "tick_length_minor": self._minor_tick_length_spin,
                "tick_rotation_minor": self._minor_tick_rotation_spin}

    def _targets(self) -> list[str]:
        """The coordinate the page edits: its tab's."""
        return [self._coordinate_axis]

    def _store_field(self, field: str) -> None:
        value = self._read_field(field)
        for axis in self._targets():
            self._coordinates[axis][field] = value

    def _show_coordinates(self) -> None:
        """Fill the page from its coordinate's settings, without applying anything."""
        values = self._coordinates[self._coordinate_axis]
        with self._form_signal_blocker():
            for field in self._COORDINATE_DEFAULTS:
                self._write_field(field, values[field])
        self._update_scale_control_state()

    #: What "Copy ticks and grid" copies from the coordinate shown to the others.
    _TICK_AND_GRID_FIELDS: tuple[str, ...] = (
        "grid_major", "grid_minor", "ticks_major", "ticks_minor",
        "tick_length", "tick_rotation", "tick_length_minor", "tick_rotation_minor",
    )

    def _copy_ticks_and_grid(self) -> None:
        """The shown coordinate's grid and ticks, given to X, Y and Z alike."""
        source = self._coordinates[self._coordinate_axis]
        for axis in axis_options.AXES:
            for field in self._TICK_AND_GRID_FIELDS:
                self._coordinates[axis][field] = source[field]
        self._queue_auto_apply()

    def _build_coordinate_page(self) -> QWidget:
        """One coordinate's settings - the X, Y and Z tabs show this one page, moved between them.

        Two short captions and two fields a row, the fields sharing the width:
        a long caption would widen the whole first column and squeeze every
        field into an unreadable stub.
        """
        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        for widget in (self._scale_combo, *self._grid_combos.values(), *self._tick_combos.values()):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            widget.setMinimumContentsLength(5)
        for edge in ("min", "max"):
            # Auto shows in the box itself, not the spin box's -1e15 floor.
            self._limit_spins[edge].setSpecialValueText(_("Auto"))
            # An explicit minimum: the spin box would otherwise ask for room
            # for all fifteen digits of its range and widen the whole panel.
            self._limit_spins[edge].setMinimumWidth(70)
        # Their ranges are huge; their values are short.
        for spin in (self._linthresh_spin, self._scale_base_spin, *self._value_spins().values()):
            spin.setMinimumWidth(70)
            spin.setMaximumWidth(120)

        def card(title: str, name: str) -> tuple[QWidget, QGridLayout, QWidget]:
            section = TitledCard(page, title, name)
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            section.card.layout().addLayout(grid)
            page_layout.addWidget(section)
            return section, grid, section.card

        def row(parent: QWidget, widgets: list[QWidget], *, fill: bool = False) -> QWidget:
            """*widgets* side by side; with *fill* the first takes the width left."""
            holder = QWidget(parent)
            layout = QHBoxLayout(holder)
            stdSizeAndlayout(layout)
            layout.setContentsMargins(0, 0, 0, 0)
            for index, widget in enumerate(widgets):
                layout.addWidget(widget, 1 if fill and index == 0 else 0)
            if not fill:
                layout.addStretch(1)
            return holder

        # Label, scale and range.
        _section, grid, box = card(_("Scale and range"), "axisScaleCard")
        grid.addWidget(QLabel(_("Label"), box), 0, 0)
        grid.addWidget(self._coordinate_label, 0, 1, 1, 3)
        grid.addWidget(QLabel(_("Scale"), box), 1, 0)
        grid.addWidget(self._scale_combo, 1, 1)
        # One box beside the scale: its base for log, its linear zone for symlog.
        self._scale_parameter_caption = QLabel(_("Base"), box)
        self._scale_parameter = QStackedWidget(box)
        self._scale_parameter.addWidget(self._scale_base_spin)
        self._scale_parameter.addWidget(self._linthresh_spin)
        grid.addWidget(self._scale_parameter_caption, 1, 2)
        grid.addWidget(self._scale_parameter, 1, 3)
        grid.addWidget(QLabel(_("Min"), box), 2, 0)
        grid.addWidget(row(box, [self._limit_spins["min"], self._limit_auto_checks["min"]], fill=True), 2, 1)
        grid.addWidget(QLabel(_("Max"), box), 2, 2)
        grid.addWidget(row(box, [self._limit_spins["max"], self._limit_auto_checks["max"]], fill=True), 2, 3)
        grid.addWidget(self._invert_check, 3, 1, 1, 3)

        # Ticks and grid: major and minor as columns.
        _section, grid, box = card(_("Ticks and grid"), "axisTicksCard")
        grid.addWidget(QLabel(_("Major"), box), 0, 1, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(QLabel(_("Minor"), box), 0, 2, 1, 2, Qt.AlignmentFlag.AlignHCenter)
        rows = (
            (_("Grid"), self._grid_combos["major"], self._grid_combos["minor"]),
            (_("Ticks"), self._tick_combos["major"], self._tick_combos["minor"]),
            (_("Length"), self._tick_length_spin, self._minor_tick_length_spin),
            (_("Rotation"), self._tick_rotation_spin, self._minor_tick_rotation_spin),
        )
        for index, (text, major, minor) in enumerate(rows, start=1):
            grid.addWidget(QLabel(text, box), index, 0)
            grid.addWidget(major, index, 1)
            grid.addWidget(minor, index, 2, 1, 2)
        copy = QPushButton(_("Copy to X, Y and Z"), box)
        copy.setToolTip(_("Give X, Y and Z this coordinate's grid and ticks."))
        copy.clicked.connect(self._copy_ticks_and_grid)
        grid.addWidget(copy, len(rows) + 1, 1, 1, 3, Qt.AlignmentFlag.AlignLeft)
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

    def _update_scale_control_state(self) -> None:
        """Enable only the scale parameters the scales actually use."""
        shown = str(self._scale_combo.currentData() or "linear")
        symlog = shown == "symlog"
        if hasattr(self, "_scale_parameter"):
            self._scale_parameter.setCurrentWidget(self._linthresh_spin if symlog else self._scale_base_spin)
            self._scale_parameter_caption.setText(_("Linear zone") if symlog else _("Base"))
            self._scale_parameter_caption.setToolTip(
                _("The linear region around zero of a symlog scale.") if symlog else _("The base of a log scale."))
        self._scale_base_spin.setEnabled(shown == "log")
        self._linthresh_spin.setEnabled(symlog)

    def _selected_scales(self) -> tuple[str, str, str]:
        """The x, y and z scales, defaulting to linear."""
        return tuple(str(self._coordinates[a]["scale"] or "linear") for a in axis_options.AXES)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Scale / tick / grid / spine options
    # ------------------------------------------------------------------
    def _extended_option_widgets(self) -> tuple[QWidget, ...]:
        """Return the controls added by the scale and decoration sections."""
        return (
            self._row_span_spin,
            self._col_span_spin,
            self._scale_combo,
            self._scale_base_spin,
            self._linthresh_spin,
            self._invert_check,
            self._tick_length_spin,
            self._tick_rotation_spin,
            self._minor_tick_length_spin,
            self._minor_tick_rotation_spin,
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
        """Populate the scale, tick, grid and spine settings from options."""
        self._row_span_spin.setValue(self._int_option(options, "row_span", 1))
        self._col_span_spin.setValue(self._int_option(options, "col_span", 1))
        self._linthresh_spin.setValue(self._float_option(options, "x_linthresh", 1.0))
        legacy_tick_length = self._float_option(options, "tick_length", 0.0)
        for axis in axis_options.AXES:
            values = self._coordinates[axis]
            scale = str(options.get(f"{axis}_scale") or "linear")
            values["scale"] = scale if scale in AXIS_SCALES else "linear"
            values["base"] = self._float_option(options, f"{axis}_scale_base", 10.0) or 10.0
            values["invert"] = bool(options.get(f"invert_{axis}", False))
            values["tick_length"] = self._float_option(options, f"{axis}_tick_length", legacy_tick_length) or 0.0
            values["tick_rotation"] = self._float_option(options, f"{axis}_tick_rotation", 0.0) or 0.0
            values["tick_length_minor"] = self._float_option(options, f"{axis}_minor_tick_length", 0.0) or 0.0
            values["tick_rotation_minor"] = self._float_option(options, f"{axis}_minor_tick_rotation", 0.0) or 0.0
            # Through axis_options so a figure saved before these existed opens
            # with the settings it actually had, rather than with four Autos.
            for which in axis_options.WHICH:
                values[f"grid_{which}"] = axis_options.grid_setting(options, axis, which)
                values[f"ticks_{which}"] = axis_options.tick_setting(options, axis, which)
            for edge in ("min", "max"):
                values[edge] = axis_options.manual_limit(options, axis, edge)
        for name, check in self._spine_checks().items():
            check.setChecked(bool(options.get(f"hide_spine_{name}", False)))
        self._show_coordinates()

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
        coordinates = self._coordinates
        scales = self._selected_scales()
        payload: dict[str, Any] = {
            "row_span": int(self._row_span_spin.value()),
            "col_span": int(self._col_span_spin.value()),
            # One threshold control for all three: symlog's linear region is a
            # property of the data's units, and an axis whose scale is not
            # symlog ignores the key entirely.
            "x_linthresh": float(self._linthresh_spin.value()) if "symlog" in scales else None,
            # Legacy global length follows X for older renderers/projects.
            "tick_length": coordinates["x"]["tick_length"] or None,
        }
        payload["y_linthresh"] = payload["z_linthresh"] = payload["x_linthresh"]
        for axis, scale in zip(axis_options.AXES, scales):
            values = coordinates[axis]
            payload[f"{axis}_scale"] = scale
            payload[f"{axis}_scale_base"] = float(values["base"]) if scale in {"log", "symlog"} else None
            payload[f"invert_{axis}"] = bool(values["invert"])
            payload[f"{axis}_tick_length"] = values["tick_length"] or None
            payload[f"{axis}_tick_rotation"] = float(values["tick_rotation"])
            payload[f"{axis}_minor_tick_length"] = values["tick_length_minor"] or None
            payload[f"{axis}_minor_tick_rotation"] = float(values["tick_rotation_minor"])
            for which in axis_options.WHICH:
                payload[axis_options.grid_key(axis, which)] = values[f"grid_{which}"]
                payload[axis_options.tick_key(axis, which)] = values[f"ticks_{which}"]
            for edge in ("min", "max"):
                payload[axis_options.limit_key(axis, edge)] = values[edge]

        # The legacy keys are written as the settings imply, so a figure
        # edited here still renders in a build that predates them - and so
        # nothing downstream reads a stale "grid: True" that the user has
        # since turned off.
        payload["grid"] = False
        payload["grid_which"] = "major"
        payload["grid_axis"] = "both"
        payload["minor_ticks"] = any(
            coordinates[axis]["ticks_minor"] not in (axis_options.AUTO, axis_options.OFF)
            for axis in axis_options.AXES
        )
        payload["limits_mode"] = getattr(axis_options, "LIMITS_MANUAL", "manual")

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
            self._projection_combo,
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
            for axis in axis_options.AXES:
                self._coordinates[axis]["label"] = str(options.get(f"{axis}_label", "")).strip()
                self._share_checks[axis].setChecked(bool(options.get(f"share{axis}", False)))
            self._projection_combo.setCurrentIndex(
                max(0, self._projection_combo.findData(projection))
            )
            self._hide_axis_check.setChecked(
                bool(options.get("hide_axis", options.get("hidden", False)))
            )
            self._font_scale.set_factor(options.get("font_scale", 1.0))
            self._load_extended_axis_options(options)

        # Translated for reading; the raw chart_type is what the signal
        # carries, since every listener looks the renderer up by that name.
        self._renderer_key = renderer
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
                self._coordinate_label,
                self._projection_combo,
                *self._share_checks.values(),
                self._hide_axis_check,
            )
            + self._extended_option_widgets()
        )

    def _clear_axis_fields(self) -> None:
        """Clear axis-specific form fields."""
        self._current_axis_id = None
        with self._form_signal_blocker():
            self._axis_label_edit.clear()
            for axis in axis_options.AXES:
                self._coordinates[axis] = dict(self._COORDINATE_DEFAULTS)
            self._projection_combo.setCurrentIndex(0)
            self._hide_axis_check.setChecked(False)
            self._font_scale.set_factor(1.0)
            self._clear_extended_axis_options()
        self._renderer_key = ""
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
        """Resolve the renderer name: the axis's chart_type first, then its options.

        The chart_type is what the figure is drawn with. ``options["renderer"]``
        once held the name as the panel *showed* it - translated, "Diagramma
        di Pareto" - which no renderer is called; reading the chart_type first
        repairs those axes too.
        """
        renderer = (
            getattr(axis_desc, "name", "")
            or options.get("renderer")
            or options.get("renderer_name")
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
            "x_label": self._coordinates["x"]["label"],
            "y_label": self._coordinates["y"]["label"],
            "z_label": self._coordinates["z"]["label"],
            "projection": str(self._projection_combo.currentData() or "rectilinear"),
            "sharex": bool(self._share_checks["x"].isChecked()),
            "sharey": bool(self._share_checks["y"].isChecked()),
            "sharez": bool(self._share_checks["z"].isChecked()),
            "hide_axis": bool(self._hide_axis_check.isChecked()),
            "font_scale": float(self._font_scale.factor()),
            # The renderer's own name, never the translated one the panel shows.
            "renderer": self._renderer_key,
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
