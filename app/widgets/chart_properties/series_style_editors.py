"""A series' line and its markers, each edited by a class of its own: LineStyleEditor and MarkerStyleEditor.

Each is one block of the Series properties: "Default (the axis style)" first - ticked, the block folds to that
one line and the series keeps nothing of its own -, then its fields. Each reads the series' style with load()
and gives back what to store with values(); a value is None when the key is to be removed. value_changed is
emitted on every edit, so the panel applies it as it does for its other controls.

A stored value the lists cannot show ("dashed", "C0", a value written elsewhere) is kept and given back for as
long as its field is left as it was loaded: writing the field's own reading back on an edit of another field
changed the series (issue #6).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from matplotlib import rcParams
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox, QWidget

from app.plugin_api.charts import dashes, markers
from app.plugin_api.charts.kwarg_spec import DEFAULT
from app.utils.i18n import _
from app.widgets.drop_down_controls.color_combo import (
    DEFAULT_COLOR_LABEL,
    MatplotlibColorCombo,
)
from app.widgets.drop_down_controls.line_combo import LineStyleCombo
from app.widgets.drop_down_controls.marker_combo import MarkerStyleCombo
from app.widgets.drop_down_controls.style_chooser import SYMBOLS, StyleChooser


def _is_default(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == DEFAULT


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _colour_combo(parent: QWidget) -> MatplotlibColorCombo:
    return MatplotlibColorCombo(parent, include_none=True, none_label=DEFAULT_COLOR_LABEL)  # untranslated, as the combos beside it


def _set_colour(combo: MatplotlibColorCombo, value: str) -> None:
    if not (value and (combo.set_current_hex(value) or combo.set_current_name(value))):
        combo.setCurrentIndex(0)


class _StyleEditor(QWidget):
    """The shared part: the Default box, a form whose rows fold, the stored-while-untouched values."""

    value_changed = Signal()

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._loading = False
        self._kept: dict[str, tuple[Any, int]] = {}  # key -> (stored value, the combo index it showed as)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setVerticalSpacing(10)
        self.default_check = QCheckBox(_("Default (the axis style)"), self)
        self.default_check.toggled.connect(self._default_toggled)
        self.form.addRow(self.default_check)
        self._fields: list[QWidget] = []

    # -- building
    def _row(self, label: str, widget: QWidget) -> None:
        self.form.addRow(label, widget) if label else self.form.addRow(widget)
        self._fields.append(widget)

    def _emit(self, *_args: Any) -> None:
        if not self._loading:
            self.value_changed.emit()

    # -- what the subclasses say
    def _owns_nothing(self, style: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def _start_from_drawn(self) -> None:
        """Default unticked: the fields start from what the style drew (else "default" is stored again)."""

    # -- kept values
    def _keep(self, key: str, raw: Any, index: int) -> None:
        self._kept[key] = (raw, index)

    def _kept_or(self, key: str, index: int, current: Any) -> Any:
        kept = self._kept.get(key)
        return kept[0] if kept is not None and index == kept[1] else current

    # -- folding
    def _default_toggled(self, on: bool) -> None:
        if not on and not self._loading:
            self._start_from_drawn()
        self.update_fields()
        self._emit()

    def update_fields(self) -> None:
        """On Default the block is its one line: the fields are hidden, not greyed."""
        shown = not self.default_check.isChecked()
        for widget in self._fields:
            self.form.setRowVisible(widget, shown)

    def load(self, style: Mapping[str, Any]) -> None:
        self._loading = True
        try:
            self._kept.clear()
            self.default_check.setChecked(self._owns_nothing(style))
            self._load(style)
            self.update_fields()
        finally:
            self._loading = False

    def clear(self) -> None:
        self.load({})

    def _load(self, style: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def values(self) -> dict[str, Any]:
        raise NotImplementedError


class LineStyleEditor(_StyleEditor):
    """The series' line: style (Standard or Custom), width, colour."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = LineStyleCombo(self)
        self.chooser = StyleChooser(self.combo, is_custom=dashes.is_custom, select=self.combo.set_current_linestyle,
                                    hint=_("e.g. ..--  or  -- . --  or  (0, (5, 2, 1, 2))"), parent=self)
        # Line width in points; the minimum is "Default" (the style's), stored as nothing.
        self.width_spin = QDoubleSpinBox(self)
        self.width_spin.setRange(0.0, 50.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setSingleStep(0.25)
        self.width_spin.setSpecialValueText(_("Default"))
        self.width_spin.setToolTip(_("Line width in points. Default leaves it to the style."))
        self.colour_combo = _colour_combo(self)
        self._row(_("Style"), self.chooser)
        self._row(_("Width"), self.width_spin)
        self._row(_("Color"), self.colour_combo)
        self.combo.currentIndexChanged.connect(self._emit)
        self.width_spin.valueChanged.connect(self._emit)
        self.colour_combo.currentIndexChanged.connect(self._emit)

    def _owns_nothing(self, style: Mapping[str, Any]) -> bool:
        return _is_default(style.get("linestyle")) and not style.get("linewidth") and not style.get("color")

    def _start_from_drawn(self) -> None:
        if self.combo.current_value() == DEFAULT:
            drawn = str(rcParams.get("lines.linestyle", "-") or "-")
            self.combo.set_current_linestyle(drawn if drawn.lower() not in ("", "none") else "-")
            self.chooser.sync()

    def _load(self, style: Mapping[str, Any]) -> None:
        # Absent is not Default: absent means "no line" (a scatter's markers only), Default "the style's line".
        raw = style.get("linestyle")
        text = raw.strip() if isinstance(raw, str) else None
        if text is not None and text.lower() == "none":
            text = "none"
        if text is not None:
            self.combo.set_current_linestyle(text)  # a pattern of its own gets its entry (charts.dashes)
        index = self.combo.findData(text) if text is not None else -1
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self._keep("linestyle", raw, self.combo.currentIndex())
        self.chooser.sync()
        self.width_spin.setValue(max(0.0, _number(style.get("linewidth"))))
        colour = style.get("color", "")
        _set_colour(self.colour_combo, str(colour or ""))
        self._keep("color", colour, self.colour_combo.currentIndex())

    def colour(self) -> str:
        """The line's colour as chosen ("" for the style's)."""
        return self.colour_combo.current_hex().strip()

    def values(self) -> dict[str, Any]:
        if self.default_check.isChecked():
            return {"linestyle": DEFAULT, "linewidth": None, "color": ""}
        linestyle = self._kept_or("linestyle", self.combo.currentIndex(), str(self.combo.current_linestyle() or "-"))
        colour = self._kept_or("color", self.colour_combo.currentIndex(), self.colour())
        return {"linestyle": linestyle, "linewidth": self.width_spin.value() or None,
                "color": colour if colour is not None else ""}


class MarkerStyleEditor(_StyleEditor):
    """The series' markers: marker (Standard, Custom, Symbol, Picture), size, every n-th, colour."""

    def __init__(self, parent: QWidget | None = None, *, repo: Callable[[], Any] | None = None,
                 line_colour: Callable[[], str] | None = None) -> None:
        super().__init__(parent)
        self._line_colour = line_colour or (lambda: "")
        self.combo = MarkerStyleCombo(self)
        self.chooser = StyleChooser(self.combo, is_custom=markers.is_custom, select=self.combo.set_current_marker,
                                    hint=_("e.g. ★  or  $\\clubsuit$  or  (5, 1, 0)"), symbols=SYMBOLS,
                                    pictures=repo, parent=self)
        # Points, as matplotlib's markersize; the minimum is "Default" (the style's), stored as nothing.
        self.size_spin = QDoubleSpinBox(self)
        self.size_spin.setRange(0.0, 1000.0)
        self.size_spin.setDecimals(1)
        self.size_spin.setSingleStep(0.5)
        self.size_spin.setSpecialValueText(_("Default"))
        self.size_spin.setToolTip(_("Marker size in points. Default leaves it to the style."))
        # A marker on every n-th point only (matplotlib's markevery): dense lines stay readable.
        self.every_spin = QSpinBox(self)
        self.every_spin.setRange(1, 100000)
        self.every_spin.setSpecialValueText(_("Every point"))
        self.every_spin.setSuffix(_(" points"))
        self.every_spin.setToolTip(_("Draw a marker on every n-th point only; the line still joins them all."))
        # The markers' own colour, or the line's (the default).
        self.same_colour_check = QCheckBox(_("Same colour as the line"), self)
        self.same_colour_check.setChecked(True)
        self.colour_combo = _colour_combo(self)
        self._row(_("Marker"), self.chooser)
        self._row(_("Size"), self.size_spin)
        self._row(_("Every"), self.every_spin)
        self._row("", self.same_colour_check)
        self._row(_("Color"), self.colour_combo)
        self.combo.currentIndexChanged.connect(self._emit)
        self.size_spin.valueChanged.connect(self._emit)
        self.every_spin.valueChanged.connect(self._emit)
        self.colour_combo.currentIndexChanged.connect(self._emit)
        self.same_colour_check.toggled.connect(self._same_colour_toggled)

    def _owns_nothing(self, style: Mapping[str, Any]) -> bool:
        return (_is_default(style.get("marker")) and not style.get("markersize") and not style.get("markevery")
                and not style.get("markercolor"))

    def _start_from_drawn(self) -> None:
        if self.combo.current_value() == DEFAULT:
            drawn = str(rcParams.get("lines.marker", "") or "")
            self.combo.set_current_marker(drawn if drawn.lower() not in ("", "none") else "o")
            self.chooser.sync()

    def _same_colour_toggled(self, same: bool) -> None:
        """Unticked: the colour starts from the line's (or the list's first), so a colour is stored -
        left on Default nothing was, and the box ticked itself back on the next reload."""
        if not same and not self._loading and not self.colour_combo.current_hex().strip():
            line = self._line_colour()
            if not (line and self.colour_combo.set_current_hex(line)):
                self.colour_combo.setCurrentIndex(min(1, self.colour_combo.count() - 1))
        self.update_fields()
        self._emit()

    def update_fields(self) -> None:
        super().update_fields()
        self.colour_combo.setEnabled(not self.same_colour_check.isChecked())

    def _load(self, style: Mapping[str, Any]) -> None:
        raw = style.get("marker")
        text = raw.strip() if isinstance(raw, str) else None
        if text is not None and text.lower() == "none":
            text = ""
        if text is not None:
            self.combo.set_current_marker(text)  # a marker of its own gets its entry (charts.markers)
        index = self.combo.findData(text) if text is not None else -1
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self._keep("marker", raw, self.combo.currentIndex())
        self.chooser.sync()
        self.size_spin.setValue(max(0.0, _number(style.get("markersize"))))
        self.every_spin.setValue(max(1, int(_number(style.get("markevery"), 1.0))))
        colour = str(style.get("markercolor") or "")
        self.same_colour_check.setChecked(not colour)
        _set_colour(self.colour_combo, colour)

    def values(self) -> dict[str, Any]:
        if self.default_check.isChecked():
            return {"marker": DEFAULT, "markersize": None, "markevery": None, "markercolor": None}
        marker = self._kept_or("marker", self.combo.currentIndex(), str(self.combo.current_marker() or ""))
        return {
            "marker": marker,
            "markersize": self.size_spin.value() or None,
            "markevery": self.every_spin.value() if self.every_spin.value() > 1 else None,
            "markercolor": None if self.same_colour_check.isChecked()
            else (self.colour_combo.current_hex().strip() or None),
        }
