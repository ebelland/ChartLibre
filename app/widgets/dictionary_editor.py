"""Generic schema-driven key/value property editor."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import (
    apply_fusion_for_item_view_styling,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.widgets.drop_down_controls.color_combo import MatplotlibColorCombo
from app.widgets.drop_down_controls.font_combo import MatplotlibFontCombo
from app.widgets.drop_down_controls.line_combo import LineStyleCombo
from app.widgets.drop_down_controls.marker_combo import MarkerStyleCombo

_KIND_BY_EXACT_KEY: dict[str, str] = {
    # Colour aliases and common artist kwargs.
    "color": "color", "c": "color", "facecolor": "color",
    "facecolors": "color", "edgecolor": "color", "edgecolors": "color",
    "markerfacecolor": "color", "markeredgecolor": "color",
    "mfc": "color", "mec": "color", "ecolor": "color",
    "labelcolor": "color",
    # Line and marker helpers.
    "linestyle": "linestyle", "ls": "linestyle", "marker": "marker",
    # Matplotlib's boxplot flier symbol: a marker under another name.
    "sym": "marker",
    "joinstyle": "joinstyle", "capstyle": "capstyle",
    # Font-family lists. The rcParams schema normally sets this explicitly,
    # but exact recognition also makes the shared editor work independently.
    "font.family": "fontlist", "font.serif": "fontlist",
    "font.sans-serif": "fontlist", "font.monospace": "fontlist",
    "font.cursive": "fontlist", "font.fantasy": "fontlist",
    "fontfamily": "fontlist", "font_family": "fontlist",
    # Numeric artist kwargs.
    "alpha": "number", "linewidth": "number", "linewidths": "number",
    "lw": "number", "markersize": "number", "ms": "number",
    "markeredgewidth": "number", "mew": "number", "capsize": "number",
    "capthick": "number", "pickradius": "number", "zorder": "number",
    "dpi": "number", "fontsize": "number", "size": "number",
    "rotation": "number", "pad": "number",
    # Boolean artist kwargs.
    "visible": "bool", "animated": "bool", "antialiased": "bool",
    "antialiaseds": "bool", "aa": "bool", "clip_on": "bool",
    "in_layout": "bool", "rasterized": "bool", "fill": "bool",
    "frameon": "bool", "show_legend": "bool", "show_in_legend": "bool",
    "sort_x": "bool", "log": "bool", "grid": "bool",
}

# Ordered from specialised endings to broad endings. For example,
# markerfacecolor must be recognised as a colour before marker is considered.
_SUFFIX_KIND_RULES: tuple[tuple[str, str], ...] = (
    # Font lists.
    (".font.family", "fontlist"), (".font.serif", "fontlist"),
    (".font.sans-serif", "fontlist"), (".font.monospace", "fontlist"),
    (".font.cursive", "fontlist"), (".font.fantasy", "fontlist"),
    ("font_family", "fontlist"), ("fontfamily", "fontlist"),
    # Colours, most specific first.
    ("markeredgecolor", "color"), ("markerfacecolor", "color"),
    ("labelcolor", "color"), ("gridcolor", "color"),
    ("edgecolor", "color"), ("facecolor", "color"),
    ("tick.color", "color"), ("tick.labelcolor", "color"),
    (".color", "color"), ("color", "color"),
    # Line and marker selectors.
    ("linestyle", "linestyle"), ("joinstyle", "joinstyle"),
    ("capstyle", "capstyle"), ("marker", "marker"),
    # Booleans.
    ("antialiased", "bool"), ("clip_on", "bool"),
    ("in_layout", "bool"), ("rasterized", "bool"),
    ("frameon", "bool"), ("visible", "bool"), ("animated", "bool"),
    (".grid", "bool"), (".fill", "bool"),
    # Numeric values. Avoid a generic 'size' rule because figsize and list-like
    # sizes are not scalar spin-box values.
    ("markeredgewidth", "number"), ("linewidth", "number"),
    ("markersize", "number"), ("fontsize", "number"),
    ("pickradius", "number"), ("capsize", "number"),
    ("capthick", "number"), ("zorder", "number"),
    ("rotation", "number"), (".dpi", "number"),
    (".alpha", "number"), ("alpha", "number"),
    ("labelpad", "number"), ("titlepad", "number"),
)
_JOINSTYLE_CHOICES = ["miter", "round", "bevel"]
_CAPSTYLE_CHOICES = ["butt", "round", "projecting"]
_GROUP_ORDER = [
    "Appearance", "Line", "Marker", "Patch", "Error bars", "Geometry",
    "Text", "Behavior", "Other",
]
#: The options nobody changes (todo H-15): real Matplotlib keywords, kept in one
#: closed group at the bottom - "Advanced" - rather than among the ones in use.
#: A renderer can send its own there with "advanced": True in the key's spec.
_ADVANCED_GROUP = "Advanced"
_ADVANCED_KEYS = frozenset({"animated", "clip_on", "in_layout", "snap", "gid", "url", "rasterized", "picker",
                            "sketch_params", "path_effects", "agg_filter", "transform", "clip_path", "clip_box"})
_MIN_ROW_HEIGHT = 30
_EDITOR_MIN_HEIGHT = 26
_DEFAULT_PROPERTY_COLUMN_WIDTH = 200
_DEFAULT_VALUE_COLUMN_WIDTH = 170
# Small: the tree takes the height its panel has, and gives it back in a short
# one - at 360 it ran past the bottom of the Axis page's Kwargs tab.
_DEFAULT_MIN_PANEL_HEIGHT = 160


def _normalize_meta(meta: object) -> dict[str, Any]:
    return dict(meta) if isinstance(meta, dict) else {"default": meta}


def _value_to_text(value: object) -> str:
    return "" if value is None else str(value)


def _parse_text_value(text: str) -> object:
    stripped = text.strip()
    if stripped == "" or stripped.lower() in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError, TypeError):
        return text


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _make_check_item(item: QTreeWidgetItem, value: Any) -> None:
    item.setFlags(
        item.flags()
        & ~Qt.ItemFlag.ItemIsUserCheckable
        & ~Qt.ItemFlag.ItemIsEditable
    )
    item.setText(1, "")
    item.setCheckState(
        1, Qt.CheckState.Checked if _truthy(value) else Qt.CheckState.Unchecked
    )


def _kind_for_key(key: str, meta: Mapping[str, Any]) -> str:
    explicit = meta.get("kind", meta.get("editor"))
    if explicit:
        return str(explicit).strip().lower()
    value_type = meta.get("type")
    if isinstance(value_type, list):
        return "enum"
    if value_type is bool:
        return "bool"
    if value_type in (int, float):
        return "number"
    lower = key.strip().lower()
    if lower in _KIND_BY_EXACT_KEY:
        return _KIND_BY_EXACT_KEY[lower]
    for suffix, kind in _SUFFIX_KIND_RULES:
        if lower.endswith(suffix):
            return kind
    default = meta.get("default")
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (int, float)):
        return "number"
    return "string"


def _group_for_key(key: str, meta: Mapping[str, Any], kind: str) -> str:
    lower = key.strip().lower()
    if meta.get("advanced") or lower in _ADVANCED_KEYS:
        return _ADVANCED_GROUP
    explicit = meta.get("group")
    if explicit:
        return str(explicit)
    if kind == "linestyle" or lower in {"linewidth", "linewidths", "lw", "alpha"}:
        return "Line"
    if kind == "marker" or lower.startswith("marker") or lower in {"mfc", "mec", "ms"}:
        return "Marker"
    if lower in {"facecolor", "facecolors", "edgecolor", "edgecolors", "hatch", "fill"}:
        return "Patch"
    if lower in {"xerr", "yerr", "ecolor", "capsize", "error_kw"}:
        return "Error bars"
    if lower in {"width", "height", "bottom", "left", "align"}:
        return "Geometry"
    if lower in {"label", "tick_label", "url", "gid"} or kind == "fontlist":
        return "Text"
    if kind == "bool" or lower in {"zorder", "picker", "snap", "clip_on", "animated", "rasterized"}:
        return "Behavior"
    if kind == "color":
        return "Appearance"
    return "Other"


class DictValueDelegate(QStyledItemDelegate):
    """Value delegate supplying type-aware inline editors."""

    def __init__(self, panel: "DictEditorPanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel

    def sizeHint(self, option, index):  # noqa: N802, ANN001, ANN201
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), _MIN_ROW_HEIGHT))
        return size

    def createEditor(self, parent: QWidget, option, index):  # noqa: N802, ANN001, ANN201
        if index.column() != 1:
            return None
        key = self._panel.key_from_index(index)
        if key is None:
            return None
        meta = self._panel.meta_for_key(key)
        kind = self._panel.kind_for_key(key)
        if kind == "bool":
            return None
        if kind == "color":
            editor = MatplotlibColorCombo(parent, include_none=True)
        elif kind == "fontlist":
            editor = MatplotlibFontCombo(
                parent,
                include_none=True,
                include_generic=True,
                extra_fonts=meta.get("choices", ()),
            )
        elif kind == "linestyle":
            editor = LineStyleCombo(parent)
        elif kind == "marker":
            editor = MarkerStyleCombo(parent)
        elif kind == "number":
            if meta.get("type") is int:
                editor = QSpinBox(parent)
                editor.setRange(
                    int(str(meta.get("min", -2147483648))),
                    int(str(meta.get("max", 2147483647))),
                )
                editor.setSingleStep(int(str(meta.get("step", 1))))
            else:
                editor = QDoubleSpinBox(parent)
                editor.setRange(
                    float(str(meta.get("min", -1.0e12))),
                    float(str(meta.get("max", 1.0e12))),
                )
                editor.setDecimals(int(str(meta.get("decimals", 6))))
                editor.setSingleStep(float(str(meta.get("step", 0.1))))
        elif kind in {"enum", "joinstyle", "capstyle", "loc"}:
            editor = QComboBox(parent)
            if kind == "joinstyle":
                choices = _JOINSTYLE_CHOICES
            elif kind == "capstyle":
                choices = _CAPSTYLE_CHOICES
            else:
                choices = list(meta.get("choices", meta.get("type", [])) or [])
            for choice in choices:
                editor.addItem("" if choice is None else str(choice), choice)
            editor.setEditable(bool(meta.get("editable")))
        else:
            editor = QLineEdit(parent)
        editor.setMinimumHeight(_EDITOR_MIN_HEIGHT)
        if isinstance(editor, QComboBox):
            editor.setProperty("cellEditor", True)
        return editor

    @staticmethod
    def _watch_untouched(editor: QSpinBox | QDoubleSpinBox) -> None:
        """Flag *editor* as holding no user value until its value or text changes."""
        editor.setProperty("untouched", True)

        def touched(*_args: object) -> None:
            editor.setProperty("untouched", False)

        editor.valueChanged.connect(touched)
        line_edit = editor.lineEdit()
        if line_edit is not None:
            line_edit.textEdited.connect(touched)

    def setEditorData(self, editor, index) -> None:  # noqa: N802, ANN001
        key = self._panel.key_from_index(index)
        if key is None:
            return
        value = self._panel.value_for_key(key)
        text = _value_to_text(value)
        if isinstance(editor, MatplotlibColorCombo):
            if text == "":
                editor.setCurrentIndex(0)
            elif not editor.set_current_hex(text):
                editor.set_current_name(text)
            return
        if isinstance(editor, MatplotlibFontCombo):
            if isinstance(value, (list, tuple)):
                family = value[0] if value else None
            else:
                parsed = _parse_text_value(text)
                family = (
                    parsed[0]
                    if isinstance(parsed, (list, tuple)) and parsed
                    else text.split(",", 1)[0].strip()
                )
            editor.set_current_font(family)
            return
        if isinstance(editor, LineStyleCombo):
            editor.set_current_linestyle(text)
            return
        if isinstance(editor, MarkerStyleCombo):
            editor.set_current_marker(text)
            return
        if isinstance(editor, QCheckBox):
            editor.setChecked(_truthy(value))
            return
        if isinstance(editor, (QDoubleSpinBox, QSpinBox)):
            if text:
                if isinstance(editor, QDoubleSpinBox):
                    editor.setValue(float(text))
                else:
                    editor.setValue(int(text))
            else:
                # Unset: the box can only show its minimum, which is not a value
                # the user chose. Committing it untouched would write that
                # minimum (a marker size of 0 hides every marker), so remember
                # that nothing was typed yet.
                self._watch_untouched(editor)
            return
        if isinstance(editor, QComboBox) and editor.isEditable():
            editor.setCurrentText(text)
            return
        if isinstance(editor, QComboBox):
            idx = editor.findData(value)
            if idx < 0:
                idx = editor.findText(text)
            if idx >= 0:
                editor.setCurrentIndex(idx)
            elif text:
                editor.insertItem(0, text, value)
                editor.setCurrentIndex(0)
            return
        if isinstance(editor, QLineEdit):
            editor.setText(text)
            editor.selectAll()

    def setModelData(self, editor, model, index) -> None:  # noqa: N802, ANN001
        key = self._panel.key_from_index(index)
        if key is None:
            return
        if isinstance(editor, MatplotlibColorCombo):
            value: object = editor.current_hex() or None
        elif isinstance(editor, MatplotlibFontCombo):
            selected = editor.current_font()
            previous = self._panel.value_for_key(key)
            fallbacks = [str(v) for v in previous] if isinstance(previous, (list, tuple)) else []
            value = (
                [selected, *[family for family in fallbacks if family != selected]]
                if selected else None
            )
        elif isinstance(editor, LineStyleCombo):
            value = editor.current_linestyle()
        elif isinstance(editor, MarkerStyleCombo):
            value = editor.current_marker()
        elif isinstance(editor, QCheckBox):
            value = editor.isChecked()
        elif isinstance(editor, (QDoubleSpinBox, QSpinBox)):
            # A row that had no value and was never edited stays without one.
            value = None if editor.property("untouched") else editor.value()
        elif isinstance(editor, QComboBox) and editor.isEditable():
            value = _parse_text_value(editor.currentText())
        elif isinstance(editor, QComboBox):
            value = editor.currentData()
        elif isinstance(editor, QLineEdit):
            value = _parse_text_value(editor.text())
        else:
            value = None
        self._panel.set_value_for_key(key, value)
        model.setData(index, self._panel._display_text(key, value), Qt.ItemDataRole.DisplayRole)
        model.setData(index, value, Qt.ItemDataRole.UserRole)


class DictEditorPanel(QWidget):
    """Qt-Designer-like grouped property editor for dictionary values."""

    valuesChanged = Signal(dict)

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config: dict[str, dict[str, Any]] = {}
        self._values: dict[str, object] = {}
        self._key_items: dict[str, QTreeWidgetItem] = {}
        self._group_items: dict[str, QTreeWidgetItem] = {}
        self._updating = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(_DEFAULT_MIN_PANEL_HEIGHT)
        layout = QVBoxLayout(self)
        stdSizeAndlayout(layout)
        self._search_row = QWidget(self)
        search_lay = QHBoxLayout(self._search_row)
        search_lay.setContentsMargins(0, 0, 0, 0)
        search_lay.setSpacing(8)
        self.search_edit = QLineEdit(self._search_row)
        self.search_edit.setPlaceholderText(_("Search properties..."))
        stdSizeAndlayout(self.search_edit)
        self.search_edit.textChanged.connect(self._apply_filter)
        search_lay.addWidget(self.search_edit, 1)
        layout.addWidget(self._search_row, 0)
        self.tree = QTreeWidget(self)
        self.tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mark_editor_panel(self.tree)
        apply_fusion_for_item_view_styling(self.tree)
        self.tree.setMinimumHeight(_DEFAULT_MIN_PANEL_HEIGHT - 32)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([_("Property"), _("Value")])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(False)
        self.tree.setIconSize(QSize(20, 20))
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QTreeWidget.EditTrigger.EditKeyPressed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.setItemDelegateForColumn(1, DictValueDelegate(self, self.tree))
        self._configure_header()
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree, 1)
        self.set_config(config or {})

    def _configure_header(self) -> None:
        header = self.tree.header()
        header.setSectionsMovable(False)
        # The value column takes the width left, so the header and the rows
        # reach the scroll bar instead of stopping short of it.
        header.setStretchLastSection(True)
        # The value column fills what the name column leaves: no sideways
        # scrolling in a panel this narrow.
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(60)
        header.resizeSection(0, _DEFAULT_PROPERTY_COLUMN_WIDTH)
        header.resizeSection(1, _DEFAULT_VALUE_COLUMN_WIDTH)

    def add_search_row_widget(self, widget: QWidget) -> None:
        widget.setParent(self._search_row)
        row_layout = self._search_row.layout()
        if isinstance(row_layout, QBoxLayout):
            row_layout.addWidget(widget, 0)

    def set_config(self, config: Mapping[str, object]) -> None:
        self.commit_pending_edits()
        selected_key = self.current_key()
        filter_text = self.filter_text()
        self._updating = True
        self.config = {key: _normalize_meta(meta) for key, meta in config.items()}
        self._values = {key: meta.get("default") for key, meta in self.config.items()}
        self._rebuild_tree()
        self._updating = False
        self.set_filter_text(filter_text)
        self.set_current_key(selected_key)
        self._emit_values_changed()

    def set_values(self, values: Mapping[str, object] | None) -> None:
        if not values:
            return
        self.commit_pending_edits()
        selected_key = self.current_key()
        filter_text = self.filter_text()
        self._updating = True
        for key, value in values.items():
            if key in self.config:
                self._values[key] = value
                self._update_item_value(key)
        self._updating = False
        self.set_filter_text(filter_text)
        self.set_current_key(selected_key)
        self._emit_values_changed()

    def get_values(self) -> dict[str, object]:
        return dict(self._values)

    def reset_to_defaults(self) -> None:
        self.set_values({key: meta.get("default") for key, meta in self.config.items()})

    def commit_pending_edits(self) -> None:
        focus_widget = QApplication.focusWidget()
        widget = focus_widget
        while widget is not None:
            if isinstance(widget, QComboBox):
                widget.hidePopup()
            widget = widget.parentWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        self.tree.clearFocus()
        for item in self._key_items.values():
            self.tree.closePersistentEditor(item, 1)

    def filter_text(self) -> str:
        return self.search_edit.text()

    def set_filter_text(self, text: str | None) -> None:
        self.search_edit.blockSignals(True)
        self.search_edit.setText(text or "")
        self.search_edit.blockSignals(False)
        self._apply_filter()

    def current_key(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        key = item.data(0, Qt.ItemDataRole.UserRole)
        return str(key) if key else None

    def set_current_key(self, key: str | None) -> None:
        if key is None:
            return
        item = self._key_items.get(key)
        if item is None:
            return
        parent = item.parent()
        if parent is not None:
            parent.setExpanded(True)
        self.tree.setCurrentItem(item, 1)
        self.tree.scrollToItem(item)

    def key_from_index(self, index) -> str | None:  # noqa: ANN001
        key = index.siblingAtColumn(0).data(Qt.ItemDataRole.UserRole)
        return str(key) if key else None

    def meta_for_key(self, key: str) -> dict[str, Any]:
        return self.config[key]

    def kind_for_key(self, key: str) -> str:
        return _kind_for_key(key, self.config[key])

    def value_for_key(self, key: str) -> object:
        return self._values.get(key)

    def set_value_for_key(self, key: str, value: object) -> None:
        selected_key = self.current_key()
        filter_text = self.filter_text()
        self._values[key] = value
        self._update_item_value(key)
        self.set_filter_text(filter_text)
        self.set_current_key(selected_key or key)
        self._emit_values_changed()

    def _rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self._key_items.clear()
        self._group_items.clear()
        grouped: dict[str, list[str]] = {}
        for key, meta in self.config.items():
            group = _group_for_key(key, meta, self.kind_for_key(key))
            grouped.setdefault(group, []).append(key)
        ordered = [group for group in _GROUP_ORDER if group in grouped]
        ordered.extend(sorted(group for group in grouped if group not in ordered and group != _ADVANCED_GROUP))
        if _ADVANCED_GROUP in grouped:
            ordered.append(_ADVANCED_GROUP)
        for group in ordered:
            group_item = QTreeWidgetItem(self.tree, [group, ""])
            group_item.setData(0, Qt.ItemDataRole.UserRole, None)
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            group_item.setExpanded(False)
            group_item.setFirstColumnSpanned(True)
            self._group_items[group] = group_item
            for key in grouped[group]:
                meta = self.config[key]
                value = self._values.get(key)
                label = str(meta.get("label", key))
                item = QTreeWidgetItem(group_item, [label, self._display_text(key, value)])
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                item.setData(1, Qt.ItemDataRole.UserRole, value)
                item.setToolTip(0, str(meta.get("description", "")))
                item.setToolTip(1, str(meta.get("description", f"{key} ({self.kind_for_key(key)})")))
                if self.kind_for_key(key) == "bool":
                    _make_check_item(item, value)
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setSizeHint(0, QSize(0, _MIN_ROW_HEIGHT))
                item.setSizeHint(1, QSize(0, _MIN_ROW_HEIGHT))
                self._key_items[key] = item
        self.tree.blockSignals(False)
        self._configure_header()
        self._apply_filter()

    def _update_item_value(self, key: str) -> None:
        item = self._key_items.get(key)
        if item is None:
            return
        value = self._values.get(key)
        self.tree.blockSignals(True)
        if self.kind_for_key(key) == "bool":
            _make_check_item(item, value)
        else:
            item.setText(1, self._display_text(key, value))
        item.setData(1, Qt.ItemDataRole.UserRole, value)
        self.tree.blockSignals(False)

    def _display_text(self, key: str, value: object) -> str:
        if self.kind_for_key(key) == "fontlist":
            if isinstance(value, str):
                parsed = _parse_text_value(value)
                value = parsed if isinstance(parsed, (list, tuple)) else value
            if isinstance(value, (list, tuple)):
                return ", ".join(str(v) for v in value)
        return _value_to_text(value)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        key_str = str(key)
        if self.kind_for_key(key_str) == "bool":
            item.setCheckState(
                1,
                Qt.CheckState.Unchecked
                if item.checkState(1) == Qt.CheckState.Checked
                else Qt.CheckState.Checked,
            )
            return

        # Always activate the value cell, even when the property-name column
        # was clicked. editItem() schedules delegate creation, so open combo
        # popups on the next event-loop turn after the editor exists.
        self.tree.setCurrentItem(item, 1)
        self.tree.editItem(item, 1)
        if self.kind_for_key(key_str) in {
            "fontlist", "color", "linestyle", "marker",
            "enum", "joinstyle", "capstyle", "loc",
        }:
            QTimer.singleShot(0, lambda current=item: self._open_combo_editor(current))

    def _open_combo_editor(self, item: QTreeWidgetItem) -> None:
        """Open the combo created by the value delegate for *item*."""
        index = self.tree.indexFromItem(item, 1)
        editor = self.tree.indexWidget(index)
        if isinstance(editor, QComboBox):
            editor.setFocus(Qt.FocusReason.MouseFocusReason)
            editor.showPopup()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 1:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        key_str = str(key)
        if self.kind_for_key(key_str) == "bool":
            value: object = item.checkState(1) == Qt.CheckState.Checked
        else:
            value = self._coerce_value_from_text(key_str, item.text(1))
        self._values[key_str] = value
        item.setData(1, Qt.ItemDataRole.UserRole, value)
        self._emit_values_changed()

    def _coerce_value_from_text(self, key: str, text: str) -> object:
        kind = self.kind_for_key(key)
        meta = self.config[key]
        if text.strip() == "":
            return None
        if kind == "bool":
            return text.strip().lower() in {"true", "1", "yes", "on"}
        if kind == "number":
            return int(text) if meta.get("type") is int else float(text)
        if kind == "fontlist":
            parsed = _parse_text_value(text)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
            return [part.strip() for part in text.split(",") if part.strip()]
        return _parse_text_value(text)

    def _apply_filter(self) -> None:
        needle = self.search_edit.text().strip().lower()
        for group, group_item in self._group_items.items():
            group_match = needle in group.lower() if needle else True
            visible_children = 0
            for index in range(group_item.childCount()):
                child = group_item.child(index)
                if child is None:
                    continue
                key = str(child.data(0, Qt.ItemDataRole.UserRole) or "")
                if not key or key not in self.config:
                    child.setHidden(True)
                    continue
                meta = self.config[key]
                haystack = " ".join([
                    key, str(meta.get("label", "")), str(meta.get("description", "")),
                    self.kind_for_key(key), _value_to_text(self._values.get(key)), group,
                ]).lower()
                visible = not needle or needle in haystack or group_match
                child.setHidden(not visible)
                if visible:
                    visible_children += 1
            group_item.setHidden(bool(needle and visible_children == 0 and not group_match))
            if needle and visible_children:
                group_item.setExpanded(True)
            elif not needle:
                group_item.setExpanded(False)

    def _emit_values_changed(self) -> None:
        if not self._updating:
            self.valuesChanged.emit(self.get_values())
