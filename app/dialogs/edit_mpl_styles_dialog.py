#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib Style Editor
=======================

A PySide6 dialog for creating and editing ``.mplstyle`` files.

Features
--------
* Style-stack panel — compose multiple built-in or file-based styles.
* Parameter-override property editor — add / edit / reset individual rcParams with
  type-aware inline editors (color picker, bool checkbox, number field,
  linestyle / marker / cmap / loc combo, cycler dialog, font-list dialog,
  plus additional enum editors).
* Live preview chart — redraws on every change with a 300 ms debounce.
* Tree picker with icons — rcParams grouped by category; each leaf carries
  a small icon that reflects its value-type (color swatch, number, bool, …).
  The last-used category is remembered between sessions via QSettings.

Serialization contract
----------------------
* Colors in the editor and saved file are stored **without** a leading ``#``
  (mpl's .mplstyle parser adds ``#`` automatically when loading).
* lists (font families, figsize, …) are saved as bare comma-separated values,
  NOT as Python ``repr``—that is the format mpl's parser expects.
* Strings are never wrapped in quotes; mpl reads everything after ``:``
  verbatim.
"""
from __future__ import annotations
import ast
from collections.abc import Callable, Iterable
import os
import re
import tempfile
from enum import Enum
from typing import Any, cast
from pathlib import Path


import cycler as _cycler
import matplotlib as mpl
from matplotlib import style as mplstyle
import numpy as np

from app.dialogs.two_panels_dialog_base import SettingsTableDialog
from app.styles.style import (
    MARGIN_CARD,
    action_presentation,
    create_action_button,
    create_compact_section_title,
    load_icon,
    mark_icon_only,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.utils.mpl_latex import (
    is_latex_rcparam,
    latex_available,
    latex_unavailable_reason,
)

mpl.use("QtAgg")
from matplotlib import image as _mimage
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from app.widgets import mpl_cursors  # noqa: F401 - native cursors on macOS
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.figure import Figure

from PySide6.QtCore import QModelIndex, Qt, QSettings, QSize, QTimer, Signal
from PySide6.QtGui import     QFont,    QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QWidget, QTreeView, QLabel, QAbstractItemView, QFileDialog, QHBoxLayout, QVBoxLayout, QComboBox, QSizePolicy, QListWidget, QListWidgetItem

# ---------------------------------------------------------------------------
# Optional external helpers (graceful fallbacks)
# ---------------------------------------------------------------------------

from app.widgets.dictionary_editor import DictEditorPanel
from app.utils.i18n import _

OverrideErrors = list[tuple[str, object, str]]
ButtonAction = (
    tuple[str | None, Callable[[], None] | None]
    | tuple[str | None, str | None, Callable[[], None] | None]
)

# ---------------------------------------------------------------------------
# rcParam type-inference
# ---------------------------------------------------------------------------


_LOC_CHOICES = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
    "center",
]
_JOINSTYLE_CHOICES = ["miter", "round", "bevel"]
_CAPSTYLE_CHOICES = ["butt", "round", "projecting"]

# Curated enums for additional rcParams
_ENUM_CHOICES: dict[str, list[str]] = {
    # images
    "image.origin": ["upper", "lower"],
    "image.interpolation": sorted(getattr(_mimage, "interpolations_names", [])) or
                           ["none", "nearest", "bilinear", "bicubic", "lanczos"],
    # axes
    "axes.autolimit_mode": ["data", "round_numbers"],
    "axes.titlelocation": ["left", "center", "right"],
    # fonts/weights
    "font.weight": ["ultralight", "light", "normal", "regular", "book",
                    "medium", "semibold", "bold", "heavy", "black"],
    "axes.titleweight": ["normal", "bold", "heavy", "light", "ultralight",
                         "semibold", "medium", "black"],
    "axes.labelweight": ["normal", "bold", "heavy", "light", "ultralight",
                         "semibold", "medium", "black"],
    "mathtext.fontset": ["dejavusans", "dejavuserif", "cm", "stix", "stixsans", "custom"],
    # saving
    "savefig.format": [],  # populated dynamically from canvas
}

_EXPLICIT_KINDS: dict[str, str] = {
    "font.family": "fontlist",
    "font.serif": "fontlist",
    "font.sans-serif": "fontlist",
    "font.monospace": "fontlist",
    "font.cursive": "fontlist",
    "font.fantasy": "fontlist",
    "text.color": "color",
    "legend.loc": "loc",
    "contour.negative_linestyle": "linestyle",
    "legend.frameon": "bool",
    "image.cmap": "cmap",
    # curated enums → 'enum'
    **{k: "enum" for k in _ENUM_CHOICES.keys()},
}

_FONT_SIZES: list[str] = [
    "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "larger", "smaller",
]
_FONT_WEIGHTS: list[str] = [
    "ultralight", "light", "normal", "regular", "book", "medium", "roman",
    "semibold", "demibold", "demi", "bold", "heavy", "extra bold", "black",
]
_FONT_STRETCHES: list[str] = [
    "ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "normal",
    "semi-expanded", "expanded", "extra-expanded", "ultra-expanded",
]

#: Choices for rcParams whose Matplotlib validator is a function rather than
#: a list of strings, by the validator's name: (choices, editable). Editable
#: when the parameter also takes a number - a font size is "large" or 12.
_VALIDATOR_CHOICES: dict[str, tuple[list[str], bool]] = {
    "validate_fontweight": (_FONT_WEIGHTS, True),
    "validate_fontsize": (_FONT_SIZES, True),
    "validate_fontstretch": (_FONT_STRETCHES, True),
    "validate_bbox": (["tight", "standard"], False),
    "validate_aspect": (["auto", "equal"], True),
    "validate_axisbelow": (["line", "True", "False"], False),
    "validate_hist_bins": (["auto", "sturges", "fd", "doane", "scott", "stone", "rice", "sqrt"], True),
}

#: Plain string rcParams that still only take a few words.
_KEY_CHOICES: dict[str, list[str]] = {
    "font.style": ["normal", "italic", "oblique"],
    "font.variant": ["normal", "small-caps"],
}


def _rc_enum_choices(key: str) -> tuple[list[str], bool] | None:
    """The values *key* accepts, if it accepts a fixed set, and whether a
    number may be typed instead. None for a free value.

    From, in order: the hand-kept lists above, the validator's own list of
    strings (Matplotlib's ValidateInStrings - 28 rcParams), and the table of
    function validators. The editor shows these as a drop-down list.
    """
    lower = (key or "").strip().lower()
    if lower in _ENUM_CHOICES:
        choices = list(_ENUM_CHOICES[lower])
        if lower == "savefig.format":
            try:
                choices = sorted(plt.gcf().canvas.get_supported_filetypes().keys()) or choices
            except Exception:
                pass
        return choices, False
    if lower in _KEY_CHOICES:
        return list(_KEY_CHOICES[lower]), False
    if lower == "image.cmap":
        return sorted(mpl.colormaps, key=str.lower), False  # pyright: ignore[reportAttributeAccessIssue]
    try:
        from matplotlib import rcsetup

        validator = rcsetup._validators.get(key)  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        return None
    if isinstance(validator, type) and issubclass(validator, Enum):
        # JoinStyle, CapStyle: the members' own values.
        return [str(member.value) for member in validator], False
    valid = getattr(validator, "valid", None)
    if isinstance(valid, dict) and valid:
        return [str(v) for v in dict.fromkeys(valid.values())], False
    return _VALIDATOR_CHOICES.get(getattr(validator, "__name__", ""))


_SUFFIX_KIND_RULES: list[tuple[str, str]] = [
    (".color", "color"),
    (".edgecolor", "color"),
    (".facecolor", "color"),
    (".labelcolor", "color"),
    (".gridcolor", "color"),
    (".tick.color", "color"),
    (".tick.labelcolor", "color"),
    (".linestyle", "linestyle"),
    (".marker", "marker"),
    (".linewidth", "number"),
    (".markersize", "number"),
    (".dpi", "number"),
    (".alpha", "number"),
    (".cmap", "cmap"),
    (".joinstyle", "joinstyle"),
    (".capstyle", "capstyle"),
]

_NON_STYLE_KEYS = frozenset({
    "backend", "backend_fallback", "date.epoch", "docstring.hardcopy",
    "figure.max_open_warning", "figure.raise_window", "interactive",
    "savefig.directory", "timezone", "tk.window_focus", "toolbar",
    "webagg.address", "webagg.open_in_browser", "webagg.port",
    "webagg.port_retries",
})

#: The generic families font.family can name; each resolves through its
#: own font.<generic> list.
_GENERIC_FONT_FAMILIES: tuple[str, ...] = ("sans-serif", "serif", "monospace", "cursive", "fantasy")


def _font_choices(*, generic: bool) -> list[str]:
    """Font families for the font helpers: the generic ones first (for
    font.family), then every family Matplotlib has found on this machine."""
    try:
        from matplotlib import font_manager

        # Names starting with "." are macOS's private system fonts.
        installed = sorted(
            {entry.name for entry in font_manager.fontManager.ttflist if not entry.name.startswith(".")},
            key=str.lower,
        )
    except Exception:
        installed = []
    return [*(_GENERIC_FONT_FAMILIES if generic else ()), *installed]


def _rcparam_kind(key: str) -> str:
    """Return a value-type tag for *key*."""
    k = (key or "").strip()
    if not k:
        return "string"
    lower = k.lower()
    if lower in _EXPLICIT_KINDS and _EXPLICIT_KINDS[lower] not in ("cmap", "enum"):
        return _EXPLICIT_KINDS[lower]
    if _rc_enum_choices(k) is not None:
        return "enum"
    if lower in ("axes.prop_cycle", "prop_cycle"):
        return "cycler"
    for suffix, kind in _SUFFIX_KIND_RULES:
        if lower.endswith(suffix):
            return kind
    try:
        cur = cast(Any, mpl.rcParams)[k]
        if isinstance(cur, bool):
            return "bool"
        if isinstance(cur, str) and is_color_like(cur):
            return "color"
        if isinstance(cur, (int, float)):
            return "number"
        if isinstance(cur, _cycler.Cycler):
            return "cycler"
    except Exception:
        pass
    return "string"

# ---------------------------------------------------------------------------
# Value-formatting helpers
# ---------------------------------------------------------------------------

def _strip_hash(s: str) -> str:
    """Remove leading ``#`` from a CSS hex color string; no-op otherwise."""
    if not isinstance(s, str):
        return s
    t = s.strip()
    if (t.startswith("#")
        and len(t) in (4, 5, 7, 9)
        and all(c in "0123456789aAbBcCdDeEfF" for c in t[1:])):
        return t[1:]
    return s

def _add_hash(s: str) -> str:
    """Add ``#`` to a bare CSS hex string for use in rcParams."""
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith("#"):
        return t
    if (len(t) in (3, 4, 6, 8)
        and all(c in "0123456789aAbBcCdDeEfF" for c in t)):
        return "#" + t
    return s



def _coerce_rc_color(value: object) -> object:
    """Return a Matplotlib-valid color object, accepting bare hex strings.

    The editor stores hex values without '#', but rcParams validation requires
    CSS hex colors to include it. Lists/tuples are only kept as RGBA tuples when
    they are numeric 3/4-tuples; other list-like values are rejected so a color
    rcParam does not accidentally receive a list of colors.
    """
    if isinstance(value, str):
        text = value.strip()
        candidate = _add_hash(text)
        to_rgba(candidate)  # pyright: ignore[reportArgumentType]
        return candidate
    if isinstance(value, tuple) and len(value) in (3, 4) and all(isinstance(v, (int, float)) for v in value):
        rgba_tuple = cast(tuple[float, ...], value)
        to_rgba(cast(Any, rgba_tuple))
        return rgba_tuple
    if isinstance(value, list) and len(value) in (3, 4) and all(isinstance(v, (int, float)) for v in value):
        rgba_tuple = tuple(float(v) for v in value)
        to_rgba(cast(Any, rgba_tuple))
        return rgba_tuple
    raise TypeError(f"Invalid Matplotlib color value: {value!r}")

def _to_table_str(val: object) -> str:
    """Stable, hash-free string for display in the editor table."""
    if isinstance(val, _cycler.Cycler):
        keys = list(val.keys)
        if len(keys) == 1:
            prop = keys[0]
            items = [_strip_hash(str(d[prop])) for d in val]
            return f"cycler('{prop}', {items!r})"
        parts = [
            f"cycler('{p}', {[ _strip_hash(str(d[p])) for d in val ]!r})"
            for p in keys
        ]
        return " + ".join(parts)
    if isinstance(val, (list, tuple)):
        return repr([_strip_hash(str(x)) for x in val])
    s = _strip_hash(str(val))
    if s.startswith("CapStyle."):
        return s.split(".", 1)[1].lower()
    if s.startswith("JoinStyle."):
        return s.split(".", 1)[1].lower()
    return s

def _to_file_str(val: object) -> str:
    """Serialize *val* for writing to a ``.mplstyle`` file."""
    if isinstance(val, (list, tuple)):
        # Comma-separated bare values — the format mpl's parser expects
        return ", ".join(_strip_hash(str(x)) for x in val)
    s = _to_table_str(val).strip()
    # cycler strings are already in the correct format
    return s



# ---------------------------------------------------------------------------
# Small icon factory for the tree picker
# ---------------------------------------------------------------------------
_ICON_SIZE = 14  # pixels, square

# ---------------------------------------------------------------------------
# RcParam Tree Picker
# ---------------------------------------------------------------------------
#: rcParams groups that are the running program's, not a chart's style:
#: backends, key bindings, GUI toolkits, internals. Left out of the picker.
SYSTEM_RC_GROUPS: frozenset[str] = frozenset(
    {
        "_internal", "agg", "animation", "backend", "backend_fallback",
        "docstring", "interactive", "keymap", "macosx", "tk", "toolbar",
        "webagg",
    }
)


def style_rc_keys() -> list[str]:
    """Every rcParam that styles a chart, sorted - the picker's contents."""
    return sorted(key for key in mpl.rcParamsDefault if key.split(".")[0] not in SYSTEM_RC_GROUPS)


#: What the preview can draw: (label, key). "this" - the chart being
#: styled - is offered only when the caller hands the editor a way to draw it.
PREVIEW_KINDS: tuple[tuple[str, str], ...] = (
    ("This figure", "this"),
    ("Lines and scatter", "lines"),
    ("Bars", "bars"),
    ("Box plot", "box"),
    ("Histogram", "hist"),
    ("Pie", "pie"),
    ("Heatmap", "heatmap"),
    ("3D surface", "surface3d"),
)


class RcParamTreePicker(QComboBox):
    """Editable combo whose drop-down is a QTreeView grouping rcParams by
    their dot-prefix category. Each leaf node shows a type-specific icon.
    The last-selected category is remembered via QSettings.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setPlaceholderText(_("Pick rcParam…"))
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(28)

        self._model = QStandardItemModel(self)
        self._view = QTreeView(self)
        self._view.setHeaderHidden(True)
        self._view.setRootIsDecorated(True)
        self._view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._view.setUniformRowHeights(True)
        self._view.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._view.clicked.connect(self._on_item_clicked)

        self.setModel(self._model)
        self.setView(self._view)

        self._selected_key: str = ""
        self._settings = QSettings("MplStyleEditor", "Dialog")
        value: object = self._settings.value("rcpicker/last_category", "")
        self._last_category: str = value if isinstance(value, str) else ""
# ------------------------------------------------------------------
    def populate(self, keys: list[str]) -> None:
        """Rebuild the tree from *keys*, grouped by dot-prefix category."""
        cats: dict[str, list[str]] = {}
        for k in keys:
            cat = (k.split(".", 1)[0] if "." in k else "other").lower()
            cats.setdefault(cat, []).append(k)

        bold: QFont = QFont()
        bold.setBold(True)
        self._model.clear()

        for cat in sorted(cats):
            cat_item: QStandardItem = QStandardItem(f" {cat}")
            cat_item.setFont(bold)
            cat_item.setSelectable(False)
            cat_item.setEditable(False)
            for k in sorted(cats[cat], key=str.lower):
                kind: str = _rcparam_kind(k)
                child: QStandardItem = QStandardItem(k)
                child.setData(k, Qt.ItemDataRole.UserRole)
                child.setToolTip(f"{k} [{kind}]")
                child.setEditable(False)
                cat_item.appendRow(child)
            self._model.appendRow(cat_item)
        self._view.expandAll()
        self._scroll_to_last_category()

    def showPopup(self) -> None:
        super().showPopup()
        self._scroll_to_last_category()

    def selected_key(self) -> str:
        return self._selected_key or self.currentText().strip()

    # ------------------------------------------------------------------
    def _scroll_to_last_category(self) -> None:
        if not self._last_category:
            return
        for r in range(self._model.rowCount()):
            item = self._model.item(r, 0)
            if item and item.text().strip().lower() == self._last_category:
                self._view.scrollTo(
                    item.index(),
                    QAbstractItemView.ScrollHint.PositionAtTop)
                return

    def _on_item_clicked(self, index: QModelIndex) -> None:
        item = self._model.itemFromIndex(index)
        if item is None or item.hasChildren():
            return  # category node — ignore clicks
        key = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
        self._selected_key = key
        cat = (key.split(".", 1)[0] if "." in key else "other").lower()
        self._last_category = cat
        self._settings.setValue("rcpicker/last_category", cat)
        self.setCurrentText(key)
        self.hidePopup()

# ---------------------------------------------------------------------------
# Main editor dialog
# ---------------------------------------------------------------------------

class MplStyleEditorDialog(SettingsTableDialog):
    """Top-level Matplotlib style editor dialog."""

    # Emit path to a temp .mplstyle file when "Apply" is clicked
    styleApplied = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        apply_callback: Callable[[str], None] | None = None,
        initial_style_text: str = "",
        figure_drawer: Callable[[Figure], None] | None = None,
    ) -> None:
        # "medium": the action rows are icons now, so the editor no longer
        # needs the large shell's width. The style stack and the preview on
        # the left, the parameters on the right.
        super().__init__(
            parent,
            title=_("Matplotlib Style Editor"),
            icon=load_icon("edit"),
            size="medium",
            settings_width=400,
        )
        #: Draws the chart being styled onto a figure, for the "This figure"
        #: preview; None when the editor was opened without one.
        self._figure_drawer = figure_drawer
        self.setModal(True)
        self.apply_callback = apply_callback
        self.style_stack: list[str] = []

        # The status, left of the buttons.
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setContentsMargins(0, 0, 0, 0)
        self.action_row.insertWidget(0, self.status, 1)
        self.add_action("apply", self._on_apply_clicked, default=True)
        self.add_action("close", self._on_close)

        # Left: stack on top, compact figure preview on bottom.
        left: QWidget = QWidget(self.settings_frame)
        ll: QVBoxLayout = QVBoxLayout(left)
        stdSizeAndlayout(ll)
        ll.addWidget(self._build_stack_group(), 0)

        preview_panel, pg = self._create_section("Figure")
        pg.setContentsMargins(0, 0, 0, 0)
        # What the preview draws: the chart being styled, or a sample of one
        # chart family, so a style can be judged where it will be used.
        self.preview_kind = QComboBox()
        for label, key in PREVIEW_KINDS:
            if key == "this" and figure_drawer is None:
                continue
            self.preview_kind.addItem(_(label), key)
        stdSizeAndlayout(self.preview_kind, minimum_contents_length=14)
        self.preview_kind.currentIndexChanged.connect(lambda _i: self._preview_timer.start())
        pg.addWidget(self.preview_kind, 0)
        self.canvas = self._make_canvas()
        self.preview_card: QWidget = QWidget()
        self.preview_card.setProperty("previewCard", True)

        card_lay: QVBoxLayout = QVBoxLayout(self.preview_card)
        card_lay.setContentsMargins(*MARGIN_CARD)
        card_lay.setSpacing(0)
        card_lay.addWidget(self.canvas, 1)
        pg.addWidget(self.preview_card, 1)
        ll.addWidget(preview_panel, 1)

        # Right: parameters editor.
        self.settings_layout.addWidget(left, 1)
        self.content_layout.addWidget(self._build_table_group(), 1)

        # Debounced preview timer.
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(300)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self.update_preview)

        # Seed the table with common keys.
        for key in [
            "figure.facecolor", "figure.edgecolor",
            "figure.dpi",
            "axes.facecolor", "axes.edgecolor",
            "axes.titlesize", "axes.labelsize",
            "axes.grid", "grid.linestyle", "grid.color",
            "lines.linewidth", "lines.linestyle", "lines.markersize",
            "font.size", "font.family", "font.weight",
            "legend.loc", "legend.frameon",
            "xtick.labelsize", "ytick.labelsize",
            "savefig.dpi", "savefig.transparent", "savefig.format",
            "axes.prop_cycle", "text.color", "image.origin",
            "axes.labelcolor", "xtick.color", "ytick.color",
            "image.cmap", "axes.autolimit_mode", "axes.titlelocation",
            "mathtext.fontset",
            "lines.solid_joinstyle", "lines.solid_capstyle",
            "lines.dash_joinstyle", "lines.dash_capstyle",
        ]:
            self._append_row(key, "")

        if initial_style_text.strip():
            self.load_style_text(initial_style_text)
        else:
            self.update_preview()

    def _on_close(self) -> None:
        self.close()

    # ======================================================================
    # Panel builders
    # ======================================================================

    def _create_section(
        self,
        title: str,
        note: str | None = None,
    ) -> tuple[QWidget, QVBoxLayout]:
        # A titled part of the dialog's own frame, not a card of its own:
        # the two-panel shell already draws the frames.
        panel = QWidget(self)
        panel.setObjectName(f"mplStyle{title.replace(' ', '')}Section")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        title_label: QLabel = create_compact_section_title(_(title), panel)
        layout.addWidget(title_label, 0)

        if note:
            note_label: QLabel = QLabel(note, panel)
            note_label.setProperty("muted", True)
            layout.addWidget(note_label, 0)

        content: QVBoxLayout = QVBoxLayout()
        stdSizeAndlayout(content)
        layout.addLayout(content, 1)
        return panel, content

    def _build_stack_group(self) -> QWidget:
        group, lay = self._create_section("Style Stack")

        self.builtin = QComboBox()
        self.builtin.setMinimumWidth(180)
        self.builtin.setMinimumHeight(28)
        self.builtin.addItems(sorted(plt.style.available))

        row = QHBoxLayout()
        stdSizeAndlayout(row)
        row.setSpacing(4)
        row.addWidget(self.builtin, 1, Qt.AlignmentFlag.AlignVCenter)

        self._add_button_actions(row, [
            ("add", "Add selected built-in style to stack",
             self._stack_add_builtin),
            ("open", "Add .mplstyle file(s) to stack",
             self._stack_add_files),
            (None, None, None),
            ("up", "Move selected entry up", self._stack_move_up),
            ("down", "Move selected entry down", self._stack_move_down),
            ("delete", "Remove selected entries", self._stack_remove_selected),
            (None, None, None),
            ("style_stack_to_editor",
             "Compute stack diff and load into parameter editor",
             self._stack_result_to_table),
        ])

        self.stack = QListWidget()
        self.stack.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.stack.setFixedHeight(140)
        mark_editor_panel(self.stack)

        lay.addLayout(row)
        lay.addWidget(self.stack)
        return group

    def _build_table_group(self) -> QWidget:
        """Create the rcParams override editor using the shared DictEditorPanel."""
        group, lay = self._create_section("Parameters Override")

        self.rc_picker = RcParamTreePicker()
        try:
            self.rc_picker.populate(style_rc_keys())
        except Exception:
            self.rc_picker.populate([])
        self.rc_picker.setMinimumHeight(28)
        stdSizeAndlayout(self.rc_picker, minimum_contents_length=36)
        self.rc_picker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # The picker on a row of its own, as wide as the panel - rcParam
        # names are long - and the buttons on the row below it.
        lay.addWidget(self.rc_picker)
        row = QHBoxLayout()
        stdSizeAndlayout(row)
        row.setSpacing(4)

        # Action ids only: the labels and tooltips live in config.json, which
        # is why these are rcparam_* rather than the generic add/open/save -
        # "Add" on this row means "insert the selected rcParam", and the button
        # should say so.
        self._add_button_actions(row, [
            ("rcparam_add", self._add_param_from_picker),
            ("rcparam_open", self._load_from_file),
            (None, None),
            ("rcparam_reset", self._reset_selected_to_defaults),
            ("rcparam_delete", self._remove_selected_rows),
            (None, None),
            ("rcparam_save", self._save_file),
        ])
        row.addStretch(1)

        self._param_schema: dict[str, object] = {}
        self.editor = DictEditorPanel({}, self)
        self.editor.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.editor.valuesChanged.connect(lambda *_: self._preview_timer.start())

        # Compatibility alias: old methods refer to self.filter.
        self.filter = self.editor.search_edit

        lay.addLayout(row)
        lay.addWidget(self.editor, 1)
        return group

    def _add_button_actions(
        self,
        layout: QHBoxLayout,
        actions: Iterable[ButtonAction],
    ) -> None:
        """Add compact action buttons to *layout*.

        Accepts both button descriptors used in this dialog:

        * ``(action_id, callback)``
          Use the shared action catalogue from ``style.py`` for icon, label and
          tooltip.
        * ``(action_id, tooltip, callback)``
          Legacy form used by the style-stack row.  The button is still created
          through ``create_action_button``.  Only the tooltip is overridden, so
          icon/label remain catalogue-driven wherever the action exists.

        ``action_id is None`` inserts a small spacer.
        """
        if actions is None:
            return

        for descriptor in actions:
            action_id: str | None
            tooltip: str | None
            callback: Callable[[], None] | None

            if len(descriptor) == 2:
                action_id, callback = descriptor
                tooltip = None
            else:
                action_id, tooltip, callback = descriptor

            if action_id is None:
                layout.addSpacing(6)
                continue

            presentation = None
            if tooltip:
                icon, label, _catalog_tooltip = action_presentation(action_id)
                presentation = (icon, label, tooltip)

            button = create_action_button(
                parent=self,
                action_id=action_id,
                action=callback,
                layout=layout,
                presentation=presentation,
            )
            # Icons with the words in the tooltip: two rows of labelled
            # buttons were what made this dialog so wide.
            button.setAccessibleName(button.text())
            mark_icon_only(button)

    # ======================================================================
    # Canvas helpers
    # ======================================================================
    def _make_canvas(self) -> FigureCanvas:
        fig: Figure = Figure(
            figsize=(4.0, 3.0),
            dpi=72,
            tight_layout=True,
            facecolor="none",
        )
        canvas: FigureCanvas = FigureCanvas(fig)
        # Follows the panel: drag the splitter or the window to see the style
        # at the size it will be used at.
        canvas.setMinimumSize(240, 180)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        canvas.setProperty("previewCanvas", True)
        return canvas

    def _draw_sample(self, fig: Figure) -> None:
        """Draw the preview chosen in the combo, under the style being edited."""
        fig.clear()
        kind = self.preview_kind.currentData() if hasattr(self, "preview_kind") else "lines"
        if kind == "this" and self._figure_drawer is not None:
            self._figure_drawer(fig)
            return

        rng = np.random.default_rng(42)
        if kind == "surface3d":
            ax = fig.add_subplot(111, projection="3d")
            grid = np.linspace(-3, 3, 40)
            xx, yy = np.meshgrid(grid, grid)
            ax.plot_surface(xx, yy, np.exp(-(xx**2 + yy**2) / 3) * np.cos(xx), cmap=None)
            ax.set_title("3D surface")
            return

        ax = fig.add_subplot(111)
        if kind == "bars":
            ax.bar(["A", "B", "C", "D", "E"], [3, 5, 2, 6, 4], label="2025")
            ax.bar(["A", "B", "C", "D", "E"], [2, 3, 1, 4, 2], label="2024", alpha=0.8)
            ax.set_title("Bars")
            ax.legend()
        elif kind == "box":
            ax.boxplot([rng.normal(m, 1.0, 80) for m in (0, 1, 0.5, 2)], tick_labels=["A", "B", "C", "D"])  # pyright: ignore[reportCallIssue]
            ax.set_title("Box plot")
        elif kind == "hist":
            ax.hist(rng.normal(0, 1, 800), bins=30, alpha=0.8, label="a")
            ax.hist(rng.normal(1.5, 0.7, 800), bins=30, alpha=0.8, label="b")
            ax.set_title("Histogram")
            ax.legend()
        elif kind == "pie":
            ax.pie([35, 25, 20, 12, 8], labels=["A", "B", "C", "D", "E"], autopct="%1.0f%%")
            ax.set_title("Pie")
        elif kind == "heatmap":
            image = ax.imshow(rng.random((12, 16)), aspect="auto")
            fig.colorbar(image, ax=ax)
            ax.set_title("Heatmap")
        else:
            x = np.linspace(0, 2 * np.pi, 200)
            ax.plot(x, np.sin(x), label="sin")
            ax.plot(x, np.cos(x), "--", label="cos")
            ax.scatter(
                x[::8], np.sin(x[::8]) + 0.15 * rng.standard_normal(len(x[::8])),
                s=26, marker="o", alpha=0.8, label="scatter",  # pyright: ignore[reportArgumentType]
            )
            ax.set_title("Lines and scatter")
            ax.legend(loc="best")
        if kind not in ("pie",):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)

    # ======================================================================
    # Style-stack actions
    # ======================================================================
    def _stack_add_builtin(self) -> None:
        name: str = self.builtin.currentText().strip()
        if not name:
            return
        self.style_stack.append(name)
        self.stack.addItem(QListWidgetItem(f"[builtin] {name}"))
        self._on_stack_changed()

    def _stack_add_files(self) -> None:
        paths: list[str]
        paths, _unused = QFileDialog.getOpenFileNames(
            self, "Add Matplotlib Style(s)", "",
            "Matplotlib Style (*.mplstyle);;All Files (*)",
        )
        for p in paths:
            self.style_stack.append(p)
            self.stack.addItem(QListWidgetItem(os.path.abspath(p)))
        if paths:
            self._on_stack_changed()

    def _stack_remove_selected(self) -> None:
        rows: list[int] = sorted(
            {i.row() for i in self.stack.selectedIndexes()}, reverse=True)
        for r in rows:
            self.stack.takeItem(r)
            del self.style_stack[r]
        self._on_stack_changed()

    def _stack_move_up(self) -> None:
        r: int = self.stack.currentRow()
        if r <= 0:
            return
        self.style_stack[r - 1], self.style_stack[r] = (
            self.style_stack[r], self.style_stack[r - 1])
        it = self.stack.takeItem(r)
        self.stack.insertItem(r - 1, it)
        self.stack.setCurrentRow(r - 1)
        self._on_stack_changed()

    def _stack_move_down(self) -> None:
        r: int = self.stack.currentRow()
        if r < 0 or r >= self.stack.count() - 1:
            return
        self.style_stack[r + 1], self.style_stack[r] = (
            self.style_stack[r], self.style_stack[r + 1])
        it = self.stack.takeItem(r)
        self.stack.insertItem(r + 1, it)
        self.stack.setCurrentRow(r + 1)
        self._on_stack_changed()

    def _stack_result_to_table(self) -> None:
        if not self.style_stack:
            show_message(self, "style.stack_empty")
            return
        try:
            with mpl.rc_context():
                mplstyle.use(cast(Any, self.style_stack))
                final = dict(mpl.rcParams)
                defaults = dict(mpl.rcParamsDefault)
                diffs = {
                    k: _to_table_str(final[k])
                    for k in final
                    if str(final[k]) != str(defaults.get(k))
                }
                self._load_dict_into_table(diffs)
                self.status.setText(
                    f"Loaded {len(diffs)} differing parameter(s) from stack.")
                self._preview_timer.start()
        except Exception as ex:
            show_message(self, "style.stack_failed", error=ex)

    def _on_stack_changed(self) -> None:
        n = len(self.style_stack)
        self.status.setText(
            f"Stack updated — {n} {'entry' if n == 1 else 'entries'}.")
        self._preview_timer.start()

    # ======================================================================
    # DictEditor helpers
    # ======================================================================
    def _schema_for_key(self, key: str, value: object = "") -> dict[str, Any]:
        """Build DictEditorPanel metadata for one rcParam key."""
        kind = _rcparam_kind(key)
        meta: dict[str, Any] = {
            "default": value,
            "kind": kind,
            "group": key.split(".", 1)[0] if "." in key else "rcParams",
            "description": str(key),
        }

        if kind == "bool":
            meta["type"] = bool
        elif kind == "number":
            meta["type"] = float
            meta["step"] = 1.0
            meta["decimals"] = 4
        elif kind == "color":
            meta["type"] = str
            meta["kind"] = "color"
        elif kind == "linestyle":
            meta["type"] = str
            meta["kind"] = "linestyle"
        elif kind == "marker":
            meta["type"] = str
            meta["kind"] = "marker"
        elif kind == "joinstyle":
            meta["type"] = _JOINSTYLE_CHOICES
            meta["kind"] = "joinstyle"
            meta["choices"] = _JOINSTYLE_CHOICES
        elif kind == "capstyle":
            meta["type"] = _CAPSTYLE_CHOICES
            meta["kind"] = "capstyle"
            meta["choices"] = _CAPSTYLE_CHOICES
        elif kind == "loc":
            meta["type"] = _LOC_CHOICES
            meta["kind"] = "loc"
            meta["choices"] = _LOC_CHOICES
        elif kind == "enum":
            choices, editable = _rc_enum_choices(key) or ([], False)
            meta["type"] = choices
            meta["kind"] = "enum"
            meta["choices"] = choices
            meta["editable"] = editable
        elif kind == "fontlist":
            meta["type"] = list
            meta["kind"] = "fontlist"
            meta["choices"] = _font_choices(generic=key.lower() == "font.family")
        elif kind == "cmap":
            try:
                choices = [str(name) for name in plt.colormaps()]  # pyright: ignore[reportAttributeAccessIssue]
                choices.sort()
            except Exception:
                choices = []
            meta["type"] = choices
            meta["kind"] = "enum"
            meta["choices"] = choices
        else:
            meta["type"] = str
        return meta

    def _editor_values(self) -> dict[str, object]:
        return self.editor.get_values()

    def _rebuild_editor(self, values: dict[str, object] | None = None) -> None:
        filter_text = self.editor.filter_text()
        current_key = self.editor.current_key()
        self.editor.set_config(self._param_schema)
        if values:
            self.editor.set_values(values)
        self.editor.set_filter_text(filter_text)
        self.editor.set_current_key(current_key)

    def _append_row(self, key: str, value: str) -> None:
        values: dict[str, object] = self._editor_values()
        self._param_schema[key] = self._schema_for_key(key, value)
        values[key] = value
        self._rebuild_editor(values)

    def _load_dict_into_table(self, d: dict[str, str]) -> None:
        self._param_schema = {
            k: self._schema_for_key(k, str(v))
            for k, v in sorted(d.items(), key=lambda kv: kv[0].lower())
        }
        self.editor.set_config(self._param_schema)
        self.editor.set_values(dict(d))
        self.editor.set_filter_text("")


    def load_style_text(self, style_text: str) -> None:
        """Load raw .mplstyle text into the override editor."""
        sanitized_text = _sanitize_mplstyle_text(style_text)
        if not sanitized_text.strip():
            return

        tmp = tempfile.NamedTemporaryFile(
            prefix="style_editor_input_",
            suffix=".mplstyle",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                tmp.write(sanitized_text)
            params = mpl.rc_params_from_file(
                str(tmp_path),
                use_default_template=False,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        self._load_dict_into_table(
            {key: _to_table_str(value) for key, value in params.items()}
        )
        self.status.setText(_("Loaded current style into editor."))
        self._preview_timer.start()

    def _reset_selected_to_defaults(self) -> None:
        key: str = self.editor.current_key() or ""
        if not key:
            show_message(self, "style.no_property_selected")
            return
        dv = cast(Any, mpl.rcParamsDefault).get(key)
        if dv is None:
            return
        value = _to_table_str(dv)
        values: dict[str, object] = self._editor_values()
        values[key] = value
        self.editor.set_values(values)
        self.status.setText(f"Reset '{key}' to default.")
        self._preview_timer.start()

    def _remove_selected_rows(self) -> None:
        key: str = self.editor.current_key() or ""
        if not key:
            return
        values: dict[str, object] = self._editor_values()
        values.pop(key, None)
        self._param_schema.pop(key, None)
        self.editor.set_config(self._param_schema)
        self.editor.set_values(values)
        self.status.setText(f"Removed '{key}'.")
        self._preview_timer.start()

    def _effective_value_for_key(self, key: str) -> object | None:
        """Return the live rcParam value for *key* given current stack + overrides."""
        try:
            with mpl.rc_context():
                if self.style_stack:
                    mplstyle.use(cast(Any, self.style_stack))
                overlay, _unused = self._collect_overrides()
                for k, v in overlay.items():
                    cast(Any, mpl.rcParams)[k] = v
                return cast(Any, mpl.rcParams).get(key)
        except Exception:
            return None

    def _add_param_from_picker(self) -> None:
        key: str = self.rc_picker.selected_key().strip()
        if not key:
            return
        if key in self._param_schema:
            self.editor.set_current_key(key)
            self.status.setText(f"'{key}' is already in the editor.")
            return
        val: object | None = self._effective_value_for_key(key)
        self._append_row(key, _to_table_str(val) if val is not None else "")
        self.editor.set_current_key(key)
        self.status.setText(f"Inserted '{key}'.")
        self._preview_timer.start()

    # ======================================================================
    # Override collection and validation
    # ======================================================================
    def _collect_overrides(
        self,
    ) -> tuple[dict[str, object], OverrideErrors]:
        """Parse editor values and return valid entries plus errors."""
        validator = mpl.RcParams(mpl.rcParams.copy())
        valid: dict[str, object] = {}
        invalid: OverrideErrors = []

        self.editor.commit_pending_edits()
        raw_values = self._editor_values()

        for key, value in raw_values.items():
            key = str(key).strip()
            if not key or key in _NON_STYLE_KEYS:
                continue
            if value is None or value == "":
                continue

            raw = _to_table_str(value).strip()
            try:
                kind = _rcparam_kind(key)
                val_obj: object = value

                if isinstance(value, str):
                    if raw.startswith("cycler("):
                        cycler_pattern = (
                            r"^cycler\(\s*['\"]?(\w*)['\"]?\s*,\s*"
                            r"(\[(?:\s*['\"][^'\"]+['\"]\s*,?)*\])"
                            r"\s*\)$"
                        )
                        m = re.match(cycler_pattern, raw)
                        if not m:
                            applogger.error("Invalid cycler() syntax")
                            return  {}, []
                        prop = m.group(1) or "color"
                        vals = ast.literal_eval(m.group(2))
                        if prop.strip().lower() == "color":
                            vals = [_add_hash(str(v)) for v in vals]
                        val_obj = _cycler.cycler(prop, vals)
                    elif (
                        (raw.startswith("[") and raw.endswith("]"))
                        or (raw.startswith("(") and raw.endswith(")"))
                    ):
                        parsed = ast.literal_eval(raw)
                        if kind == "color":
                            val_obj = _coerce_rc_color(parsed)
                        else:
                            val_obj = parsed
                    elif kind == "fontlist":
                        val_obj = [s.strip() for s in raw.split(",") if s.strip()]
                    elif kind == "bool":
                        val_obj = raw.lower() in ("true", "1", "yes", "on")
                    elif kind == "number":
                        try:
                            val_obj = int(raw)
                        except ValueError:
                            val_obj = float(raw)
                    elif kind == "color":
                        val_obj = _coerce_rc_color(raw)
                elif kind == "color":
                    val_obj = _coerce_rc_color(value)

                cast(Any, validator)[key] = val_obj
                valid[key] = val_obj
            except Exception as ex:
                invalid.append((key, raw, str(ex)))

        # Why here: this is the single funnel every override passes through on
        # its way to the preview, the saved style, and the applied figure.  A
        # usetex entry that survives it makes every text draw raise on a machine
        # without a TeX installation.
        if not latex_available():
            for key in [k for k in valid if is_latex_rcparam(k)]:
                del valid[key]
                invalid.append((key, "", latex_unavailable_reason()))

        return valid, invalid

    # ======================================================================
    # Preview
    # ======================================================================
    def update_preview(self) -> None:
        style_dict: dict[str, object]
        invalid: OverrideErrors
        style_dict, invalid = self._collect_overrides()
        try:
            with mpl.rc_context():
                if self.style_stack:
                    missing = [
                        p for p in self.style_stack
                        if os.path.isabs(p) and not os.path.exists(p)
                    ]
                    if missing:
                        self.status.setText(
                            f"Warning: {len(missing)} stack file(s) missing;"
                            " preview uses available ones.")
                    mplstyle.use(cast(Any, self.style_stack))
                for k, v in style_dict.items():
                    cast(Any, mpl.rcParams)[k] = v

                self._draw_sample(self.canvas.figure)
                self.canvas.draw_idle()

            if invalid:
                last = invalid[-1]
                self.status.setText(
                    f"Preview updated. {len(style_dict)} applied; "
                    f"{len(invalid)} invalid "
                    f"(last: {last[0]} = {last[1]!r})")
                self.status.setToolTip(f"Last error for '{last[0]}':\n{last[2]}")
            else:
                self.status.setText(
                    f"Preview updated — "
                    f"stack: {len(self.style_stack)} "
                    f"overrides: {len(style_dict)}")
                self.status.setToolTip("")
        except Exception as ex:
            self.status.setText(_("Preview failed — see error dialog."))
            show_message(self, "style.preview_failed", error=ex)

    # ======================================================================
    # Save / Load / Apply
    # ======================================================================
    def _save_file(self) -> None:
        path: str
        path, _unused = QFileDialog.getSaveFileName(
            self, _("Save Matplotlib Style"), "",
            "Matplotlib Style (*.mplstyle)")
        if not path:
            return
        if not path.endswith(".mplstyle"):
            path += ".mplstyle"

        style_dict: dict[str, object]
        invalid: OverrideErrors
        style_dict, invalid = self._collect_overrides()
        if invalid:
            show_message(self, "style.invalid_entries", entries=len(invalid))

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "# Generated by MplStyleEditorDialog"
                    " — hex colors saved without '#'\n")
                for k in sorted(style_dict):
                    if k in _NON_STYLE_KEYS:
                        continue
                    f.write(f"{k}: {_to_file_str(style_dict[k])}\n")
            self.status.setText(
                f"Saved {len(style_dict)} parameter(s): "
                f"{os.path.basename(path)}")
        except Exception as ex:
            show_message(self, "style.save_failed", error=ex)

    def _load_from_file(self) -> None:
        path: str
        path, _unused = QFileDialog.getOpenFileName(
            self, _("Open Matplotlib Style"), "",
            "Matplotlib Style (*.mplstyle);;All Files (*)")
        if not path:
            return
        try:
            params = mpl.rc_params_from_file(path, use_default_template=False)
        except Exception as ex:
            show_message(self, "style.load_failed", error=ex)
            return
        self._load_dict_into_table(
            {k: _to_table_str(v) for k, v in params.items()})
        self.status.setText(f"Loaded: {os.path.basename(path)}")
        self._preview_timer.start()

    def _current_merged_style(self) -> dict[str, object]:
        """
        Return a merged rcParams dict (style stack applied, then overrides).
        Non-style keys are filtered out.
        """
        with mpl.rc_context():
            if self.style_stack:
                try:
                    mplstyle.use(cast(Any, self.style_stack))
                except Exception:
                    pass
            overrides: dict[str, object]
            overrides, _unused = self._collect_overrides()
            for k, v in overrides.items():
                try:
                    cast(Any, mpl.rcParams)[k] = v
                except Exception:
                    pass
            merged: dict[str, object] = {k: v for k, v in mpl.rcParams.items()
                      if k not in _NON_STYLE_KEYS}
        return merged

    def _write_temp_style_file(self, style_dict: dict[str, object]) -> str:
        """
        Create a temporary .mplstyle file from style_dict and return its path.
        Uses the same serialization contract as _save_file().
        """
        tmp = tempfile.NamedTemporaryFile(prefix="applied_", suffix=".mplstyle", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()  # reopen with text

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("# Temporary style generated by MplStyleEditorDialog\n")
                f.write("# Hex colors saved without '#'\n")
                for k in sorted(style_dict):
                    if k in _NON_STYLE_KEYS:
                        continue
                    f.write(f"{k}: {_to_file_str(style_dict[k])}\n")
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)  # Python 3.8+
            except Exception:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
            raise
        return str(tmp_path)

    def _on_apply_clicked(self) -> None:
        """
        Build merged rcParams -> write temp .mplstyle -> emit/callback with file path.
        """
        try:
            style_dict: dict[str, object] = self._current_merged_style()
            temp_path: str = self._write_temp_style_file(style_dict)
        except Exception as ex:
            show_message(self, "style.apply_failed", error=ex)
            return

        # Emit signal with path
        try:
            self.styleApplied.emit(temp_path)
        except Exception:
            pass

        # Call user callback with path
        if callable(self.apply_callback):
            try:
                self.apply_callback(temp_path)
            except Exception as ex:
                show_message(self, "style.apply_callback_failed", error=ex)
                return

        self.status.setText(f"Applied style written to: {temp_path}")


def _sanitize_mplstyle_text(style_text: str) -> str:
    """Return safe .mplstyle text suitable for Matplotlib parsing.

    The editor can receive raw mplstyle text from files, text boxes, or stored
    figure options. Matplotlib rc files support keys that are undesirable or
    unsafe in an embedded chart-preview/editor context, for example backend,
    toolbar, interactive, and webagg runtime settings. This helper removes
    those runtime-only keys while preserving comments, blank lines, and normal
    style declarations.
    """
    forbidden_keys = {
        "backend",
        "backend_fallback",
        "interactive",
        "toolbar",
        "webagg.address",
        "webagg.open_in_browser",
        "webagg.port",
        "webagg.port_retries",
    }

    out_lines: list[str] = []
    for line in (style_text or "").splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        # Skip include/import-like directives. They make a stored style depend
        # on external files and can break preview/apply when paths move.
        if stripped.startswith("@"):
            continue

        if ":" in stripped:
            key = stripped.split(":", 1)[0].strip().lower()
            if key in forbidden_keys:
                continue

        out_lines.append(line)

    sanitized = "\n".join(out_lines).replace("\r\n", "\n").replace("\r", "\n")
    return sanitized.rstrip("\n") + "\n"
