"""Remember what the user last typed in a dialog, and where a window was.

``import_data_dialog`` already persisted its entries by hand: one read function,
one write function, and a literal key per field.  Doing that for every dialog
would mean a hundred more lines that all say the same thing and all drift.

Instead this module walks a dialog's own attributes - ``self._table_combo``,
``self._skip_rows_spin`` - and persists whichever are input widgets it knows
how to read.  The attribute name is the storage key, so the JSON stays legible
and a renamed attribute simply forgets its old value instead of crashing.

    class MyDialog(QDialog):
        def __init__(...):
            ...
            restore_dialog_state(self, "my_dialog")   # after the widgets exist

        def accept(self):
            save_dialog_state(self, "my_dialog")
            super().accept()

Widgets are read and written through :data:`_ACCESSORS`; anything not listed
there is ignored, which is what keeps a table view or a chart canvas from
ending up in config.json.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QWidget,
)

from app.logs.logger import applogger
from app.utils.config import get_section, set_section

CONFIG_SECTION = "dialog_state"
GEOMETRY_SECTION = "window_geometry"

# Widget type -> (read, write).  Order matters: the first isinstance match
# wins, so subclasses must be listed before their bases (QAbstractButton is
# last because QCheckBox, QRadioButton and QPushButton are all buttons).
_ACCESSORS: list[tuple[type, Callable[[Any], Any], Callable[[Any, Any], None]]] = [
    (QComboBox, lambda w: w.currentText(), lambda w, v: _set_combo(w, v)),
    (QLineEdit, lambda w: w.text(), lambda w, v: w.setText(str(v))),
    (QPlainTextEdit, lambda w: w.toPlainText(), lambda w, v: w.setPlainText(str(v))),
    (QSpinBox, lambda w: w.value(), lambda w, v: w.setValue(int(v))),
    (QDoubleSpinBox, lambda w: w.value(), lambda w, v: w.setValue(float(v))),
    (QSlider, lambda w: w.value(), lambda w, v: w.setValue(int(v))),
    (QTabWidget, lambda w: w.currentIndex(), lambda w, v: w.setCurrentIndex(int(v))),
    (QSplitter, lambda w: w.sizes(), lambda w, v: _set_splitter(w, v)),
    (QAbstractButton, lambda w: w.isChecked(), lambda w, v: w.setChecked(bool(v))),
]


def _set_combo(combo: QComboBox, value: Any) -> None:
    """Select *value* in a combo, or type it when the combo is editable.

    A remembered entry whose row no longer exists - a table that was dropped,
    a renderer that was renamed - must not clear the user's other choices, so
    a miss is simply left alone.
    """
    text = str(value)
    index = combo.findText(text)
    if index >= 0:
        combo.setCurrentIndex(index)
    elif combo.isEditable():
        combo.setEditText(text)


def _set_splitter(splitter: QSplitter, value: Any) -> None:
    """Restore splitter sizes, ignoring a stale list of the wrong length."""
    if isinstance(value, list) and len(value) == splitter.count():
        splitter.setSizes([int(size) for size in value])


def _stateful_widgets(owner: object) -> dict[str, QWidget]:
    """Return ``attribute name -> widget`` for every readable widget on *owner*.

    Only the object's own attributes are considered - not the whole child tree
    - because an attribute name is stable and meaningful while a generated
    child's position is neither.
    """
    found: dict[str, QWidget] = {}
    for name, value in vars(owner).items():
        if not isinstance(value, QWidget):
            continue
        if any(isinstance(value, widget_type) for widget_type, _, _ in _ACCESSORS):
            found[name.lstrip("_")] = value
    return found


def _accessor(widget: QWidget) -> tuple[Callable[[Any], Any], Callable[[Any, Any], None]] | None:
    """Return the read/write pair for a widget, or None if it has none."""
    for widget_type, read, write in _ACCESSORS:
        if isinstance(widget, widget_type):
            return read, write
    return None


# ----------------------------------------------------------------------
# Dialog entries
# ----------------------------------------------------------------------
def dialog_entries(owner: object, *, inputs_only: bool = False) -> dict[str, Any]:
    """The current entries of *owner*, by attribute name, as save_dialog_state stores them.

    ``inputs_only`` leaves out what only arranges the window (splitter
    sizes, the tab shown) and the buttons that act rather than hold a choice: what an
    operation records about how it was run.
    """
    state: dict[str, Any] = {}
    for name, widget in _stateful_widgets(owner).items():
        if inputs_only and (
            isinstance(widget, (QSplitter, QTabWidget))
            or (isinstance(widget, QAbstractButton) and not widget.isCheckable())
        ):
            continue
        accessor = _accessor(widget)
        if accessor is None:
            continue
        try:
            state[name] = accessor[0](widget)
        except Exception:  # noqa: BLE001 - any widget, and closing must never fail
            applogger.exception(
                # A dialog closing must never raise a modal error box.
                "Could not read %s.%s", type(owner).__name__, name,
            )
    return state


def save_dialog_state(owner: object, key: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Store the current entries of *owner* under ``dialog_state.<key>``.

    *extra* is stored beside them: what is not an attribute widget - a form's
    parameters, a choice of columns - under names no attribute has, read back
    with :func:`stored_dialog_state`. Returns what was stored, which is handy
    in tests and costs nothing.
    """
    state = dialog_entries(owner)
    state.update(extra or {})
    section = get_section(CONFIG_SECTION)
    section[key] = state
    set_section(CONFIG_SECTION, section)
    return state


def restore_dialog_state(owner: object, key: str) -> None:
    """Re-apply the entries stored under ``dialog_state.<key>``.

    Call this once the widgets exist and have been populated: restoring a
    combo before its rows are loaded would silently select nothing.  A field
    that fails to restore is logged and skipped, never raised, because a stale
    config file must not stop a dialog from opening.
    """
    state = get_section(CONFIG_SECTION).get(key)
    if not isinstance(state, dict):
        return
    set_dialog_entries(owner, state)


def stored_dialog_state(key: str) -> dict[str, Any]:
    """What is stored under ``dialog_state.<key>``: entries, and whatever a dialog added."""
    state = get_section(CONFIG_SECTION).get(key)
    return dict(state) if isinstance(state, dict) else {}


def set_dialog_entries(owner: object, state: Mapping[str, Any], *, quietly: bool = True) -> None:
    """Put *state*'s values into *owner*'s widgets, by attribute name.

    *quietly* blocks the widgets' signals, as restoring a window does; a
    Revert lets them through, so whatever hangs on a combo follows it.
    """
    widgets = _stateful_widgets(owner)
    for name, value in state.items():
        widget = widgets.get(name)
        accessor = _accessor(widget) if widget is not None else None
        if widget is None or accessor is None:
            continue
        try:
            if quietly:
                widget.blockSignals(True)
            accessor[1](widget, value)
        except Exception:  # noqa: BLE001 - any widget, any stale value: skipped, never fatal
            applogger.exception(
                "Could not restore %s.%s", type(owner).__name__, name,
            )
        finally:
            if quietly:
                widget.blockSignals(False)


# ----------------------------------------------------------------------
# Window geometry
# ----------------------------------------------------------------------
def save_window_geometry(window: QWidget, key: str) -> None:
    """Remember the position, size and maximised state of a window.

    Stored as plain numbers rather than as Qt's ``saveGeometry()`` blob so the
    values stay readable and editable in config.json.  While maximised the
    restored size is read from ``normalGeometry`` instead, so closing a
    maximised window still records somewhere sensible to come back to.
    """
    normal = window.normalGeometry() if window.isMaximized() else window.geometry()
    section = get_section(GEOMETRY_SECTION)
    section[key] = {
        "x": int(normal.x()),
        "y": int(normal.y()),
        "width": int(normal.width()),
        "height": int(normal.height()),
        "maximized": bool(window.isMaximized()),
    }
    set_section(GEOMETRY_SECTION, section)


def restore_window_geometry(window: QWidget, key: str) -> bool:
    """Re-apply a remembered geometry; returns False when there is none.

    The saved position is only honoured if it still lands on a screen: a window
    restored onto a monitor that has since been unplugged would be invisible
    and unreachable.
    """
    state = get_section(GEOMETRY_SECTION).get(key)
    if not isinstance(state, dict):
        return False

    width = int(state.get("width", 0) or 0)
    height = int(state.get("height", 0) or 0)
    if width > 0 and height > 0:
        window.resize(width, height)

    x, y = state.get("x"), state.get("y")
    if x is not None and y is not None and _is_on_a_screen(window, int(x), int(y)):
        window.move(int(x), int(y))

    if bool(state.get("maximized", False)):
        window.showMaximized()
    return True


def _is_on_a_screen(window: QWidget, x: int, y: int) -> bool:
    """Return True when (x, y) falls inside one of the available screens."""
    from PySide6.QtGui import QGuiApplication

    for screen in QGuiApplication.screens():
        if screen.availableGeometry().contains(x, y):
            return True
    return False


def clear_state(key: str) -> None:
    """Forget the entries and geometry stored for one key."""
    for section_name in (CONFIG_SECTION, GEOMETRY_SECTION):
        section = get_section(section_name)
        if section.pop(key, None) is not None:
            set_section(section_name, section)


__all__ = [
    "clear_state",
    "restore_dialog_state",
    "restore_window_geometry",
    "save_dialog_state",
    "save_window_geometry",
]
