"""Layout scale and platform: margins, spacings, sizes, the paths and the platform the UI is on."""
from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QKeySequence
from app.utils.config import get_constant


# This module is in app/styles/style/; the UI directory is app/styles.
_UI_DIR = Path(__file__).resolve().parent.parent
_APP_DIR = _UI_DIR.parent
_STYLES_DIR = _APP_DIR / "styles"
_IS_MACOS = platform.system().lower() == "darwin"
_IS_WINDOWS = platform.system().lower().startswith("win")

#: Public alias, for the few callers outside this module that need the
#: platform check but have no reason to reach into a private module constant
#: (main_window.py's menu bar, at the moment).
IS_MACOS: bool = _IS_MACOS
IS_WINDOWS: bool = _IS_WINDOWS

#: Qt::ExpandedClientAreaHint (Qt 6.9+): content drawn under the native
#: title bar. PySide6 lists the value only under its deprecated alias
#: MaximizeUsingFullscreenGeometryHint, so it is built from the number.
EXPANDED_CLIENT_AREA_HINT: Qt.WindowType = Qt.WindowType(0x00400000)

MenuCallback = Callable[..., Any]
ShortcutLike = QKeySequence.StandardKey | QKeySequence | str | int | None


# ----------------------------------------------------------------------
# Layout scale
# ----------------------------------------------------------------------
# One scale for the whole app.  Before this existed there were seven different
# margin conventions across 58 hand-written call sites, so two sibling dialogs
# could not be made to line up without editing both.
SPACING_DEFAULT: int = 8

MARGIN_CARD: tuple[int, int, int, int] = (10, 10, 10, 10)
MARGIN_DIALOG: tuple[int, int, int, int] = (12, 12, 12, 12)
MARGIN_PANEL: tuple[int, int, int, int] = (6, 6, 6, 6)

# Canonical dialog sizes, keyed by the role a dialog plays.  Anything that needs
# a size picks one of these instead of inventing another number.
DIALOG_SIZE_SMALL: QSize = QSize(*get_constant("dialog_size_small", [560, 360]))
DIALOG_SIZE_MEDIUM: QSize = QSize(*get_constant("dialog_size_medium", [900, 640]))
DIALOG_SIZE_LARGE: QSize = QSize(*get_constant("dialog_size_large", [1020, 700]))

DIALOG_SIZES: dict[str, QSize] = {
    "small": DIALOG_SIZE_SMALL,
    "medium": DIALOG_SIZE_MEDIUM,
    "large": DIALOG_SIZE_LARGE,
}

# How many characters a combo box is guaranteed to show.  This is the single
# biggest driver of how narrow a side panel can be made: a combo asking for 24
# characters puts a ~200 px floor under every panel that contains one, which
# is what used to pin the main window's left panel at about 300 px however the
# splitter was dragged.  Six characters plus elision keeps the panel free to
# shrink; the full value stays readable through the combo's tooltip.
COMBO_MIN_CONTENTS_LENGTH: int = get_constant("combo_min_contents_length", 6)

# Room for the drop-down arrow and the frame, on top of the text itself.
_COMBO_CHROME_WIDTH: int = 34

# Splitter handle: wide enough to grab without aiming, narrow enough not to
# read as a divider.
SPLITTER_HANDLE_WIDTH: int = get_constant("splitter_handle_width", 6)

# Floor for the resizable side panels.  Not zero: a panel dragged to nothing
# looks like a bug and cannot be grabbed again.
PANEL_MIN_WIDTH: int = get_constant("panel_min_width", 180)

@dataclass(frozen=True, slots=True)
class PlatformStyle:
    """Resolved platform styling information."""

    system: str
    qss_file: Path | None


@dataclass(slots=True)
class MenuItem:
    """Typed descriptor for an item created by ``create_menu``."""

    text: str
    tooltip: str = ""
    shortcut: ShortcutLike = None
    callback: MenuCallback | None = None
    checkable: bool = False
    # A str is looked up through load_icon (action id, SVG name, or Fluent
    # glyph token), same as before. A QIcon is used as-is - for a menu item
    # that has to render pixel-identical to an icon built elsewhere from raw
    # SVG source (icon_from_svg_source), such as a plugin's own artwork,
    # rather than to whatever a same-named catalogue/file token resolves to.
    icon: str | QIcon | None = None
    action_id: str | None = None
    checked: bool = False
    enabled: bool = True
    #: Nested items, for an entry that opens a submenu rather than doing
    #: something - "Open recent" and its list of files. A submenu item's
    #: own callback is never called: Qt gives the entry to the submenu.
    submenu: "list[MenuItem | None] | None" = None


# ----------------------------------------------------------------------
