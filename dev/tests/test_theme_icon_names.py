"""Every ThemeIcon name in config.json is one an icon theme can resolve.

A typo in a theme name is silent: QIcon.fromTheme returns a null icon and
the button falls back to its SVG, or is blank. So each configured name must
be one Qt itself standardises (QIcon.ThemeIcon, read from Qt so the list
grows with it) or one of the extra names below.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PySide6.QtGui import QIcon

CONFIG = Path(__file__).resolve().parents[2] / "config.json"

#: Names outside Qt's enum that the application uses anyway. Each one is in
#: the freedesktop Icon Naming Specification or is shipped by every major
#: icon theme (Adwaita, Breeze, Papirus) - which is the bar for adding one.
EXTRA_THEME_ICON_NAMES: frozenset[str] = frozenset(
    {
        "accessories-calculator",
        "applications-system",
        "document-edit",
        # The user manual: freedesktop's own name for "open the help
        # document", which is exactly what this action does.
        "help-contents",
        # Connecting to a server database: the freedesktop Status icon for
        # network activity, the closest standard name to what this does.
        "network-transmit-receive",
        # No Qt ThemeIcon enum member covers "fetch this from the web" -
        # GoDown is a plain navigation arrow, not a download. Breeze,
        # Papirus and Adwaita all ship this Icon Naming Specification
        # emblem, originally for a downloads folder.
        "emblem-downloads",
        "object-select",
        # No icon theme has a chart, and a plotting application's Plot button
        # is the one place a drawing of our own is the honest answer. Breeze
        # and Papirus ship this name; Adwaita does not, and there the SVG in
        # app/icons takes over - which is the fallback doing its job rather
        # than a gap.
        "office-chart-line",
        "open-menu",
        "preferences-system",
        "system-run",
        # Likewise: Breeze has it, Adwaita dropped it, and "the rows matching
        # a condition" has no better standard name.
        "view-filter",
        "view-sort-ascending",
        # Import from another database: the same MIME-type naming family as
        # "x-office-spreadsheet" above, this one for LibreOffice Base's .odb.
        "x-office-database",
        "x-office-spreadsheet",
        # Developer-menu stubs (todo.txt P3-4/P3-5/P3-6): freedesktop Icon
        # Naming Specification categories, not covered by Qt's own smaller
        # ThemeIcon enum.
        "applications-development",
        "applications-graphics",
        "preferences-desktop-locale",
        # Added to config.json while this check was not running, and found
        # when it was brought back: each one is in the freedesktop Icon
        # Naming Specification or ships with Breeze, which is enough for the
        # themed icon to be tried before the application's own drawing.
        "applications-internet",
        "document-export",
        "edit-find-replace",
        "object-flip-horizontal",
        "text-x-generic",
        "view-grid",
        "view-hidden",
        "view-list-details",
        "view-list-tree",
        "view-more",
        "view-sidebar",
        "view-sort-descending",
        "view-statistics",
        "view-visible",
    }
)


def _standard_theme_icon_names() -> set[str]:
    """Every freedesktop name QIcon.ThemeIcon knows: DocumentOpen -> document-open."""
    return {
        re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
        for name in QIcon.ThemeIcon.__members__
    }


def _configured_names(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ThemeIcon":
                found.update([value] if isinstance(value, str) else list(value))
            else:
                found |= _configured_names(value)
    elif isinstance(node, list):
        for value in node:
            found |= _configured_names(value)
    return found


def test_every_configured_theme_icon_name_is_known() -> None:
    names = _configured_names(json.loads(CONFIG.read_text(encoding="utf-8")))
    assert names, "no ThemeIcon names found in config.json"
    unknown = sorted(names - _standard_theme_icon_names() - EXTRA_THEME_ICON_NAMES)
    assert unknown == [], f"not a known theme icon name: {unknown}"
