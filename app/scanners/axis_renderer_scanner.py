"""
Static discovery and dynamic loading of axis renderers.

Renderer classes are found by AST-scanning ``app/charts`` - and, for a
renderer the Renderer Helper (Developer menu) scaffolded, ``user/charts``
alongside it, see ``app.utils.config.USER_CHARTS_DIR`` - at import time.
The classes themselves are loaded from disk on demand and cached.
"""

from __future__ import annotations

from pathlib import Path

from app.logs.logger import applogger
from app.scanners.class_discovery import (
    discover_classes_merged,
    import_class_from_discovery_entry,
)
from app.utils.config import USER_CHARTS_DIR


# Chart types that have been renamed, old name -> current name.
#
# Why this exists: chart_type is stored per axis in the database, so renaming a
# renderer orphans figures already built with the old name unless aliases are
# resolved here.
CHART_TYPE_ALIASES: dict[str, str] = {
    "Horizontal Histogram": "Histogram",
}


def import_class_from_file(renderer: dict):
    """
    Return the renderer class described by a discovery entry.
    """
    return import_class_from_discovery_entry(
        renderer,
        module_prefix="_dynamic_axis_renderer",
    )


def get_renderer(name: str) -> dict | None:
    """
    Return the discovery entry whose ``Name`` matches, else None.

    Renamed chart types are resolved through CHART_TYPE_ALIASES, so figures
    saved under a previous name keep rendering.
    """
    wanted = str(name or "").strip()

    entry = next(
        (renderer for renderer in renderers if renderer["value"] == wanted),
        None,
    )

    if entry is not None:
        return entry

    alias = CHART_TYPE_ALIASES.get(wanted)

    if alias is None:
        return None

    applogger.info(
        "Chart type %r was renamed to %r; resolving through the alias table.",
        wanted,
        alias,
    )

    return next(
        (renderer for renderer in renderers if renderer["value"] == alias),
        None,
    )


def _discover_axis_renderers() -> list:
    """
    Scan app/charts, then user/charts, for classes that directly inherit
    from BaseAxisRenderer - built-in first, so a name collision (which the
    Renderer Helper already refuses to create) resolves to the built-in
    one rather than silently shadowing it.
    """
    builtin_root = Path(__file__).resolve().parent.parent / "charts"

    return discover_classes_merged(
        roots=(builtin_root, USER_CHARTS_DIR),
        base_class_name="BaseAxisRenderer",
        value_attr="Name",
        string_attrs=(
            "Category",
            "Description",
        ),
        string_list_attrs=(
            "RequiredRoles",
            "OptionalRoles",
        ),
        require_value_attr=True,
    )


# Scanned once at import time; the renderer set is fixed for the process.
renderers: list[dict] = _discover_axis_renderers()