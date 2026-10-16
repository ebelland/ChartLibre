"""
Static discovery and dynamic loading of data operation dialogs.

Dialog classes are found by AST-scanning ``app/data_operations`` - and,
for an operation the Data Operation Builder (Developer menu) scaffolded,
``user/data_operations`` alongside it, see
``app.utils.config.USER_DATA_OPERATIONS_DIR`` - at import time. The
classes themselves are loaded from disk on demand and cached.
"""

from __future__ import annotations

from pathlib import Path

from app.scanners.class_discovery import (
    discover_classes_merged,
    import_class_from_discovery_entry,
)
from app.utils.config import USER_DATA_OPERATIONS_DIR


def import_class_from_file(operation: dict):
    """
    Return the data operation dialog class described by a discovery entry.
    """
    return import_class_from_discovery_entry(
        operation,
        module_prefix="_dynamic_data_operation",
    )


def _discover_data_operations() -> list:
    """
    Scan app/data_operations, then user/data_operations, for classes
    that directly inherit from DataOperationDialogBase - built-in first,
    so a name collision (which the Data Operation Builder already
    refuses to create) resolves to the built-in one rather than silently
    shadowing it.
    """
    builtin_root = Path(__file__).resolve().parent.parent / "data_operations"

    return discover_classes_merged(
        roots=(builtin_root, USER_DATA_OPERATIONS_DIR),
        base_class_name="DataOperationDialogBase",
        value_attr="Name",
        string_attrs=(
            "Description",
            # The SVG source itself, read as a literal like any other string
            # attribute.  Nothing here has to know it is markup: the widget
            # hands it to style.icon_from_svg_source and Qt does the rest.
            "Icon",
            "Keywords",
        ),
        string_list_attrs=("Goals",),
        require_value_attr=False,
    )


# Scanned once at import time; the operation set is fixed for the process.
data_operations: list[dict] = _discover_data_operations()