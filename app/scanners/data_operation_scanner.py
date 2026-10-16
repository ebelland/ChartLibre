"""
Static discovery and dynamic loading of data operation dialogs.

Dialog classes are found by AST-scanning ``app/data_operations`` - and,
for an operation the Data Operation Builder (Developer menu) scaffolded,
``user/data_operations`` alongside it, see
``app.utils.config.USER_DATA_OPERATIONS_DIR`` - at import time. The
classes themselves are loaded from disk on demand and cached. How a family
is found is app.scanners.plugin_family's; this declares what this one is.
"""

from __future__ import annotations

from pathlib import Path

from app.scanners.plugin_family import PluginFamily
from app.utils.config import USER_DATA_OPERATIONS_DIR

#: Classes that directly inherit from DataOperationDialogBase, built-in first,
#: so a name collision (which the Data Operation Builder already refuses to
#: create) resolves to the built-in one rather than silently shadowing it.
FAMILY = PluginFamily(
    label="data operations",
    base_class="DataOperationDialogBase",
    builtin_root=Path(__file__).resolve().parent.parent / "data_operations",
    user_root=USER_DATA_OPERATIONS_DIR,
    module_prefix="_dynamic_data_operation",
    # Icon is the SVG source itself, read as a literal like any other string
    # attribute: nothing here has to know it is markup (the widget hands it to
    # style.icon_from_svg_source).
    string_attrs=("Description", "Icon", "Keywords", "Category"),
    string_list_attrs=("Goals",),
    int_attrs=("Order",),
)


def import_class_from_file(operation: dict):
    """
    Return the data operation dialog class described by a discovery entry.
    """
    return FAMILY.load(operation)


# Scanned once at import time; the operation set is fixed for the process.
data_operations: list[dict] = FAMILY.discover()
