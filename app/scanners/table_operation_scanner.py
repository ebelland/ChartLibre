"""Discovery of table operations (the Analysis panel's plugins).

Classes directly subclassing TableOperationDialogBase in app/table_operations
(and user/table_operations, for the user's own) are found by an AST scan at
import time and loaded from disk when opened. How a family is found is
app.scanners.plugin_family's; this declares what this one is.
"""
from __future__ import annotations

from pathlib import Path

from app.scanners.plugin_family import PluginFamily
from app.utils.config import USER_CONTENT_DIR

#: Where the user's own table operations live, beside the data operations.
USER_TABLE_OPERATIONS_DIR: Path = USER_CONTENT_DIR / "table_operations"

FAMILY = PluginFamily(
    label="table operations (analyses)",
    base_class="TableOperationDialogBase",
    builtin_root=Path(__file__).resolve().parent.parent / "table_operations",
    user_root=USER_TABLE_OPERATIONS_DIR,
    module_prefix="_dynamic_table_operation",
    string_attrs=("Description", "Category", "Icon", "Keywords"),
    string_list_attrs=("Goals",),
)


def import_class_from_file(operation: dict):
    """The table operation dialog class a discovery entry describes."""
    return FAMILY.load(operation)


# Scanned once at import time; the operation set is fixed for the process.
table_operations: list[dict] = FAMILY.discover()
