"""Discovery of table operations, as series_operation_scanner discovers series operations.

Classes directly subclassing TableOperationDialogBase in app/table_operations
(and user/table_operations, for the user's own) are found by an AST scan at
import time and loaded from disk when opened.
"""
from __future__ import annotations

from pathlib import Path

from app.scanners.class_discovery import discover_classes_merged, import_class_from_discovery_entry
from app.utils.config import USER_CONTENT_DIR

#: Where the user's own table operations live, beside the series operations.
USER_TABLE_OPERATIONS_DIR: Path = USER_CONTENT_DIR / "table_operations"


def import_class_from_file(operation: dict):
    """The table operation dialog class a discovery entry describes."""
    return import_class_from_discovery_entry(operation, module_prefix="_dynamic_table_operation")


def _discover_table_operations() -> list[dict]:
    builtin_root = Path(__file__).resolve().parent.parent / "table_operations"
    return discover_classes_merged(
        roots=(builtin_root, USER_TABLE_OPERATIONS_DIR),
        base_class_name="TableOperationDialogBase",
        value_attr="Name",
        string_attrs=("Description", "Category", "Icon"),
        string_list_attrs=(),
        require_value_attr=False,
    )


# Scanned once at import time; the operation set is fixed for the process.
table_operations: list[dict] = _discover_table_operations()
