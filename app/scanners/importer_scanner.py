"""Discovery of the file importers (todo I-01).

Classes directly subclassing FileImporter in app/importers (and user/importers,
for a lab's own formats) are found by an AST scan and loaded from disk when a
file of theirs is opened. How a family is found is app.scanners.plugin_family's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.scanners.plugin_family import PluginFamily
from app.utils.config import USER_CONTENT_DIR

USER_IMPORTERS_DIR: Path = USER_CONTENT_DIR / "importers"

FAMILY = PluginFamily(
    label="file importers",
    base_class="FileImporter",
    builtin_root=Path(__file__).resolve().parent.parent / "importers",
    user_root=USER_IMPORTERS_DIR,
    module_prefix="_dynamic_importer",
    string_attrs=("Description", "Category"),
    string_list_attrs=("Suffixes", "Packages"),
    int_attrs=("Order",),
)

# Scanned once at import time; the set is fixed for the process.
importers: list[dict[str, Any]] = FAMILY.discover()


def importer_for(path: str | Path) -> dict[str, Any] | None:
    """The importer whose suffixes hold *path*'s - a user's own before a built-in one."""
    suffix = Path(path).suffix.lower()
    found = [entry for entry in importers if suffix in [str(s).lower() for s in entry.get("suffixes") or ()]]
    return found[-1] if found else None


def load_importer(entry: dict[str, Any]):
    """An instance of the importer an entry describes."""
    return FAMILY.load(entry)()


def importer_suffixes() -> list[str]:
    """Every suffix an importer reads."""
    return sorted({str(s).lower() for entry in importers for s in entry.get("suffixes") or ()})
