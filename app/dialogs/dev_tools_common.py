"""Shared plumbing for the Developer menu's code-scaffolding tools.

Renderer Helper, Series Operation Builder and Function Creator all do the
same four things - turn a display name into a file/class name, write a
templated .py file into the matching user/ folder (kept out of the app's
own shipped source, see app.utils.config.USER_CONTENT_DIR), import it back
fresh to confirm it is both syntactically valid and the shape its scanner
looks for, and open it in the system's default editor - so that sequence
lives here once rather than three times with three chances to drift.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from app.scanners.class_discovery import discover_classes_merged


def slug(text: str) -> str:
    """A lowercase_with_underscores fragment from arbitrary text."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip()).strip("_")
    return cleaned.lower()


def class_name(text: str, *, suffix: str, fallback: str = "Custom") -> str:
    """A PascalCase<suffix> class name from arbitrary text.

    Prefixed with *fallback* when the text has no letters to start an
    identifier with (e.g. "3D Thing") - a class name may not start with a
    digit.
    """
    parts = re.split(r"[^0-9a-zA-Z]+", text.strip())
    camel = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not camel or not camel[0].isalpha():
        camel = f"{fallback}{camel}"
    return f"{camel}{suffix}"


def discover_both_roots(
    *,
    builtin_root: Path,
    user_root: Path,
    base_class_name: str,
    value_attr: str | None = "Name",
    require_value_attr: bool = True,
) -> list[dict]:
    """Fresh-scan a built-in folder and its user/ counterpart together.

    Never reads a scanner's own module-level cache, built once at import
    time - that snapshot would miss a file a dialog using this just wrote
    moments earlier in the same run. A thin, dialog-facing wrapper around
    class_discovery.discover_classes_merged, which every scanner
    (axis_renderer_scanner.py and its siblings) uses the same way for its
    own module-level cache.
    """
    return discover_classes_merged(
        roots=(builtin_root, user_root),
        base_class_name=base_class_name,
        value_attr=value_attr,
        require_value_attr=require_value_attr,
    )


def open_in_editor(path: Path) -> None:
    """Open *path* with the system's default handler for a .py file."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def import_check(path: Path, expected_class_name: str) -> str:
    """Import *path* fresh and confirm *expected_class_name* is in it.

    Returns an empty string on success, else a message describing what
    went wrong, meant to be shown to the user verbatim (it is deliberately
    not translated - it is either a Python exception's own message or a
    file-shape problem naming the exact class not found).
    """
    module_name = f"_dev_tool_check_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return "Could not create an import spec for the new file."

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
        return f"{type(exc).__name__}: {exc}"
    finally:
        sys.modules.pop(module_name, None)

    if getattr(module, expected_class_name, None) is None:
        return f'The file imported, but class "{expected_class_name}" was not found in it.'
    return ""
