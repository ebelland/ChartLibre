"""
Generic static class discovery and dynamic class loading.

This module is intentionally independent from axis renderers, chart renderers,
data operation dialogs, or any other plugin-like class family.

It provides:
- AST-based discovery of classes that directly inherit from a named base class.
- Static extraction of literal class attributes.
- Cached dynamic class loading from disk.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.logs.logger import applogger


@lru_cache(maxsize=None)
def _load_class(file_path: str, class_name: str, mtime_ns: int, module_prefix: str):
    """
    Import ``class_name`` from ``file_path`` exactly once per file revision.

    ``mtime_ns`` is part of the cache key only. It makes the cache miss when the
    file changes on disk.
    """
    del mtime_ns  # cache key only

    path = Path(file_path)
    module_name = f"{module_prefix}_{path.stem}_{abs(hash(file_path))}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        applogger.error(f"Could not create import spec for: {path}.")
        return None

    module = importlib.util.module_from_spec(spec)

    # Register so that relative imports and module-level singletons behave
    # consistently during exec_module.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - a plugin file: any failure is its own, and reported
        applogger.error(
            "Class %s failed to load from %s (%s: %s).",
            class_name,
            path,
            type(exc).__name__,
            exc,
        )
        sys.modules.pop(module_name, None)
        return None

    cls = getattr(module, class_name, None)

    if cls is None:
        applogger.error(
            f"File '{path}' loaded as module '{module_name}', "
            f"but class '{class_name}' was not found."
        )
        return None

    if not isinstance(cls, type):
        applogger.error(
            f"'{class_name}' found in '{path}', but it is not a class "
            f"(got {type(cls)!r})."
        )
        return None

    return cls


@lru_cache(maxsize=None)
def _resolved_path(raw_path: str) -> Path:
    """Resolve a discovery entry's path once per raw string, not per call.

    The renderer/operation file *set* does not move while the process runs -
    only a file's own content can, which the mtime check below re-reads on
    every call so a file the Developer-menu scaffolding tools just wrote (or
    a plugin author just edited) hot-reloads without a restart. Resolving
    the same path string again on every render is pure syscall overhead on a
    value that never changes for it.
    """
    return Path(raw_path).resolve()


#: What ast.literal_eval raises for an expression that is not a literal.
_NOT_A_LITERAL: tuple[type[Exception], ...] = (ValueError, TypeError, SyntaxError, RecursionError)

def import_class_from_discovery_entry(
    entry: dict[str, Any],
    *,
    module_prefix: str = "_dynamic_discovered",
):
    """
    Return the class described by a discovery entry.

    Required entry keys:
        - path
        - name
    """
    file_path = _resolved_path(str(entry["path"]))

    try:
        mtime_ns = file_path.stat().st_mtime_ns
    except OSError:
        applogger.error(f"{file_path} not found.")
        return None

    return _load_class(
        str(file_path),
        str(entry["name"]),
        mtime_ns,
        module_prefix,
    )


def extract_class_string_attr(
    class_node: ast.ClassDef,
    attr_name: str,
) -> str | None:
    """
    Extract a class-level string attribute from a class body.

    Accepted:
        Name = "Scatter"
        Name: str = "Scatter"

    Rejected:
        Name = SOME_CONST
        Name = 123
    """
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    try:
                        value = ast.literal_eval(stmt.value)
                    except _NOT_A_LITERAL:
                        return None
                    return value if isinstance(value, str) else None

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                if stmt.value is None:
                    return None
                try:
                    value = ast.literal_eval(stmt.value)
                except _NOT_A_LITERAL:
                    return None
                return value if isinstance(value, str) else None

    return None


def extract_class_int_attr(
    class_node: ast.ClassDef,
    attr_name: str,
) -> int | None:
    """
    Extract a class-level int attribute from a class body.

    Accepted:
        Order = 20
        Order: int = 20

    Rejected: anything that is not an int literal (a bool included).
    """
    for stmt in class_node.body:
        target, value_node = None, None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value_node = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target, value_node = stmt.target, stmt.value
        if isinstance(target, ast.Name) and target.id == attr_name and value_node is not None:
            try:
                value = ast.literal_eval(value_node)
            except _NOT_A_LITERAL:
                return None
            return value if isinstance(value, int) and not isinstance(value, bool) else None
    return None


def extract_class_string_list_attr(
    class_node: ast.ClassDef,
    attr_name: str,
) -> list[str] | None:
    """
    Extract a class-level list[str] attribute from a class body.

    Accepted:
        RequiredRoles = ["x", "y"]
        RequiredRoles: list[str] = ["x", "y"]

    Rejected:
        RequiredRoles = ("x", "y")
        RequiredRoles = ["x", 1]
        RequiredRoles = SOME_CONST
    """
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    return literal_string_list(stmt.value)

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                if stmt.value is None:
                    return None
                return literal_string_list(stmt.value)

    return None


def extract_class_number_list_attr(class_node: ast.ClassDef, attr_name: str) -> list[float] | None:
    """A class-level literal list (or tuple) of numbers - a fit function's ``p0 = [1.0, 0.5]`` - else None."""
    for stmt in class_node.body:
        target, value_node = None, None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value_node = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target, value_node = stmt.target, stmt.value
        if isinstance(target, ast.Name) and target.id == attr_name and value_node is not None:
            try:
                value = ast.literal_eval(value_node)
            except _NOT_A_LITERAL:
                return None
            if isinstance(value, (list, tuple)) and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
                return [float(item) for item in value]
            return None
    return None


def literal_string_list(node: ast.AST) -> list[str] | None:
    """
    Return a literal list[str] from an AST node, else None.
    """
    try:
        value = ast.literal_eval(node)
    except _NOT_A_LITERAL:
        return None

    if not isinstance(value, list):
        return None

    if not all(isinstance(item, str) for item in value):
        return None

    return value


def directly_inherits_from(
    class_node: ast.ClassDef,
    base_class_name: str,
) -> bool:
    """
    Return True if class_node directly inherits from base_class_name.

    Strictly matches:
        class X(BaseClass):

    Also supports:
        class X(module.BaseClass):
    """
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == base_class_name:
            return True

        if isinstance(base, ast.Attribute) and base.attr == base_class_name:
            return True

    return False


def discover_classes(
    *,
    root: Path,
    base_class_name: str,
    value_attr: str | None = "Name",
    string_attrs: tuple[str, ...] = (),
    string_list_attrs: tuple[str, ...] = (),
    int_attrs: tuple[str, ...] = (),
    number_list_attrs: tuple[str, ...] = (),
    require_value_attr: bool = True,
    skip_init: bool = True,
) -> list[dict[str, Any]]:
    """
    Discover classes under root that directly inherit from base_class_name.

    Returns entries shaped like:
        {
            "name": class name,
            "path": file path as posix string,
            "value": value_attr content or class name,
            ...
        }
    """
    discovered: list[dict[str, Any]] = []

    if not root.is_dir():
        return discovered

    for path in root.rglob("*.py"):
        if skip_init and path.name == "__init__.py":
            continue

        src = path.read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            if not directly_inherits_from(node, base_class_name):
                continue

            value = None

            if value_attr is not None:
                value = extract_class_string_attr(node, value_attr)

                if require_value_attr and value is None:
                    continue

            entry: dict[str, Any] = {
                "name": node.name,
                "path": path.as_posix(),
                "value": value if value is not None else node.name,
            }

            for attr_name in string_attrs:
                entry[attr_name_to_key(attr_name)] = extract_class_string_attr(
                    node,
                    attr_name,
                )

            for attr_name in string_list_attrs:
                entry[attr_name_to_key(attr_name)] = extract_class_string_list_attr(
                    node,
                    attr_name,
                )

            for attr_name in int_attrs:
                entry[attr_name_to_key(attr_name)] = extract_class_int_attr(node, attr_name)

            for attr_name in number_list_attrs:
                entry[attr_name_to_key(attr_name)] = extract_class_number_list_attr(node, attr_name)

            discovered.append(entry)

    return discovered


def discover_classes_merged(
    *,
    roots: tuple[Path, ...],
    base_class_name: str,
    value_attr: str | None = "Name",
    string_attrs: tuple[str, ...] = (),
    string_list_attrs: tuple[str, ...] = (),
    int_attrs: tuple[str, ...] = (),
    number_list_attrs: tuple[str, ...] = (),
    require_value_attr: bool = True,
    family: str = "",
) -> list[dict[str, Any]]:
    """Discover across several roots at once, earlier roots winning ties.

    For a plugin family split across a built-in folder and a user one (see
    app.utils.config.USER_CHARTS_DIR and its siblings) - pass the built-in
    root first so a name collision resolves to the built-in entry rather
    than silently shadowing it with a user one. A root that does not exist
    (the user folder, before anything has ever been scaffolded into it)
    contributes nothing rather than raising - see discover_classes' own
    handling of a missing root.
    """
    discovered: list[dict[str, Any]] = []
    seen_values: set[Any] = set()
    for root in roots:
        for entry in discover_classes(
            root=root,
            base_class_name=base_class_name,
            value_attr=value_attr,
            string_attrs=string_attrs,
            string_list_attrs=string_list_attrs,
            int_attrs=int_attrs,
            number_list_attrs=number_list_attrs,
            require_value_attr=require_value_attr,
        ):
            if entry["value"] in seen_values:
                applogger.warning(
                    "%s '%s' in %s is shadowed by an earlier one with the "
                    "same name; rename it to make it visible.",
                    base_class_name,
                    entry["value"],
                    entry["path"],
                )
                continue
            seen_values.add(entry["value"])
            discovered.append(entry)
    if family and not discovered:
        # Plugins are files that can be removed: a family left with none is an
        # installation problem, and the feature that lists them will be empty.
        applogger.error(
            "No %s found (classes deriving from %s in %s); the application will have none.",
            family,
            base_class_name,
            ", ".join(str(root) for root in roots),
        )
    return discovered


def attr_name_to_key(attr_name: str) -> str:
    """
    Convert class attribute names to discovery dictionary keys.

    Examples:
        Description -> description
        RequiredRoles -> required
        OptionalRoles -> optional
    """
    explicit = {
        "Name": "value",
        "Description": "description",
        "RequiredRoles": "required",
        "OptionalRoles": "optional",
    }

    if attr_name in explicit:
        return explicit[attr_name]

    return attr_name[:1].lower() + attr_name[1:]