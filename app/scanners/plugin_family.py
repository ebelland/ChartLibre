"""One plugin family: where its classes live, what they declare, how they are found and loaded.

Charts (axis renderers), data operations, table operations, importers and
fit functions are the same kind of thing: classes deriving from a base class,
found by an AST scan of a built-in folder and a user one, their metadata read
from literal class attributes, the class itself imported only when opened. A family is declared
here by its parameters - nothing else in the application needs to know how
they are found, and adding a family is one more declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.scanners.class_discovery import discover_classes_merged, import_class_from_discovery_entry


@dataclass(frozen=True, slots=True)
class PluginFamily:
    """The declaration of a plugin family."""

    #: What the family is called in the log: "data operations".
    label: str
    #: The base class its plugins derive from, by name.
    base_class: str
    #: Where the built-in plugins are; the user's own folder is scanned after it.
    builtin_root: Path
    #: None: the built-in folder alone (a test's own library).
    user_root: Path | None
    #: Prefix of the module name a plugin file is loaded under.
    module_prefix: str
    #: The attributes read from each class, as entry keys (class_discovery.attr_name_to_key).
    string_attrs: tuple[str, ...] = ()
    string_list_attrs: tuple[str, ...] = ()
    int_attrs: tuple[str, ...] = ()
    #: Literal lists of numbers (a fit function's starting values).
    number_list_attrs: tuple[str, ...] = ()
    #: The attribute that names a plugin: "Name", or "name" for the fit functions.
    value_attr: str = "Name"
    #: A plugin without a Name is listed under its class name, unless this is set.
    require_name: bool = False

    def discover(self) -> list[dict[str, Any]]:
        """Every plugin found, built-in first; an error is logged when there is none."""
        return discover_classes_merged(
            roots=tuple(root for root in (self.builtin_root, self.user_root) if root is not None),
            base_class_name=self.base_class,
            value_attr=self.value_attr,
            string_attrs=self.string_attrs,
            string_list_attrs=self.string_list_attrs,
            int_attrs=self.int_attrs,
            number_list_attrs=self.number_list_attrs,
            require_value_attr=self.require_name,
            family=self.label,
        )

    def load(self, entry: dict[str, Any]):
        """The plugin class a discovery entry describes, imported from its file."""
        return import_class_from_discovery_entry(entry, module_prefix=self.module_prefix)
