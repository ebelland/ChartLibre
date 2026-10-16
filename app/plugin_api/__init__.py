"""What a plugin imports: the base classes and helpers of each plugin family.

Charts, data operations, table operations and fit functions are plugins - a
file dropped into a family's folder (app/<family>/ or user/<family>/) is the
whole install, and deleting it the whole removal. What they share lives here,
outside those folders, so that nothing in a plugin folder is anything but a
plugin (or belongs to one):

    app.plugin_api.charts            BaseAxisRenderer, render_figure, grids, kwarg_spec ...
    app.plugin_api.data_operations   DataOperationDialogBase, parameters, results
    app.plugin_api.table_operations  TableOperationDialogBase, column roles, table profile
    app.plugin_api.functions         base_function, optimizers, starting points

These modules used to live inside the family folders. Plugins written then -
the user's own, in user/ - import them by those paths, so the old names still
resolve, to the same module objects, and log a warning naming the new path.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

#: family -> modules that moved into app.plugin_api.<family>.
MOVED: dict[str, tuple[str, ...]] = {
    "charts": ("axis_options", "base", "descriptor_grid", "grids", "kwarg_spec", "layout_presets", "render_figure"),
    "data_operations": ("dialog_base", "parameter_form", "parameter_spec", "results"),
    "table_operations": ("column_roles", "dialog_base", "table_profile"),
    "functions": ("base", "monte_carlo", "optimizers", "starting_point"),
}

#: old dotted name -> new dotted name.
OLD_TO_NEW: dict[str, str] = {
    f"app.{family}.{name}": f"app.plugin_api.{family}.{name}"
    for family, names in MOVED.items()
    for name in names
}


class _Alias(importlib.abc.Loader):
    """Hands back the module already imported under its new name."""

    def __init__(self, new_name: str) -> None:
        self._new_name = new_name

    def create_module(self, spec):  # noqa: ANN001
        return importlib.import_module(self._new_name)

    def exec_module(self, module) -> None:  # noqa: ANN001
        """Nothing to run: the module is the real one."""


class _OldPathFinder(importlib.abc.MetaPathFinder):
    """Resolves ``app.charts.base`` and the like to their new place."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        new_name = OLD_TO_NEW.get(fullname)
        if new_name is None:
            # A submodule of a moved package (app.data_operations.dialog_base.common).
            for old, new in OLD_TO_NEW.items():
                if fullname.startswith(old + "."):
                    new_name = new + fullname[len(old):]
                    break
        if new_name is None:
            return None
        from app.logs.logger import applogger

        applogger.warning("%s moved to %s; update the import of this plugin.", fullname, new_name)
        return importlib.util.spec_from_loader(fullname, _Alias(new_name))


def install_old_path_aliases() -> None:
    """Make the old import paths work; once."""
    if not any(isinstance(finder, _OldPathFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _OldPathFinder())
