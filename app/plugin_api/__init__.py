"""What a plugin imports: the base classes and helpers of each plugin family.

Charts, data operations and fit functions are plugins - a
file dropped into a family's folder (app/<family>/ or user/<family>/) is the
whole install, and deleting it the whole removal. What they share lives here,
outside those folders, so that nothing in a plugin folder is anything but a
plugin (or belongs to one):

    app.plugin_api.charts            BaseAxisRenderer, render_figure, grids, kwarg_spec ...
    app.plugin_api.data_operations   DataOperationDialogBase, RolesOperationDialogBase (a table's columns
                                     cast into roles), QuickAnswerDialogBase, parameters, results; column
                                     roles, table profile, next steps, guided plan (the roles' helpers)
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
    "functions": ("base", "monte_carlo", "optimizers", "starting_point"),
}

#: old dotted name -> new dotted name.
OLD_TO_NEW: dict[str, str] = {
    f"app.{family}.{name}": f"app.plugin_api.{family}.{name}"
    for family, names in MOVED.items()
    for name in names
}
# The table operations' bases are the data operations' since one engine runs both (ID48).
OLD_TO_NEW.update({
    "app.table_operations.dialog_base": "app.plugin_api.data_operations.roles_base",
    "app.plugin_api.table_operations.dialog_base": "app.plugin_api.data_operations.roles_base",
    "app.plugin_api.table_operations.quick_answer": "app.plugin_api.data_operations.quick_answer",
})
# Their helpers too: app.plugin_api.table_operations is gone, its modules in app.plugin_api.data_operations.
for _name in ("chart_source", "column_roles", "guided_plan", "next_steps", "table_profile"):
    OLD_TO_NEW[f"app.table_operations.{_name}"] = f"app.plugin_api.data_operations.{_name}"
    OLD_TO_NEW[f"app.plugin_api.table_operations.{_name}"] = f"app.plugin_api.data_operations.{_name}"
# The packages themselves, so that their submodules can be found under them.
OLD_TO_NEW["app.table_operations"] = "app.data_operations"
OLD_TO_NEW["app.plugin_api.table_operations"] = "app.plugin_api.data_operations"


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
            # A submodule of a moved package (app.data_operations.dialog_base.common): the longest old name wins.
            for old in sorted(OLD_TO_NEW, key=len, reverse=True):
                if fullname.startswith(old + "."):
                    new_name = OLD_TO_NEW[old] + fullname[len(old):]
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
