"""Renderer Helper: scaffold a new chart-type renderer file (todo.txt P3-6).

A chart type lives entirely in one file under app/charts/ (built in) or
user/charts/ (user-authored, kept out of the app's own source tree - see
app.utils.config.USER_CHARTS_DIR for why), as a class that directly
subclasses BaseAxisRenderer - axis_renderer_scanner.py AST-scans both
folders at import time and picks it up with no registration step. This
dialog is a short form for the handful of fields a renderer's class body
needs (Name, Category, Description, Link, required/optional roles) that
writes a new, immediately-importable file into user/charts/ with those
fields already filled in and a render_axis stub left for the user to write
the actual drawing code into. See BaseAxisRenderer's own docstring
(app/charts/base.py) for the full contract, and app/charts/text.py for a
short, complete renderer to read alongside it.
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFormLayout, QLineEdit, QWidget

from app.developer_helpers.developer_helper_base import DeveloperDialogBase
from app.utils.config import USER_CHARTS_DIR
from app.utils.i18n import _

#: Where axis_renderer_scanner.py itself looks for the built-in renderers -
#: used here only to check a new Name against them, never written to.
BUILTIN_CHARTS_DIR: Path = Path(__file__).resolve().parent.parent / "charts"

#: Where this dialog writes - a user-authored renderer, kept separate from
#: the app's own shipped source. Created on first use.
CHARTS_DIR: Path = USER_CHARTS_DIR

#: Offered in the Category combo, existing families first (Matplotlib's own
#: taxonomy - see BaseAxisRenderer.Category) with "User" - the sensible
#: catch-all for a renderer that doesn't fit one of them - pre-selected.
EXISTING_CATEGORIES: tuple[str, ...] = (
    "User",
    "Pairwise data",
    "Statistical distributions",
    "Diagnostic plots",
    "Gridded data",
    "Irregularly gridded data",
    "3D and volumetric data",
)


def _parse_roles(text: str) -> list[str]:
    """Split a comma/whitespace-separated roles field into clean names."""
    return [role for role in re.split(r"[,\s]+", text.strip()) if role]


def _render_axis_body(required_roles: list[str]) -> str:
    """A working stub if the roles look like x/y, else a TODO placeholder.

    A renderer whose required roles are exactly the x/y everyone reaches
    for first gets an actual ax.plot() - so the new chart type visibly
    works the moment it is picked, instead of drawing nothing until the
    TODO is addressed. Anything else is too renderer-specific to guess at.
    """
    if "x" in required_roles and "y" in required_roles:
        return (
            "        for sd in valid_series:\n"
            "            ax.plot(\n"
            "                sd.df[\"x\"],\n"
            "                sd.df[\"y\"],\n"
            "                label=sd.name,\n"
            "                **self.get_kwargs(axis_options),\n"
            "            )\n"
        )
    hint_role = required_roles[0] if required_roles else "x"
    return (
        "        for sd in valid_series:\n"
        f"            # TODO: draw sd onto ax. sd.df[\"{hint_role}\"] holds the\n"
        f"            # \"{hint_role}\" role's column - every name in RequiredRoles/\n"
        "            # OptionalRoles is a column of sd.df, aliased there by the\n"
        "            # series' own SQL query (SELECT ... AS " + hint_role + ").\n"
        "            pass\n"
    )


def _roles_literal(roles: list[str]) -> str:
    return "[" + ", ".join(repr(role) for role in roles) + "]"


def render_stub_source(
    *,
    class_name: str,
    name: str,
    category: str,
    description: str,
    link: str,
    required_roles: list[str],
    optional_roles: list[str],
) -> str:
    """Return the full source text of the scaffolded renderer file."""
    role_lines = "\n".join(
        f"    {role}   {'required' if role in required_roles else 'optional'}, TODO: what it holds"
        for role in (*required_roles, *optional_roles)
    ) or "    (none declared)"

    return f'''"""{name} renderer.

TODO: describe what this chart type draws and when to reach for it, the
way every renderer under app/charts/ does at the top of its own file -
app/charts/text.py is a short, complete example to read alongside this
one, and BaseAxisRenderer's own docstring (app/charts/base.py) documents
every class attribute below.

Role columns:
{role_lines}

This file was scaffolded by the Renderer Helper (Developer menu). It is
already a valid, discoverable renderer - axis_renderer_scanner.py picks up
any class in app/charts/ that directly subclasses BaseAxisRenderer, with
no registration step - but render_axis below only draws a placeholder.
Fill it in with real Matplotlib calls, then delete this paragraph.
"""
from __future__ import annotations

from typing import Any

from app.charts.base import BaseAxisRenderer, SeriesData


class {class_name}(BaseAxisRenderer):
    """TODO: one line describing this chart type."""

    Name: str = {name!r}
    Category: str = {category!r}
    Description: str = {description!r}
    Link: str = {link!r}

    RequiredRoles: list[str] = {_roles_literal(required_roles)}
    OptionalRoles: list[str] = {_roles_literal(optional_roles)}

    #: Matplotlib keyword arguments forwarded verbatim to the plot call, as
    #: {{name: metadata}}. See BaseAxisRenderer.Kwargs' own docstring
    #: (app/charts/base.py) for the metadata shape ("default", "type",
    #: "min"/"max", "kind", "group", "description") that drives the
    #: generated editor UI automatically - a new keyword needs no widget code.
    Kwargs: dict[str, object] = {{}}

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {{}}
        valid_series = self.valid_series(series)
        if not valid_series:
            return

{_render_axis_body(required_roles)}'''


class RendererHelperDialog(DeveloperDialogBase):
    """Scaffold a new BaseAxisRenderer file from a short form."""

    TITLE = "Renderer Helper"
    ICON = "renderer_helper"
    CARD_NAME = "rendererHelperCard"
    HINT = (
        "Writes a new file under user/charts/ with a class already "
        "filled in from these fields - no registration step, the "
        "scanner picks it up as soon as the file exists. render_axis "
        "is left for you to write the actual drawing code into."
    )
    NAME_PLACEHOLDER = "e.g. Ridgeline Plot"
    FILE_PLACEHOLDER = "e.g. ridgeline.py"
    DESCRIPTION_PLACEHOLDER = "One line, shown in the chart picker."
    CLASS_SUFFIX = "AxisRenderer"
    CREATE_ACTION = "create_renderer"
    BUILTIN_DIR = BUILTIN_CHARTS_DIR
    USER_DIR = CHARTS_DIR
    BASE_CLASS_NAME = "BaseAxisRenderer"
    CREATED_MESSAGE = "dev.renderer_created"
    FAILED_MESSAGE = "dev.renderer_create_failed"
    BAD_FILE_NAME = "File name must be a valid Python file name, e.g. my_chart.py."
    FILE_EXISTS = "A file named \"{file}\" already exists under user/charts/."
    NAME_TAKEN = "\"{name}\" is already used by an existing chart type."
    NOT_DISCOVERED = (
        "The class imported, but axis_renderer_scanner did not discover it as "
        "a renderer - check that it subclasses BaseAxisRenderer directly."
    )

    def add_fields(self, form: QFormLayout, card: QWidget) -> None:
        self._category_combo = QComboBox(card)
        self._category_combo.setEditable(True)
        self._category_combo.addItems(EXISTING_CATEGORIES)
        self._category_combo.setCurrentText("User")
        form.addRow(_("Category:"), self._category_combo)

        super().add_fields(form, card)

        self._link_edit = QLineEdit(card)
        self._link_edit.setPlaceholderText(_("Matplotlib documentation URL (optional)"))
        form.addRow(_("Link:"), self._link_edit)

        self._required_roles_edit = QLineEdit(card)
        self._required_roles_edit.setText("x, y")
        self._required_roles_edit.setPlaceholderText(_("comma-separated, e.g. x, y"))
        form.addRow(_("Required roles:"), self._required_roles_edit)

        self._optional_roles_edit = QLineEdit(card)
        self._optional_roles_edit.setPlaceholderText(_("comma-separated, optional"))
        form.addRow(_("Optional roles:"), self._optional_roles_edit)

    def validate_fields(self) -> str:
        if not _parse_roles(self._required_roles_edit.text()):
            return _("At least one required role is needed.")
        return ""

    def render_source(self, *, class_name: str, name: str, description: str) -> str:
        return render_stub_source(
            class_name=class_name,
            name=name,
            category=self._category_combo.currentText().strip() or "User",
            description=description,
            link=self._link_edit.text().strip(),
            required_roles=_parse_roles(self._required_roles_edit.text()),
            optional_roles=_parse_roles(self._optional_roles_edit.text()),
        )
