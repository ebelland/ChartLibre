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

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.dev_tools_common import (
    class_name as _class_name,
    discover_both_roots,
    import_check as _generic_import_check,
    open_in_editor,
    slug as _slug,
)
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils.config import USER_CHARTS_DIR
from app.utils.i18n import _
from app.utils.messages import show_message

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


class RendererHelperDialog(QDialog):
    """Scaffold a new BaseAxisRenderer file from a short form."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Renderer Helper"))
        self.setWindowIcon(load_icon("renderer_helper"))
        self.setModal(True)

        root = QVBoxLayout(self)
        # None, not "medium" - see series_operation_builder_dialog's own
        # note on why a forced 900x640 read as loose gaps between every
        # row rather than as a window that was simply too big: with every
        # child here at stretch 0, Qt spreads unclaimed surplus roughly
        # evenly between them instead of leaving it in one place. Sizing
        # to the layout's own sizeHint removes the surplus this card
        # never asked for.
        apply_dialog_shell(self, root, size=None)

        card = CardFrame(self, "rendererHelperCard")
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_("Renderer Helper"), card))

        hint = QLabel(
            _(
                "Writes a new file under user/charts/ with a class already "
                "filled in from these fields - no registration step, the "
                "scanner picks it up as soon as the file exists. render_axis "
                "is left for you to write the actual drawing code into."
            ),
            card,
        )
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._name_edit = QLineEdit(card)
        self._name_edit.setPlaceholderText(_("e.g. Ridgeline Plot"))
        self._name_edit.textEdited.connect(self._on_name_edited)
        form.addRow(_("Name:"), self._name_edit)

        self._file_edit = QLineEdit(card)
        self._file_edit.setPlaceholderText(_("e.g. ridgeline.py"))
        self._file_edit.textEdited.connect(self._on_file_edited_by_user)
        form.addRow(_("File name:"), self._file_edit)

        self._category_combo = QComboBox(card)
        self._category_combo.setEditable(True)
        self._category_combo.addItems(EXISTING_CATEGORIES)
        self._category_combo.setCurrentText("User")
        form.addRow(_("Category:"), self._category_combo)

        self._description_edit = QLineEdit(card)
        self._description_edit.setPlaceholderText(
            _("One line, shown in the chart picker.")
        )
        form.addRow(_("Description:"), self._description_edit)

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

        card_layout.addLayout(form)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        create_action_button(
            parent=self,
            action_id="create_renderer",
            action=self._on_create,
            layout=action_row,
        )
        card_layout.addLayout(action_row)
        # Belt and braces: if this dialog is ever resized past its content,
        # the surplus collapses here rather than spreading back out
        # between the rows above.
        card_layout.addStretch(1)

        root.addWidget(card, 1)

        self._file_edited_by_user = False

    # ------------------------------------------------------------------
    # Auto-deriving the file name from the chart's Name
    # ------------------------------------------------------------------
    def _on_name_edited(self, text: str) -> None:
        if self._file_edited_by_user:
            return
        self._file_edit.setText(f"{_slug(text)}.py" if text.strip() else "")

    def _on_file_edited_by_user(self, _text: str) -> None:
        self._file_edited_by_user = True

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def _on_create(self) -> None:
        problem = self._validate()
        if problem:
            show_message(self, "dev.validation_error", detail=problem)
            return

        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        class_name = _class_name(name, suffix="AxisRenderer")
        category = self._category_combo.currentText().strip() or "User"
        description = self._description_edit.text().strip()
        link = self._link_edit.text().strip()
        required_roles = _parse_roles(self._required_roles_edit.text())
        optional_roles = _parse_roles(self._optional_roles_edit.text())

        source = render_stub_source(
            class_name=class_name,
            name=name,
            category=category,
            description=description,
            link=link,
            required_roles=required_roles,
            optional_roles=optional_roles,
        )

        target = CHARTS_DIR / file_name
        try:
            CHARTS_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        except OSError as exc:
            applogger.exception("Could not write the new renderer file.")
            show_message(self, "dev.validation_error", detail=str(exc))
            return

        error = _import_check(target, class_name)
        if error:
            show_message(
                self,
                "dev.renderer_create_failed",
                path=str(target),
                error=error,
            )
            return

        applogger.info(
            "Renderer Helper: wrote %s (%s, chart_type=%r).", target, class_name, name
        )
        show_message(self, "dev.renderer_created", name=name, path=str(target))
        open_in_editor(target)
        self.accept()

    def _validate(self) -> str:
        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        description = self._description_edit.text().strip()
        required_roles = _parse_roles(self._required_roles_edit.text())

        if not name:
            return _("Name cannot be empty.")
        if not description:
            return _("Description cannot be empty.")
        if not file_name.endswith(".py") or not _slug(file_name[:-3]):
            return _("File name must be a valid Python file name, e.g. my_chart.py.")
        if (CHARTS_DIR / file_name).exists():
            return _("A file named \"{file}\" already exists under user/charts/.").format(
                file=file_name
            )
        if not required_roles:
            return _("At least one required role is needed.")
        if _renderer_name_taken(name):
            return _(
                "\"{name}\" is already used by an existing chart type."
            ).format(name=name)
        return ""


def _renderer_name_taken(name: str) -> bool:
    """True when an existing renderer, built-in or user's own, already
    declares this Name."""
    entries = discover_both_roots(
        builtin_root=BUILTIN_CHARTS_DIR,
        user_root=CHARTS_DIR,
        base_class_name="BaseAxisRenderer",
    )
    return any(entry["value"] == name for entry in entries)


def _import_check(path: Path, class_name: str) -> str:
    """Import *path* fresh and confirm it discovers as a renderer.

    Returns an empty string on success, else a message describing what
    went wrong - the same two-step check todo.txt's own design sketch
    asks for: the file must both import cleanly (dev_tools_common's own
    generic check) and be the shape axis_renderer_scanner actually looks
    for (a class directly subclassing BaseAxisRenderer, naming Name as a
    plain string literal).
    """
    error = _generic_import_check(path, class_name)
    if error:
        return error

    entries = discover_both_roots(
        builtin_root=BUILTIN_CHARTS_DIR,
        user_root=CHARTS_DIR,
        base_class_name="BaseAxisRenderer",
    )
    if not any(entry["name"] == class_name for entry in entries):
        return _(
            "The class imported, but axis_renderer_scanner did not discover it as "
            "a renderer - check that it subclasses BaseAxisRenderer directly."
        )
    return ""
