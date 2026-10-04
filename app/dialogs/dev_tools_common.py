"""Shared plumbing for the Developer menu's code-scaffolding tools.

Renderer Helper, Series Operation Builder and Function Creator all do the
same four things - turn a display name into a file/class name, write a
templated .py file into the matching user/ folder (kept out of the app's
own shipped source, see app.utils.config.USER_CONTENT_DIR), import it back
fresh to confirm it is both syntactically valid and the shape its scanner
looks for, and open it in the system's default editor - so that sequence
lives here once rather than three times with three chances to drift.
:class:`ScaffoldDialog` is the window they share.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from app.logs.logger import applogger
from app.scanners.class_discovery import discover_classes_merged
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.utils.messages import show_message


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


class ScaffoldDialog(QDialog):
    """The form all three tools share: Name, File name, Description, Create.

    A subclass names its folders, scanner base class and texts as class
    attributes, adds its own rows in :meth:`add_fields`, checks them in
    :meth:`validate_fields`, and writes the file's text in
    :meth:`render_source`. Everything else - the file name following the
    Name until edited by hand, the checks every tool makes, writing the file,
    importing it back and opening it - happens here once.

    Texts are kept untranslated here and passed through ``_`` when shown,
    so they follow the language chosen at run time.
    """

    #: Window title (also the card's title), icon and the card's object name.
    TITLE: ClassVar[str] = ""
    ICON: ClassVar[str] = ""
    CARD_NAME: ClassVar[str] = ""
    #: The paragraph under the title.
    HINT: ClassVar[str] = ""
    NAME_PLACEHOLDER: ClassVar[str] = ""
    FILE_PLACEHOLDER: ClassVar[str] = ""
    DESCRIPTION_PLACEHOLDER: ClassVar[str] = ""
    #: What a file name derived from the Name ends with ("_dialog.py", ".py").
    FILE_SUFFIX: ClassVar[str] = ".py"
    #: The class name's suffix ("Function", "AxisRenderer"...).
    CLASS_SUFFIX: ClassVar[str] = ""
    #: The Create button's action id in config.json.
    CREATE_ACTION: ClassVar[str] = ""
    #: Where the scanner finds the built-in ones, and where this writes.
    BUILTIN_DIR: ClassVar[Path] = Path()
    USER_DIR: ClassVar[Path] = Path()
    #: What the new class must directly subclass, and the attribute its
    #: scanner reads the name from.
    BASE_CLASS_NAME: ClassVar[str] = ""
    VALUE_ATTR: ClassVar[str] = "Name"
    REQUIRE_VALUE_ATTR: ClassVar[bool] = True
    #: Messages (config.json) for created and failed, and the texts of the
    #: checks that name this kind of file.
    CREATED_MESSAGE: ClassVar[str] = ""
    FAILED_MESSAGE: ClassVar[str] = ""
    BAD_FILE_NAME: ClassVar[str] = ""
    FILE_EXISTS: ClassVar[str] = ""
    NAME_TAKEN: ClassVar[str] = ""
    NOT_DISCOVERED: ClassVar[str] = ""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_(self.TITLE))
        self.setWindowIcon(load_icon(self.ICON))
        self.setModal(True)

        root = QVBoxLayout(self)
        # None, not "medium": a title, a hint and a few fields are far short
        # of the 900x640 every "medium" dialog gets, and with every child of
        # the card at stretch 0, Qt spreads the surplus evenly between them
        # rather than leaving it at the bottom - a forced oversize read as
        # loose gaps between every row. Sized to the layout's own sizeHint,
        # there is no surplus.
        apply_dialog_shell(self, root, size=None)

        card = CardFrame(self, self.CARD_NAME)
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_(self.TITLE), card))
        hint = QLabel(_(self.HINT), card)
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        form = QFormLayout()
        stdSizeAndlayout(form)
        self._name_edit = QLineEdit(card)
        self._name_edit.setPlaceholderText(_(self.NAME_PLACEHOLDER))
        self._name_edit.textEdited.connect(self._on_name_edited)
        form.addRow(_("Name:"), self._name_edit)
        self._file_edit = QLineEdit(card)
        self._file_edit.setPlaceholderText(_(self.FILE_PLACEHOLDER))
        self._file_edit.textEdited.connect(self._on_file_edited_by_user)
        form.addRow(_("File name:"), self._file_edit)
        self._description_edit = QLineEdit(card)
        self._description_edit.setPlaceholderText(_(self.DESCRIPTION_PLACEHOLDER))
        self.add_fields(form, card)
        card_layout.addLayout(form)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(parent=self, action_id="close", action=self.reject, layout=action_row)
        create_action_button(parent=self, action_id=self.CREATE_ACTION, action=self._on_create, layout=action_row)
        card_layout.addLayout(action_row)
        # Should the window be dragged larger, the surplus collects here
        # rather than spreading back out between the rows above.
        card_layout.addStretch(1)
        root.addWidget(card, 1)

        self._file_edited_by_user = False

    # -- What a subclass supplies -----------------------------------------

    def add_fields(self, form: QFormLayout, card: QWidget) -> None:
        """Add the Description row (``self._description_edit``) and any others, in order."""
        form.addRow(_("Description:"), self._description_edit)

    def validate_fields(self) -> str:
        """A message for what is wrong with this tool's own fields, or ''."""
        return ""

    def render_source(self, *, class_name: str, name: str, description: str) -> str:
        """The new file's text."""
        raise NotImplementedError

    # -- The file name follows the Name until edited by hand ---------------

    def _on_name_edited(self, text: str) -> None:
        if self._file_edited_by_user:
            return
        self._file_edit.setText(f"{slug(text)}{self.FILE_SUFFIX}" if text.strip() else "")

    def _on_file_edited_by_user(self, _text: str) -> None:
        self._file_edited_by_user = True

    # -- Create -----------------------------------------------------------

    def _on_create(self) -> None:
        problem = self._validate()
        if problem:
            show_message(self, "dev.validation_error", detail=problem)
            return

        name = self._name_edit.text().strip()
        new_class = class_name(name, suffix=self.CLASS_SUFFIX)
        source = self.render_source(
            class_name=new_class, name=name, description=self._description_edit.text().strip()
        )
        target = self.USER_DIR / self._file_edit.text().strip()
        try:
            self.USER_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        except OSError as exc:
            applogger.exception("%s: could not write %s.", self.TITLE, target)
            show_message(self, "dev.validation_error", detail=str(exc))
            return

        error = self._import_check(target, new_class)
        if error:
            show_message(self, self.FAILED_MESSAGE, path=str(target), error=error)
            return

        applogger.info("%s: wrote %s (%s, name=%r).", self.TITLE, target, new_class, name)
        show_message(self, self.CREATED_MESSAGE, name=name, path=str(target))
        open_in_editor(target)
        self.accept()

    def _validate(self) -> str:
        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        if not name:
            return _("Name cannot be empty.")
        if not self._description_edit.text().strip():
            return _("Description cannot be empty.")
        if not file_name.endswith(".py") or not slug(file_name[:-3]):
            return _(self.BAD_FILE_NAME)
        if (self.USER_DIR / file_name).exists():
            return _(self.FILE_EXISTS).format(file=file_name)
        problem = self.validate_fields()
        if problem:
            return problem
        if any(entry["value"] == name for entry in self._discovered()):
            return _(self.NAME_TAKEN).format(name=name)
        return ""

    def _discovered(self) -> list[dict]:
        return discover_both_roots(
            builtin_root=self.BUILTIN_DIR,
            user_root=self.USER_DIR,
            base_class_name=self.BASE_CLASS_NAME,
            value_attr=self.VALUE_ATTR,
            require_value_attr=self.REQUIRE_VALUE_ATTR,
        )

    def _import_check(self, path: Path, new_class: str) -> str:
        """Import *path* fresh, then confirm its scanner finds *new_class* in it."""
        error = import_check(path, new_class)
        if error:
            return error
        if not any(entry["name"] == new_class for entry in self._discovered()):
            return _(self.NOT_DISCOVERED)
        return ""
