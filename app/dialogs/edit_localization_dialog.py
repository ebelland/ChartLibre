"""Edit Localization: a table editor for one language's .po catalogue
(todo.txt P3-5).

Every row is a source string ``_()``/``tr()`` wraps somewhere in the app
(``i18n.source_translator_calls()`` - the same AST sweep
app/tests/test_localization.py's own safety net runs) unioned with
whatever the selected language's catalogue already has, so a string that
is translated but no longer used anywhere still shows up to edit or
remove by hand, and a string used but never added to the catalogue shows
up with an empty translation rather than silently failing the next full
test run instead. "Missing only" filters to exactly that empty-
translation case - what would otherwise only surface at
`python -m pytest app/tests/test_localization.py`.

Saving writes through :func:`app.utils.i18n._write_po` (round-trips
losslessly with :func:`app.utils.i18n._parse_po`, see that function's own
docstring) and recompiles the .mo immediately via
:func:`app.utils.i18n.compile_catalog`, so a translation shows up the next
time that language is selected without restarting the app.

"Translate missing (auto)" is a soft dependency on the ``translate``
PyPI package (a thin wrapper around the MyMemory translation API, no key
needed) - offered only when it is importable, and always a starting
point to review, never treated as a finished translation.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.settings_dialog import LANGUAGE_NAMES
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils import i18n
from app.utils.i18n import _
from app.utils.messages import show_message

#: A catalogue this new is seeded with every known source string and
#: nothing else - see _default_header.
_NEW_CATALOG_HEADER_TEMPLATE = (
    "Project-Id-Version: ChartLibre\n"
    "Language: {code}\n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/plain; charset=UTF-8\n"
    "Content-Transfer-Encoding: 8bit\n"
)


def _po_path(language: str) -> Path:
    return i18n.LOCALES_DIR / language / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"


def _leading_comment(path: Path) -> str:
    """Return a .po file's leading ``#`` comment lines, verbatim.

    _write_po does not reconstruct these on its own (see its own
    docstring) - reading them here before a save is what keeps them from
    being silently dropped on an existing catalogue's first re-save
    through this dialog.
    """
    if not path.is_file():
        return ""
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("#"):
            lines.append(raw_line)
            continue
        break
    return "\n".join(lines)


def _translate_available() -> bool:
    try:
        import translate  # noqa: F401
    except ImportError:
        return False
    return True


def _auto_translate(text: str, *, to_lang: str) -> str | None:
    """Return a machine translation of *text*, or None if unavailable.

    Never raises - a translation service being unreachable is an everyday
    occurrence for a tool like this, not a bug to surface as a crash.
    """
    try:
        from translate import Translator
    except ImportError:
        return None
    try:
        return str(Translator(to_lang=to_lang).translate(text))
    except Exception:  # noqa: BLE001 - network/service failure, not ours to fix
        applogger.warning(
            "Automatic translation failed for %r (%s).",
            text[:60],
            to_lang,
            show_dialog=False,
            raise_error=False,
        )
        return None


class EditLocalizationDialog(QDialog):
    """Browse, edit and extend one language's translation catalogue."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Edit Localization"))
        self.setWindowIcon(load_icon("edit_localization"))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="large")

        card = CardFrame(self, "editLocalizationCard")
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_("Edit Localization"), card))

        top_row = QHBoxLayout()
        stdSizeAndlayout(top_row)
        top_row.addWidget(QLabel(_("Language:"), card))
        self._language_combo = self._build_language_combo(card)
        self._language_combo.currentIndexChanged.connect(self._reload_table)
        top_row.addWidget(self._language_combo)
        create_action_button(
            parent=self,
            action_id="new",
            action=self._on_new_language,
            layout=top_row,
            presentation=(load_icon("new"), _("New language…"), _("Create a new, empty catalogue")),
        )
        top_row.addStretch(1)
        self._missing_only_check = QCheckBox(_("Missing only"), card)
        self._missing_only_check.toggled.connect(self._apply_filter)
        top_row.addWidget(self._missing_only_check)
        card_layout.addLayout(top_row)

        self._search_edit = QLineEdit(card)
        self._search_edit.setPlaceholderText(_("Search source or translation…"))
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filter)
        card_layout.addWidget(self._search_edit)

        self._table = QTableWidget(0, 2, card)
        self._table.setHorizontalHeaderLabels([_("Source"), _("Translation")])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        mark_editor_panel(self._table)
        card_layout.addWidget(self._table, 1)

        self._status_label = QLabel(card)
        self._status_label.setWordWrap(True)
        card_layout.addWidget(self._status_label)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        create_action_button(
            parent=self,
            action_id="info",
            action=self._show_manual_instructions,
            layout=action_row,
            presentation=(
                load_icon("info"),
                _("Localized manual…"),
                _("How to produce a translated copy of the user manual"),
            ),
        )
        if _translate_available():
            create_action_button(
                parent=self,
                action_id="run",
                action=self._on_auto_translate_missing,
                layout=action_row,
                presentation=(
                    load_icon("run"),
                    _("Translate missing (auto)"),
                    _("Fill every empty translation with a machine translation to review"),
                ),
            )
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        create_action_button(
            parent=self, action_id="apply", action=self._on_save, layout=action_row
        )
        card_layout.addLayout(action_row)

        root.addWidget(card, 1)

        self._entries: dict[str, str] = {}
        self._reload_table()

    # ------------------------------------------------------------------
    # Language combo
    # ------------------------------------------------------------------
    def _build_language_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        # "en" is the source language, not a translation target - the
        # msgid *is* the English text, so there is nothing to edit into it.
        for code in i18n.available_languages():
            if code == i18n.DEFAULT_LANGUAGE:
                continue
            combo.addItem(LANGUAGE_NAMES.get(code, code), code)
        if combo.count() == 0:
            combo.addItem(LANGUAGE_NAMES.get("it", "it"), "it")
        return combo

    def _current_language(self) -> str:
        return str(self._language_combo.currentData() or "it")

    def _on_new_language(self) -> None:
        code, ok = QInputDialog.getText(
            self,
            _("New language"),
            _("Two-letter language code (e.g. \"fr\", \"de\"):"),
        )
        code = code.strip().lower()
        if not ok or not code:
            return
        if not (code.isalpha() and 2 <= len(code) <= 3):
            show_message(self, "dev.validation_error", detail=_("Not a valid language code."))
            return
        if code in i18n.available_languages():
            show_message(
                self, "dev.validation_error",
                detail=_("\"{code}\" already has a catalogue.").format(code=code),
            )
            return

        entries = {string: "" for string in sorted(i18n.source_translator_calls())}
        entries[""] = _NEW_CATALOG_HEADER_TEMPLATE.format(code=code)
        path = _po_path(code)
        i18n._write_po(
            entries, path,
            header_comment=f"# {code} translation for ChartLibre.\n# Message ids are the English source strings.",
        )
        i18n.compile_catalog(code, force=True)

        self._language_combo.addItem(LANGUAGE_NAMES.get(code, code), code)
        self._language_combo.setCurrentIndex(self._language_combo.count() - 1)

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    def _reload_table(self) -> None:
        language = self._current_language()
        path = _po_path(language)
        catalog = i18n._parse_po(path) if path.is_file() else {}
        catalog.pop("", None)

        merged: dict[str, str] = dict.fromkeys(sorted(i18n.source_translator_calls()), "")
        merged.update(catalog)  # existing translations, and any orphaned extra entries
        self._entries = merged
        self._populate_table()
        self._status_label.setText(
            _("{count} string(s), {missing} missing translation.").format(
                count=len(merged),
                missing=sum(1 for value in merged.values() if not value.strip()),
            )
        )

    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._entries))
        for row, (msgid, msgstr) in enumerate(self._entries.items()):
            source_item = QTableWidgetItem(msgid)
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            source_item.setData(Qt.ItemDataRole.UserRole, msgid)
            self._table.setItem(row, 0, source_item)
            self._table.setItem(row, 1, QTableWidgetItem(msgstr))
        self._apply_filter()

    def _apply_filter(self) -> None:
        missing_only = self._missing_only_check.isChecked()
        search = self._search_edit.text().strip().lower()
        for row in range(self._table.rowCount()):
            source_item = self._table.item(row, 0)
            translation_item = self._table.item(row, 1)
            source_text = source_item.text() if source_item else ""
            translation_text = translation_item.text() if translation_item else ""
            is_missing = not translation_text.strip()
            matches_search = (
                not search
                or search in source_text.lower()
                or search in translation_text.lower()
            )
            hide = (missing_only and not is_missing) or not matches_search
            self._table.setRowHidden(row, hide)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _on_save(self) -> None:
        language = self._current_language()
        path = _po_path(language)
        header_comment = _leading_comment(path)
        if not header_comment:
            header_comment = (
                f"# {LANGUAGE_NAMES.get(language, language)} translation for ChartLibre.\n"
                "# Message ids are the English source strings."
            )

        entries = dict(self._entries)
        for row in range(self._table.rowCount()):
            source_item = self._table.item(row, 0)
            translation_item = self._table.item(row, 1)
            if source_item is None or translation_item is None:
                continue
            msgid = str(source_item.data(Qt.ItemDataRole.UserRole))
            entries[msgid] = translation_item.text()

        existing = i18n._parse_po(path) if path.is_file() else {}
        entries[""] = existing.get("", _NEW_CATALOG_HEADER_TEMPLATE.format(code=language))

        try:
            i18n._write_po(entries, path, header_comment=header_comment)
            i18n.compile_catalog(language, force=True)
        except OSError as exc:
            applogger.exception("Could not save the localization catalogue.")
            show_message(self, "dev.validation_error", detail=str(exc))
            return

        applogger.info("Edit Localization: saved %s (%d entries).", path, len(entries) - 1)
        self._reload_table()
        show_message(self, "dev.localization_saved", language=LANGUAGE_NAMES.get(language, language))

    # ------------------------------------------------------------------
    # Auto-translate
    # ------------------------------------------------------------------
    def _on_auto_translate_missing(self) -> None:
        if not ask_before_auto_translate(self):
            return
        language = self._current_language()
        translated_count = 0
        for row in range(self._table.rowCount()):
            source_item = self._table.item(row, 0)
            translation_item = self._table.item(row, 1)
            if source_item is None or translation_item is None:
                continue
            if translation_item.text().strip():
                continue
            guess = _auto_translate(source_item.text(), to_lang=language)
            if guess:
                translation_item.setText(guess)
                translated_count += 1
        self._apply_filter()
        show_message(self, "dev.localization_auto_translated", count=translated_count)

    # ------------------------------------------------------------------
    # Manual
    # ------------------------------------------------------------------
    def _show_manual_instructions(self) -> None:
        QMessageBox.information(
            self,
            _("Localized manual"),
            _(
                "docs/manual/user_manual.typ is not translated by this tool - "
                "it is prose inside Typst markup, not a string catalogue. To "
                "produce a translated copy:\n\n"
                "1. Copy user_manual.typ to user_manual_<lang>.typ (e.g. "
                "user_manual_it.typ).\n"
                "2. Translate the prose, leaving every #-command, code block "
                "and image reference untouched.\n"
                "3. Compile it: typst compile user_manual_<lang>.typ.\n\n"
                "The application text itself does not need this step - it "
                "comes from the catalogue this dialog edits."
            ),
        )


def ask_before_auto_translate(parent: QWidget) -> bool:
    """Confirm before filling the table with machine translations.

    A separate function so a test can call it without going through the
    real QMessageBox.
    """
    return (
        QMessageBox.question(
            parent,
            _("Translate missing (auto)"),
            _(
                "This fills every empty translation with a machine "
                "translation from an online service - a starting point to "
                "review, not a finished translation. Continue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        == QMessageBox.StandardButton.Yes
    )
