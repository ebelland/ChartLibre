"""Edit Localization: a table editor for one language's .po catalogue
(todo.txt P3-5).

Every row is a source string ``_()``/``tr()`` wraps somewhere in the app
(``i18n.source_translator_calls()`` - the same AST sweep
dev/tests/test_localization.py's own safety net runs) unioned with
whatever the selected language's catalogue already has, so a string that
is translated but no longer used anywhere still shows up to edit or
remove by hand, and a string used but never added to the catalogue shows
up with an empty translation rather than silently failing the next full
test run instead. "Missing only" filters to exactly that empty-
translation case - what would otherwise only surface at
`python -m pytest dev/tests/test_localization.py`.

Saving writes through :func:`app.utils.i18n._write_po` (round-trips
losslessly with :func:`app.utils.i18n._parse_po`, see that function's own
docstring) and recompiles the .mo immediately via
:func:`app.utils.i18n.compile_catalog`, so a translation shows up the next
time that language is selected without restarting the app.

"Translate missing (auto)" is a soft dependency on ``deep_translator``
(Google Translate or MyMemory, no key needed; see
app.utils.machine_translation) - offered only when it is importable, run
on a worker thread so the dialog stays usable and can be stopped, and
always a starting point to review, never treated as a finished translation.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
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
    QPushButton,
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
from app.utils import i18n, machine_translation
from app.utils.background import BackgroundTask, run_in_background
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
        self._translate_button: QPushButton | None = None
        self._provider_combo: QComboBox | None = None
        if machine_translation.available():
            self._provider_combo = QComboBox(card)
            self._provider_combo.setToolTip(_("Translation service"))
            for key, name in machine_translation.PROVIDERS:
                self._provider_combo.addItem(name, key)
            action_row.addWidget(self._provider_combo)
            self._translate_button = create_action_button(
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
        self._translation_task: BackgroundTask | None = None
        #: [done, total], written by the worker thread, read by the timer.
        self._translation_progress: list[int] = [0, 0]
        #: (source, translation) pairs the worker has finished and the
        #: timer has not yet put in the table. list.append is atomic, so
        #: the worker appends and the GUI thread drains without a lock.
        self._arrived: list[tuple[str, str]] = []
        self._translation_filled = 0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._show_translation_progress)
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
        # OK saves and closes, like every other OK in the application. A
        # message box saying so was one more click standing between the
        # user and the window they came from - and it read as the dialog
        # refusing to close.
        self.accept()

    # ------------------------------------------------------------------
    # Auto-translate
    # ------------------------------------------------------------------
    def _on_auto_translate_missing(self) -> None:
        """Start filling the empty translations - or, while running, stop."""
        if self._translation_task is not None:
            self._translation_task.cancel()
            return
        missing = [
            str(source_item.data(Qt.ItemDataRole.UserRole))
            for row in range(self._table.rowCount())
            if (source_item := self._table.item(row, 0)) is not None
            and (translation_item := self._table.item(row, 1)) is not None
            and not translation_item.text().strip()
        ]
        if not missing:
            show_message(self, "dev.localization_auto_translated", count=0)
            return
        if not ask_before_auto_translate(self):
            return

        language = self._current_language()
        provider = str(self._provider_combo.currentData()) if self._provider_combo else "google"
        progress = self._translation_progress
        progress[:] = [0, len(missing)]
        arrived = self._arrived
        arrived.clear()
        self._translation_filled = 0

        def work(cancel_event: object) -> machine_translation.TranslationRun:
            def report(done: int, total: int) -> None:
                progress[:] = [done, total]

            return machine_translation.translate_all(
                missing,
                language=language,
                provider=provider,
                # Google refuses addresses that ask too often: what it does
                # not translate goes to the other service.
                fallbacks=[key for key, _name in machine_translation.PROVIDERS if key != provider],
                cancel_event=cancel_event,  # pyright: ignore[reportArgumentType]
                progress=report,
                on_result=lambda source, text: arrived.append((source, text)),
            )

        self._translation_task = run_in_background(
            work, self._on_translation_finished, self._on_translation_failed
        )
        self._set_translating(True)

    def _set_translating(self, running: bool) -> None:
        if self._translate_button is not None:
            self._translate_button.setText(_("Stop") if running else _("Translate missing (auto)"))
        if self._provider_combo is not None:
            self._provider_combo.setEnabled(not running)
        self._language_combo.setEnabled(not running)
        if running:
            self._progress_timer.start()
            self._show_translation_progress()
        else:
            self._progress_timer.stop()

    def _show_translation_progress(self) -> None:
        self._fill_arrived()
        done, total = self._translation_progress
        self._status_label.setText(
            _("Translating {done} of {total}…").format(done=done, total=total)
        )

    def _fill_arrived(self) -> int:
        """Put the translations that have arrived into their empty cells.

        Run by the progress timer while the service works, so the table
        fills block by block and Stop keeps everything already there.
        Only into a cell still empty: the table stays editable meanwhile,
        and a hand-typed translation wins.
        """
        if not self._arrived:
            return 0
        count = len(self._arrived)  # the worker may append meanwhile
        batch = dict(self._arrived[:count])
        del self._arrived[:count]
        filled = 0
        for row in range(self._table.rowCount()):
            source_item = self._table.item(row, 0)
            translation_item = self._table.item(row, 1)
            if source_item is None or translation_item is None:
                continue
            guess = batch.get(str(source_item.data(Qt.ItemDataRole.UserRole)))
            if guess and not translation_item.text().strip():
                translation_item.setText(guess)
                filled += 1
        self._translation_filled += filled
        return filled

    def _on_translation_finished(self, run: machine_translation.TranslationRun) -> None:
        self._translation_task = None
        self._set_translating(False)
        self._fill_arrived()
        filled = self._translation_filled
        self._apply_filter()
        self._status_label.setText(
            _("{count} string(s), {missing} missing translation.").format(
                count=self._table.rowCount(),
                missing=sum(
                    1 for row in range(self._table.rowCount())
                    if (item := self._table.item(row, 1)) is not None and not item.text().strip()
                ),
            )
        )
        left = run.failed + run.rejected
        switched = "; ".join(
            _("{first} refused the requests, continued with {second}").format(first=first, second=second)
            for first, second in run.switched
        )
        if not left and not run.stopped:
            if switched:
                show_message(self, "dev.localization_auto_translate_partial", count=filled, reasons=switched)
            else:
                show_message(self, "dev.localization_auto_translated", count=filled)
            return
        reasons: list[str] = [switched] if switched else []
        if run.stopped:
            reasons.append(_("stopped"))
        if run.failed:
            why = (
                _("{services} refused for their request limit - Google for a while, MyMemory until tomorrow").format(
                    services=", ".join(run.limited)
                )
                if run.limited
                else run.error or _("no answer")
            )
            reasons.append(
                _("{count} not translated by the service ({error})").format(count=run.failed, error=why)
            )
        if run.rejected:
            reasons.append(
                _("{count} discarded because a placeholder such as {{name}} or %s was lost").format(
                    count=run.rejected
                )
            )
        show_message(
            self, "dev.localization_auto_translate_partial",
            count=filled, reasons="; ".join(reasons),
        )

    def _on_translation_failed(self, error: BaseException) -> None:
        self._translation_task = None
        self._set_translating(False)
        self._reload_status_only()
        applogger.warning(
            "Automatic translation could not start: %s", error,
            show_dialog=False, raise_error=False,
        )
        show_message(self, "dev.validation_error", detail=str(error))

    def _reload_status_only(self) -> None:
        self._status_label.setText(
            _("{count} string(s), {missing} missing translation.").format(
                count=len(self._entries),
                missing=sum(1 for value in self._entries.values() if not value.strip()),
            )
        )

    def done(self, result: int) -> None:
        """Stop a running translation when the dialog closes, however it closes."""
        if self._translation_task is not None:
            self._translation_task.cancel()
        super().done(result)

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
