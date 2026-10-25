"""Application preferences, in one place the user can reach.

These values all lived in ``config.json`` and could only be changed by editing
it by hand - which meant that in practice they were never changed, and that a
typo in one of them showed up as an unstyled window or an untranslated menu
with nothing to point at.

Everything here is machine-level and so belongs in config.json rather than in
a .dhub file: it describes this installation, not any particular figure.

Nothing here applies live, and the dialog says so rather than leaving the user
to wonder.

The style used to preview itself as you moved through the list, because seeing
a stylesheet is the point of choosing it.  That had to go: applying a style
re-polishes every widget in the application, and ``QApplication.setStyle`` for
a Qt plugin destroys and rebuilds the QStyle underneath them all.  When one of
those widgets is a ``QWebEngineView`` - the HTML results pane, when PySide6's
addons are installed - the repolish reaches into Chromium's own delegate and
takes the process down with a segmentation fault, in ``CrBrowserMain``, below
anything Python can catch.

The language never applied live either, for a duller reason: menus and labels
read the catalogue when they are built, so switching mid-session translates
whatever is constructed next and leaves the rest in the old language.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import applogger
from app.dialogs.two_panels_dialog_base import PanelDialog
from app.styles.style import (
    CONFIG_APP_STYLE,
    available_app_styles,
    configure_combo_width,
    load_icon,
    resolve_app_style,
    stdSizeAndlayout,
)
from app.data.repo._common import SQL_GUARD_KEY, sql_guard_enabled
from app.plugin_api.charts.render_figure import DEFAULT_MPL_STYLE_KEY
from app.utils.config import DEVELOPER_MODE_KEY, MPLSTYLES_DIR, developer_mode, get_language, get_value, set_value
from app.utils.machine_translation import DEEPL_KEY_SETTING
from app.utils.i18n import AUTO_LANGUAGE, available_languages, platform_language, tr

#: Config key and choices for the default chart export format.  The chart
#: panel's save dialog offers all of them whatever is chosen here; this only
#: decides which one it starts on, which is the one most saves use.
CONFIG_SAVE_FORMAT: str = "save_format"

#: Extension and file-dialog filter per format, in the order the save dialog
#: lists them.  One table, so the dialog's choices and the save dialog's
#: filters cannot come to disagree about what is supported.
SAVE_FORMAT_FILTERS: dict[str, tuple[str, str]] = {
    "PNG": ("png", "PNG Image (*.png)"),
    "JPEG": ("jpg", "JPEG Image (*.jpg *.jpeg)"),
    "SVG": ("svg", "SVG Vector (*.svg)"),
    "PDF": ("pdf", "PDF Document (*.pdf)"),
}
SAVE_FORMATS: tuple[str, ...] = tuple(SAVE_FORMAT_FILTERS)
DEFAULT_SAVE_FORMAT: str = "PNG"

#: Language codes have no display names in the catalogue, so they are named
#: here.  A code with no entry shows as itself rather than being hidden: a new
#: locale folder should appear in this list the moment it is added, named or
#: not.
#:
#: Deliberately not translated: a language is listed under its own name, so
#: that someone who has landed in a language they cannot read can still find
#: the one they can.
LANGUAGE_NAMES: dict[str, str] = {
    # Lingue già presenti (corretto 'pol' in 'pl')
    "en": "English",
    "it": "Italiano",
    "fr": "Français",
    "es": "Español",    
    "de": "Deutsch",
    "pt": "Português",
    "zh": "中文",
    "pl": "Polski",
    "ja": "日本語",
    "ko": "한국어",
    "ru": "Русский",
    "ar": "العربية",
    "hi": "हिन्दी",
    "nl": "Nederlands",
    "tr": "Türkçe",
    "sv": "Svenska",
}


def language_choices() -> list[tuple[str, str]]:
    """Return the (stored value, label) pairs for the language combo.

    Auto comes first and is the honest default: it stores a sentinel rather
    than the resolved code, so a machine that later changes its locale follows
    it instead of staying on whatever it happened to be the day the setting
    was saved.  The label names the language it currently resolves to, because
    "Auto" on its own does not tell the user what they are about to get.
    """
    resolved = platform_language()
    auto_label = tr("Auto") + f" ({LANGUAGE_NAMES.get(resolved, resolved)})"
    return [
        (AUTO_LANGUAGE, auto_label),
        *((code, LANGUAGE_NAMES.get(code, code)) for code in available_languages()),
    ]


def normalized_save_format(value: object) -> str:
    """Return a supported export format, defaulting rather than raising."""
    clean = str(value or "").strip().upper()
    if clean in ("JPG", "JPEG"):
        return "JPEG"
    return clean if clean in SAVE_FORMATS else DEFAULT_SAVE_FORMAT


def _mpl_style_choices() -> list[tuple[str, str] | None]:
    """The default-chart-style choices, grouped with separators.

    ChartLibre's own default first, then Matplotlib's; then the shipped
    folders - Apps, Journals - and any other folder after them, one group
    each. A style is stored as its path under mplstyles without the
    extension and shown as "Apps.jmp-classic", the way the Figure panel
    names them.
    """
    paths = sorted(
        path.relative_to(MPLSTYLES_DIR).with_suffix("")
        for path in MPLSTYLES_DIR.rglob("*.mplstyle")
    ) if MPLSTYLES_DIR.exists() else []

    def entry(path: Any) -> tuple[str, str]:
        return path.as_posix(), path.as_posix().replace("/", ".")

    default = [path for path in paths if path.as_posix() == "ChartLibreDefault"]
    top_level = [path for path in paths if len(path.parts) == 1 and path not in default]
    folders: dict[str, list[Any]] = {}
    for path in paths:
        if len(path.parts) > 1:
            folders.setdefault(path.parts[0], []).append(path)
    # Apps, then Journals, then whatever else is there.
    order = sorted(folders, key=lambda name: ({"apps": 0, "journals": 1}.get(name.lower(), 2), name.lower()))

    choices: list[tuple[str, str] | None] = [entry(path) for path in default]
    choices.append(("", tr("Matplotlib default")))
    choices.extend(entry(path) for path in top_level)
    for name in order:
        choices.append(None)
        choices.extend(entry(path) for path in folders[name])
    return choices


class SettingsDialog(PanelDialog):
    """Read and write the machine-level preferences in config.json.

    Nothing is written until Save, and nothing is applied until the next
    start: a dialog that persisted on every keystroke would make Cancel a lie,
    and one that applied a style live crashed the application - see the module
    docstring.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, title=tr("Settings"), icon=load_icon("settings"), size="small")

        # Only to preselect the combo: nothing here changes the running app.
        self._entry_style = resolve_app_style()
        # The stored setting, not the resolved language: a user who chose Auto
        # must find Auto selected when they reopen this, not the code it
        # happened to resolve to on this machine.
        self._entry_language = get_language()

        # A titled frame per group, each with its own form; their label
        # columns are given one width at the end, so the fields line up from
        # one frame to the next as they did in a single form.
        forms: list[QFormLayout] = []
        card = self
        form = self._section(tr("Appearance"), forms)

        # Automatic, then the stylesheets shipped in app/styles: nothing else.
        self._style_combo = self._combo(
            card,
            [(key, tr(label)) for key, label in available_app_styles()],
            self._entry_style,
        )
        form.addRow(tr("App style"), self._style_combo)

        self._language_combo = self._combo(
            card,
            language_choices(),
            self._entry_language,
        )
        form.addRow(tr("Language"), self._language_combo)

        form = self._section(tr("Charts"), forms)

        self._format_combo = self._combo(
            card,
            [(name, name) for name in SAVE_FORMATS],
            normalized_save_format(get_value(CONFIG_SAVE_FORMAT)),
        )
        form.addRow(tr("Default export format"), self._format_combo)

        self._mpl_style_combo = self._combo(
            card,
            _mpl_style_choices(),
            str(get_value(DEFAULT_MPL_STYLE_KEY) or ""),
        )
        self._mpl_style_combo.setToolTip(tr("The chart style of every figure that has none of its own."))
        form.addRow(tr("Default chart style"), self._mpl_style_combo)

        # Every report's "How to read this" box (H-04): on for a new install - students first.
        from app.plugin_api import reading_guide

        form = self._section(tr("Reports"), forms)
        self._how_to_read_check = QCheckBox(tr("A \"How to read this\" box in every report"), card)
        self._how_to_read_check.setToolTip(tr("Under each report's main result, two sentences on what its key "
                                              "numbers - a p-value, R², a Cpk - say about these data."))
        self._how_to_read_check.setChecked(reading_guide.enabled())
        form.addRow(tr("Explain"), self._how_to_read_check)

        self._build_assistant_section(card, forms)

        # What most people never change, closed - unless something in it is
        # not as it came (todo H-10).
        form = self._section(
            tr("Advanced"), forms, collapsible=True, collapsed=sql_guard_enabled() and not developer_mode()
        )
        # Red: switching it off lets typed SQL change the database.
        self._sql_guard_check = QCheckBox(
            tr("Refuse SQL that would change the database"), card
        )
        self._sql_guard_check.setProperty("destructive", True)
        self._sql_guard_check.setToolTip(
            tr(
                "Checked when a query is saved or typed, and when a project is "
                "opened: UPDATE, DELETE, DROP and the like are refused. Charts "
                "are never slowed down by it."
            )
        )
        self._sql_guard_check.setChecked(sql_guard_enabled())
        form.addRow(tr("SQL guard"), self._sql_guard_check)

        # The tools for contributors, and what only they use (todo H-13).
        self._developer_check = QCheckBox(tr("Show the tools for extending the app"), card)
        self._developer_check.setToolTip(
            tr(
                "Edit Localization, Data Operation Builder, Function Creator and "
                "Renderer Helper on the Project page, and the DeepL key below."
            )
        )
        self._developer_check.setChecked(developer_mode())
        form.addRow(tr("Developer mode"), self._developer_check)

        # The one secret in here: shown as dots, kept in user.json on this
        # computer - never in config.json or a project - and sent only to
        # DeepL. Edit Localization offers DeepL once it is set (todo A-08).
        self._deepl_key_edit = QLineEdit(card)
        self._deepl_key_edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self._deepl_key_edit.setPlaceholderText(tr("Free plan keys end in :fx"))
        self._deepl_key_edit.setToolTip(
            tr(
                "Used by Edit Localization to translate with DeepL. Kept on this "
                "computer only, and sent to nobody but DeepL. Leave it empty to "
                "translate with Google or MyMemory."
            )
        )
        self._deepl_key_edit.setText(str(get_value(DEEPL_KEY_SETTING, "") or ""))
        form.addRow(tr("DeepL API key"), self._deepl_key_edit)
        deepl_label = form.labelForField(self._deepl_key_edit)
        self._developer_check.toggled.connect(self._deepl_key_edit.setVisible)
        if deepl_label is not None:
            self._developer_check.toggled.connect(deepl_label.setVisible)
        self._developer_check.toggled.emit(self._developer_check.isChecked())

        note = QLabel(tr("The style and the language are applied the next time the app starts."), self)
        note.setProperty("muted", True)
        note.setWordWrap(True)
        self.body_layout.addWidget(note, 0)
        self.body_layout.addStretch(1)
        _align_labels(forms)

        self.add_action("save", self._save)
        self.add_action("close", self.reject)

    def _section(self, title: str, forms: list[QFormLayout], *, collapsible: bool = False,
                 collapsed: bool = False) -> QFormLayout:
        """A titled frame and its form; the form, to fill."""
        form = QFormLayout()
        stdSizeAndlayout(form)
        self.add_card(title, f"settings{title.replace(' ', '')}Card", collapsible=collapsible,
                      collapsed=collapsed).layout().addLayout(form)
        forms.append(form)
        return form

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _combo(
        parent: QWidget, choices: Sequence[tuple[str, str] | None], current: str
    ) -> QComboBox:
        """Return a combo of (stored value, shown label), on *current*.

        A ``None`` entry is a separator line.

        The stored value is carried as item data rather than being recovered
        from the label, so translating a label cannot change what is written
        to config.json.
        """
        combo = QComboBox(parent)
        for choice in choices:
            if choice is None:
                combo.insertSeparator(combo.count())
                continue
            value, label = choice
            combo.addItem(label, value)

        index = combo.findData(current)
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)
        configure_combo_width(combo)
        return combo

    @staticmethod
    def _value(combo: QComboBox) -> str:
        return str(combo.currentData() or "")

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def _build_assistant_section(self, card: QWidget, forms: list[QFormLayout]) -> None:
        """The assistant: its provider, the key (kept by the system), what it may do and what it may send."""
        from app.ai import settings as assistant
        from app.scanners.ai_provider_scanner import ai_providers, provider_class

        form = self._section(tr("Assistant"), forms)
        self._assistant_in_navigation = QCheckBox(tr("Show the Assistant in the navigation bar"), card)
        self._assistant_in_navigation.setChecked(assistant.in_navigation())
        form.addRow(tr("Navigation"), self._assistant_in_navigation)
        names = [str(entry["value"]) for entry in ai_providers]
        self._assistant_provider = self._combo(card, [(name, name) for name in names], assistant.provider_name())
        form.addRow(tr("Provider"), self._assistant_provider)
        self._assistant_key_names = {str(entry["value"]): provider_class(entry).KEY_NAME for entry in ai_providers}
        self._assistant_key = QLineEdit(card)
        self._assistant_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        key_name = self._assistant_key_names.get(assistant.provider_name(), "")
        self._assistant_key_loaded = assistant.api_key(key_name) if key_name else ""
        self._assistant_key.setText(self._assistant_key_loaded)
        if assistant.keychain_available():
            self._assistant_key.setToolTip(tr("Kept by the system's keychain on this computer - never in a file "
                                              "or a project - and sent only to the provider."))
        else:
            self._assistant_key.setEnabled(False)
            self._assistant_key.setPlaceholderText(tr("pip install keyring to keep it here"))
            self._assistant_key.setToolTip(tr("Without the keyring package the key is read from the environment "
                                              "(ANTHROPIC_API_KEY for Claude)."))
        # Test: the smallest request the provider answers, with the key as typed - before it is saved.
        self._assistant_check = QPushButton(tr("Try"), card)
        self._assistant_check.setToolTip(tr("Sends the provider the smallest request it answers, with this key: "
                                            "whether the key, the account's credit and the connection work."))
        self._assistant_check.clicked.connect(self._check_assistant_key)
        self._assistant_check_task: Any = None
        self._assistant_check_result = QLabel(card)
        self._assistant_check_result.setWordWrap(True)
        self._assistant_check_result.setProperty("muted", True)
        self._assistant_check_result.hide()
        key_field = QWidget(card)
        key_column = QVBoxLayout(key_field)
        key_column.setContentsMargins(0, 0, 0, 0)
        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(self._assistant_key, 1)
        key_row.addWidget(self._assistant_check)
        key_column.addLayout(key_row)
        key_column.addWidget(self._assistant_check_result)
        form.addRow(tr("API key"), key_field)
        self._assistant_provider.currentIndexChanged.connect(self._assistant_provider_changed)
        self._assistant_provider_changed()
        self._assistant_permission = self._combo(card, [(key, tr(label)) for key, label in assistant.PERMISSIONS],
                                                 assistant.permission())
        self._assistant_permission.setToolTip(tr("What the assistant may do. Every change to the data is confirmed "
                                                 "first and can be undone."))
        form.addRow(tr("May"), self._assistant_permission)
        self._assistant_consent = QCheckBox(tr("Send questions and data to the provider"), card)
        self._assistant_consent.setToolTip(tr("The assistant sends your questions, the tables' columns and "
                                              "summaries - and sample rows, below - to the provider. Nothing is "
                                              "sent before this is checked."))
        self._assistant_consent.setChecked(assistant.consented())
        form.addRow(tr("Data"), self._assistant_consent)
        self._assistant_rows = QSpinBox(card)
        self._assistant_rows.setRange(0, 200)
        self._assistant_rows.setValue(assistant.sample_rows())
        self._assistant_rows.setSpecialValueText(tr("None"))
        self._assistant_rows.setToolTip(tr("How many rows of a table the assistant may send as an example."))
        form.addRow(tr("Sample rows"), self._assistant_rows)

    def _assistant_provider_changed(self) -> None:
        """A provider that needs no key (Demo): the key and its test are not for it."""
        from app.ai import settings as assistant

        needs_key = bool(self._assistant_key_names.get(self._value(self._assistant_provider)))
        self._assistant_key.setEnabled(needs_key and assistant.keychain_available())
        self._assistant_check.setEnabled(needs_key)
        self._assistant_check_result.hide()

    def _check_assistant_key(self) -> None:
        """Try the key on a worker thread; say what came of it under the field."""
        import shiboken6

        from app.ai import settings as assistant
        from app.scanners.ai_provider_scanner import ai_providers, provider_class
        from app.utils.background import run_in_background

        name = self._value(self._assistant_provider)
        entry = next((entry for entry in ai_providers if entry["value"] == name), None)
        if entry is None:
            return
        provider = provider_class(entry)()
        key = self._assistant_key.text().strip() or assistant.api_key(provider.KEY_NAME)

        def say(text: str) -> None:
            if shiboken6.isValid(self._assistant_check_result):  # the dialog may be closed by now
                self._assistant_check_result.setText(text)
                self._assistant_check_result.show()
                self._assistant_check.setEnabled(True)
                self._grow_to_fit()

        if provider.unavailable() or not key:
            say(tr(provider.unavailable()) if provider.unavailable() else tr("Type the key first."))
            return
        self._assistant_check.setEnabled(False)
        self._assistant_check_result.setText(tr("Trying the key..."))
        self._assistant_check_result.show()
        self._grow_to_fit()
        self._assistant_check_task = run_in_background(
            lambda _cancel: provider.check(key),
            lambda problem: say(tr(problem) if problem else tr("The key works: {provider} answered.").format(
                provider=name)),
            lambda error: say(str(error) or type(error).__name__),
        )

    def _grow_to_fit(self) -> None:
        """Taller when a wrapped line appeared since it opened: a shown window does not grow for one itself."""
        layout = self.layout()
        if layout is None:
            return
        layout.activate()
        height = (layout.totalHeightForWidth(self.width()) if layout.hasHeightForWidth()
                  else layout.totalSizeHint().height())
        if height > self.height():
            self.resize(self.width(), height)

    def _save_assistant(self) -> None:
        from app.ai import settings as assistant

        assistant.set_in_navigation(self._assistant_in_navigation.isChecked())
        provider = self._value(self._assistant_provider)
        assistant.save(provider=provider, permission_level=self._value(self._assistant_permission),
                       consent=self._assistant_consent.isChecked(), rows=self._assistant_rows.value(),
                       assistant_mode=assistant.mode())
        key = self._assistant_key.text().strip()
        key_name = self._assistant_key_names.get(provider, "")
        if key_name and self._assistant_key.isEnabled() and key != self._assistant_key_loaded:
            assistant.set_api_key(key_name, key)

    def _save(self) -> None:
        """Write every preference, then close.

        Written one key at a time through ``set_value``, which re-reads and
        re-writes the whole document each time.  That is slower than assembling
        one write and entirely deliberate: this dialog is not the only writer
        of config.json - dialog geometry is saved on close - and a single
        blind write would drop whatever else changed while it was open.
        """
        style_key = self._value(self._style_combo)
        set_value(CONFIG_APP_STYLE, style_key)
        set_value("language", self._value(self._language_combo))
        set_value(CONFIG_SAVE_FORMAT, self._value(self._format_combo))
        set_value(DEFAULT_MPL_STYLE_KEY, self._value(self._mpl_style_combo))
        set_value(SQL_GUARD_KEY, bool(self._sql_guard_check.isChecked()))
        from app.plugin_api import reading_guide

        set_value(reading_guide.SETTING_KEY, bool(self._how_to_read_check.isChecked()))
        set_value(DEEPL_KEY_SETTING, self._deepl_key_edit.text().strip())
        set_value(DEVELOPER_MODE_KEY, bool(self._developer_check.isChecked()))
        self._save_assistant()

        applogger.info(
            "Settings saved: style=%s language=%s format=%s",
            style_key,
            self._value(self._language_combo),
            self._value(self._format_combo),
        )
        self.accept()


def _align_labels(forms: list[QFormLayout]) -> None:
    """One label width for every form, so their fields start at the same x."""
    labels = []
    for form in forms:
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            if item is not None and item.widget() is not None:
                labels.append(item.widget())
    width = max((label.sizeHint().width() for label in labels), default=0)
    for label in labels:
        label.setMinimumWidth(width)
