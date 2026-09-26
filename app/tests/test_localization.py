"""Guards for the ``_()`` sweep.

The application's user-facing strings go through ``app.utils.i18n._``, the
conventional gettext alias.  Three things can go wrong quietly, and each has a
test here, because none of them shows up as a crash in the language the strings
are already written in.
"""
from __future__ import annotations

import ast
from pathlib import Path


from app.utils import i18n

APP_DIR = Path(__file__).resolve().parent.parent
PO_PATH = i18n.LOCALES_DIR / "it" / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"


def _modules() -> list[Path]:
    return [
        path
        for path in sorted(APP_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts and "tests" not in path.parts
    ]


def _translator_calls(tree: ast.AST) -> list[str]:
    return [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("_", "tr")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


# ----------------------------------------------------------------------
# The alias
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------
def test_every_translated_string_is_in_the_italian_catalogue() -> None:
    """A wrapped string with no entry renders as English, mid-sentence."""
    catalog = i18n._parse_po(PO_PATH)
    missing = sorted(
        {
            message
            for path in _modules()
            for message in _translator_calls(
                ast.parse(path.read_text(encoding="utf-8"))
            )
            if message not in catalog
        }
    )

    assert missing == [], f"{len(missing)} strings have no Italian: {missing[:5]}"


def test_the_catalogue_defines_each_message_once() -> None:
    """Two entries for one msgid means one of them is dead and nobody knows
    which. They were identical this time; the next pair need not be, and then
    which translation shows depends on parse order."""
    import re
    from collections import Counter

    ids = re.findall(r'^msgid "(.*)"$', PO_PATH.read_text(encoding="utf-8"), re.M)
    repeated = sorted(message for message, count in Counter(ids).items() if count > 1)

    assert repeated == []


def test_labels_held_in_tables_are_translated_too() -> None:
    """The gap the ``_()`` sweep cannot see.

    A label defined as module-level data and passed as ``tr(label)`` carries no
    literal at the call site, so neither xgettext nor
    ``test_every_translated_string_is_in_the_italian_catalogue`` can find it.
    "Dark" reached the settings dialog untranslated exactly this way. Each of
    these tables has to be named here by hand - which is the cost of holding UI
    text as data, and the reason to keep such tables few.
    """
    from app.styles.style import APP_STYLE_LABELS

    # LANGUAGE_NAMES is deliberately absent: a language picker names each
    # language in that language - Italiano stays Italiano in the English UI -
    # so those are endonyms rather than strings to translate, and the dialog
    # passes them without tr() for the same reason.
    #
    # RESIZE_MODES lived here too, until the Settings dialog stopped offering
    # a global default fit mode at all - see app/dialogs/settings_dialog.py.
    # Its replacement, app.widgets.chart_panel.RESIZE_MODE_CHOICES, is covered
    # by test_the_fit_mode_choices_are_translated below instead: it carries a
    # tooltip alongside each label, which this flat list of labels has no
    # room for.
    catalog = i18n._parse_po(PO_PATH)
    labels = [
        *APP_STYLE_LABELS.values(),
    ]
    missing = sorted({label for label in labels if label not in catalog})

    assert missing == []


def test_the_demo_project_names_and_summaries_are_translated() -> None:
    """The Load demo picker reads both from DEMO_PROJECTS as data.

    Same blind spot as the fit-mode choices above: no literal at the call
    site, so xgettext and the main sweep cannot see them.
    """
    from app.data.demo_project import DEMO_PROJECTS

    catalog = i18n._parse_po(PO_PATH)
    missing = {
        text
        for demo in DEMO_PROJECTS
        for text in (demo.file_name, demo.summary)
        if text and text not in catalog
    }

    assert sorted(missing) == []


def test_the_catalogue_has_no_empty_translations() -> None:
    """An empty msgstr silently falls back, so it reads as untranslated."""
    catalog = i18n._parse_po(PO_PATH)
    blank = sorted(key for key, value in catalog.items() if key and not value.strip())

    assert blank == []


def test_switching_language_actually_changes_the_strings() -> None:
    """The whole point, asserted once end to end."""
    previous = i18n.language()
    try:
        i18n.set_language("it")
        assert i18n._("Settings") == "Impostazioni"
        assert i18n._("Axis properties") == "Proprietà dell'asse"
        i18n.set_language("en")
        assert i18n._("Settings") == "Settings"
    finally:
        i18n.set_language(previous)


# ----------------------------------------------------------------------
# What must NOT be translated
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Qt's own strings, which ours cannot reach
# ----------------------------------------------------------------------


