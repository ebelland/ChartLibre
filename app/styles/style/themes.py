"""Application styles: the QSS themes, their palettes, and applying one."""
from __future__ import annotations

import platform
from pathlib import Path


from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QWidget
from app.logs.logger import applogger
from app.styles.palettes import themed_qss
from app.utils.config import get_value
from app.styles.style.metrics import PlatformStyle, _STYLES_DIR
from app.styles.style.icons import _substitute_qss_symbols, substitute_ui_font
from app.styles.style import icons as _icons


# Styles
# ----------------------------------------------------------------------
def _load_qss(name: str) -> tuple[str | None, Path | None]:
    """Load a QSS file by name from the app styles directory."""
    return _read_qss_path(_STYLES_DIR / name)


def _read_qss_path(path: Path) -> tuple[str | None, Path | None]:
    """Load a QSS file by path, from anywhere on disk.

    Shared by the two shipped sheets and by a stylesheet the user picked
    themselves (see QSS_FILE_PREFIX): a sheet that has gone missing or
    cannot be read leaves the application on no sheet at all rather than
    raising, which is the same answer :func:`_load_qss` has always given.
    """
    if not path.exists():
        applogger.warning("QSS not found: %s", path)
        return None, None

    try:
        return path.read_text(encoding="utf-8"), path
    except Exception:
        applogger.exception("Failed to read QSS: %s", path)
        return None, None


#: What the user can choose between, and what each choice does.
#:
#: Two different mechanisms sit behind one setting, deliberately.  A *theme* is
#: one of this application's stylesheets plus a palette; a *Qt style* is a
#: plugin Qt itself provides - Fusion, Windows, and whatever the desktop has
#: installed - drawn entirely by Qt with no sheet of ours on top.  They are
#: offered together because from where the user sits they answer one question,
#: and kept apart in code because they have nothing else in common.
APP_STYLE_AUTOMATIC: str = "automatic"

#: theme key -> (stylesheet file, palette key)
APP_STYLE_QSS: dict[str, tuple[str, str]] = {
    "fluent_win11": ("fluent_win11.qss", "light"),
    "macos_native": ("macos_native.qss", "light"),
    # The dark theme is the Fluent *structure* with the dark palette: the
    # sheets carry no opaque surface colours worth speaking of, so a second
    # 1,000-line file would differ from this one only in tokens.
    "dark": ("fluent_win11.qss", "dark"),
}

#: Prefix marking a Qt style plugin rather than one of our themes.
QT_STYLE_PREFIX: str = "qt:"

#: Prefix marking a ``.qss`` file the user picked themselves, followed by
#: its absolute path. The same shape as QT_STYLE_PREFIX and stored in the
#: same config key, so one setting still answers "what does this app look
#: like" - a shipped theme, an installed Qt style, or a sheet of your own.
#: Its palette is the light one: a sheet written elsewhere cannot be asked
#: which of ours it was built against, and light is what both shipped
#: sheets use.
QSS_FILE_PREFIX: str = "file:"

CONFIG_APP_STYLE: str = "app_style"

#: Labels for the themes.  Qt styles are labelled by their own key, since that
#: is the name their documentation and their users know them by.
APP_STYLE_LABELS: dict[str, str] = {
    APP_STYLE_AUTOMATIC: "Automatic",
    "fluent_win11": "Fluent (Windows 11)",
    "macos_native": "macOS native",
    "dark": "Dark",
}


def available_qt_styles() -> list[str]:
    """Return the Qt style plugins this installation actually has.

    Enumerated rather than listed.  Fusion and Windows ship with Qt; Breeze,
    Oxygen and QtCurve are KDE plugins that have to be installed separately and
    are absent from a stock macOS or Windows machine.  Hard-coding those names
    would put three entries in the menu that silently do nothing - so they
    appear here exactly when they can be applied, and not otherwise.
    """
    return list(QtWidgets.QStyleFactory.keys())


def qss_file_style_key(path: Path | str) -> str:
    """Return the stored app-style key for a ``.qss`` file on disk."""
    return f"{QSS_FILE_PREFIX}{Path(path).expanduser().resolve()}"


def qss_file_style_label(key: str) -> str:
    """Return the label to show for a ``file:`` app-style key."""
    return f"File: {Path(key[len(QSS_FILE_PREFIX):]).name}"


def available_app_styles() -> list[tuple[str, str]]:
    """Return every selectable style as (stored key, label to show).

    A stylesheet the user picked from disk is one of the entries when it
    is the current choice: it is the only style whose set is not knowable
    in advance, and leaving it out would mean reopening Settings on a
    combo that quietly showed something the app is not wearing.
    """
    styles = [(key, APP_STYLE_LABELS[key]) for key in APP_STYLE_LABELS]
    styles.extend(
        (f"{QT_STYLE_PREFIX}{name}", f"Qt: {name}") for name in available_qt_styles()
    )

    current = resolve_app_style()
    if current.startswith(QSS_FILE_PREFIX):
        styles.append((current, qss_file_style_label(current)))
    return styles


def _automatic_qss_name() -> str | None:
    """Return the stylesheet this desktop would pick for itself."""
    system = platform.system().lower()
    if system.startswith("win"):
        return "fluent_win11.qss"
    if system == "darwin":
        return "macos_native.qss"
    return None


def resolve_app_style(preference: str | None = None) -> str:
    """Return a valid app-style key, falling back to automatic.

    An unknown value reads as automatic rather than as "no stylesheet": a typo
    in config.json, or a Qt plugin that was uninstalled since it was chosen,
    should leave the app looking normal rather than unstyled.
    """
    clean = str(preference or "").strip()
    if not clean:
        clean = str(get_value(CONFIG_APP_STYLE, "") or "").strip()

    if clean.startswith(QSS_FILE_PREFIX):
        raw = clean[len(QSS_FILE_PREFIX) :].strip()
        path = Path(raw).expanduser()
        if raw and path.is_file():
            return qss_file_style_key(path)
        applogger.warning(
            "The stylesheet %r is no longer there; using the automatic style.",
            raw,
        )
        return APP_STYLE_AUTOMATIC

    if clean.startswith(QT_STYLE_PREFIX):
        name = clean[len(QT_STYLE_PREFIX) :]
        available = {key.lower(): key for key in available_qt_styles()}
        if name.lower() in available:
            return f"{QT_STYLE_PREFIX}{available[name.lower()]}"
        applogger.warning(
            "The Qt style %r is not installed; using the automatic style.",
            name,
        )
        return APP_STYLE_AUTOMATIC

    lowered = clean.lower()
    return lowered if lowered in APP_STYLE_QSS else APP_STYLE_AUTOMATIC


def apply_platform_style(
    app: QApplication, preference: str | None = None
) -> PlatformStyle:
    """Apply the configured theme or Qt style.

    ``preference`` overrides what config.json says, so the settings dialog can
    show a choice before committing it.

    Every path here sets *both* the stylesheet and the palette, including to
    nothing.  A style change has to undo the previous one completely: leaving a
    dark palette under a light sheet is how a theme switcher produces a window
    that belongs to neither theme.
    """
    system = platform.system().lower()
    style_key = resolve_app_style(preference)

    if style_key.startswith(QT_STYLE_PREFIX):
        name = style_key[len(QT_STYLE_PREFIX) :]
        # A Qt style draws itself; none of ours is a dark theme, so the glyphs
        # stay dark whatever palette the plugin brings with it.
        _icons._ACTIVE_THEME_IS_DARK = False
        app.setStyleSheet("")
        app.setPalette(QtWidgets.QStyleFactory.create(name).standardPalette())
        app.setStyle(name)
        return PlatformStyle(system, None)

    if style_key.startswith(QSS_FILE_PREFIX):
        # A sheet of the user's own, read from wherever they keep it.
        # Otherwise treated exactly like a shipped one: the same palette
        # substitution runs over it, so a sheet that uses none of our
        # @TOKEN@ placeholders simply passes through unchanged.
        qss, path = _read_qss_path(Path(style_key[len(QSS_FILE_PREFIX) :]))
        palette_key = "light"
    else:
        if style_key == APP_STYLE_AUTOMATIC:
            qss_name, palette_key = _automatic_qss_name(), "light"
        else:
            qss_name, palette_key = APP_STYLE_QSS[style_key]

        qss, path = _load_qss(qss_name) if qss_name else ("", None)

    # No app.setStyle() override here, deliberately: macos_native.qss's own
    # header explains why most standard controls (QMenu, QScrollBar,
    # QLineEdit...) are left unstyled on purpose - Qt's native Aqua/Fluent
    # style plugins already draw those authentically (QPushButton,
    # QComboBox and QCheckBox are the exceptions that file's own header
    # explains too), and forcing Fusion app-wide to fix one narrow
    # rendering gap (see apply_fusion_for_item_view_styling below) would
    # throw that away for every other widget in the application. The gap
    # gets a surgical fix
    # instead, on just the specific views that need it.

    # Read the sheet first, then theme it: themed_qss hands back the palette it
    # resolved, and the palette has to be installed even when there is no sheet
    # to go with it - a desktop with no QSS of its own still gets the theme's
    # colours through Qt.
    themed, palette = themed_qss(qss or "", palette_key)
    themed = substitute_ui_font(themed)
    # Images a sheet needs (a combo's chevrons): url(@ICON:name@), drawn from
    # the system's own symbols rather than shipped as artwork.
    themed = _substitute_qss_symbols(themed)
    _icons._ACTIVE_THEME_IS_DARK = palette.dark
    app.setPalette(palette.qpalette())

    if not themed:
        # Nothing to install, so clear whatever the previous choice left rather
        # than leaving half of it behind.
        app.setStyleSheet("")
        return PlatformStyle(system, None)

    app.setStyleSheet(themed)
    return PlatformStyle(system, path)


def apply_fusion_for_item_view_styling(view: QWidget) -> None:
    """Force Fusion on one view whose selection/header styling must render
    as this application's own QSS says, not as the native platform style's
    own approximation of it.

    The narrow case this exists for: TableListPanel and TablePreviewPanel
    both style QAbstractItemView selection (a rounded "capsule" on the
    sidebar, a flat highlight on the file list) and header colours
    directly, and the native macOS and Windows Qt styles paint a fair
    amount of that themselves - a table's selected-row highlight in
    particular - in ways a stylesheet cannot always override. Fusion draws
    itself entirely from the palette and the stylesheet, so it is the one
    Qt style guaranteed to render what these files actually say.

    Scoped to the one view that needs it via ``QWidget.setStyle`` rather
    than ``QApplication.setStyle`` - the rest of the application keeps
    whatever native style (or Fusion, if that is what was actually chosen)
    it already had. See macos_native.qss's own header for why every other
    standard control is deliberately left native.

    The scroll bars are put back on the application's style afterwards,
    and that is not a detail: a widget's style *is* inherited by its
    children, so Fusion on the view reached the two scroll bars as well
    and drew them as Fusion draws them with no stylesheet to follow - a
    square black handle between two stepper arrows, beside the rounded
    native ones on every other scrolling widget in the window. Nothing
    about a scroll bar was the reason for Fusion here; the selection and
    the header were.
    """
    view.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    application_style = QtWidgets.QApplication.style()
    for reader in ("verticalScrollBar", "horizontalScrollBar"):
        # getattr: the helper's argument is a QWidget, and only a scroll
        # area has these. Nothing else needs putting back.
        scrollbar = getattr(view, reader, None)
        bar = scrollbar() if callable(scrollbar) else None
        if isinstance(bar, QtWidgets.QScrollBar):
            bar.setStyle(application_style)


# ----------------------------------------------------------------------
