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

    A sheet that has gone missing or cannot be read leaves the application
    on no sheet at all rather than raising, which is the same answer
    :func:`_load_qss` has always given.
    """
    if not path.exists():
        applogger.warning("QSS not found: %s", path)
        return None, None

    try:
        return path.read_text(encoding="utf-8"), path
    except (OSError, UnicodeDecodeError):
        applogger.exception("Failed to read QSS: %s", path)
        return None, None


#: What the user can choose between, and what each choice does.
#:
#: A *theme* is one of this application's stylesheets (the .qss files in
#: app/styles) plus a palette; "Automatic" picks the one for this desktop.
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

# Any other sheet dropped into app/styles/ is a theme too, on the light
# palette: the Settings dropdown lists the .qss files shipped here and
# "Automatic", nothing else.
for _sheet in sorted(_STYLES_DIR.glob("*.qss")):
    if _sheet.name not in {name for name, _palette in APP_STYLE_QSS.values()}:
        APP_STYLE_QSS[_sheet.stem] = (_sheet.name, "light")

CONFIG_APP_STYLE: str = "app_style"

#: Labels for the themes; a sheet without one is shown by its file name.
APP_STYLE_LABELS: dict[str, str] = {
    APP_STYLE_AUTOMATIC: "Automatic",
    "fluent_win11": "Fluent (Windows 11)",
    "macos_native": "macOS native",
    "dark": "Dark",
}
for _key in APP_STYLE_QSS:
    APP_STYLE_LABELS.setdefault(_key, _key.replace("_", " ").title())


def available_app_styles() -> list[tuple[str, str]]:
    """Return every selectable style as (stored key, label to show):
    Automatic, then each stylesheet shipped in app/styles."""
    return [(key, APP_STYLE_LABELS[key]) for key in APP_STYLE_LABELS]


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
    in config.json, or a Qt style or stylesheet path chosen in an older
    version (``qt:...``, ``file:...``), should leave the app looking normal
    rather than unstyled.
    """
    clean = str(preference or "").strip()
    if not clean:
        clean = str(get_value(CONFIG_APP_STYLE, "") or "").strip()

    lowered = clean.lower()
    return lowered if lowered in APP_STYLE_QSS else APP_STYLE_AUTOMATIC


_POPUP_CONTAINER_SHEET = "background: palette(base); border: none;"
_combo_init_patched = False


def _frameless_combo_popups() -> None:
    """Give every combo's pop-up container a frameless sheet as the combo is made (todo P-02).

    On macOS the container paints a grey menu panel inside the pop-up window,
    which the system already frames with its rounded edge and shadow: two
    frames. The application's sheet does not reach that private widget, a
    sheet on the container itself does. Set once per combo, when it is built
    (every combo here is built from Python, subclasses included) - not from
    an application-wide event filter, which crashed PySide on objects being
    destroyed.
    """
    global _combo_init_patched
    if _combo_init_patched:
        return
    original = QtWidgets.QComboBox.__init__

    def init(self: QtWidgets.QComboBox, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        container = self.view().parentWidget()
        if container is not None and not container.styleSheet():
            container.setStyleSheet(_POPUP_CONTAINER_SHEET)

    QtWidgets.QComboBox.__init__ = init  # type: ignore[method-assign]
    _combo_init_patched = True


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
    if str(qss_name).startswith("macos_native"):
        _frameless_combo_popups()
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
    # Owned by the view: QWidget.setStyle does not take ownership, and an
    # unowned Python style was freed while the view still drew with it - a
    # "QCommonStyle already deleted" seen at random, a crash at worst.
    fusion = QtWidgets.QStyleFactory.create("Fusion")
    fusion.setParent(view)
    view.setStyle(fusion)

    application_style = QtWidgets.QApplication.style()
    for reader in ("verticalScrollBar", "horizontalScrollBar"):
        # getattr: the helper's argument is a QWidget, and only a scroll
        # area has these. Nothing else needs putting back.
        scrollbar = getattr(view, reader, None)
        bar = scrollbar() if callable(scrollbar) else None
        if isinstance(bar, QtWidgets.QScrollBar):
            bar.setStyle(application_style)


# ----------------------------------------------------------------------
