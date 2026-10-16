"""Icons: action icons from the catalogue, SF Symbols, Segoe Fluent glyphs, theme icons and SVG."""
from __future__ import annotations

import re
import tempfile
import sys
from functools import lru_cache
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import QByteArray, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QImage, QPainter, QPen, QPixmap
from app.logs.logger import applogger
from app.styles.style.metrics import _IS_MACOS, _IS_WINDOWS
from app.styles.style.catalogue import ActionSpec, _catalog, action


# Icons
# ----------------------------------------------------------------------

_FLUENT_GLYPH_MIN = 0xE700
_FLUENT_GLYPH_MAX = 0xF8FF
# Fallback ink only.  The real colour comes from _symbol_tint(), the same
# source the SF Symbol path uses - the two glyph backends were tinted
# differently, so the Windows glyphs stayed a fixed dark grey and vanished on
# a dark theme while the macOS ones followed the palette.
_FLUENT_DEFAULT_COLOR = "#404040"
# Keyed by what changes the bitmap: the glyph or symbol, the logical size,
# the tint, and the pixel ratio it was rasterised for.  The ratio belongs in
# the key - without it a 1x icon built during startup is handed back on a
# Retina screen, which is the blur this was meant to fix.
_FLUENT_ICON_CACHE: dict[tuple[str, int, str, float], QIcon] = {}
_SF_SYMBOL_ICON_CACHE: dict[tuple[str, int, str, float], QIcon] = {}

# The AppKit module, looked up once.  None means "not available"; the flag
# distinguishes that from "not looked up yet", so the warning is logged once
# rather than for every button.
_SF_SYMBOL_BRIDGE: Any | None = None
_SF_SYMBOL_BRIDGE_CHECKED: bool = False

# The oldest pyobjc-core that is safe to import, per Python version.
#
# pyobjc-core is a C extension built against CPython's internals, and it is
# unusually sensitive to them.  A wheel built during a Python beta keeps its
# cp3XX tag after those internals move underneath it, so pip installs it
# happily and the interpreter then dies on import - a native crash, below the
# level any try/except can reach.  Refusing to import is the only defence.
#
# Each entry reads: on this Python or newer, require this pyobjc-core.
_MIN_PYOBJC_CORE: tuple[tuple[tuple[int, int], tuple[int, ...]], ...] = (
    ((3, 14), (12, 0)),  # 11.1's cp314 wheel predates the 3.14 release
    ((3, 10), (11, 0)),
)


def _normalize_icon_name(icon: str | None) -> str:
    """Return a normalized action id, SVG icon id, SF Symbol id or glyph id."""
    return str(icon or "").strip()



def action_presentation(action_id: str) -> tuple[QIcon, str, str]:
    """Return the icon, label and tooltip configured for an action.

    The action catalogue is the only source for button text, tooltips and icon
    tokens. Missing action entries intentionally produce an empty presentation,
    which makes catalogue mistakes visible during development instead of hiding
    them behind hard-coded fallbacks.
    """
    spec = action(action_id)
    text = spec.translated_text()
    tooltip = spec.translated_description() or text
    return icon_from_action_spec(spec), text, tooltip


def _is_fluent_glyph(value: str | None) -> bool:
    """Return True for a Segoe Fluent private-use glyph or hex code."""
    text = _normalize_icon_name(value)
    if not text:
        return False

    if len(text) == 1:
        return _FLUENT_GLYPH_MIN <= ord(text) <= _FLUENT_GLYPH_MAX

    candidate = text.upper()
    if candidate.startswith("U+"):
        candidate = candidate[2:]
    elif candidate.startswith("0X"):
        candidate = candidate[2:]

    if not (3 <= len(candidate) <= 6):
        return False

    try:
        codepoint = int(candidate, 16)
    except ValueError:
        return False

    return _FLUENT_GLYPH_MIN <= codepoint <= _FLUENT_GLYPH_MAX


def _fluent_glyph_char(value: str) -> str:
    """Return the actual glyph character for a Fluent code or glyph."""
    text = _normalize_icon_name(value)
    if len(text) == 1:
        return text

    candidate = text.upper()
    if candidate.startswith("U+"):
        candidate = candidate[2:]
    elif candidate.startswith("0X"):
        candidate = candidate[2:]

    return chr(int(candidate, 16))


#: The body font the Fluent sheet asks for, best first, resolved against what
#: is actually installed by :func:`ui_font_family`.
#:
#: "Segoe UI Variable" is deliberately *not* in this list. It reads like the
#: Windows 11 UI font's name and is what the sheet used to name, but no such
#: family is ever installed: Windows 11 registers the variable face under
#: three optical sizes - "Segoe UI Variable Small", "... Text" and
#: "... Display" - and nothing under the bare name. Text is the one meant for
#: body UI at this sheet's 10pt.
_UI_FONT_STACK: tuple[str, ...] = (
    "Segoe UI Variable Text",
    "Segoe UI",
    "Inter",
    "Arial",
)


@lru_cache(maxsize=1)
def ui_font_family() -> str:
    """Return a ``font-family`` value naming only fonts that exist here.

    The same rule as :func:`_best_fluent_font_family` below, for the body
    font rather than the icon font, and for the same reason: a QSS
    ``font-family`` that names a missing family makes Qt go looking for it,
    and on Windows that lookup walks every alias the system knows before
    giving up -

        Populating font family aliases took 200 ms. Replace uses of missing
        font family "Segoe UI Variable" with one that exists to avoid this
        cost.

    - which is a fifth of a second of the app's startup, plus a warning in
    every log, spent to arrive at the second name in the list. Checking
    ``families()`` first costs nothing extra: the font database is populated
    on the first query either way, and a family that *is* there never
    reaches the alias path.

    Falls back to whatever Qt itself considers the UI font, which is by
    definition installed, rather than to another guess.
    """
    families = set(QFontDatabase.families())
    for family in _UI_FONT_STACK:
        if family in families:
            return f'"{family}"'
    return f'"{QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()}"'


#: What :func:`ui_font_family` replaces in a sheet. Substituted after the
#: palette's own tokens (see themed_qss) because it is not a palette
#: property: the same theme resolves to a different family on a machine with
#: different fonts installed.
UI_FONT_FAMILY_TOKEN = "@UI_FONT_FAMILY@"


def substitute_ui_font(qss: str) -> str:
    """Resolve :data:`UI_FONT_FAMILY_TOKEN` in *qss*.

    A sheet that does not use the token - the macOS one, or a sheet of the
    user's own - passes through untouched, and without the font database
    being read at all.
    """
    if UI_FONT_FAMILY_TOKEN not in qss:
        return qss
    return qss.replace(UI_FONT_FAMILY_TOKEN, ui_font_family())


def _best_fluent_font_family() -> str | None:
    """Return an installed Fluent icon font, or None.

    Do not fall back to a generic/system font here. On Windows, asking Qt to
    render a private-use Fluent glyph with a missing font can trigger DirectWrite
    fallback through legacy bitmap fonts such as 8514oem or Fixedsys, which
    produces noisy CreateFontFaceFromHDC warnings.
    """
    families = set(QFontDatabase.families())
    for family in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if family in families:
            return family
    return None


# ----------------------------------------------------------------------
# Theme icons (freedesktop / QIcon.fromTheme)
# ----------------------------------------------------------------------
# The third icon backend, and the one that covers Linux.  SF Symbols are
# macOS's and Segoe Fluent is Windows's; neither exists elsewhere, which left
# every other desktop on the SVGs shipped in app/icons - drawn once, by hand,
# and never matching the system they were drawn onto.  QIcon.fromTheme reads
# the icon theme the desktop is already using, so the application's Open
# looks like every other Open on that machine.
#
# The names are freedesktop's, from config.json's ThemeIcon field; the test
# suite checks each one (dev/tests/test_theme_icon_names.py), because a typo
# in a theme name is silent - fromTheme returns a null icon and the SVG
# quietly takes over.

_THEME_ICON_CACHE: dict[str, QIcon] = {}


#: Appended to a theme name when the plain one is not in the theme.
#:
#: Not a detail: GNOME's Adwaita stopped shipping the full-colour action
#: icons at version 45 and now carries only the monochrome set, so on a
#: current GNOME desktop ``document-open`` is missing and
#: ``document-open-symbolic`` is the icon. KDE's Breeze and Papirus still
#: ship both. Asking for the plain name first and this second is what covers
#: the two, and in the right order: where both exist, the plain one is the
#: theme's full-colour artwork and the more native answer.
_THEME_ICON_SYMBOLIC_SUFFIX: str = "-symbolic"


#: Themes to fall back to when the desktop names none, best first.
#:
#: Not a preference: these are the sets that are actually installed on the
#: machines that need them.  Adwaita comes with GNOME and with GTK itself,
#: Breeze with KDE, Papirus is the most widely installed third-party set, and
#: hicolor is last because the specification guarantees it exists and it
#: contains almost no action icons - it is the answer only when there is no
#: other.
_FALLBACK_ICON_THEMES: tuple[str, ...] = (
    "Adwaita",
    "breeze",
    "Papirus",
    "gnome",
    "oxygen",
    "hicolor",
)

#: What themeName() says when the desktop has not chosen one.  Qt returns the
#: empty string; hicolor is treated the same way because a theme with no
#: action icons in it is not a theme this application can draw from.
_UNSET_ICON_THEMES: frozenset[str] = frozenset({"", "hicolor"})


#: Where icon themes live on a freedesktop system, per the XDG base
#: directory spec and the Icon Theme Specification.
_XDG_ICON_DIRS: tuple[str, ...] = (
    "~/.icons",
    "~/.local/share/icons",
    "/usr/local/share/icons",
    "/usr/share/icons",
)


def _ensure_xdg_icon_search_paths() -> None:
    """Make sure Qt is looking where icon themes are actually installed.

    Qt does not populate ``themeSearchPaths`` itself: the platform theme
    plugin does, and that plugin is what a desktop session provides. Under
    GNOME or KDE the paths are therefore already right. On the desktops
    :func:`ensure_icon_theme` exists for - a bare window manager, XFCE with
    no settings daemon - and under the offscreen platform, ``themeSearchPaths``
    is just ``[':/icons']``, Qt's own resource path, and nothing under
    ``/usr/share/icons`` is ever looked at.

    That made the probe below answer "no themes installed" on a machine
    with several, so no theme was set, and every one of the catalogue's
    theme icons came back null - the whole Linux icon backend silently off
    on exactly the desktops it was written to cover.

    Only directories that exist are added, and only once; a path Qt already
    has is left where it is, so a desktop that ordered them itself keeps its
    order and its precedence.
    """
    paths = [str(directory) for directory in QIcon.themeSearchPaths()]
    known = set(paths)
    for candidate in _XDG_ICON_DIRS:
        resolved = str(Path(candidate).expanduser())
        if resolved not in known and Path(resolved).is_dir():
            paths.append(resolved)
            known.add(resolved)
    if len(paths) != len(QIcon.themeSearchPaths()):
        QIcon.setThemeSearchPaths(paths)


def _installed_icon_themes() -> list[str]:
    """Return the theme names actually present in Qt's search paths."""
    _ensure_xdg_icon_search_paths()
    found: list[str] = []
    for directory in QIcon.themeSearchPaths():
        root = Path(str(directory))
        if not root.is_dir():
            continue
        for name in _FALLBACK_ICON_THEMES:
            if name in found:
                continue
            # index.theme is what makes a directory a theme rather than a
            # folder of loose files; Qt will not read one without it.
            if (root / name / "index.theme").is_file():
                found.append(name)
    return found


def ensure_icon_theme() -> str:
    """Give Qt an icon theme to read when the desktop has not named one.

    The theme backend is what covers Linux, and it is silent when it fails:
    ``fromTheme`` returns a null icon and the shipped SVG takes over, so a
    machine with no theme set looks exactly like one from before theme icons
    existed.  GNOME and KDE set the name through their platform integration,
    but a bare window manager - i3, or XFCE without its settings daemon -
    sets nothing, and that is a whole class of Linux desktop getting the old
    hand-drawn icons for no reason.

    A *fallback* theme is set in every case, which is Qt's own mechanism for
    "look here for what the chosen theme is missing", and the theme name
    itself only when nothing has claimed it.  Nothing is overridden: a user
    who chose Papirus keeps Papirus.

    Harmless where there are no themes at all, which is what Windows and
    macOS look like: the probe finds nothing and nothing is set.  Returns the
    theme in use afterwards, or "" when there is still none.
    """
    installed = _installed_icon_themes()
    if not installed:
        return str(QIcon.themeName() or "")

    if str(QIcon.fallbackThemeName() or "") in _UNSET_ICON_THEMES:
        QIcon.setFallbackThemeName(installed[0])

    current = str(QIcon.themeName() or "")
    if current in _UNSET_ICON_THEMES:
        QIcon.setThemeName(installed[0])
        applogger.info(
            "No icon theme was set; using %r of %s.", installed[0], installed
        )
        # Anything asked for before now was answered against no theme at all.
        _THEME_ICON_CACHE.clear()
        current = installed[0]

    return current


def _theme_icon_names(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Return a ThemeIcon field as the ordered names to try."""
    if not value:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _theme_icon_candidates(token: str) -> tuple[str, ...]:
    """Return the names to try for one configured theme icon, best first."""
    if token.endswith(_THEME_ICON_SYMBOLIC_SUFFIX):
        return (token,)
    return (token, token + _THEME_ICON_SYMBOLIC_SUFFIX)


def _create_theme_icon(name: str | None) -> QIcon:
    """Return the desktop's own icon for a freedesktop *name*, or an empty one.

    Empty is the honest answer on a desktop with no icon theme installed, and
    on Windows and macOS, where Qt has no theme to read at all.  The caller
    falls through to the shipped SVG, which is why those are still here.

    Not tinted, unlike the two glyph backends: a theme icon is already drawn
    for the desktop's own light or dark variant, and painting over it would
    replace artwork the user chose with a flat silhouette.
    """
    token = _normalize_icon_name(name)
    if not token:
        return QIcon()

    cached = _THEME_ICON_CACHE.get(token)
    if cached is not None:
        return cached

    icon = QIcon()
    for candidate in _theme_icon_candidates(token):
        try:
            # hasThemeIcon first: fromTheme on a missing name can return a
            # non-null icon carrying nothing, which would end the search on
            # the wrong candidate and hand a blank icon to a toolbar.
            if not QIcon.hasThemeIcon(candidate):
                continue
            found = QIcon.fromTheme(candidate)
        except Exception:  # pragma: no cover - defensive; fromTheme does not raise
            applogger.exception("Failed to read theme icon %r", candidate)
            continue
        if not found.isNull():
            icon = found
            break

    # Deliberately cached even when null. A missing name stays missing for
    # the life of the process, and the alternative is asking the theme engine
    # for it again on every button that names it.
    _THEME_ICON_CACHE[token] = icon
    return icon


def theme_icon_is_available(name: str | None) -> bool:
    """True when the running desktop can actually supply this icon."""
    return not _create_theme_icon(name).isNull()


def _create_fluent_icon(
    glyph: str,
    *,
    size: int = 20,
    color: str = _FLUENT_DEFAULT_COLOR,
) -> QIcon:
    """Create a QIcon by painting a Segoe Fluent glyph into a pixmap.

    Returns an empty QIcon when no Fluent icon font is installed. That lets the
    caller fall back to the configured SVG icon without causing DirectWrite font
    fallback warnings.
    """
    font_family = _best_fluent_font_family()
    if font_family is None:
        return QIcon()

    glyph_char = _fluent_glyph_char(glyph)
    color_name = str(color or _FLUENT_DEFAULT_COLOR)
    ratio = _icon_device_pixel_ratio()
    cache_key = (glyph_char, int(size), color_name, round(ratio, 2))

    cached = _FLUENT_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # The pixmap carries the display's pixel ratio, so the coordinates below
    # stay logical while the glyph is rasterised at full resolution.
    pixmap = create_hidpi_pixmap(size, size)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont(font_family)
        font.setPixelSize(max(1, int(size * 0.85)))
        painter.setFont(font)
        qcolor = QColor(color_name)
        painter.setPen(qcolor if qcolor.isValid() else QColor(_FLUENT_DEFAULT_COLOR))
        painter.drawText(
            QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, glyph_char
        )
    finally:
        painter.end()

    icon = QIcon(pixmap)
    _FLUENT_ICON_CACHE[cache_key] = icon
    return icon


def _pyobjc_core_is_safe_to_import() -> bool:
    """Say whether the installed pyobjc-core can be imported without crashing.

    Read from the distribution metadata rather than by importing the package:
    the failure being guarded against is a segmentation fault inside the import
    itself, so asking the question has to cost nothing native.

    An unknown or unparseable version is treated as safe.  The guard exists to
    catch one known-bad combination, not to police the dependency: refusing to
    run on a version this table has not heard of would age badly.
    """
    try:
        import importlib.metadata

        installed = importlib.metadata.version("pyobjc-core")
    except Exception:
        return True

    try:
        parsed = tuple(int(part) for part in installed.split(".")[:3])
    except ValueError:
        return True

    for python_version, minimum in _MIN_PYOBJC_CORE:
        if sys.version_info[:2] >= python_version:
            if parsed >= minimum:
                return True
            applogger.warning(
                "pyobjc-core %s is not built for Python %d.%d and importing it "
                "would crash the application, so the SVG icons are used "
                "instead. Upgrade it with: pip install -U "
                "'pyobjc-framework-Cocoa>=%s'",
                installed,
                sys.version_info[0],
                sys.version_info[1],
                ".".join(str(part) for part in minimum),
            )
            return False

    return True


def _sf_symbol_bridge() -> Any | None:
    """Return the AppKit module, or None with one explanation in the log.

    SF Symbols are drawn by AppKit, which reaches Python through pyobjc.  It is
    a real dependency and it is easy not to have: without it every symbol
    lookup fails, and the previous version of this code swallowed that in a
    bare ``except``, so the whole icon set silently fell back to SVG with
    nothing anywhere to say why.
    """
    global _SF_SYMBOL_BRIDGE, _SF_SYMBOL_BRIDGE_CHECKED

    if _SF_SYMBOL_BRIDGE_CHECKED:
        return _SF_SYMBOL_BRIDGE

    # Cleared up front so that both failure paths below leave the memo saying
    # "looked, found nothing" rather than whatever it happened to hold.
    _SF_SYMBOL_BRIDGE_CHECKED = True
    _SF_SYMBOL_BRIDGE = None
    if not _pyobjc_core_is_safe_to_import():
        return None

    try:
        import AppKit  # type: ignore[import-not-found]
    except ImportError:
        applogger.warning(
            "SF Symbols need pyobjc, which is not installed; using the SVG "
            "icons instead. Install it with: pip install pyobjc-framework-Cocoa",
        )
        return None

    _SF_SYMBOL_BRIDGE = AppKit
    return AppKit


def _create_sf_symbol_icon(symbol: str, *, size: int = 20) -> QIcon:
    """Create a QIcon from a macOS SF Symbol id.

    Returns an empty QIcon when the symbol cannot be drawn, so the caller falls
    back to the SVG.  Every failure is logged once per symbol: an icon that is
    quietly missing is the kind of thing nobody reports and nobody fixes.

    The symbol is rasterised at the display's pixel density rather than at the
    logical size: a 20 px bitmap stretched across a Retina screen's 40 device
    pixels is the whole of why these icons looked soft next to the SVGs, which
    are drawn as vectors and so never had the problem.
    """
    name = _normalize_icon_name(symbol)
    if not name or not _IS_MACOS:
        return QIcon()

    ratio = _icon_device_pixel_ratio()
    # The tint belongs in the key, exactly as it does for the Fluent glyphs.
    # Without it, symbols built once under the light palette were handed back
    # unchanged after a switch to the dark theme - dark ink on a dark button,
    # cached and therefore permanent until restart.
    tint = _symbol_tint()
    cache_key = (name, int(size), tint, round(ratio, 2))
    cached = _SF_SYMBOL_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    appkit = _sf_symbol_bridge()
    if appkit is None:
        return QIcon()

    icon = QIcon()
    try:
        image = appkit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            name, None
        )
        if image is None:
            # A name this version of macOS does not know: worth saying, since
            # the catalogue is hand-written and a typo looks like nothing.
            applogger.warning(
                "Unknown SF Symbol %r; using the SVG icon instead.",
                name,
            )
            _SF_SYMBOL_ICON_CACHE[cache_key] = icon
            return icon

        # Ask for the symbol at the *device* size.  A point size of 20 gives a
        # 20 px bitmap, which Qt then has to stretch over 40 physical pixels on
        # a Retina display; asking for 40 up front means no resampling at all.
        configuration = appkit.NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(
            float(size) * ratio, appkit.NSFontWeightRegular, 2  # 2 = .medium scale
        )
        configured = image.imageWithSymbolConfiguration_(configuration) or image
        configured.setTemplate_(True)

        tiff = configured.TIFFRepresentation()
        if tiff is None:
            raise ValueError("the symbol produced no TIFF representation")

        rep = appkit.NSBitmapImageRep.imageRepWithData_(tiff)
        if rep is None:
            raise ValueError("the TIFF representation could not be read")

        # An empty properties dictionary, not None: the parameter is typed as
        # an NSDictionary and pyobjc will not convert None for it.
        png_data = rep.representationUsingType_properties_(
            appkit.NSBitmapImageFileTypePNG, {}
        )
        if png_data is None:
            raise ValueError("the symbol could not be encoded as PNG")

        # No format argument: Pylance's PySide6 stubs type that overload
        # incorrectly, and Qt detects PNG from the bytes anyway.
        qimage = QImage.fromData(bytes(png_data))
        if qimage.isNull():
            raise ValueError("Qt could not read the PNG data")

        icon = QIcon(
            _tinted_pixmap(QPixmap.fromImage(qimage), size, tint, ratio=ratio)
        )
    except Exception as error:
        applogger.warning(
            "SF Symbol %r could not be rendered (%s); using the SVG icon instead.",
            name,
            error,
        )
        icon = QIcon()

    # Cached either way, including the empty icon: a symbol that failed once
    # will fail every time, and this runs for every button on every menu.
    _SF_SYMBOL_ICON_CACHE[cache_key] = icon
    return icon


def _icon_device_pixel_ratio() -> float:
    """Return how many physical pixels the display gives one logical pixel.

    Read from the primary screen rather than from the application: the ratio
    is a property of a display, and the application-level accessor is not part
    of QCoreApplication's interface even where it happens to resolve.  Icons
    are cached, so they cannot follow a window between screens of different
    densities anyway; the primary screen is the honest approximation.

    Falls back to 2.0 on macOS when no screen exists yet, since icons are
    sometimes built while the app is still starting and every Mac that can run
    this is Retina.  Clamped to 3: beyond that the bitmaps grow faster than
    anyone can see the difference.
    """
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return 2.0 if _IS_MACOS else 1.0

    return min(max(float(screen.devicePixelRatio()), 1.0), 3.0)


def create_hidpi_pixmap(width: int, height: int) -> QPixmap:
    """Return a transparent pixmap that paints sharply on a Retina display.

    The bitmap is allocated at the display's physical pixel count and labelled
    with its ratio, which is what tells Qt to draw it one device pixel per
    pixel rather than interpolating.  Because the ratio is set on the pixmap,
    QPainter scales for it: callers keep drawing in the logical coordinates
    they already use, and only the result is sharper.

    Use this anywhere an icon is painted by hand.  Every such place in the app
    allocated the bitmap at logical size, so all of them were soft on a Mac
    while the SVG icons beside them - vectors, rasterised on demand - were not.
    """
    ratio = _icon_device_pixel_ratio()
    pixmap = QPixmap(
        max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))
    )
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


#: Whether the theme currently applied is a dark one.  Set by
#: apply_platform_style, which is the only thing that knows.
_ACTIVE_THEME_IS_DARK: bool = False

#: Glyph ink, light and dark.  Near-black rather than pure black because the
#: system draws its own symbols that way and it is what makes them read as
#: crisp; near-white for the same reason in reverse.
SYMBOL_INK_LIGHT: str = "#eaeaea"
SYMBOL_INK_DARK: str = "#1a1a1a"


def _symbol_tint() -> str:
    """Return the colour to paint SF Symbols and Fluent glyphs in.

    Decided by the theme this application applied, not by reading ButtonText
    off the palette.  The palette is not a reliable answer here: a Qt style
    plugin installs its own, and on a Mac whose system appearance is dark that
    palette comes back light-on-dark even though the app is wearing a light
    theme - which painted white glyphs onto white buttons.

    The rule is therefore the simple one: light ink only under a dark theme,
    dark ink under everything else.
    """
    return SYMBOL_INK_LIGHT if _ACTIVE_THEME_IS_DARK else SYMBOL_INK_DARK


def _opaque_bounding_rect(pixmap: QPixmap) -> QRect:
    """Return the bounding rectangle of a pixmap's non-transparent pixels.

    Some SF Symbols pad their own canvas off-centre - ``square.and.arrow.down``
    (the Save glyph) measured 10px of empty space above its ink against 4px
    below, out of an 82px-tall source canvas, while most symbols split that
    evenly. Centring the pixmap's *rectangle* still carries that imbalance
    straight into the icon, which is what made Save sit visibly lower than
    Reload or Copy in the same menu although every icon was the same
    dimensions.  Falls back to the full pixmap rect when there is no ink to
    measure or the scan fails - a blank or unreadable glyph is nothing to
    special-case.
    """
    try:
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        width, height = image.width(), image.height()
        if width <= 0 or height <= 0:
            return pixmap.rect()

        stride = image.bytesPerLine()
        buffer = bytes(image.constBits())[: stride * height]
        rows = np.frombuffer(buffer, dtype=np.uint8).reshape((height, stride))
        alpha = rows[:, 3 : width * 4 : 4]

        ys, xs = np.nonzero(alpha > 10)
        if ys.size == 0:
            return pixmap.rect()
        return QRect(
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min()) + 1,
            int(ys.max() - ys.min()) + 1,
        )
    except Exception:
        return pixmap.rect()


def _tinted_pixmap(
    pixmap: QPixmap,
    size: int,
    color: str = _FLUENT_DEFAULT_COLOR,
    *,
    ratio: float = 1.0,
) -> QPixmap:
    """Repaint a pixmap's opaque parts in one colour at the display's density.

    ``size`` is in logical pixels and ``ratio`` says how many real ones each of
    those is worth.  The bitmap is built at the physical size and then labelled
    with its ratio, so Qt draws it one-for-one instead of interpolating - the
    difference between a sharp icon and a smeared one.

    The result is always exactly ``size * ratio`` square, whatever the source's
    own proportions.  SF Symbols are not square - a trash can is narrower than
    it is tall, a document narrower still - and ``scaled(..., KeepAspectRatio)``
    on its own returns a pixmap sized to *that* glyph rather than to the slot
    every icon is supposed to share.  Qt's menu icon column takes its width
    from the pixmap it is handed, so those items sat at a different horizontal
    offset from their text than the ones next to them - a document icon flush
    against its label, a trash can with a visible gap, in the same menu.  The
    Segoe Fluent path this is the counterpart of never had the problem: a
    glyph painted with ``Qt.AlignCenter`` into a fixed-size rect is centred by
    construction.  Centring here, after scaling, is what makes the two paths
    agree.

    Centred on the scaled pixmap's *ink*, not its rectangle - see
    ``_opaque_bounding_rect``. A glyph whose own canvas already sits its ink
    off-centre would otherwise carry that offset straight through, evenly
    padded rectangle and all.
    """
    physical = max(1, int(round(size * ratio)))
    scaled = (
        pixmap
        if pixmap.width() == physical and pixmap.height() == physical
        else pixmap.scaled(
            physical,
            physical,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )

    qcolor = QColor(color)
    if qcolor.isValid():
        painter = QPainter(scaled)
        try:
            # SourceIn keeps the alpha and replaces the colour.
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(scaled.rect(), qcolor)
        finally:
            painter.end()

    ink = _opaque_bounding_rect(scaled)
    ink_center_x = ink.x() + ink.width() / 2.0
    ink_center_y = ink.y() + ink.height() / 2.0
    offset_x = round(physical / 2.0 - ink_center_x)
    offset_y = round(physical / 2.0 - ink_center_y)

    if scaled.width() != physical or scaled.height() != physical or offset_x or offset_y:
        canvas = QPixmap(physical, physical)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        try:
            painter.drawPixmap(offset_x, offset_y, scaled)
        finally:
            painter.end()
        scaled = canvas

    # Without this Qt treats the bitmap as logical pixels and draws it at twice
    # the intended size.
    scaled.setDevicePixelRatio(ratio)
    return scaled

_SVG_SOURCE_ICON_CACHE: dict[tuple[str, int, float], QIcon] = {}


#: The document a bare icon body is wrapped in.  An operation declares its
#: artwork as path data alone - see ``DataOperationDialogBase.Icon`` - which
#: is the readable half; the frame around it is always the same and repeating
#: it in eleven plugins would only be eleven places to get the viewBox wrong.
SVG_ICON_DOCUMENT: str = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
)


def svg_icon_document(body: str, color: str | None = None) -> str:
    """Return icon path data wrapped in the standard SVG document."""
    return SVG_ICON_DOCUMENT.format(color=color or _symbol_tint(), body=body)


def icon_from_svg_source(
    svg: str | None,
    *,
    size: int = 20,
    color: str | None = None,
) -> QIcon:
    """Return a QIcon drawn from an SVG document held in memory.

    For icons that travel inside the thing they belong to rather than in
    ``app/icons``: a data operation is a plugin, and one .py file dropped
    into ``app/data_operations`` has to be the whole of it, artwork
    included.  See ``DataOperationDialogBase.Icon``.

    Unlike the theme and glyph paths this rasterises once at a chosen size, because the
    QIcon is built from a pixmap rather than from a file Qt can re-read.  So
    the pixel ratio belongs in the key, exactly as it does for the SF Symbol
    and Fluent glyph paths.

    Malformed markup returns an empty icon and says so once: an operation that
    silently lost its icon is the kind of thing nobody reports.
    """
    source = (svg or "").strip()
    if not source:
        return QIcon()

    if "<svg" not in source.lower():
        # Path data on its own: wrap it. This used to be done by the widget
        # that lists the operations, so every *other* caller got an empty icon
        # from perfectly good markup - which is why the operation dialogs had
        # no window icon and nothing said so.
        source = svg_icon_document(source, color)

    ratio = _icon_device_pixel_ratio()
    # Keyed on the document itself.  These come from class attributes, so the
    # same str object arrives every time and Python has already cached its
    # hash; there is nothing to gain from digesting it first.
    cache_key = (source, int(size), round(ratio, 2))  # colour is inside source
    cached = _SVG_SOURCE_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    renderer = QSvgRenderer(QByteArray(source.encode("utf-8")))
    if not renderer.isValid():
        applogger.warning(
            "An embedded icon could not be parsed as SVG (starts %r); the "
            "widget will have no icon.",
            source[:60],
        )
        icon = QIcon()
    else:
        # KeepAspectRatio rather than the default stretch: these are hand-drawn
        # and a non-square viewBox should letterbox, not distort.
        renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = create_hidpi_pixmap(size, size)
        painter = QPainter(pixmap)
        try:
            renderer.render(painter, QRectF(0, 0, size, size))
        finally:
            painter.end()
        icon = QIcon(pixmap)

    # Cached either way: markup that failed to parse will fail every time, and
    # this runs for every operation button the window builds.
    _SVG_SOURCE_ICON_CACHE[cache_key] = icon
    return icon


#: Images a stylesheet asks for as url(@ICON:name@): (SF Symbol, Segoe
#: Fluent glyph, theme icon), rendered to a PNG when the sheet is applied.
QSS_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "combo_chevrons": ("chevron.up.chevron.down", "E70D", "go-down"),
    "chevron_down": ("chevron.down", "E70D", "go-down"),
    "chevron_up": ("chevron.up", "E70E", "go-up"),
}

_QSS_SYMBOL_RE = re.compile(r"@ICON:([A-Za-z0-9_]+)@")


#: Images drawn here rather than taken from a symbol font: name -> a function
#: that paints the 48 x 48 pixmap. A check mark is white whatever the theme,
#: because it sits on the highlight colour of a checked box.
def _paint_checkbox_tick(pixmap: QPixmap) -> None:
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#ffffff"))
        pen.setWidthF(5.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline([QPointF(12, 25), QPointF(21, 34), QPointF(37, 15)])
    finally:
        painter.end()


QSS_DRAWN: dict[str, Callable[[QPixmap], None]] = {"checkbox_tick": _paint_checkbox_tick}


def _qss_symbol_file(name: str) -> str | None:
    """Render one QSS_SYMBOLS entry to a PNG and return its path, or None."""
    painter_for = QSS_DRAWN.get(name)
    if painter_for is not None:
        pixmap = QPixmap(48, 48)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter_for(pixmap)
        target = Path(tempfile.gettempdir()) / "chartlibre-qss" / f"{name}.png"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            return target.as_posix() if pixmap.save(str(target), "PNG") else None
        except Exception:
            applogger.debug("Could not draw the QSS image %s.", name, exc_info=True)
            return None
    spec = QSS_SYMBOLS.get(name)
    if spec is None:
        return None
    sf_symbol, fluent, theme = spec
    icon = QIcon()
    if _IS_MACOS:
        icon = _create_sf_symbol_icon(sf_symbol, size=24)
    elif _IS_WINDOWS and _is_fluent_glyph(fluent):
        icon = _create_fluent_icon(fluent, color=_symbol_tint())
    if icon.isNull():
        icon = symbol_icon("", theme)
    if icon.isNull():
        return None
    target = Path(tempfile.gettempdir()) / "chartlibre-qss" / f"{name}.png"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not icon.pixmap(QSize(48, 48)).save(str(target), "PNG"):
            return None
    except Exception:
        applogger.debug("Could not render the QSS symbol %s.", name, exc_info=True)
        return None
    return target.as_posix()


def _substitute_qss_symbols(qss: str) -> str:
    """Replace every url(@ICON:name@) with a rendered PNG, or drop the image."""
    def image(match: re.Match[str]) -> str:
        return _qss_symbol_file(match.group(1)) or ""

    return _QSS_SYMBOL_RE.sub(image, qss) if "@ICON:" in qss else qss


def colored_icon(icon: QIcon, color: QColor | str, size: int = 20) -> QIcon:
    """*icon* repainted in one colour, at the display's pixel density.

    For the few places where an icon carries meaning by its colour - a
    recent project drawn as a blue document, the way Finder shows one -
    rather than taking the ink every other glyph shares.
    """
    if icon.isNull():
        return icon
    ratio = _icon_device_pixel_ratio()
    physical = max(1, int(round(size * ratio)))
    source = icon.pixmap(QSize(physical, physical))
    if source.isNull():
        return icon
    name = color.name() if isinstance(color, QColor) else str(color)
    return QIcon(_tinted_pixmap(source, size, name, ratio=ratio))


def symbol_icon(sf_symbol: str, theme_icon: str = "") -> QIcon:
    """An SF Symbol on macOS, else a theme icon, else an empty icon.

    For icons chosen from data rather than from the action catalogue - one
    per chart type in the navigation list.
    """
    if _IS_MACOS and sf_symbol:
        icon = _create_sf_symbol_icon(sf_symbol)
        if not icon.isNull():
            return icon
    for name in _theme_icon_names(theme_icon):
        icon = _create_theme_icon(name)
        if not icon.isNull():
            return icon
    return QIcon()


def icon_from_action_spec(spec: "ActionSpec") -> QIcon:
    """Return the icon for an action using the catalogue priority order.

    One rule, applied in order, and every step falls through when it produces
    nothing:

    0. ``SVG``, when the catalogue draws the icon itself: an action no system
       set has a glyph for (a table's rows told apart from its columns).

    1. On macOS, ``SFSymbol`` - the system's own icon set, so the app looks
       like a Mac app.
    2. On Windows, ``SegoeFluent`` - likewise.
    3. Anywhere, ``ThemeIcon`` through ``QIcon.fromTheme``. This is what
       covers Linux, where neither of the above exists and where the desktop
       already has an icon theme the user chose. It sits after the two
       platform sets rather than before them because on a Mac or a Windows
       machine there is no theme to read, so it would only ever be a wasted
       lookup - and on the rare Linux-style desktop running under either, the
       native set is still the more native answer.
    4. An empty ``QIcon``.

    There used to be a fifth step, and 67 files behind it: an SVG shipped in
    ``app/icons``, drawn once by hand, matching no system in particular. It
    was the answer for a Mac with no pyobjc, a PC with no Fluent font, and a
    Linux box with no icon theme. Those are the three cases that now show no
    icon at all, which was a deliberate choice: three real icon sets cover
    every machine anyone actually runs, and a hand-drawn fallback for the
    rest is a maintenance burden charged to everybody.

    ``report_icon_sources`` says which backend answers for each action on the
    machine it runs on, and what is left under ``"none"`` is what nothing can
    draw there.
    """
    if spec.svg:
        # Drawn for this action because no system glyph says it: preferred
        # wherever it is given, so the icon is the same on every platform.
        icon = icon_from_svg_source(spec.svg)
        if not icon.isNull():
            return icon

    if _IS_MACOS and spec.sf_symbol:
        icon = _create_sf_symbol_icon(spec.sf_symbol)
        if not icon.isNull():
            return icon

    if _IS_WINDOWS and spec.segoe_fluent and _is_fluent_glyph(spec.segoe_fluent):
        icon = _create_fluent_icon(spec.segoe_fluent, color=_symbol_tint())
        if not icon.isNull():
            return icon

    for name in _theme_icon_names(spec.theme_icon):
        icon = _create_theme_icon(name)
        if not icon.isNull():
            return icon

    return QIcon()


#: Which backend answered for an action, in the order they are tried.
ICON_SOURCES: tuple[str, ...] = ("svg", "sf_symbol", "segoe_fluent", "theme", "none")


def icon_source_for_action(spec: "ActionSpec") -> str:
    """Return which backend :func:`icon_from_action_spec` would use.

    The same chain, asked rather than drawn. Kept beside it deliberately: two
    implementations of one priority order would drift, and the whole point of
    this is to be able to trust the answer.
    """
    if spec.svg and not icon_from_svg_source(spec.svg).isNull():
        return "svg"

    if _IS_MACOS and spec.sf_symbol and not _create_sf_symbol_icon(spec.sf_symbol).isNull():
        return "sf_symbol"

    if (
        _IS_WINDOWS
        and spec.segoe_fluent
        and _is_fluent_glyph(spec.segoe_fluent)
        and not _create_fluent_icon(spec.segoe_fluent, color=_symbol_tint()).isNull()
    ):
        return "segoe_fluent"

    if any(
        theme_icon_is_available(name) for name in _theme_icon_names(spec.theme_icon)
    ):
        return "theme"

    return "none"


def report_icon_sources() -> dict[str, list[str]]:
    """Return the action ids each icon backend answers for, on this machine.

    What lands under ``"none"`` is what this machine cannot draw at all - the
    only report that matters now that there is no shipped fallback beneath the
    three real icon sets. Run it, do not guess: a mistyped theme name fails
    silently, because ``fromTheme`` returns a null icon and says nothing.
    """
    report: dict[str, list[str]] = {source: [] for source in ICON_SOURCES}
    for action_id, spec in sorted(_catalog().items()):
        report[icon_source_for_action(spec)].append(action_id)
    return report


def load_icon(icon: str | None) -> QIcon:
    """Return an icon for an action id, a Fluent glyph, or a theme name.

    Action ids go through :func:`icon_from_action_spec` and follow the
    catalogue's priority order. Anything else is taken literally: a Fluent
    glyph token is drawn as Fluent, and any other token is asked of the icon
    theme - which is what the handful of menu items that name an icon without
    being catalogue actions now pass, since the SVG names they used to pass
    no longer exist. An unknown token is an empty ``QIcon``.
    """
    token = _normalize_icon_name(icon)
    if not token:
        return QIcon()

    spec = _catalog().get(token)
    if spec is not None:
        return icon_from_action_spec(spec)

    if _is_fluent_glyph(token):
        return _create_fluent_icon(token, color=_symbol_tint())

    return _create_theme_icon(token)


# ----------------------------------------------------------------------
