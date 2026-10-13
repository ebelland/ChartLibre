"""The action catalogue: config.json's actions as ActionSpec, and the menu items made from them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from app.logs.logger import applogger
from app.utils.config import get_section
from app.utils.i18n import tr
from app.styles.style.metrics import MenuItem, ShortcutLike


def _theme_icon_field(value: object) -> str | tuple[str, ...] | None:
    """Normalise a catalogue ThemeIcon into one name or an ordered several.

    A list is allowed because no single freedesktop name is in every theme.
    "office-chart-line" is Breeze's and Papirus's, and GNOME has no chart icon
    at all, so a plotting application's Plot button is either two names or a
    blank button on GNOME. Written best-first: the name that means the thing,
    then whatever is closest and widely shipped.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        names = tuple(str(name).strip() for name in value if str(name).strip())
        if not names:
            return None
        return names[0] if len(names) == 1 else names
    text = str(value).strip()
    return text or None



# The action catalogue
# ----------------------------------------------------------------------
# config.json is the only source of an action's icon, label, tooltip and
# shortcut.  Read once into ActionSpec and cached, because every menu and
# every button asks for one while the window is being built.
@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Everything the UI needs to present one action."""

    action_id: str
    text: str
    description: str
    shortcut: ShortcutLike = None
    checkable: bool = False
    sf_symbol: str | None = None
    segoe_fluent: str | None = None
    theme_icon: str | tuple[str, ...] | None = None
    #: The icon drawn here, as SVG path data (the body of SVG_ICON_DOCUMENT),
    #: for an action no system set has a glyph for - a table's rows told apart
    #: from its columns. Inked in the theme's colour, like the system glyphs.
    svg: str | None = None

    @classmethod
    def from_config(cls, action_id: str, entry: dict[str, Any]) -> "ActionSpec":
        return cls(
            action_id=action_id,
            text=str(entry.get("text") or action_id),
            description=str(entry.get("description") or ""),
            shortcut=(entry.get("shortcut") or None),
            sf_symbol=(entry.get("SFSymbol") or entry.get("sf_symbol") or None),
            checkable=bool(entry.get("checkable", False)),
            segoe_fluent=(entry.get("SegoeFluent") or entry.get("segoe_fluent") or None),
            theme_icon=_theme_icon_field(
                entry.get("ThemeIcon") or entry.get("theme_icon")
            ),
            svg=(entry.get("SVG") or entry.get("svg") or None),
        )

    def translated_text(self) -> str:
        return tr(self.text)

    def translated_description(self) -> str:
        return tr(self.description) if self.description else ""

_cache: dict[str, ActionSpec] | None = None
# The section the cache was built from.  The catalogue lives in config.json
# and is parsed once per file signature, so this object stays the same one
# until config.json itself changes on disk - a settings write to user.json
# rebuilds the merged view around it but not the section itself.  Comparing
# identity is therefore enough to notice an edit without re-parsing on every
# button.
_cache_source: object | None = None


def _actions_section() -> dict[str, Any]:
    """Return the ``actions`` object from config.json, or an empty one.

    Not ``load_config()[...]``: a config.json without the section would raise
    while the first menu is being built, taking the window with it.
    """
    return get_section("actions")


def _catalog() -> dict[str, ActionSpec]:
    global _cache, _cache_source
    section = _actions_section()
    if _cache is not None and _cache_source is section:
        return _cache

    _cache_source = section
    _cache = {
        action_id: ActionSpec.from_config(action_id, entry)
        for action_id, entry in section.items()
        if isinstance(entry, dict)
    }
    return _cache



def action(action_id: str) -> ActionSpec:
    key = str(action_id or "").strip()
    spec = _catalog().get(key)
    if spec is not None:
        return spec

    applogger.warning(
        "No action %r in config.json; the widget will have no label or icon.",
        action_id,
    )
    # Empty, not a placeholder labelled with the id: a button reading
    # "series_outliers" looks like a working button with an odd name, and the
    # gap goes unnoticed.  A blank one does not.
    return ActionSpec(key, "", "")


def action_menu_item(
    action_id: str,
    callback,
    *,
    checkable: bool | None = None,
    checked: bool = False,
    enabled: bool = True,
    shortcut: ShortcutLike | Ellipsis = ...,  # type: ignore[valid-type]
) -> MenuItem:
    spec = action(action_id)
    return MenuItem(
        text=spec.translated_text(),
        tooltip=spec.translated_description(),
        shortcut=spec.shortcut if shortcut is ... else shortcut,
        callback=callback,
        checkable=spec.checkable if checkable is None else checkable,
        # Pass the action id to style.load_icon(); icon selection remains fully
        # catalogue-driven.
        icon=spec.action_id,
        action_id=spec.action_id,
        checked=checked,
        enabled=enabled,
    )
