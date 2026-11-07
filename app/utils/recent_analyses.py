"""The analyses opened last, newest first: the rail's Recent section (and how many it shows, a Setting)."""

from __future__ import annotations

from typing import Any

from app.utils.config import get_value, set_value

#: How many the rail shows (Settings > General); 0 hides the section.
COUNT_KEY = "recent_analyses_count"
DEFAULT_COUNT = 3
MAX_COUNT = 10
#: The class names of the analyses opened last, newest first.
LIST_KEY = "recent_analyses"


def count() -> int:
    """How many recent analyses the rail shows."""
    try:
        return max(0, min(MAX_COUNT, int(get_value(COUNT_KEY, DEFAULT_COUNT))))
    except (TypeError, ValueError):
        return DEFAULT_COUNT


def names() -> list[str]:
    """The class names of the analyses opened last, newest first (as many as kept, not only those shown)."""
    stored = get_value(LIST_KEY, [])
    return [str(name) for name in stored if name] if isinstance(stored, list) else []


def remember(operation: dict[str, Any]) -> bool:
    """*operation* opened now: first in the list. Whether the list changed."""
    name = str(operation.get("name") or "")
    if not name or name == "NewPlotTabDialog":
        return False
    before = names()
    after = [name, *(item for item in before if item != name)][:MAX_COUNT]
    if after == before:
        return False
    set_value(LIST_KEY, after)
    return True


def operations(known: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The recent analyses among *known* (the scanner's), newest first, as many as count() says."""
    by_name = {str(item.get("name")): item for item in known}
    return [by_name[name] for name in names() if name in by_name][:count()]
