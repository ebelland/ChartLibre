"""``import_class_from_discovery_entry`` resolves a path once, not per call.

``Path(entry["path"]).resolve()`` used to run on every single call - once per
axis, on every chart redraw - even though the renderer/operation file *set*
never moves while the process runs. ``_resolved_path`` caches that part by
the raw path string; the mtime check right after it is untouched, which is
what has to keep working: a file the Developer-menu scaffolding tools (or a
plugin author) just edited on disk must still hot-reload without a restart.
"""
from __future__ import annotations

import time
from pathlib import Path

from app.scanners.class_discovery import (
    _resolved_path,
    import_class_from_discovery_entry,
)

_V1 = "class Thing:\n    def value(self):\n        return 1\n"
_V2 = "class Thing:\n    def value(self):\n        return 2\n"


def test_resolving_the_same_raw_path_twice_returns_the_same_object(tmp_path: Path) -> None:
    """Not just an equal Path - the point is the cache is actually hit."""
    target = tmp_path / "a_file.py"
    target.write_text(_V1, encoding="utf-8")

    first = _resolved_path(str(target))
    second = _resolved_path(str(target))
    assert first is second


def test_editing_the_file_on_disk_still_hot_reloads(tmp_path: Path) -> None:
    """The behaviour the mtime check exists for, still intact once path
    resolution is cached separately from it."""
    target = tmp_path / "hot_reload_target.py"
    target.write_text(_V1, encoding="utf-8")

    entry = {"path": str(target), "name": "Thing"}
    cls_before = import_class_from_discovery_entry(entry, module_prefix="_test_hot_reload")
    assert cls_before is not None
    assert cls_before().value() == 1

    # Over a second, not a sub-second sleep: _load_class's own lru_cache keys
    # on the file's nanosecond mtime and correctly misses well before this,
    # but CPython's SourceFileLoader writes a __pycache__ .pyc alongside the
    # temp file and validates it against the source's mtime at *whole-second*
    # resolution - two edits inside the same second both pass that check and
    # the stale compiled bytecode gets served regardless. Nothing here to fix
    # (it is import machinery, not this module); the test just has to clear
    # it the same way a person editing a file and coming back later would.
    time.sleep(1.1)
    target.write_text(_V2, encoding="utf-8")

    cls_after = import_class_from_discovery_entry(entry, module_prefix="_test_hot_reload")
    assert cls_after is not None
    assert cls_after().value() == 2
