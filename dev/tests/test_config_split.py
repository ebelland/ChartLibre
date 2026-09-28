"""``config.json`` ships; ``user.json`` accumulates.

The two used to be one file, and the cost showed up in the repository rather
than in a bug report: running the suite rewrote ``last_database`` to a pytest
temporary directory and left it in ``git status``, so a real diff had to be
picked out from around it.  ``app_style``, the window geometry and twelve
dialogs' remembered entries were versioned the same way, committed by whoever
happened to run the application before committing.

These tests hold the boundary in both directions - a setting must not land in
the shipped catalogue, and the catalogue must not be rewritten by the program
that reads it - and cover the migration, which only ever runs once on any
given machine and therefore gets no second chance to be right.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.utils import config


@pytest.fixture(autouse=True)
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point both files at throwaway ones and clear the caches between tests."""
    application = tmp_path / "config.json"
    user = tmp_path / "user.json"
    monkeypatch.setattr(config, "CONFIG_PATH", application)
    monkeypatch.setattr(config, "USER_CONFIG_PATH", user)
    monkeypatch.setattr(config, "_migrated_from", None)
    config._cache.clear()
    config._cache_stamp.clear()
    config._merged = None
    return application, user


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Which file a write lands in
# ----------------------------------------------------------------------
def test_a_setting_is_written_to_the_user_file(
    temp_config: tuple[Path, Path],
) -> None:
    application, user = temp_config
    _write(application, {"actions": {"open": {"label": "Open"}}})

    config.set_value("last_database", "/tmp/a.dhub")
    config.set_section("dialog_state", {"x": {"y": 1}})
    config.update_section("chart_panel", copy_dpi=300)

    saved = _read(user)
    assert saved["last_database"] == "/tmp/a.dhub"
    assert saved["dialog_state"] == {"x": {"y": 1}}
    assert saved["chart_panel"] == {"copy_dpi": 300}
    # The read-modify-write dance is exactly what used to drop sections.
    assert set(saved) == {"last_database", "dialog_state", "chart_panel"}






# ----------------------------------------------------------------------
# Reading the two as one
# ----------------------------------------------------------------------


def test_the_user_file_wins(temp_config: tuple[Path, Path]) -> None:
    """During the changeover a key can exist in both. The one the application
    wrote is the newer one."""
    application, user = temp_config
    _write(application, {"language": "en"})
    _write(user, {"language": "it"})

    assert config.get_value("language") == "it"




def test_a_corrupt_user_file_does_not_take_the_catalogue_with_it(
    temp_config: tuple[Path, Path],
) -> None:
    """A hand-edited file with a stray comma must not stop the application -
    and must not cost it its labels and icons either."""
    application, user = temp_config
    _write(application, {"actions": {"open": {"label": "Open"}}})
    user.write_text("{ this is not json", encoding="utf-8")

    assert config.get_section("actions") == {"open": {"label": "Open"}}




# ----------------------------------------------------------------------
# Migration, which happens once and cannot be retried
# ----------------------------------------------------------------------
PRE_SPLIT = {
    "actions": {"open": {"label": "Open"}},
    "messages": {"chart.no_series": {"text": "Add a series."}},
    "last_database": "/tmp/old.dhub",
    "app_style": "dark",
    "language": "it",
    "window_geometry": {"main_window": [0, 0, 800, 600]},
}




def test_nothing_is_lost_in_the_move(temp_config: tuple[Path, Path]) -> None:
    """The point of migrating rather than shipping a stripped config.json:
    the file on somebody's machine is the one holding their settings."""
    application, _user = temp_config
    _write(application, PRE_SPLIT)

    for name, expected in PRE_SPLIT.items():
        assert config.load_config()[name] == expected










# ----------------------------------------------------------------------
# The repository's own copy
# ----------------------------------------------------------------------
