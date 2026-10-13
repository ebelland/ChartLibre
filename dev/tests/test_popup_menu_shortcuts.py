"""Off macOS the app menu is a popup: its shortcuts are carried by the window (UI-07)."""
from __future__ import annotations

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.main_window import main_menus
from app.main_window.main_window import MainWindow


def test_the_window_carries_the_popup_menus_shortcuts(qapp, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_menus, "IS_MACOS", False)
    repo = SqliteRepo(db_path=tmp_path / "m.dhub")
    window = MainWindow(repo, tmp_path / "m.dhub")
    try:
        window._build_app_menu()
        carried = {action.shortcut().toString() for action in window.actions()}
        assert {"Ctrl+Z", "Ctrl+S"} <= carried
        before = len(window.actions())
        window._build_app_menu()  # a rebuild replaces them, it does not add a second set
        assert len(window.actions()) == before
    finally:
        window.close()
        repo.close()
