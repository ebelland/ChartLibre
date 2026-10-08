"""Off macOS - no menu bar - Help is a row of the rail, and opens the Help menu."""
from __future__ import annotations

from pathlib import Path

import pytest

import app.main_window.main_window as main_window_module
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger


@pytest.mark.parametrize("macos", [False, True])
def test_help_is_on_the_rail_where_there_is_no_menu_bar(qapp, tmp_path: Path, monkeypatch, macos: bool) -> None:
    monkeypatch.setattr(main_window_module, "IS_MACOS", macos)
    repo = SqliteRepo(db_path=tmp_path / "help.dhub")
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    try:
        assert (window._left_panel.button("help") is not None) == (not macos)
        if macos:
            return
        shown: list[bool] = []
        monkeypatch.setattr(window, "_popup_help_menu", lambda: shown.append(True))
        window._left_panel.select("help")
        assert shown == [True]
        texts = [action.text() for action in window._help_menu.actions() if not action.isSeparator()]
        assert texts  # the manual and the credits
    finally:
        window.close()
        applogger.set_status_bar(None)
        repo.close()
