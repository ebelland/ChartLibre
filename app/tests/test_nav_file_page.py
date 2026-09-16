"""The rail's "File" tile switches to an embedded File page.

NavigationBar's popup-menu File button (off macOS only, InstantPopup +
window._file_menu) becomes a real "nav_file" page tile on every platform -
New/Open/Import/Save/Save As as flat buttons, plus an Open Recent popup that
rebuilds itself from user.json on every show - mirroring how Database's own
modal dialog became a page.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _file_tile_index(window: MainWindow) -> int:
    return window._nav_action_ids.index("nav_file")


def test_the_file_tile_switches_to_the_file_page(window: MainWindow) -> None:
    index = _file_tile_index(window)
    window._set_nav_index(index)

    assert window._left_stack.currentIndex() == index
    assert window._left_rail.buttons[index].isChecked()


def test_the_page_offers_the_file_actions(window: MainWindow) -> None:
    from PySide6.QtWidgets import QPushButton

    page = window._left_stack.widget(_file_tile_index(window))
    labels = {button.text() for button in page.findChildren(QPushButton)}
    for expected in ("New", "Open", "Import", "Save", "Open recent"):
        assert any(expected.lower() in label.lower() for label in labels), (
            expected, labels
        )


def test_no_recent_projects_shows_a_disabled_placeholder(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.dialogs.main_window.get_recent_databases", lambda: []
    )
    window._set_nav_index(_file_tile_index(window))
    menu = window._file_recent_button.menu()

    menu.aboutToShow.emit()

    actions = menu.actions()
    assert len(actions) == 1
    assert not actions[0].isEnabled()


def test_the_recent_menu_reflects_the_current_list(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    made_up = [Path("/tmp/one.dhub"), Path("/tmp/two.dhub")]
    monkeypatch.setattr(
        "app.dialogs.main_window.get_recent_databases", lambda: made_up
    )
    window._set_nav_index(_file_tile_index(window))
    menu = window._file_recent_button.menu()

    menu.aboutToShow.emit()

    names = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "one.dhub" in names
    assert "two.dhub" in names


def test_clicking_new_calls_the_new_file_handler(window: MainWindow) -> None:
    window._set_nav_index(_file_tile_index(window))
    page = window._left_stack.widget(_file_tile_index(window))

    from PySide6.QtWidgets import QPushButton

    new_button = next(
        b for b in page.findChildren(QPushButton) if b.text() == "New"
    )
    with patch.object(window, "_on_new_file") as handler:
        new_button.click()
    handler.assert_called_once()
