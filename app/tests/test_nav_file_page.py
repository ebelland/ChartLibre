"""The rail's "File" tile switches to an embedded File page.

NavigationBar's popup-menu File button (off macOS only, InstantPopup +
window._file_menu) becomes a real "nav_file" page tile on every platform -
Workspace (New/Open/Import/Load demo) and Save as flat buttons in their own
sections, plus an Open Recent list (one button per line, not a dropdown)
that rebuilds itself from user.json on every visit - mirroring how
Database's own modal dialog became a page.
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
    from PySide6.QtWidgets import QLabel, QPushButton

    page = window._left_stack.widget(_file_tile_index(window))
    labels = {button.text() for button in page.findChildren(QPushButton)}
    labels |= {label.text() for label in page.findChildren(QLabel)}
    for expected in ("New", "Open", "Import", "Load demo", "Save", "Open recent"):
        assert any(expected.lower() in label.lower() for label in labels), (
            expected, labels
        )


def test_the_page_is_split_into_sections(window: MainWindow) -> None:
    from PySide6.QtWidgets import QLabel

    page = window._left_stack.widget(_file_tile_index(window))
    titles = {
        label.text()
        for label in page.findChildren(QLabel)
        if label.property("sectionTitle")
    }
    assert titles == {"Workspace", "Save", "Open recent"}


def test_no_recent_projects_shows_a_disabled_placeholder(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.dialogs.main_window.get_recent_databases", lambda: []
    )
    window._set_nav_index(_file_tile_index(window))

    from PySide6.QtWidgets import QLabel

    labels = [
        label.text()
        for label in window._file_page.findChildren(QLabel)
        if label.property("muted")
    ]
    assert "No recent projects" in labels


def test_the_recent_list_reflects_the_current_list(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    made_up = [Path("/tmp/one.dhub"), Path("/tmp/two.dhub")]
    monkeypatch.setattr(
        "app.dialogs.main_window.get_recent_databases", lambda: made_up
    )
    window._set_nav_index(_file_tile_index(window))

    from PySide6.QtWidgets import QPushButton

    names = {button.text() for button in window._file_page.findChildren(QPushButton)}
    assert "one.dhub" in names
    assert "two.dhub" in names


def test_the_recent_list_is_one_button_per_line_not_a_dropdown(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each recent entry is a plain button with no menu attached - the
    dropdown (window._file_recent_button.menu()) this list replaced."""
    made_up = [Path("/tmp/one.dhub"), Path("/tmp/two.dhub")]
    monkeypatch.setattr(
        "app.dialogs.main_window.get_recent_databases", lambda: made_up
    )
    window._set_nav_index(_file_tile_index(window))

    from PySide6.QtWidgets import QPushButton

    one_button = next(
        b for b in window._file_page.findChildren(QPushButton) if b.text() == "one.dhub"
    )
    assert one_button.menu() is None


def test_clicking_a_recent_entry_opens_it(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    made_up = [Path("/tmp/one.dhub")]
    monkeypatch.setattr(
        "app.dialogs.main_window.get_recent_databases", lambda: made_up
    )

    from PySide6.QtWidgets import QPushButton

    # Patched before the list is (re)built, not after: each row's click
    # handler is functools.partial(self._on_open_recent, path), which - by
    # design, so a rebuilt list can't ever call a stale path - captures
    # today's bound method at build time rather than re-resolving the
    # attribute on every click the way a plain `self._on_open_recent`
    # connection would.
    with patch.object(window, "_on_open_recent") as handler:
        window._set_nav_index(_file_tile_index(window))
        button = next(
            b for b in window._file_page.findChildren(QPushButton) if b.text() == "one.dhub"
        )
        button.click()
    handler.assert_called_once_with(Path("/tmp/one.dhub"))


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
