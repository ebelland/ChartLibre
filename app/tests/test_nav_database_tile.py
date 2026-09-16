"""The rail's "Database" tile opens the Database Info dialog, not a page.

NavigationBar lists "nav_database" among its action_ids (a rail tile, index
3) but _left_stack only has three real pages (Data / Chart options / Series
operations) - there is no embedded fourth page for it yet, only the existing
modal DatabaseInfoDialog. _set_nav_index special-cases that id to open the
dialog instead of switching to a page that does not exist, and resyncs the
rail's checked tile to whichever page is still actually showing afterward.
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


def _database_tile_index(window: MainWindow) -> int:
    return window._left_rail.action_ids.index("nav_database")


def test_the_database_tile_opens_the_dialog_not_a_missing_page(
    window: MainWindow,
) -> None:
    with patch.object(window, "_on_database_info") as opener:
        window._set_nav_index(_database_tile_index(window))
    opener.assert_called_once()


def test_clicking_it_does_not_change_the_visible_page(window: MainWindow) -> None:
    window._set_nav_index(0)
    before = window._left_stack.currentIndex()

    with patch.object(window, "_on_database_info"):
        window._set_nav_index(_database_tile_index(window))

    assert window._left_stack.currentIndex() == before


def test_the_rail_resyncs_to_the_still_visible_page(window: MainWindow) -> None:
    window._set_nav_index(1)

    with patch.object(window, "_on_database_info"):
        window._set_nav_index(_database_tile_index(window))

    assert window._left_rail.buttons[1].isChecked()
    database_button = window._left_rail.buttons[_database_tile_index(window)]
    assert not database_button.isChecked()


def test_on_database_info_opens_the_dialog_directly(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No recursion back into _set_nav_index(3), which would loop forever."""
    import app.dialogs.main_window as main_window_module

    opened: list[object] = []

    class _StubDialog:
        def __init__(self, repo, parent=None):
            opened.append(repo)

        def exec(self):
            return None

    monkeypatch.setattr(main_window_module, "DatabaseInfoDialog", _StubDialog)
    window._on_database_info()
    assert opened == [window._repo]
