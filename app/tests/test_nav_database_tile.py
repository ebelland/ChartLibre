"""The rail's "Database" tile switches to the embedded Database page.

NavigationBar lists "nav_database" among its action_ids (a rail tile), and
_left_stack now has a matching fourth page (see
main_window._create_database_page) with the Query Builder / Optimize DB
actions and the embedded DatabaseInfoPanel - no more modal dialog, no more
special-casing in _set_nav_index to fall back to one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger
from app.widgets.database_info_panel import DatabaseInfoPanel
from PySide6.QtWidgets import QPushButton


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _database_tile_index(window: MainWindow) -> int:
    return window._left_rail.action_ids.index("nav_database")


def test_the_database_tile_switches_to_the_database_page(window: MainWindow) -> None:
    index = _database_tile_index(window)
    window._set_nav_index(index)

    assert window._left_stack.currentIndex() == index
    assert window._left_rail.buttons[index].isChecked()
    assert window._left_stack.currentWidget().findChild(DatabaseInfoPanel) is not None


def test_on_database_info_lands_on_the_same_page(window: MainWindow) -> None:
    window._on_database_info()

    assert window._left_stack.currentIndex() == _database_tile_index(window)


def test_the_page_offers_query_builder_and_optimize_db(window: MainWindow) -> None:
    page = window._left_stack.widget(_database_tile_index(window))
    labels = {button.text() for button in page.findChildren(QPushButton)}
    # Exact catalogue wording is covered by the action-catalogue tests;
    # this only needs to know the two actions are reachable from the page.
    assert any("query" in label.lower() for label in labels)
    assert any("optimi" in label.lower() for label in labels)


def test_the_embedded_panel_shows_the_connected_database(window: MainWindow) -> None:
    panel = window._database_info_panel
    assert isinstance(panel, DatabaseInfoPanel)
    assert panel._repo is window._repo
