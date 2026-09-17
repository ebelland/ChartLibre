"""The rail's "Developer" tile switches to an embedded Developer page.

The old Developer menu group (Edit Localization, Series Operation Builder,
Function Creator, Renderer Helper) is gone from _app_menu_items entirely -
not hidden-but-built the way File/Database's groups still are - because off
macOS there was no menu bar to carry it, and the rail's own flattened popup
that used to carry it on every platform no longer has any button pointing
at it (settings_button and help_button each got their own narrower target
once File/Database became pages), making the whole group unreachable
outside of macOS's native bar. A page, reachable from the rail on every
platform, replaces it - see main_window._create_developer_page.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QPushButton

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _developer_tile_index(window: MainWindow) -> int:
    return window._left_rail.action_ids.index("nav_developer")


def test_the_developer_tile_switches_to_the_developer_page(window: MainWindow) -> None:
    index = _developer_tile_index(window)
    window._set_nav_index(index)

    assert window._left_stack.currentIndex() == index
    assert window._left_rail.buttons[index].isChecked()


def test_the_page_offers_every_dev_tool(window: MainWindow) -> None:
    page = window._left_stack.widget(_developer_tile_index(window))
    labels = {button.text() for button in page.findChildren(QPushButton)}
    for expected in (
        "Edit Localization",
        "Series Operation Builder",
        "Function Creator",
        "Renderer Helper",
    ):
        assert expected in labels


@pytest.mark.parametrize(
    "handler_name, label",
    [
        ("_on_edit_localization", "Edit Localization"),
        ("_on_series_operation_builder", "Series Operation Builder"),
        ("_on_function_creator", "Function Creator"),
        ("_on_renderer_helper", "Renderer Helper"),
    ],
)
def test_clicking_a_tool_calls_its_handler(
    window: MainWindow, handler_name: str, label: str
) -> None:
    page = window._left_stack.widget(_developer_tile_index(window))
    button = next(b for b in page.findChildren(QPushButton) if b.text() == label)
    with patch.object(window, handler_name) as handler:
        button.click()
    handler.assert_called_once()
