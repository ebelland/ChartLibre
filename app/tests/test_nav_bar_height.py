"""The nav rail must not force the window taller than it needs to be.

NavigationBar stacks Fluent-style tiles that are each setFixedSize (64px
tall, on purpose - not an accident to relax). Stacked vertically that sums
to well over a modest window height, and without an explicit
setMinimumHeight(0) on the rail itself, that sum became its
minimumSizeHint - which propagates up through the containing frame and the
splitter to the window, so the rail ended up dictating a floor at least as
tall as its own content ("the nav bar is as large as the containing frame").
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def test_the_rail_does_not_pin_a_minimum_height(window: MainWindow) -> None:
    assert window._left_rail.minimumHeight() == 0


def test_the_rails_natural_content_is_in_fact_taller_than_that(
    window: MainWindow,
) -> None:
    """Otherwise the test above would be checking nothing: this confirms
    the rail really does have enough fixed-size tiles stacked up that a
    minimum-height floor was a real risk, not a hypothetical one."""
    assert window._left_rail.sizeHint().height() > 200


def test_the_window_can_shrink_below_the_rails_natural_height_off_macos(
    repo: SqliteRepo, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off macOS the rail carries its full complement of tiles (workspace,
    file, four pages, settings, help - eight fixed-size tiles, ~560px+),
    tall enough to exceed the window's other UI's own natural floor if
    nothing relaxes it. On macOS the rail alone (five tiles, ~356px) never
    was the binding constraint, so this would not have caught the bug the
    other two tests guard against - it has to run with the full tile set."""
    import app.dialogs.main_window as main_window_module

    monkeypatch.setattr(main_window_module, "IS_MACOS", False)
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    try:
        rail_natural_height = built._left_rail.sizeHint().height()
        assert rail_natural_height > 500
        assert built.minimumSizeHint().height() < rail_natural_height
    finally:
        built.close()
        applogger.set_status_bar(None)
