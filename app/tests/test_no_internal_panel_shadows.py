"""The elevation shadow belongs to the window, not the internal cards.

leftPanelCard and chartSurfaceCard used to each carry their own
QGraphicsDropShadowEffect (Fluent "elevated surface" styling). Sitting flush
against sibling panels inside a splitter, a drop shadow has nowhere to
render into and just looked wrong - "ombra su main window non sui pannelli
frame interni. Andavano bene prima" (shadow belongs on the main window, not
the internal frame panels - they were fine before [the shadow was added]).
"""
from __future__ import annotations

from pathlib import Path

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


def test_the_left_panel_carries_no_drop_shadow(window: MainWindow) -> None:
    assert window._left_panel.graphicsEffect() is None


def test_the_chart_surface_carries_no_drop_shadow(window: MainWindow) -> None:
    assert window._chart_surface.graphicsEffect() is None
