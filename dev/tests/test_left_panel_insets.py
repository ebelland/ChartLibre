"""The left panel's cards keep the same gap from its right and bottom edges as from the rail (todo W-01).

On Windows' white page the cards ran flush to the panel's right and bottom
edges; the left side had the rail's spacing, the other two had nothing.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QScrollArea, QApplication, QFrame, QWidget

import app.styles.style as style
from app.data.sqlite_repo import SqliteRepo
from app.main_window.main_window import MainWindow
from app.logs.logger import applogger
from app.widgets.nav_bar import NavigationBar


@pytest.fixture(params=["fluent_win11", "macos_native"])
def window(qapp: QApplication, tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[MainWindow]:
    style.apply_platform_style(qapp, request.param)
    repo = SqliteRepo(db_path=tmp_path / "insets.dhub")
    built = MainWindow(repo=repo, db_path=repo.db_path)
    built.resize(1300, 850)
    built.show()
    qapp.processEvents()
    yield built
    built.close()
    applogger.set_status_bar(None)
    style.apply_platform_style(qapp)


def _x(widget: QWidget, window: QWidget, right: bool = False) -> int:
    return widget.mapTo(window, QPoint(widget.width() if right else 0, 0)).x()


def test_the_file_page_cards_have_equal_gaps_on_three_sides(qapp: QApplication, window: MainWindow) -> None:
    navigation = window.findChild(NavigationBar)
    assert navigation is not None and navigation.select("file")
    qapp.processEvents()
    rail, panel = navigation.bar, navigation.panels
    card = window.findChild(QFrame, "fileWorkspaceCard")
    assert card is not None

    left_gap = _x(card, window) - _x(rail, window, right=True)
    right_gap = _x(panel, window, right=True) - _x(card, window, right=True)
    page = navigation.panel_stack.currentWidget()
    # On a short window the page scrolls: its scroll bar is not part of the gap.
    scroll = page.findChild(QScrollArea)
    if scroll is not None and scroll.verticalScrollBar().isVisible():
        right_gap -= scroll.verticalScrollBar().width()
    stack_bottom = page.mapTo(window, QPoint(0, page.height())).y()
    panel_bottom = panel.mapTo(window, QPoint(0, panel.height())).y()

    # Within a pixel: the panel draws a hairline on its own edge.
    assert left_gap >= 6
    assert abs(right_gap - left_gap) <= 1
    assert abs((panel_bottom - stack_bottom) - left_gap) <= 1


def test_the_chart_surface_has_round_corners_its_bars_do_not_cover(
    qapp: QApplication, window: MainWindow, request: pytest.FixtureRequest
) -> None:
    """Windows: the chart pane is rounded like the left pane (bugs2.pptx), and its jump
    bar is rounded with it - square, it covered the curve (todo W-02, when the pane
    was made square instead).

    Windows only: on macOS the chart pane is a plain white surface with no
    outline at all.
    """
    if request.node.callspec.params["window"] != "fluent_win11":
        pytest.skip("the macOS sheet draws no outline on the chart surface")
    surface = window.findChild(QFrame, "chartSurfaceCard")
    assert surface is not None
    image = surface.grab().toImage()
    page = image.pixelColor(image.width() // 2, image.height() // 2).lightnessF()

    def outlined(x: int, y: int) -> bool:
        return image.pixelColor(x, y).lightnessF() < page - 0.03

    # Straight sides away from the corner, and the curve itself drawn, not under a bar's white.
    assert outlined(0, 14) and outlined(14, 0)
    assert any(outlined(x, y) for x in range(1, 5) for y in range(1, 5))
    assert not outlined(0, 0)  # the corner itself is cut off by the curve