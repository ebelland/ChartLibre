"""The left panel's cards keep the same gap from its right and bottom edges as from the rail (todo W-01).

On Windows' white page the cards ran flush to the panel's right and bottom
edges; the left side had the rail's spacing, the other two had nothing.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QFrame, QWidget

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
    stack_bottom = page.mapTo(window, QPoint(0, page.height())).y()
    panel_bottom = panel.mapTo(window, QPoint(0, panel.height())).y()

    # Within a pixel: the panel draws a hairline on its own edge.
    assert left_gap >= 6
    assert abs(right_gap - left_gap) <= 1
    assert abs((panel_bottom - stack_bottom) - left_gap) <= 1


def test_the_chart_surface_has_square_corners_under_its_flush_bars(
    qapp: QApplication, window: MainWindow, request: pytest.FixtureRequest
) -> None:
    """todo W-02: the jump bar's square background covered a rounded corner.

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

    # The outline runs straight to the corner along both sides. Rounded,
    # its first pixels were hidden under the bars' white and the corner
    # read as an outline that stopped short.
    assert all(outlined(0, y) for y in range(0, 5))
    assert all(outlined(x, 0) for x in range(0, 5))
