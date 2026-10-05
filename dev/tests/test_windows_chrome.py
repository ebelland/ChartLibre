"""The Windows chrome, drawn here in the Fluent style: rail rows and caption buttons.

There is no Windows in the test run, but the stylesheet and the layout are
Qt's own and render the same anywhere, so the geometry is checked directly:
the labels sit to the right of the icons in the ordinary font, and the three
caption buttons are as tall as the strip, side by side with nothing between.
"""
from __future__ import annotations

from typing import Any, cast

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
import shiboken6
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QToolButton, QVBoxLayout, QWidget

import app.styles.style as style
from app.widgets.custom_title_bar import (
    CAPTION_BUTTON_WIDTH,
    CUSTOM_TITLE_BAR_HEIGHT,
    CustomTitleBar,
    caption_icon,
)
from app.widgets.nav_bar import NavBarItem, NavigationBar


@pytest.fixture(scope="module")
def fluent(qapp: QApplication):
    # Once for the file: applying a sheet repolishes every widget the run has
    # made so far, which took 30 s per test when done per test.
    style.apply_platform_style(qapp, "fluent_win11")
    yield
    style.apply_platform_style(qapp)


def _rail(qapp: QApplication) -> tuple[QWidget, NavigationBar]:
    """A rail of five rows - four actions and one page - in a host window."""
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    page = QWidget()
    icon = QIcon()
    rail = NavigationBar(
        {
            "Tools": {
                "workspace": NavBarItem("Workspace", icon, None),
                "file": NavBarItem("File", icon, None),
                "tables": NavBarItem("Tables", icon, page),
                "series_operations": NavBarItem("Series operations", icon, None),
            },
            "Charts": {"chart:1": NavBarItem("A chart with a long title", icon, None)},
        },
        host,
    )
    layout.addWidget(rail)
    host.resize(240, 760)
    host.show()
    qapp.processEvents()
    return host, rail


def _rows(rail: NavigationBar) -> list[QToolButton]:
    return list(rail._buttons.values())


def test_a_row_is_the_icon_then_its_label_on_one_line(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        for button in _rows(rail):
            assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            assert "\n" not in button.text()  # the old tiles broke long labels onto two lines
            assert button.height() == 36 and button.width() > 150
    finally:
        host.close()


def test_the_rail_uses_the_application_font_without_a_size_of_its_own(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        rows = _rows(rail)
        assert {button.font().pointSizeF() for button in rows} == {10.0}  # the Fluent sheet's one size
        assert not any(button.font().bold() for button in rows)
        rail.select("tables")
        qapp.processEvents()
        assert not rail._buttons["tables"].font().bold()  # a bold selected row is wider than its neighbours' text
    finally:
        host.close()


def test_collapsing_keeps_icons_only_and_expanding_restores_the_labels(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        labels = [button.text() for button in _rows(rail)]
        rail.set_compact(True)
        qapp.processEvents()
        assert rail.bar.width() == 56
        assert all(button.text() == "" for button in _rows(rail))
        assert all(button.width() <= 56 for button in _rows(rail))

        rail.set_compact(False)
        qapp.processEvents()
        assert rail.bar.width() == 200
        assert [button.text() for button in _rows(rail)] == labels
        assert all(button.width() > 150 for button in _rows(rail))
    finally:
        host.close()


def test_the_collapsed_icons_stay_together_at_the_top(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        rail.set_compact(True)
        host.resize(80, 900)
        qapp.processEvents()
        tops = [button.geometry().top() for button in rail._buttons.values()]
        gaps = [b - a for a, b in zip(tops, tops[1:])]
        assert max(gaps) <= 36 + 4 + 8  # a row, the spacing and a section's gap - not the height shared out
    finally:
        host.close()


def test_rebuilding_a_section_leaves_no_old_row_drawn(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        old = _rows(rail)
        rail.set_section("Charts", {"chart:2": NavBarItem("Another chart", QIcon(), None)})
        qapp.processEvents()
        assert not any(button.isVisible() for button in old if shiboken6.isValid(button))
        assert list(rail._buttons)[-1] == "chart:2"
    finally:
        host.close()


def _title_bar(qapp: QApplication) -> tuple[QMainWindow, CustomTitleBar]:
    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    # A plain QMainWindow stands in for MainWindow: the strip only asks it for its title, icon and the window verbs.
    bar = CustomTitleBar(cast(Any, window), is_macos=False, parent=central)
    layout.addWidget(bar)
    layout.addStretch(1)
    window.setCentralWidget(central)
    window.resize(700, 120)
    window.show()
    qapp.processEvents()
    return window, bar


def test_the_caption_buttons_fit_the_strip_and_touch_each_other(qapp: QApplication, fluent) -> None:
    window, bar = _title_bar(qapp)
    try:
        buttons = [bar.minimize_button, bar.maximize_button, bar.close_button]
        for button in buttons:
            assert button.size().width() == CAPTION_BUTTON_WIDTH
            assert button.size().height() == CUSTOM_TITLE_BAR_HEIGHT == bar.height()  # not cut off at the bottom
        for left, right in zip(buttons, buttons[1:]):
            assert right.geometry().left() == left.geometry().right() + 1  # no gap between them
        assert buttons[-1].geometry().right() == bar.width() - 1  # the close button is flush with the edge
        assert bar.sidebar_button is not None and bar.sidebar_button.height() <= bar.height()
    finally:
        window.close()


def test_the_restore_glyph_is_not_the_maximise_glyph(qapp: QApplication) -> None:
    maximise = caption_icon("maximize", "#202020").pixmap(QSize(12, 12)).toImage()
    restore = caption_icon("restore", "#202020").pixmap(QSize(12, 12)).toImage()
    assert maximise != restore


def test_the_close_glyph_turns_white_on_hover(qapp: QApplication) -> None:
    icon = caption_icon("close", "#202020", "#ffffff")
    normal = icon.pixmap(QSize(12, 12), QIcon.Mode.Normal).toImage()
    active = icon.pixmap(QSize(12, 12), QIcon.Mode.Active).toImage()

    def ink(image) -> int:
        return max(
            (image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height())),
            key=lambda colour: colour.alpha(),
        ).red()

    assert ink(normal) < 100 and ink(active) > 200



def test_a_hidden_bar_leaves_its_header_above_the_panels(qapp: QApplication, fluent) -> None:
    """macOS's title strip holds the button that brings the bar back: it must stay reachable."""
    host = QWidget()
    layout = QHBoxLayout(host)
    header = QWidget()
    header.setFixedHeight(30)
    page = QWidget()
    rail = NavigationBar({"Tools": {"tables": NavBarItem("Tables", QIcon(), page)}}, host, header=header)
    layout.addWidget(rail)
    host.resize(500, 400)
    host.show()
    try:
        rail.select("tables")
        rail.set_bar_hidden(True)
        qapp.processEvents()
        assert rail.bar.isHidden() and header.isVisible()
        assert rail.panels.isAncestorOf(header)
        rail.set_bar_hidden(False)
        qapp.processEvents()
        assert rail.bar.isAncestorOf(header) and header.isVisible()
    finally:
        host.close()
