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
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

import app.styles.style as style
from app.widgets.custom_title_bar import (
    CAPTION_BUTTON_WIDTH,
    CUSTOM_TITLE_BAR_HEIGHT,
    CustomTitleBar,
    caption_icon,
)
from app.widgets.nav_bar import NavigationBar


@pytest.fixture
def fluent(qapp: QApplication):
    style.apply_platform_style(qapp, "fluent_win11")
    yield
    style.apply_platform_style(qapp)


def _rail(qapp: QApplication) -> tuple[QWidget, NavigationBar]:
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    rail = NavigationBar(host, is_macos=False)
    layout.addWidget(rail)
    host.resize(240, 760)
    host.show()
    qapp.processEvents()
    return host, rail


def test_a_row_is_the_icon_then_its_label_on_one_line(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        for button in rail._all_tiles():
            assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            assert "\n" not in button.text()  # the old tiles broke long labels onto two lines
            assert button.height() == 36 and button.width() > 150
    finally:
        host.close()


def test_the_rail_uses_the_application_font_without_a_size_of_its_own(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        assert rail.buttons[0].font().pointSizeF() == rail.chart_list.font().pointSizeF() == 10.0
        assert not rail.buttons[0].font().bold()
        rail.select_page(1)
        qapp.processEvents()
        assert not rail.buttons[1].font().bold()  # a bold selected row is wider than its neighbours' text
    finally:
        host.close()


def test_collapsing_keeps_icons_only_and_expanding_restores_the_labels(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        labels = [button.text() for button in rail._all_tiles()]
        rail.set_compact(True)
        qapp.processEvents()
        assert rail.width() == 56
        assert all(button.text() == "" for button in rail._all_tiles())
        assert all(button.width() <= 56 for button in rail._all_tiles())

        rail.set_compact(False)
        qapp.processEvents()
        assert rail.width() == 200
        assert [button.text() for button in rail._all_tiles()] == labels
        assert all(button.width() > 150 for button in rail._all_tiles())
    finally:
        host.close()


def test_the_collapsed_icons_stay_together_at_the_top(qapp: QApplication, fluent) -> None:
    host, rail = _rail(qapp)
    try:
        rail.set_compact(True)
        host.resize(80, 900)
        qapp.processEvents()
        tops = [button.geometry().top() for button in rail.buttons]
        gaps = [b - a for a, b in zip(tops, tops[1:])]
        assert max(gaps) <= 36 + 10  # a row and the layout spacing, not the height shared out
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

