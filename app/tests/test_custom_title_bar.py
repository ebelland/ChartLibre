"""CustomTitleBar: Windows chrome vs macOS traffic lights.

is_macos is a constructor parameter, not re-imported from app.styles.style
- the same reason NavigationBar takes one too: main_window.py already
decided which of IS_WINDOWS/IS_MACOS triggered building this at all, and
re-deriving the answer independently inside the widget would silently
ignore that decision (and any test monkeypatching it).
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QMainWindow

from app.widgets.custom_title_bar import (
    CUSTOM_TITLE_BAR_HEIGHT,
    CustomTitleBar,
    _MAC_TITLE_BAR_HEIGHT,
)


@pytest.fixture
def window(qapp) -> QMainWindow:
    """A plain QMainWindow: every attribute CustomTitleBar touches
    (windowIcon, windowTitle, showMinimized, close, windowTitleChanged,
    isMaximized, showMaximized, showNormal, windowHandle) is already a
    real QMainWindow/QWidget member - MainWindow itself adds nothing this
    widget cares about."""
    built = QMainWindow()
    built.setWindowTitle("ChartLibre | test.dhub")
    yield built
    built.close()


def test_windows_gets_an_icon_title_and_three_right_aligned_buttons(window) -> None:
    bar = CustomTitleBar(window, is_macos=False)
    try:
        assert bar.height() == CUSTOM_TITLE_BAR_HEIGHT
        assert bar.title_label is not None
        assert bar.title_label.text() == window.windowTitle()
        layout = bar.layout()
        assert layout.indexOf(bar.minimize_button) < layout.indexOf(bar.close_button)
    finally:
        bar.deleteLater()


def test_windows_title_label_tracks_the_window_title(window) -> None:
    bar = CustomTitleBar(window, is_macos=False)
    try:
        window.setWindowTitle("ChartLibre | renamed.dhub")
        assert bar.title_label.text() == "ChartLibre | renamed.dhub"
    finally:
        bar.deleteLater()


def test_macos_gets_three_left_aligned_traffic_lights_and_no_title(window) -> None:
    bar = CustomTitleBar(window, is_macos=True)
    try:
        assert bar.height() == _MAC_TITLE_BAR_HEIGHT
        assert bar.title_label is None
        # AppKit's own left-to-right order: close, minimize, zoom. Read
        # from the controls row rather than from the widget's own layout:
        # that one is a column now, so that set_compact can move the
        # sidebar toggle onto a second line under the lights.
        row = bar._controls_row
        assert (
            row.indexOf(bar.close_button)
            < row.indexOf(bar.minimize_button)
            < row.indexOf(bar.maximize_button)
        )
    finally:
        bar.deleteLater()


def test_macos_traffic_lights_are_appkits_own_colors(window) -> None:
    bar = CustomTitleBar(window, is_macos=True)
    try:
        assert "#FF5F57" in bar.close_button.styleSheet()
        assert "#FEBC2E" in bar.minimize_button.styleSheet()
        assert "#28C840" in bar.maximize_button.styleSheet()
    finally:
        bar.deleteLater()


def test_macos_zoom_toggles_maximized_without_touching_its_own_icon(window) -> None:
    bar = CustomTitleBar(window, is_macos=True)
    try:
        style_before = bar.maximize_button.styleSheet()
        bar.maximize_button.click()
        assert window.isMaximized()
        assert bar.maximize_button.styleSheet() == style_before
    finally:
        bar.deleteLater()


def test_windows_maximize_button_swaps_its_icon_on_toggle(window) -> None:
    bar = CustomTitleBar(window, is_macos=False)
    try:
        icon_before = bar.maximize_button.icon()
        bar.maximize_button.click()
        assert window.isMaximized()
        assert bar.maximize_button.icon().cacheKey() != icon_before.cacheKey()
    finally:
        bar.deleteLater()
