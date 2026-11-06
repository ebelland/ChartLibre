"""The chart's navigation buttons: Matplotlib's own pan, zoom and view history, shown the app's way.

Matplotlib's bar is hidden and its actions shown by the chart's own buttons;
what these pin is that nothing was lost on the way - the cursor's coordinates
still reach the panel, Pan and Zoom still switch Matplotlib's mode and show it
checked, Back follows the view history, and the zoom controls are still there.
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QApplication, QSlider, QToolButton

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel
from dev.tests._figure_factory import create_renderer_showcase_db


@pytest.fixture(scope="module")
def showcase(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SqliteRepo, dict[str, int]]]:
    path = tmp_path_factory.mktemp("navigation") / "showcase.dhub"
    ids = create_renderer_showcase_db(path, n_points=200)
    repo = SqliteRepo(db_path=path)
    yield repo, ids
    repo.close()


def _settle(seconds: float = 0.3) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def panel(qapp, showcase) -> Iterator[ChartPanel]:
    repo, ids = showcase
    widget = ChartPanel(repo, ids["Scatter Plot"])
    widget.set_resize_mode("FIT", persist=False)
    widget.resize(900, 600)
    widget.show()
    _settle()
    yield widget
    widget.hide()
    widget.deleteLater()


def _buttons(panel: ChartPanel) -> dict[str, QToolButton]:
    """The navigation pill's buttons, by the Matplotlib action each one shows."""
    actions = {action: name for name, action in panel._toolbar._actions.items()}
    pill = panel.findChild(QToolButton, "chartToolButton").parentWidget()
    return {actions[button.defaultAction()]: button for button in pill.findChildren(QToolButton)
            if button.defaultAction() in actions}


def test_matplotlib_bar_is_hidden_and_its_actions_are_in_the_pill(panel: ChartPanel) -> None:
    assert not panel._toolbar.isVisible()
    assert set(_buttons(panel)) == {"home", "back", "forward", "pan", "zoom"}
    assert all(button.isVisible() for button in _buttons(panel).values())


def test_cursor_coordinates_reach_the_label(panel: ChartPanel) -> None:
    axes = panel.figure.axes[0]
    x, y = axes.transData.transform((axes.get_xlim()[0] * 0.5 + axes.get_xlim()[1] * 0.5,
                                     axes.get_ylim()[0] * 0.5 + axes.get_ylim()[1] * 0.5))
    MouseEvent("motion_notify_event", panel.canvas, x, y)._process()
    text = panel._coordinates_label.text()
    # Matplotlib's own wording of the point - "(x, y) = (10.96, 10.31)" in
    # 3.11, "x=10.96 y=10.31" before - so the numbers are what is checked.
    expected = axes.format_coord(*axes.transData.inverted().transform((x, y))).split()
    assert text and text.split() == " ".join(expected).split(), text


def test_pan_and_zoom_switch_matplotlib_mode_and_show_checked(panel: ChartPanel) -> None:
    buttons = _buttons(panel)
    buttons["pan"].click()
    assert panel._toolbar.mode.name == "PAN"
    assert buttons["pan"].isChecked() and not buttons["zoom"].isChecked()
    buttons["zoom"].click()
    assert panel._toolbar.mode.name == "ZOOM"
    assert buttons["zoom"].isChecked() and not buttons["pan"].isChecked()
    buttons["zoom"].click()
    assert panel._toolbar.mode.name == "NONE"


def test_back_follows_the_view_history(panel: ChartPanel) -> None:
    buttons = _buttons(panel)
    toolbar = panel._toolbar
    toolbar.push_current()
    axes = panel.figure.axes[0]
    low, high = axes.get_xlim()
    axes.set_xlim(low, (low + high) / 2.0)
    toolbar.push_current()
    assert buttons["back"].isEnabled()
    buttons["back"].click()
    assert axes.get_xlim() == pytest.approx((low, high))


def test_zoom_controls_are_still_there(panel: ChartPanel) -> None:
    assert panel.findChild(QSlider, "chartZoomSlider") is not None
    assert panel._zoom_label.text().endswith("%")
