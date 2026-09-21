"""The status bar's contents need room at the window's left and right edges.

Reported as "Status bar: no padding on left and right edges" (todo.txt P1-3):
the project label on one end and the state label on the other sat wherever
the active style's own internal offset left them - two pixels in on Fusion,
flush under others - rather than at a padding this app chose.

The fix is setContentsMargins rather than a rule in the sheet, and that is
not a style preference: see test_qss_padding_on_a_status_bar_is_a_no_op
below for why the sheet cannot do it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel, QMainWindow

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


def test_the_bar_has_horizontal_padding(window: MainWindow) -> None:
    margins = window.statusBar().contentsMargins()
    assert margins.left() > 0
    assert margins.right() > 0


def test_the_padding_is_symmetric(window: MainWindow) -> None:
    """Both ends or neither - a bar padded on one side only reads as a
    layout mistake, which is what the report was about in the first place."""
    margins = window.statusBar().contentsMargins()
    assert margins.left() == margins.right()


def test_the_bar_keeps_its_fixed_height(window: MainWindow) -> None:
    """The padding is horizontal only: the bar is a fixed 24px strip, and
    vertical margins would eat into the labels inside it rather than move
    them."""
    margins = window.statusBar().contentsMargins()
    assert (margins.top(), margins.bottom()) == (0, 0)
    assert window.statusBar().height() == 24


def test_the_first_label_really_is_pushed_off_the_edge(window: MainWindow) -> None:
    """The margins above are only worth asserting if Qt lays the widgets
    out by them - this is the effect the report asked for, measured."""
    window.resize(900, 600)
    window.statusBar().adjustSize()
    left_margin = window.statusBar().contentsMargins().left()

    assert window._status_project.x() >= left_margin


def test_the_padding_survives_a_state_change(window: MainWindow) -> None:
    """_set_status_state re-applies a whole stylesheet to the bar on every
    state change - busy, error, back to normal - and each of those makes Qt
    re-lay the bar out. The margins are set once at construction, so a
    relayout that dropped them would take the padding away the first time
    anything happened, which is exactly when nobody is looking at the
    left edge."""
    before = window.statusBar().contentsMargins()

    for state in ("busy", "error", "success", "normal"):
        window._set_status_state(state)

    after = window.statusBar().contentsMargins()
    assert (after.left(), after.right()) == (before.left(), before.right())
    assert after.left() > 0


def test_the_padding_survives_a_temporary_message(window: MainWindow) -> None:
    """showMessage hides the bar's normal widgets and shows them again
    afterwards - another relayout, and the one that runs most often."""
    bar = window.statusBar()
    bar.showMessage("Database saved.")
    bar.clearMessage()

    assert bar.contentsMargins().left() > 0
    assert window._status_project.x() >= bar.contentsMargins().left()


def test_qss_padding_on_a_status_bar_is_a_no_op(qapp) -> None:
    """Why this is not done in the stylesheet, pinned down.

    QStatusBar is one of the widgets Qt honours only background and border
    for - a "padding" is parsed, accepted, and then ignored, so the sheet
    version of this fix looks exactly like a working one in the diff and
    changes nothing on screen. This test fails the day Qt starts honouring
    it, which is the day this could move back into fluent_win11.qss.
    """
    window = QMainWindow()
    try:
        bar = window.statusBar()
        bar.setObjectName("paddingProbe")
        label = QLabel("project.dhub", bar)
        bar.addWidget(label, 0)
        window.resize(600, 400)
        window.show()
        qapp.processEvents()
        unpadded_x = label.x()

        bar.setStyleSheet("QStatusBar#paddingProbe { padding: 0 24px; }")
        qapp.processEvents()

        assert label.x() == unpadded_x

        bar.setContentsMargins(24, 0, 24, 0)
        qapp.processEvents()

        assert label.x() > unpadded_x
    finally:
        window.close()
        applogger.set_status_bar(None)


def _leftmost_ink(widget) -> int | None:
    """The x of the first column this widget paints anything dark into.

    Where the text *lands*, rather than where the widget holding it was
    put - which is the only way to see a string QStatusBar paints itself,
    since it belongs to no child widget whose geometry could be read.
    """
    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    image.fill(0xFFFFFFFF)
    widget.render(image)
    for x in range(image.width()):
        for y in range(image.height()):
            if image.pixelColor(x, y).lightness() < 128:
                return x
    return None


def test_a_temporary_message_is_not_moved_by_the_margins(qapp) -> None:
    """A known limit of the fix above, measured rather than asserted in prose.

    showMessage() text is painted by QStatusBar itself, not by any of the
    labels, at an offset hard-coded in Qt that no margin or sheet reaches.
    So a transient message - "Database saved.", and everything AppLogger
    routes here - sits a few pixels left of the project label it replaces,
    however much padding the bar is given. Nothing in this app can move it
    short of not using showMessage() at all (see todo.txt P1-3); what this
    test does is stop the next person reading that gap as the margins
    above having failed, and fail if a future Qt fixes it.
    """
    window = QMainWindow()
    try:
        bar = window.statusBar()
        bar.setObjectName("messageProbe")
        bar.setStyleSheet(
            "QStatusBar#messageProbe { background: #ffffff; color: #000000; }"
        )
        label = QLabel("project.dhub", bar)
        bar.addWidget(label, 0)
        bar.setContentsMargins(24, 0, 24, 0)
        window.resize(600, 400)
        window.show()
        qapp.processEvents()

        label_ink = _leftmost_ink(bar)

        bar.showMessage("Database saved.")
        qapp.processEvents()
        message_ink = _leftmost_ink(bar)

        assert bar.currentMessage() == "Database saved."
        assert label_ink is not None and message_ink is not None
        # The margin moved the label and left the message where it was.
        assert message_ink < label_ink
    finally:
        window.close()
        applogger.set_status_bar(None)
