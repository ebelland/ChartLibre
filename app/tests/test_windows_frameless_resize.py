"""Edge-drag resizing for the frameless window (Windows only).

Qt.FramelessWindowHint (see main_window.py __init__) leaves the OS with no
resize handles of its own - "cannot resize main window under Win11" was
that gap: setMouseTracking/installEventFilter were wired up in __init__ but
MainWindow had no eventFilter at all, so neither of them did anything.
eventFilter now hit-tests the margin around _central_host's own edges and
calls QWindow.startSystemResize, the same technique WindowsTitleBar already
uses for moving the window.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger
import app.dialogs.main_window as main_window_module


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


@pytest.fixture
def window(
    repo: SqliteRepo, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
):
    monkeypatch.setattr(main_window_module, "IS_WINDOWS", True)
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    built.resize(900, 700)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _move_event(pos: QPoint) -> QMouseEvent:
    pf = QPointF(pos)
    return QMouseEvent(
        QEvent.Type.MouseMove, pf, pf, pf,
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    )


def _press_event(pos: QPoint) -> QMouseEvent:
    pf = QPointF(pos)
    return QMouseEvent(
        QEvent.Type.MouseButtonPress, pf, pf, pf,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


def test_a_point_at_the_left_edge_is_detected(window: MainWindow) -> None:
    rect = window._central_host.rect()
    edges = window._resize_edge_at(QPoint(0, rect.height() // 2))
    assert edges == Qt.Edge.LeftEdge


def test_a_point_at_the_top_left_corner_is_both_edges(window: MainWindow) -> None:
    edges = window._resize_edge_at(QPoint(0, 0))
    assert edges == (Qt.Edge.TopEdge | Qt.Edge.LeftEdge)


def test_a_point_in_the_interior_is_no_edge(window: MainWindow) -> None:
    rect = window._central_host.rect()
    edges = window._resize_edge_at(QPoint(rect.width() // 2, rect.height() // 2))
    assert edges == Qt.Edge(0)


def test_hovering_an_edge_sets_a_resize_cursor(window: MainWindow) -> None:
    rect = window._central_host.rect()
    window.eventFilter(window._central_host, _move_event(QPoint(0, rect.height() // 2)))
    assert window._central_host.cursor().shape() == Qt.CursorShape.SizeHorCursor


def test_hovering_the_interior_uses_the_ordinary_cursor(window: MainWindow) -> None:
    rect = window._central_host.rect()
    window.eventFilter(
        window._central_host, _move_event(QPoint(0, rect.height() // 2))
    )
    window.eventFilter(
        window._central_host,
        _move_event(QPoint(rect.width() // 2, rect.height() // 2)),
    )
    assert window._central_host.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_pressing_at_an_edge_starts_a_system_resize(window: MainWindow) -> None:
    handle = MagicMock()
    window.windowHandle = lambda: handle  # type: ignore[method-assign]

    rect = window._central_host.rect()
    consumed = window.eventFilter(
        window._central_host, _press_event(QPoint(0, rect.height() // 2))
    )

    assert consumed is True
    handle.startSystemResize.assert_called_once_with(Qt.Edge.LeftEdge)


def test_pressing_in_the_interior_does_not_start_a_resize(window: MainWindow) -> None:
    handle = MagicMock()
    window.windowHandle = lambda: handle  # type: ignore[method-assign]

    rect = window._central_host.rect()
    consumed = window.eventFilter(
        window._central_host,
        _press_event(QPoint(rect.width() // 2, rect.height() // 2)),
    )

    assert consumed is False
    handle.startSystemResize.assert_not_called()


def test_a_maximized_window_is_not_resized_from_its_edges(window: MainWindow) -> None:
    handle = MagicMock()
    window.windowHandle = lambda: handle  # type: ignore[method-assign]
    window.showMaximized()

    rect = window._central_host.rect()
    window.eventFilter(window._central_host, _press_event(QPoint(0, rect.height() // 2)))

    handle.startSystemResize.assert_not_called()


def test_elsewhere_the_filter_does_nothing_special(
    repo: SqliteRepo, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    monkeypatch.setattr(main_window_module, "IS_WINDOWS", False)
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    try:
        handle = MagicMock()
        built.windowHandle = lambda: handle  # type: ignore[method-assign]
        rect = built._central_host.rect()

        built.eventFilter(built._central_host, _press_event(QPoint(0, rect.height() // 2)))

        handle.startSystemResize.assert_not_called()
    finally:
        built.close()
        applogger.set_status_bar(None)
