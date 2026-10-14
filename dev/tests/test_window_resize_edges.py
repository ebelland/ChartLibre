"""Windows frameless resizing: the bottom edge and corners are the status bar's (P0-4).

The window has no native frame, so the app hit-tests its own edges. The status
bar sits below the central area, outside it, which left the real bottom edge
and corner - the bar's strip - with nothing to grab.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt

from app.main_window.main_frame import resize_edges

SIZE = QSize(800, 600)
MARGIN = 6
E = Qt.Edge


def edges(x: int, y: int, **kw) -> Qt.Edge:
    return resize_edges(QPoint(x, y), SIZE, MARGIN, **kw)


def test_the_interior_has_no_edge() -> None:
    assert edges(400, 300) == Qt.Edge(0)


def test_each_side_and_corner_of_a_whole_window() -> None:
    assert edges(0, 300) == E.LeftEdge
    assert edges(799, 300) == E.RightEdge
    assert edges(400, 0) == E.TopEdge
    assert edges(400, 599) == E.BottomEdge
    assert edges(0, 0) == E.TopEdge | E.LeftEdge
    assert edges(799, 0) == E.TopEdge | E.RightEdge
    assert edges(799, 599) == E.BottomEdge | E.RightEdge
    assert edges(0, 599) == E.BottomEdge | E.LeftEdge


def test_the_central_area_above_a_status_bar_has_no_bottom_edge() -> None:
    assert edges(400, 599, bottom=False) == Qt.Edge(0)
    assert edges(799, 599, bottom=False) == E.RightEdge  # the side is still the window's side


def test_the_status_bar_gives_the_bottom_edge_and_both_corners() -> None:
    """A 24 px strip: its top is the central area's bottom, not a window edge."""
    bar = QSize(800, 24)

    def at(x: int, y: int) -> Qt.Edge:
        return resize_edges(QPoint(x, y), bar, MARGIN, top=False, corner=14)

    assert at(400, 23) == E.BottomEdge
    assert at(400, 0) == Qt.Edge(0)  # the bar's top is not the window's
    assert at(795, 20) == E.BottomEdge | E.RightEdge
    assert at(2, 20) == E.BottomEdge | E.LeftEdge
    assert at(790, 22) == E.BottomEdge | E.RightEdge  # inside the wider corner zone, 14 px
    assert at(795, 5) == E.RightEdge  # up the side, above the bottom band, it is the side only
