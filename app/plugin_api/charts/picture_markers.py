"""Markers drawn as pictures from the project (__images__): a series' marker "image:<name>".

The pictures come from the project being drawn: render_figure sets the source for the time of a render, so
renderers need no repository of their own. An SVG is drawn through Qt's own SVG renderer (no new package), at
twice the size it shows, so it stays sharp; a single-colour picture (an icon) takes the series' colour, a
multicolour one or a photo keeps its own.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

import numpy as np

PREFIX = "image:"
MAX_POINTS = 2000  # more pictures than this draw slowly: an even sample of them is drawn

_source: ContextVar[Callable[[str], Any] | None] = ContextVar("picture_source", default=None)


def is_picture(marker: Any) -> bool:
    return isinstance(marker, str) and marker.startswith(PREFIX) and len(marker) > len(PREFIX)


def name_of(marker: str) -> str:
    return marker[len(PREFIX):]


@contextmanager
def pictures_from(get_image: Callable[[str], Any] | None) -> Iterator[None]:
    """Pictures are read with *get_image(name)* while the block runs."""
    token = _source.set(get_image)
    try:
        yield
    finally:
        _source.reset(token)


@lru_cache(maxsize=64)
def _rgba(kind: str, data: bytes, pixels: int) -> np.ndarray | None:
    """The picture as an RGBA float array, its longer side *pixels*."""
    from PySide6.QtCore import QByteArray, QRectF, Qt
    from PySide6.QtGui import QImage, QPainter

    if kind == "svg":
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(QByteArray(data))
        if not renderer.isValid():
            return None
        size = renderer.defaultSize()
        w, h = max(size.width(), 1), max(size.height(), 1)
        scale = pixels / max(w, h)
        image = QImage(max(1, round(w * scale)), max(1, round(h * scale)), QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, image.width(), image.height()))
        painter.end()
    else:
        image = QImage.fromData(data)
        if image.isNull():
            return None
        image = image.scaled(pixels, pixels, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation).convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.bytesPerLine() * height)
    array = raw.reshape(height, image.bytesPerLine())[:, : width * 4].reshape(height, width, 4) / 255.0
    return array.copy()


def _single_colour(array: np.ndarray) -> bool:
    opaque = array[..., 3] > 0.5
    if opaque.sum() < 4:
        return False
    return float(array[..., :3][opaque].std(axis=0).max()) < 0.08


def picture(marker: str, size_points: float, dpi: float, colour: Any = None) -> np.ndarray | None:
    """The array drawn for *marker* at *size_points*, tinted with *colour* when it is single-coloured."""
    get = _source.get()
    if get is None:
        return None
    stored = get(name_of(marker))
    if stored is None:
        return None
    pixels = max(8, min(512, round(size_points * dpi / 72.0 * 2)))
    array = _rgba(stored.kind, bytes(stored.data), pixels)
    if array is None:
        return None
    if colour is not None and _single_colour(array):
        from matplotlib.colors import to_rgb

        try:
            array = array.copy()
            array[..., :3] = to_rgb(colour)
        except ValueError:
            pass
    return array


def draw(ax, xs: Any, ys: Any, marker: str, *, size_points: float | None, colour: Any = None, zorder: float = 2.5,
         label: str = "") -> bool:
    """Put the picture at every (x, y); False when there is nothing to draw it with (no picture of that name)."""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    from app.logs.logger import applogger

    size = float(size_points or 0) or 12.0
    array = picture(marker, size, ax.figure.dpi, colour)
    if array is None:
        applogger.warning("Picture marker %r: no such picture in the project; the style's marker is used.", marker)
        return False
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) > MAX_POINTS:
        applogger.warning("%d picture markers: an even sample of %d is drawn.", len(xs), MAX_POINTS)
        keep = np.linspace(0, len(xs) - 1, MAX_POINTS).astype(int)
        xs, ys = xs[keep], ys[keep]
    zoom = 0.5  # drawn at twice the size: half of it shows
    for x, y in zip(xs, ys, strict=False):
        try:
            if not (np.isfinite(float(x)) and np.isfinite(float(y))):
                continue
        except (TypeError, ValueError):
            pass
        box = AnnotationBbox(OffsetImage(array, zoom=zoom), (x, y), frameon=False, pad=0, zorder=zorder)
        box._chartlibre_picture = marker
        ax.add_artist(box)
    ax.update_datalim(np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
                      if np.issubdtype(np.asarray(xs).dtype, np.number) and np.issubdtype(np.asarray(ys).dtype,
                                                                                          np.number) else [])
    ax.autoscale_view()
    if label:  # the legend shows the series' colour as a square: a picture has no legend handle of its own
        ax.add_line(Line2D([], [], linestyle="", marker="s", color=colour or "0.4", label=label))
    return True
