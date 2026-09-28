"""Dataclasses describing a figure, its axes, and their series.

These mirror the three descriptor tables in the database
(``__figure_descriptors__``, ``__axis_descriptors__``, ``__series_descriptors__``)
and are what the render pipeline consumes.  ``SqliteRepo.load_figure_descriptor``
is the only place that builds the tree.

The nesting is the same in both directions: a figure owns its axes, an axis
owns its series, and each level keeps the id of its parent so a single node can
be updated without walking down from the root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, kw_only=True)
class Descriptor:
    """What a figure, an axis and a series all have.

    ``name`` is what the node is called - a figure's title, an axis' chart
    type (the renderer's ``Name``), a series' label. ``index`` is its place
    among its siblings: an axis' slot in the figure's grid, a series' draw
    order on its axis. ``parent_id`` is the node above it - the figure of an
    axis, the axis of a series, None for a figure. ``options`` is the rest
    of its settings, the JSON each descriptor table stores.
    """

    id: int
    name: str = ""
    index: int = 0
    parent_id: int | None = None
    options: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class SeriesDescriptor(Descriptor):
    """One plotted series: where its data comes from and how it looks.

    ``roles`` maps a renderer's role names ("x", "y", "value", ...) to columns
    of ``sql_query``; ``options`` is the series' style - whatever the
    renderer accepts, plus the markers the series-operation dialogs use to
    find series they generated.
    """

    sql_query: str = ""
    roles: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class AxisDescriptor(Descriptor):
    """One axis of a figure, its renderer, and the series drawn on it.

    ``name`` is the chart type: a renderer's ``Name``, resolved through the
    scanner at render time rather than stored as a class reference, so a
    project file stays readable and survives a renderer being renamed in
    code. ``index`` is the axis' slot in the figure's grid.
    """

    title: str = ""
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    series: list[SeriesDescriptor] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class FigureDescriptor(Descriptor):
    """A figure: its grid, its options, and its axes.

    ``nrows``/``ncols`` are the subplot grid.  An axis whose ``index`` falls
    outside it is never drawn, which is why the operation dialogs grow the
    grid when they add one.
    """

    nrows: int = 1
    ncols: int = 1
    axes: list[AxisDescriptor] = field(default_factory=list)
