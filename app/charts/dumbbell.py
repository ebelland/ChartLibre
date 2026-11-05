"""Dumbbell plot: two values per category, joined, so the change is the length of the bar.

Before and after a treatment, two methods on the same samples, last year and
this year: two dots on one row per category and a line between them. The
direction and size of each change are read at once, and sorting by the
difference ranks the categories by how much they moved.

Roles: label the category, start and end its two values.

  https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hlines.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from app.plugin_api.charts.base import (
    LEGEND_OPTIONS,
    BaseAxisRenderer,
    SeriesData,
    merge,
)
from app.utils.coercion import column_of, to_numbers


class DumbbellAxisRenderer(BaseAxisRenderer):
    """Two dots per row joined by a line."""

    Name: str = "Dumbbell Plot"
    Category: str = "Pairwise data"
    Description: str = (
        "Two values per category - before and after, two methods, two years - as two dots joined by a "
        "line: how far and which way each one moved."
    )
    Keywords: str = "dumbbell connected dot plot before after prima dopo confronto differenza manubrio"
    Goals: ClassVar[list[str]] = ["compare"]
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hlines.html"

    RequiredRoles: ClassVar[list[str]] = ["label", "start", "end"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {
        "markersize": {"default": 8.0, "type": float, "min": 1.0, "max": 30.0, "group": "Appearance",
                       "description": "Size of the dots."},
        "linewidth": {"default": 2.5, "type": float, "min": 0.2, "max": 12.0, "step": 0.1, "group": "Appearance",
                      "description": "Width of the line between the dots."},
    }

    Options: ClassVar[dict[str, Any]] = merge(
        LEGEND_OPTIONS,
        {
            "start_name": {"default": "Start", "type": str, "group": "Names",
                           "description": "What the start value is, for the legend."},
            "end_name": {"default": "End", "type": str, "group": "Names",
                         "description": "What the end value is, for the legend."},
            "start_color": {"default": "#7A8797", "type": str, "kind": "color", "group": "Colours",
                            "description": "Dot of the start value."},
            "end_color": {"default": "#2F6FE4", "type": str, "kind": "color", "group": "Colours",
                          "description": "Dot of the end value."},
            "line_color": {"default": "#C9CFD9", "type": str, "kind": "color", "group": "Colours",
                           "description": "Line between the dots."},
            "sort": {"default": "none", "type": ["none", "start", "end", "difference"], "group": "Layout",
                     "description": "Order the rows by the start value, the end value, or the change."},
            "difference_labels": {"default": False, "type": bool, "group": "Layout",
                                  "description": "Write the change beside each row's end dot."},
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="the rows of two series would share their positions")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        labels = np.asarray(sd.df["label"]).astype(str)
        start = to_numbers(column_of(sd.df, "start")).to_numpy(dtype=float)
        end = to_numbers(column_of(sd.df, "end")).to_numpy(dtype=float)
        keep = np.isfinite(start) & np.isfinite(end)
        labels, start, end = labels[keep], start[keep], end[keep]
        if not labels.size:
            return
        sort = str(self.opt("sort", style) or "none")
        if sort != "none":
            key = {"start": start, "end": end, "difference": end - start}[sort]
            order = np.argsort(key, kind="stable")[::-1]
            labels, start, end = labels[order], start[order], end[order]
        # The first row on top.
        labels, start, end = labels[::-1], start[::-1], end[::-1]
        rows = np.arange(labels.size)

        kwargs = self.get_kwargs(style)
        size = float(kwargs.get("markersize", 8.0) or 8.0)
        width = float(kwargs.get("linewidth", 2.5) or 2.5)
        ax.hlines(rows, start, end, colors=str(self.opt("line_color", style)), linewidth=width, zorder=2)
        ax.plot(start, rows, linestyle="none", marker="o", markersize=size, zorder=3,
                color=str(self.opt("start_color", style)), label=str(self.opt("start_name", style) or "Start"))
        ax.plot(end, rows, linestyle="none", marker="o", markersize=size, zorder=3,
                color=str(self.opt("end_color", style)), label=str(self.opt("end_name", style) or "End"))
        if bool(self.opt("difference_labels", style)):
            for row, first, last in zip(rows, start, end):
                ahead = last >= first
                ax.annotate(f"{last - first:+,.3g}", (last, row), xytext=(8 if ahead else -8, 0),
                            textcoords="offset points", va="center", ha="left" if ahead else "right",
                            fontsize="small", color="0.3")
        ax.set_yticks(rows, labels)
        ax.set_ylim(-0.7, labels.size - 0.3)
        ax.margins(x=0.08)
        if bool(self.opt("show_legend", style)):
            ax.legend()
        self.apply_annotations(ax, options)
