"""Slope chart: each category's value at two moments, joined by a line.

Two columns of values, one line per category from the first to the second.
Where a dumbbell keeps the values on one scale row by row, the slope chart
lays them side by side, so what stands out is the change in rank - lines that
cross - and the few categories whose line runs against the rest. Rising
lines and falling ones can take different colours.

Roles: label the category, start and end its two values.

  https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import column_of, to_numbers


class SlopeAxisRenderer(BaseAxisRenderer):
    """One line per category between two columns of values."""

    Name: str = "Slope Chart"
    WhenToUse: str = "Use it to compare how several items changed between two moments."
    Category: str = "Pairwise data"
    Description: str = (
        "Each category's value at two moments, joined by a line: which rose, which fell, which "
        "overtook which."
    )
    Keywords: str = "slope chart slopegraph before after prima dopo rank classifica pendenza"
    Goals: ClassVar[list[str]] = ["compare"]
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html"

    RequiredRoles: ClassVar[list[str]] = ["label", "start", "end"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {
        "linewidth": {"default": 2.0, "type": float, "min": 0.2, "max": 10.0, "step": 0.1, "group": "Appearance",
                      "description": "Width of each line."},
        "markersize": {"default": 6.0, "type": float, "min": 0.0, "max": 30.0, "group": "Appearance",
                       "description": "Size of the dots at the two ends."},
    }

    Options: ClassVar[dict[str, Any]] = {
        "start_name": {"default": "Before", "type": str, "group": "Names",
                       "description": "Heading of the first column."},
        "end_name": {"default": "After", "type": str, "group": "Names",
                     "description": "Heading of the second column."},
        "colour_by": {"default": "direction", "type": ["direction", "category"], "group": "Colours",
                      "description": "Colour a line by whether it rises or falls, or each category its own."},
        "up_color": {"default": "#2F6FE4", "type": str, "kind": "color", "group": "Colours",
                     "description": "Lines that rise."},
        "down_color": {"default": "#E5484D", "type": str, "kind": "color", "group": "Colours",
                       "description": "Lines that fall."},
        "value_labels": {"default": True, "type": bool, "group": "Labels",
                         "description": "Write the category and its value at both ends."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="two sets of lines on the same two columns would tangle")
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
        kwargs = self.get_kwargs(style)
        width = float(kwargs.get("linewidth", 2.0) or 2.0)
        size = float(kwargs.get("markersize", 6.0) or 0.0)
        by_direction = str(self.opt("colour_by", style)) == "direction"
        up, down = str(self.opt("up_color", style)), str(self.opt("down_color", style))
        show_labels = bool(self.opt("value_labels", style))
        for index, (name, first, last) in enumerate(zip(labels, start, end)):
            colour = (up if last >= first else down) if by_direction else f"C{index % 10}"
            ax.plot([0, 1], [first, last], color=colour, linewidth=width, marker="o", markersize=size,
                    solid_capstyle="round", zorder=3, label=name)
            if show_labels:
                ax.annotate(f"{name}  {first:,.4g}", (0, first), xytext=(-8, 0), textcoords="offset points",
                            ha="right", va="center", fontsize="small", color=colour)
                ax.annotate(f"{last:,.4g}  {name}", (1, last), xytext=(8, 0), textcoords="offset points",
                            ha="left", va="center", fontsize="small", color=colour)
        for where in (0, 1):
            ax.axvline(where, color="0.85", linewidth=0.8, zorder=1)
        ax.set_xticks([0, 1], [str(self.opt("start_name", style) or "Before"),
                               str(self.opt("end_name", style) or "After")])
        ax.set_xlim(-0.6 if show_labels else -0.1, 1.6 if show_labels else 1.1)
        ax.margins(y=0.08)
        self.apply_annotations(ax, options)
