"""Lollipop chart: a value per category as a thin stem and a dot.

The same data as a bar chart with far less ink: with many categories, or
values that differ little, a row of solid bars becomes a block of colour and
the eye reads the bars' area instead of their ends. A stem and a dot keep
only the end, which is the value. Sorting by value turns the chart into a
ranking.

Roles: x the category, y its value.

  https://matplotlib.org/stable/gallery/lines_bars_and_markers/stem_plot.html
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


class LollipopAxisRenderer(BaseAxisRenderer):
    """A stem from the baseline and a dot at each category's value."""

    Name: str = "Lollipop Chart"
    Category: str = "Pairwise data"
    Description: str = (
        "A value per category as a thin stem with a dot at its end: a bar chart with less ink, "
        "clearer with many categories, a ranking when sorted."
    )
    Keywords: str = "lollipop stem dot bar ranking classifica categorie lecca-lecca"
    Goals: ClassVar[list[str]] = ["compare"]
    Link: str = "https://matplotlib.org/stable/gallery/lines_bars_and_markers/stem_plot.html"

    RequiredRoles: ClassVar[list[str]] = ["x", "y"]

    Kwargs: ClassVar[dict[str, Any]] = {
        "markersize": {"default": 7.0, "type": float, "min": 1.0, "max": 30.0, "group": "Appearance",
                       "description": "Size of the dot."},
        "linewidth": {"default": 1.6, "type": float, "min": 0.2, "max": 8.0, "step": 0.1, "group": "Appearance",
                      "description": "Width of the stem."},
    }

    Options: ClassVar[dict[str, Any]] = merge(
        LEGEND_OPTIONS,
        {
            "horizontal": {"default": True, "type": bool, "group": "Layout",
                           "description": "Categories down the y axis, values along x: long names stay readable."},
            "sort": {"default": "none", "type": ["none", "ascending", "descending"], "group": "Layout",
                     "description": "Order the categories by the first series' value."},
            "baseline": {"default": 0.0, "type": float, "group": "Layout",
                         "description": "Where the stems start."},
            "value_labels": {"default": False, "type": bool, "group": "Layout",
                             "description": "Write each value beside its dot."},
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series)
        if not valid:
            return
        horizontal = bool(self.opt("horizontal", options))
        baseline = float(self.opt("baseline", options) or 0.0)
        tables = []
        for sd in valid:
            names = np.asarray(sd.df["x"]).astype(str)
            values = to_numbers(column_of(sd.df, "y")).to_numpy(dtype=float)
            keep = np.isfinite(values)
            tables.append(dict(zip(names[keep], values[keep])))
        categories = list(dict.fromkeys(name for table in tables for name in table))
        if not categories:
            return
        sort = str(self.opt("sort", options) or "none")
        if sort != "none":
            categories.sort(key=lambda name: tables[0].get(name, -np.inf), reverse=sort == "descending")
        if horizontal:
            # The first category on top, as it is read.
            categories = categories[::-1]

        positions = np.arange(len(categories), dtype=float)
        count = len(valid)
        spread = 0.6 / count if count > 1 else 0.0
        for index, (sd, table) in enumerate(zip(valid, tables)):
            style = self.merge_style(options, sd.style or {})
            kwargs = self.get_kwargs(style)
            colour = self.series_color(style, index)
            at = positions + (index - (count - 1) / 2.0) * spread
            values = np.array([table.get(name, np.nan) for name in categories])
            keep = np.isfinite(values)
            at, values = at[keep], values[keep]
            width = float(kwargs.get("linewidth", 1.6) or 1.6)
            size = float(kwargs.get("markersize", 7.0) or 7.0)
            if horizontal:
                ax.hlines(at, baseline, values, colors=colour, linewidth=width, zorder=2)
                ax.plot(values, at, linestyle="none", marker="o", markersize=size, color=colour, label=sd.name,
                        zorder=3)
            else:
                ax.vlines(at, baseline, values, colors=colour, linewidth=width, zorder=2)
                ax.plot(at, values, linestyle="none", marker="o", markersize=size, color=colour, label=sd.name,
                        zorder=3)
            if bool(self.opt("value_labels", style)):
                for where, value in zip(at, values):
                    text = f"{value:,.4g}"
                    if horizontal:
                        ax.annotate(text, (value, where), xytext=(6 if value >= baseline else -6, 0),
                                    textcoords="offset points", va="center",
                                    ha="left" if value >= baseline else "right", fontsize="small")
                    else:
                        ax.annotate(text, (where, value), xytext=(0, 6 if value >= baseline else -6),
                                    textcoords="offset points", ha="center",
                                    va="bottom" if value >= baseline else "top", fontsize="small")

        if horizontal:
            ax.set_yticks(positions, categories)
            ax.axvline(baseline, color="0.6", linewidth=0.8, zorder=1)
            ax.margins(x=0.08)
        else:
            ax.set_xticks(positions, categories)
            ax.axhline(baseline, color="0.6", linewidth=0.8, zorder=1)
            ax.margins(y=0.08)
            if len(categories) > 8:
                ax.tick_params(axis="x", rotation=45)
        if count > 1 and bool(self.opt("show_legend", options)):
            ax.legend()
        self.apply_annotations(ax, options)
