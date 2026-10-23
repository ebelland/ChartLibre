"""Multi-vari chart (AN-06): where a measure varies - within a cell, between cells, between groups.

JMP's Variability Chart and Minitab's Multi-Vari: the outer factor's levels
side by side along x, the inner factor's levels within each, every value as a
point, each cell's mean marked and joined across the inner levels, and each
outer group's mean drawn as a line across it. The three sources of variation a
quality study separates are read at a glance: the spread of the points within
a cell (repeatability, within piece), the step from cell to cell (the inner
factor - operators), and the step from group to group (the outer factor -
parts). Gage R&R draws its study with it.

Roles:
    y      required, the measurements
    x      required, the outer factor (parts)
    group  optional, the inner factor (operators)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.plugin_api.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.utils.coercion import to_numbers


class MultiVariAxisRenderer(BaseAxisRenderer):
    """Values by an outer and an inner factor, with the cell and group means."""

    Name: str = "Multi-Vari Chart"
    Category: str = "Quality and process"
    Description: str = (
        "Each value by an outer factor and an inner one, with the cell means joined and each group's mean: "
        "where the variation is - within, between cells, between groups."
    )
    Keywords: str = "multi-vari multivari variability chart variabilità gage r&r gauge misura operatori parti"
    Goals: list[str] = ["process", "compare"]
    Link: str = "https://www.jmp.com/support/help/en/18.0/#page/jmp/variability-gauge-charts.shtml"

    RequiredRoles: list[str] = ["y", "x"]
    OptionalRoles: list[str] = ["group"]
    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "markersize": {
            "default": 4.0, "type": float, "min": 0.5, "max": 20.0, "group": "Points",
            "description": "Size of each value's point.",
        },
        "alpha": {
            "default": 0.6, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Points",
            "description": "Opacity of the points.",
        },
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "cell_means": {
                "default": True, "type": bool, "group": "Means",
                "description": "Mark each cell's mean and join them across the inner factor.",
            },
            "group_means": {
                "default": True, "type": bool, "group": "Means",
                "description": "Draw each outer group's mean as a line across the group.",
            },
            "separators": {
                "default": True, "type": bool, "group": "Means",
                "description": "A faint line between one outer group and the next.",
            },
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="one study at a time")
        if sd is None:
            return
        frame = pd.DataFrame({"y": to_numbers(sd.df["y"]), "x": sd.df["x"].astype(str)})
        frame["group"] = sd.df["group"].astype(str) if "group" in sd.df.columns else ""
        frame = frame.dropna(subset=["y"])
        if frame.empty:
            return
        kwargs = self.get_kwargs(options)
        size = float(kwargs.get("markersize", 4.0) or 4.0)
        alpha = float(kwargs.get("alpha", 0.6) if kwargs.get("alpha") is not None else 0.6)
        outers = list(dict.fromkeys(frame["x"]))
        inners = list(dict.fromkeys(frame["group"]))
        width = len(inners)
        colours = [f"C{index % 10}" for index in range(width)]

        ticks, labels = [], []
        for outer_index, outer in enumerate(outers):
            base = outer_index * (width + 1)
            part = frame[frame["x"] == outer]
            means_x, means_y = [], []
            for inner_index, inner in enumerate(inners):
                values = part.loc[part["group"] == inner, "y"].to_numpy(float)
                if not values.size:
                    continue
                position = base + inner_index
                ax.plot(np.full(values.size, position), values, linestyle="none", marker="o", markersize=size,
                        alpha=alpha, color=colours[inner_index],
                        label=str(inner) if outer_index == 0 and inner else "_nolegend_")
                means_x.append(position)
                means_y.append(float(values.mean()))
            if bool(self.opt("cell_means", options)) and means_x:
                ax.plot(means_x, means_y, color="0.2", linewidth=1.2, marker="_", markersize=14, zorder=4)
            if bool(self.opt("group_means", options)) and means_x:
                ax.hlines(float(part["y"].mean()), base - 0.4, base + width - 0.6, colors="C3", linewidth=1.6,
                          zorder=3)
            if bool(self.opt("separators", options)) and outer_index:
                ax.axvline(base - 1.0, color="0.85", linewidth=0.8, zorder=0)
            ticks.append(base + (width - 1) / 2.0)
            labels.append(str(outer))
        ax.set_xticks(ticks, labels)
        ax.set_xlim(-1.0, len(outers) * (width + 1) - 1.0)
        if inners != [""] and bool(self.opt("show_legend", options)):
            ax.legend()
        self.apply_annotations(ax, options)
