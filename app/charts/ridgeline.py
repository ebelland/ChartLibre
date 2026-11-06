"""Ridgeline plot: the distribution of each group as a density curve, stacked and overlapping.

Ten violins side by side are hard to compare; ten histograms in a grid each
take an axis. A ridgeline draws each group's smoothed density on its own
baseline, one above the other and overlapping a little, like a mountain
range: a shift, a widening or a second peak from one group to the next is
seen down the column.

Roles: y the values, group optional (several series stack when there is none).

  https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import column_of, to_numbers


class RidgelineAxisRenderer(BaseAxisRenderer):
    """One density curve per group on stacked baselines."""

    Name: str = "Ridgeline Plot"
    WhenToUse: str = "Use it to compare the distribution shapes of many groups."
    Category: str = "Statistical distributions"
    Description: str = (
        "Each group's distribution as a smooth density curve, stacked and slightly overlapping: "
        "shifts, spreads and second peaks compared down the column."
    )
    Keywords: str = "ridgeline joyplot density kde distribuzioni gruppi densità creste"
    Goals: ClassVar[list[str]] = ["compare", "describe"]
    Link: str = "https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html"

    RequiredRoles: ClassVar[list[str]] = ["y"]
    OptionalRoles: ClassVar[list[str]] = ["group"]

    Kwargs: ClassVar[dict[str, Any]] = {
        "alpha": {"default": 0.75, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Appearance",
                  "description": "Opacity of each curve's fill."},
        "linewidth": {"default": 1.2, "type": float, "min": 0.0, "max": 6.0, "step": 0.1, "group": "Appearance",
                      "description": "Width of each curve's outline."},
    }

    Options: ClassVar[dict[str, Any]] = {
        "overlap": {"default": 1.6, "type": float, "min": 0.2, "max": 5.0, "step": 0.1, "group": "Layout",
                    "description": "Height of the tallest curve, in rows: above 1 the ridges overlap."},
        "bandwidth": {"default": None, "type": float, "min": 0.01, "max": 5.0, "step": 0.05, "group": "Density",
                      "description": "Smoothing, as a factor of Scott's rule. Empty uses Scott's rule."},
        "points": {"default": 256, "type": int, "min": 32, "max": 4096, "group": "Density",
                   "description": "Points along each curve."},
        "median": {"default": True, "type": bool, "group": "Layout",
                   "description": "Mark each group's median under its curve."},
        "single_colour": {"default": "", "type": str, "kind": "color", "group": "Appearance",
                          "description": "One colour for every ridge. Empty gives each group a colour of its own."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series)
        if not valid:
            return
        groups: dict[str, np.ndarray] = {}
        for sd in valid:
            values = to_numbers(column_of(sd.df, "y")).to_numpy(dtype=float)
            if "group" in sd.df.columns:
                labels = np.asarray(sd.df["group"]).astype(str)
                frame = pd.DataFrame({"g": labels, "v": values}).dropna()
                for label, part in frame.groupby("g", sort=False):
                    groups[str(label)] = part["v"].to_numpy(float)
            else:
                groups[sd.name] = values[np.isfinite(values)]
        groups = {name: values for name, values in groups.items() if values.size >= 2}
        if not groups:
            return

        everything = np.concatenate(list(groups.values()))
        low, high = float(everything.min()), float(everything.max())
        pad = (high - low) * 0.08 or 1.0
        grid = np.linspace(low - pad, high + pad, int(self.opt("points", options) or 256))
        bandwidth = self.opt_typed("bandwidth", options)
        curves = {}
        for name, values in groups.items():
            if np.ptp(values) == 0:
                curve = np.exp(-0.5 * ((grid - values[0]) / (pad / 4)) ** 2)
            else:
                curve = gaussian_kde(values, bw_method=float(bandwidth) if bandwidth else "scott")(grid)
            curves[name] = curve
        tallest = max(float(curve.max()) for curve in curves.values()) or 1.0
        overlap = float(self.opt("overlap", options) or 1.6)
        kwargs = self.get_kwargs(options)
        alpha = float(kwargs.get("alpha", 0.75) if kwargs.get("alpha") is not None else 0.75)
        width = float(kwargs.get("linewidth", 1.2) if kwargs.get("linewidth") is not None else 1.2)
        single = str(self.opt("single_colour", options) or "").strip()
        names = list(curves)
        count = len(names)
        # The first group on top; lower ridges are drawn over the ones behind them.
        for row, name in enumerate(names):
            base = float(count - 1 - row)
            height = curves[name] / tallest * overlap
            colour = single or f"C{row % 10}"
            zorder = 2 + row * 2
            ax.fill_between(grid, base, base + height, color=colour, alpha=alpha, linewidth=0, zorder=zorder)
            ax.plot(grid, base + height, color="white", linewidth=width + 1.2, zorder=zorder + 0.5)
            ax.plot(grid, base + height, color=colour, linewidth=width, zorder=zorder + 1)
            ax.axhline(base, color="0.75", linewidth=0.6, zorder=zorder)
            if bool(self.opt("median", options)):
                median = float(np.median(groups[name]))
                ax.vlines(median, base, base + float(np.interp(median, grid, height)), colors="0.2",
                          linewidth=1.0, zorder=zorder + 1)
        ax.set_yticks([float(count - 1 - row) for row in range(count)], names)
        ax.set_ylim(-0.1, count - 1 + overlap + 0.1)
        ax.set_xlim(grid[0], grid[-1])
        ax.tick_params(axis="y", length=0)
        self.apply_annotations(ax, options)
