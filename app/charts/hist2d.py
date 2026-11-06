"""2D histogram: x/y pairs counted in a grid of rectangular bins.

The square-binned sibling of the hexbin: where a dense scatter piles points on
points, each cell's colour says how many landed in it. Square bins line up
with the axes, so a row or a column of cells reads as a conditional
distribution, and the edges can be set to round numbers. An optional weight
sums a quantity per cell instead of counting rows.

Roles: x, y, weight optional.

  https://matplotlib.org/stable/plot_types/stats/hist2d.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from matplotlib.colors import LogNorm

from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import column_of, to_numbers


class Hist2dAxisRenderer(BaseAxisRenderer):
    """Counts of x/y pairs in rectangular bins, as colours."""

    Name: str = "2D Histogram"
    Category: str = "Statistical distributions"
    Description: str = (
        "x/y pairs counted in a grid of rectangular bins and shown as colours: where a scatter with too "
        "many points actually piles up, on bins aligned with the axes."
    )
    Keywords: str = "2d histogram hist2d density densità istogramma bidimensionale bins heat many points"
    Goals: ClassVar[list[str]] = ["relate"]
    Link: str = "https://matplotlib.org/stable/plot_types/stats/hist2d.html"

    RequiredRoles: ClassVar[list[str]] = ["x", "y"]
    OptionalRoles: ClassVar[list[str]] = ["weight"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {
        "bins": {"default": 40, "type": int, "min": 2, "max": 1000, "group": "Binning",
                 "description": "Bins along each axis."},
        "cmin": {"default": None, "type": float, "min": 0.0, "group": "Binning",
                 "description": "Cells with fewer counts than this are left blank."},
        "cmap": {"default": None, "type": str, "kind": "colormap", "group": "Appearance",
                 "description": "Colormap of the counts. Empty uses the style's."},
        "alpha": {"default": None, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Appearance",
                  "description": "Opacity of the cells."},
    }

    Options: ClassVar[dict[str, Any]] = {
        "count_scale": {"default": "linear", "type": ["linear", "log"], "group": "Binning",
                        "description": "'log' colours by the logarithm of the count, so a few crowded cells do "
                                       "not wash out the rest."},
        "density": {"default": False, "type": bool, "group": "Binning",
                    "description": "Show a probability density instead of counts."},
        "colorbar": {"default": False, "type": bool, "group": "Colorbar",
                     "description": "Add a colorbar for the counts."},
        "colorbar_label": {"default": "count", "type": str, "group": "Colorbar",
                           "description": "Label beside the colorbar."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="a second histogram would cover the first")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        x = to_numbers(column_of(sd.df, "x")).to_numpy(dtype=float)
        y = to_numbers(column_of(sd.df, "y")).to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        weights = None
        if "weight" in sd.df.columns:
            weights = to_numbers(column_of(sd.df, "weight")).to_numpy(dtype=float)
            keep &= np.isfinite(weights)
            weights = weights[keep]
        x, y = x[keep], y[keep]
        if x.size < 2:
            return
        kwargs = self.get_kwargs(style)
        if str(self.opt("count_scale", style)) == "log" and not bool(self.opt("density", style)):
            kwargs["norm"] = LogNorm()
        *_, image = ax.hist2d(x, y, weights=weights, density=bool(self.opt("density", style)), **kwargs)
        if bool(self.opt("colorbar", style)):
            self.add_colorbar(ax, image, style)
        ax.grid(False)
        self.apply_annotations(ax, options)
