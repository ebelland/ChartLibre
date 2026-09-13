"""Hexbin: 2D density of x/y pairs, binned into hexagons.

A scatter of enough points stops being readable long before it stops being
drawable - past a few thousand, markers pile on markers and the picture shows
where the axes are, not where the data is. Hexbin answers the question a
dense scatter cannot: how many points landed in this neighbourhood, read off
as a colour instead of counted by eye through overplotted dots. Optionally a
third column is aggregated per hexagon instead of counted - the mean
lifetime in each bin, say - which is the same shape as a weighted 2D
histogram.

  https://matplotlib.org/stable/plot_types/stats/hexbin.html
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger

#: Options this renderer consumes itself, stripped before the rest reaches
#: ``ax.hexbin`` - which rejects anything it does not recognise.
_RENDERER_ONLY_KWARGS: tuple[str, ...] = (
    "count_scale",
    "reduce_function",
    "colorbar",
    "colorbar_label",
)

#: reduce_function choice -> the callable ax.hexbin's C/reduce_C_function
#: wants. Only used when the optional "value" role is mapped; matplotlib's
#: own default (mean) applies when it is not, since C is then None and
#: reduce_C_function is simply never passed.
_REDUCERS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "count": len,
}


class HexbinAxisRenderer(BaseAxisRenderer):
    """Renderer for a hexagonally-binned 2D density (or aggregate) map.

    Role columns:
        x, y      required numeric coordinates, any layout - no grid needed.
        value     optional; when mapped, each hexagon shows Reduce function
                  applied to the values that landed in it instead of a
                  plain count.
    """

    Name: str = "Hexbin"
    Category: str = "Statistical distributions"
    Description: str = (
        "2D density of x/y pairs binned into hexagons - where a scatter "
        "with too many points to read individually actually piles up."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/stats/hexbin.html"

    RequiredRoles: list[str] = ["x", "y"]
    OptionalRoles: list[str] = ["value"]

    #: A second series would draw a second, overlapping density map - not a
    #: comparison, just one obscuring the other.
    MaxSeries: int | None = 1

    Kwargs: dict[str, object] = {
        "gridsize": {
            "default": 40,
            "type": int,
            "min": 2,
            "max": 500,
            "group": "Binning",
            "description": "Hexagons across the x range. Higher resolves finer structure; lower reads better with fewer points.",
        },
        "mincnt": {
            "default": None,
            "type": int,
            "min": 0,
            "group": "Binning",
            "description": "Hexagons with fewer points than this are left blank instead of drawn at the low end of the colormap.",
        },
        "count_scale": {
            "default": "linear",
            "type": ["linear", "log"],
            "group": "Binning",
            "description": (
                "'log' colours by log10(count) instead of the raw count - "
                "the usual fix when a few crowded bins wash out every other "
                "one on a linear scale. Ignored when Value is mapped, since "
                "the cells then hold an aggregate, not a count."
            ),
        },
        "reduce_function": {
            "default": "mean",
            "type": ["mean", "sum", "max", "min", "count"],
            "group": "Binning",
            "description": "How the Value role's numbers are combined within one hexagon. Ignored when Value is not mapped.",
        },
        "cmap": {
            "default": "viridis",
            "type": str,
            "group": "Appearance",
            "description": "Colormap the count/aggregate is mapped through, e.g. 'viridis' or 'coolwarm'.",
        },
        "alpha": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "group": "Appearance",
            "description": "Opacity of the hexagons.",
        },
        "edgecolors": {
            "default": "face",
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Hexagon border colour. 'face' matches the fill, which is invisible - set a colour to see the grid.",
        },
        "linewidths": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
            "group": "Appearance",
            "description": "Hexagon border width. Only visible with Edgecolors set to an actual colour.",
        },
        "colorbar": {
            "default": False,
            "type": bool,
            "group": "Colorbar",
            "description": (
                "Add a colorbar for the count/aggregate scale. It is a "
                "second Axes taken out of this one's space, so a "
                "constrained or compressed figure layout places it best."
            ),
        },
        "colorbar_label": {
            "default": None,
            "type": str,
            "group": "Colorbar",
            "description": "Label written alongside the colorbar. Empty leaves it unlabelled.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self.single_series(
            series, reason="a second density map would only obscure the first"
        )
        if sd is None:
            return

        if "x" not in sd.df.columns or "y" not in sd.df.columns:
            applogger.warning(
                "Series '%s' skipped: Hexbin needs x and y roles.",
                sd.name,
                show_dialog=False,
                raise_error=False,
            )
            return

        x = pd.to_numeric(sd.df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sd.df["y"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)

        value = None
        if "value" in sd.df.columns:
            value = pd.to_numeric(sd.df["value"], errors="coerce").to_numpy(dtype=float)
            finite &= np.isfinite(value)
            value = value[finite]

        x, y = x[finite], y[finite]
        if x.size < 2:
            applogger.info("Series '%s' skipped: no finite x/y pairs.", sd.name)
            return

        style = self.merge_style(axis_options, sd.style or {})
        kwargs = self.get_kwargs(style)
        for key in _RENDERER_ONLY_KWARGS:
            kwargs.pop(key, None)

        if value is not None:
            reducer = _REDUCERS.get(
                str(self.opt("reduce_function", style) or "mean"), np.mean
            )
            kwargs["C"] = value
            kwargs["reduce_C_function"] = reducer
        elif str(self.opt("count_scale", style) or "linear") == "log":
            kwargs["bins"] = "log"

        collection = ax.hexbin(x, y, **kwargs)

        if bool(self.opt("colorbar", style)):
            self._colorbar(ax, collection, style)

        self.apply_annotations(ax, axis_options)

    def _colorbar(self, ax: Any, mappable: Any, options: dict[str, Any]) -> None:
        """Add a colorbar for *mappable* beside *ax*.

        ``use_gridspec=False`` for the same reason every other renderer's
        own ``_colorbar`` uses it - see contour.py's for the full account.
        """
        figure = ax.get_figure()
        if figure is None:
            return
        colorbar = figure.colorbar(mappable, ax=ax, use_gridspec=False)
        label = self.opt("colorbar_label", options)
        if label:
            colorbar.set_label(str(label))
