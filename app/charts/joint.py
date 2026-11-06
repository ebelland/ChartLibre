"""Joint plot: a scatter of x against y with each one's distribution along its margin.

A scatter shows how two measures go together but hides how each is spread on
its own - a skew, two modes, a clump of outliers on one axis. The joint plot
adds the two marginal distributions, a histogram or a density curve along the
top for x and along the right for y, drawn on axes that share the scatter's
scales: zoom into the scatter and the margins follow.

Roles: x, y, group optional (one colour, and one marginal curve, per group).

  https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from scipy.stats import gaussian_kde

from app.plugin_api.charts.base import (
    LEGEND_OPTIONS,
    BaseAxisRenderer,
    SeriesData,
    merge,
)
from app.utils.coercion import column_of, to_numbers


class JointAxisRenderer(BaseAxisRenderer):
    """A scatter with marginal distributions on its top and right."""

    Name: str = "Joint Plot"
    Category: str = "Statistical distributions"
    Description: str = (
        "A scatter of x against y with each one's distribution - histogram or density - along the top "
        "and the right margin, sharing the scatter's scales."
    )
    Keywords: str = "joint plot marginal histogram scatter distribuzioni marginali congiunta"
    Goals: ClassVar[list[str]] = ["relate", "describe"]
    Link: str = "https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html"

    RequiredRoles: ClassVar[list[str]] = ["x", "y"]
    OptionalRoles: ClassVar[list[str]] = ["group"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {
        "s": {"default": 16.0, "type": float, "min": 1.0, "max": 400.0, "group": "Points",
              "description": "Area of each point, in points squared."},
        "alpha": {"default": 0.7, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Points",
                  "description": "Opacity of the points."},
    }

    Options: ClassVar[dict[str, Any]] = merge(
        LEGEND_OPTIONS,
        {
            "margin_kind": {"default": "histogram", "type": ["histogram", "density"], "group": "Margins",
                            "description": "Draw each margin as a histogram or as a smooth density curve."},
            "bins": {"default": 30, "type": int, "min": 2, "max": 500, "group": "Margins",
                     "description": "Bins of the marginal histograms."},
            "margin_size": {"default": 0.22, "type": float, "min": 0.08, "max": 0.5, "step": 0.01,
                            "group": "Margins",
                            "description": "Depth of each margin, as a fraction of the scatter's size."},
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="the margins would mix two series' distributions")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        x = to_numbers(column_of(sd.df, "x")).to_numpy(dtype=float)
        y = to_numbers(column_of(sd.df, "y")).to_numpy(dtype=float)
        groups = np.asarray(sd.df["group"]).astype(str) if "group" in sd.df.columns else np.full(x.size, "")
        keep = np.isfinite(x) & np.isfinite(y)
        x, y, groups = x[keep], y[keep], groups[keep]
        if x.size < 2:
            return
        kwargs = {key: value for key, value in self.get_kwargs(style).items() if value is not None}
        names = list(dict.fromkeys(groups))
        single = names == [""]
        colours = [self.series_color(style, 0)] if single else [f"C{index % 10}" for index in range(len(names))]

        depth = float(self.opt("margin_size", style) or 0.22)
        gap = 0.03
        top = ax.inset_axes((0.0, 1.0 + gap, 1.0, depth), sharex=ax)
        right = ax.inset_axes((1.0 + gap, 0.0, depth, 1.0), sharey=ax)
        for margin in (top, right):
            for side in ("top", "right", "left", "bottom"):
                margin.spines[side].set_visible(False)
            margin.grid(False)
            margin.set_facecolor("none")
        top.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, left=False, bottom=False)
        right.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, left=False, bottom=False)

        kind = str(self.opt("margin_kind", style) or "histogram")
        bins = int(self.opt("bins", style) or 30)
        x_edges = np.histogram_bin_edges(x, bins=bins)
        y_edges = np.histogram_bin_edges(y, bins=bins)
        x_grid = np.linspace(x_edges[0], x_edges[-1], 256)
        y_grid = np.linspace(y_edges[0], y_edges[-1], 256)
        for name, colour in zip(names, colours):
            mask = groups == name
            ax.scatter(x[mask], y[mask], color=colour, edgecolors="white", linewidths=0.4,
                       label=name if not single else sd.name, zorder=3, **kwargs)
            if kind == "density" and mask.sum() > 2 and np.ptp(x[mask]) > 0 and np.ptp(y[mask]) > 0:
                top.fill_between(x_grid, gaussian_kde(x[mask])(x_grid), color=colour, alpha=0.35, linewidth=0)
                top.plot(x_grid, gaussian_kde(x[mask])(x_grid), color=colour, linewidth=1.2)
                right.fill_betweenx(y_grid, gaussian_kde(y[mask])(y_grid), color=colour, alpha=0.35, linewidth=0)
                right.plot(gaussian_kde(y[mask])(y_grid), y_grid, color=colour, linewidth=1.2)
            else:
                top.hist(x[mask], bins=x_edges, color=colour, alpha=0.6 if not single else 0.85,
                         histtype="stepfilled", linewidth=0)
                right.hist(y[mask], bins=y_edges, color=colour, alpha=0.6 if not single else 0.85,
                           histtype="stepfilled", orientation="horizontal", linewidth=0)
        top.set_ylim(bottom=0)
        right.set_xlim(left=0)
        if not single and bool(self.opt("show_legend", style)):
            ax.legend()
        self.apply_annotations(ax, options)
