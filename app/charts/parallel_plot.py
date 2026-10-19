"""Parallel plot: each row a line across one vertical axis per variable (todo C-02).

As JMP's Graph > Parallel Plot: several measurements side by side, each on
its own scale, every row a polyline through its values - patterns across
many columns at once, groups as colours, outliers as lines that cross the
others. Each axis is scaled from its own minimum (bottom) to its maximum
(top), written at its ends; or all on one scale, or standardised.

Roles: as Pair Plot - column_1 .. column_6 (none mapped: every numeric
column), and group, which colours the lines.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.lines import Line2D

from app.charts.pair_plot import PAIR_COLUMN_ROLES, PairPlotAxisRenderer
from app.logs.logger import applogger
from app.plugin_api.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge


class ParallelPlotAxisRenderer(BaseAxisRenderer):
    """One vertical axis per variable; a line per row."""

    Name: str = "Parallel Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Each row a line across one vertical axis per column, each on its own scale: patterns, groups "
        "and outliers across many measurements at once."
    )
    Keywords: str = "parallel coordinates coordinate parallele grafico parallelo multivariate multivariata profile profilo"
    Goals: list[str] = ["relate", "groups", "anomalies"]
    Link: str = "https://pandas.pydata.org/docs/reference/api/pandas.plotting.parallel_coordinates.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = [*PAIR_COLUMN_ROLES, "group"]

    Kwargs: dict[str, Any] = {
        "linewidth": {"default": 0.8, "type": float, "min": 0.1, "max": 10.0, "group": "Line",
                      "description": "Width of each row's line."},
        "alpha": {"default": 0.5, "type": float, "min": 0.0, "max": 1.0, "group": "Line",
                  "description": "Opacity of the lines: lower for many rows."},
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "scale": {
                "default": "range",
                "type": ["range", "standardize", "common"],
                "group": "Axes",
                "description": "Each axis from its minimum to its maximum, each in standard deviations "
                               "from its mean, or every column on one common scale.",
            },
            "max_rows": {
                "default": 2000, "type": int, "min": 10, "max": 1_000_000, "group": "Axes",
                "description": "Draw at most this many rows, a random sample of them.",
            },
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series, require_roles=False)
        if not valid:
            return
        variables = PairPlotAxisRenderer._variables(valid[0])
        if len(variables) < 2:
            applogger.info("Parallel plot: it needs at least two numeric columns; '%s' has %d.",
                           valid[0].name, len(variables))
            return
        layers = PairPlotAxisRenderer._layers(self, valid, variables)  # type: ignore[arg-type]
        if not layers:
            return
        everything = np.vstack([matrix for _label, _colour, matrix in layers])
        scale = str(self.opt("scale", options) or "range")
        with np.errstate(all="ignore"):
            low, high = np.nanmin(everything, axis=0), np.nanmax(everything, axis=0)
            mean, sd = np.nanmean(everything, axis=0), np.nanstd(everything, axis=0, ddof=1)

        def scaled(matrix: np.ndarray) -> np.ndarray:
            if scale == "standardize":
                return (matrix - mean) / np.where(sd > 0, sd, 1.0)
            if scale == "range":
                return (matrix - low) / np.where(high > low, high - low, 1.0)
            return matrix

        kwargs = self.get_kwargs(options)
        limit = int(self.opt_typed("max_rows", options) or 2000)
        rng = np.random.default_rng(0)
        positions = np.arange(len(variables))
        for label, colour, matrix in layers:
            rows = matrix if len(matrix) <= limit else matrix[rng.choice(len(matrix), limit, replace=False)]
            lines = scaled(rows)
            ax.plot(positions, lines.T, color=colour, linewidth=kwargs.get("linewidth", 0.8),
                    alpha=kwargs.get("alpha", 0.5))
        for position in positions:
            ax.plot([position, position], [0, 1] if scale == "range" else ax.get_ylim(), color="0.35",
                    linewidth=0.8, zorder=0)
        ax.set_xticks(positions, [name for _column, name in variables])
        ax.set_xlim(-0.1, len(variables) - 0.9)
        if scale == "range":
            # The axes are the vertical lines: no frame, and room for each one's ends above and below.
            ax.set_ylim(-0.09, 1.09)
            ax.set_yticks([])
            for side in ("top", "bottom", "left", "right"):
                ax.spines[side].set_visible(False)
            for position, (bottom, top) in enumerate(zip(low, high)):
                ax.text(position, 1.02, f"{top:.4g}", ha="center", va="bottom", fontsize="x-small")
                ax.text(position, -0.02, f"{bottom:.4g}", ha="center", va="top", fontsize="x-small")
        elif scale == "standardize":
            ax.set_ylabel("Standard deviations from the mean")
        ax.grid(False)
        if bool(self.opt("show_legend", options)) and len(layers) > 1:
            handles = [Line2D([], [], color=colour, label=label) for label, colour, _m in layers]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
