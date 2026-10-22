"""Radar (spider) chart: one spoke per variable, a polygon per group.

Several measurements of one thing on one picture: a profile. Each variable is
a spoke from the centre, scaled from its own minimum (centre) to its maximum
(rim), so columns in different units are comparable; a polygon joins a group's
values around the spokes. Meant for a few groups over a handful of variables -
more than six spokes or five polygons stop being read as a shape.

Roles: as Pair Plot - column_1 .. column_6 (none mapped: every numeric
column), and group, which gives one polygon each. Needs a polar axes, which
``Projection`` asks for.

  https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.charts.pair_plot import PAIR_COLUMN_ROLES, PairPlotAxisRenderer
from app.logs.logger import applogger
from app.plugin_api.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge


class RadarAxisRenderer(BaseAxisRenderer):
    """A profile polygon per group over one spoke per variable."""

    Name: str = "Radar Chart"
    Category: str = "Diagnostic plots"
    Description: str = (
        "One spoke per column and a polygon per group: how profiles of several measurements "
        "compare, each column on its own scale."
    )
    Keywords: str = "radar spider web star polar profile ragnatela multivariate"
    Goals: list[str] = ["compare", "relate"]
    Projection: str = "polar"
    Link: str = "https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = [*PAIR_COLUMN_ROLES, "group"]

    Kwargs: dict[str, Any] = {
        "linewidth": {"default": 1.5, "type": float, "min": 0.1, "max": 10.0, "group": "Line",
                      "description": "Width of each polygon's outline."},
        "alpha": {"default": 0.2, "type": float, "min": 0.0, "max": 1.0, "group": "Line",
                  "description": "Opacity of each polygon's fill. 0 draws outlines only."},
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "summary": {
                "default": "mean",
                "type": ["mean", "median", "rows"],
                "group": "Data",
                "description": "One polygon per group at its mean or median, or one per row.",
            },
            "scale": {
                "default": "range",
                "type": ["range", "common"],
                "group": "Data",
                "description": "Each spoke from its own minimum to its maximum, or every spoke on one scale.",
            },
            "max_rows": {
                "default": 12, "type": int, "min": 1, "max": 1000, "group": "Data",
                "description": "With one polygon per row: draw at most this many, the first ones.",
            },
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series, require_roles=False)
        if not valid:
            return
        variables = PairPlotAxisRenderer._variables(valid[0])
        if len(variables) < 3:
            applogger.info("Radar chart: it needs at least three numeric columns; '%s' has %d.",
                           valid[0].name, len(variables))
            return
        layers = PairPlotAxisRenderer._layers(self, valid, variables)  # type: ignore[arg-type]
        if not layers:
            return
        if not hasattr(ax, "set_theta_zero_location"):
            applogger.warning("Radar chart: the axis is not polar; set the axis to polar and redraw.")
            return

        everything = np.vstack([matrix for _label, _colour, matrix in layers])
        summary = str(self.opt("summary", options) or "mean")
        scale = str(self.opt("scale", options) or "range")
        with np.errstate(all="ignore"):
            low, high = np.nanmin(everything, axis=0), np.nanmax(everything, axis=0)
        if scale == "common":
            low, high = np.full_like(low, np.nanmin(low)), np.full_like(high, np.nanmax(high))
        span = np.where(high > low, high - low, 1.0)

        kwargs = self.get_kwargs(options)
        limit = int(self.opt_typed("max_rows", options) or 12)
        count = len(variables)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        closed = np.append(angles, angles[0])
        handles = []
        for label, colour, matrix in layers:
            if summary == "rows":
                rows = matrix[:limit]
            else:
                with np.errstate(all="ignore"):
                    rows = (np.nanmean if summary == "mean" else np.nanmedian)(matrix, axis=0)[None, :]
            for index, row in enumerate(rows):
                radii = (row - low) / span
                radii = np.append(radii, radii[0])
                (line,) = ax.plot(closed, radii, color=colour, linewidth=kwargs.get("linewidth", 1.5),
                                  label=label if index == 0 else None)
                if kwargs.get("alpha"):
                    ax.fill(closed, radii, color=colour, alpha=kwargs["alpha"])
                if index == 0:
                    handles.append(line)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(angles, [name for _column, name in variables])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], [] if scale == "range" else None)
        if bool(self.opt("show_legend", options)) and len(layers) > 1:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.1), borderaxespad=0.0)
        self.apply_annotations(ax, options)
