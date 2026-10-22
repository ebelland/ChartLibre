"""Wind rose: how often each compass direction occurs, split by strength.

The polar bar chart of a direction column: the circle is cut into equal
sectors, a bar grows from the centre in each by how many observations fell in
it, and - given a value column such as the wind speed - the bar is stacked in
bands of that value, so a sector reads as "from the west, mostly light, now and
then a gale". Also for any angle: the hour of the day, the phase of a cycle, a
heading.

Roles: x is the direction in degrees (0 = north, clockwise; the Angle unit
option says radians or hours), y optional, the value to split each sector by.
Needs a polar axes, which ``Projection`` asks for.

  https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_bar.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.logs.logger import applogger
from app.plugin_api.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.utils.coercion import to_numbers

_PERIOD = {"degrees": 360.0, "radians": 2 * np.pi, "hours": 24.0}


class WindRoseAxisRenderer(BaseAxisRenderer):
    """Polar bars: counts per direction sector, stacked by bands of a value."""

    Name: str = "Wind Rose"
    Category: str = "Pairwise data"
    Description: str = (
        "How often each direction occurs, as bars around a circle, stacked by bands of a "
        "second value such as the speed. Also for any angle or hour of the day."
    )
    Keywords: str = "wind rose polar bar direction compass angle circular rosa dei venti direzione"
    Goals: list[str] = ["describe"]
    Projection: str = "polar"
    Link: str = "https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_bar.html"

    RequiredRoles: list[str] = ["x"]
    OptionalRoles: list[str] = ["y"]

    #: A second series would be drawn over the first one's bars.
    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "edgecolor": {"default": "white", "type": str, "kind": "color", "group": "Appearance",
                      "description": "Outline of each bar."},
        "linewidth": {"default": 0.5, "type": float, "min": 0.0, "max": 5.0, "group": "Appearance",
                      "description": "Width of each bar's outline."},
        "alpha": {"default": None, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Appearance",
                  "description": "Opacity of the bars."},
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "sectors": {"default": 16, "type": int, "min": 3, "max": 72, "group": "Sectors",
                        "description": "Equal sectors the circle is cut into; 16 are the compass points."},
            "angle_unit": {"default": "degrees", "type": ["degrees", "radians", "hours"], "group": "Sectors",
                           "description": "The unit of x: a full circle is 360, 2 pi or 24."},
            "bands": {"default": 5, "type": int, "min": 1, "max": 12, "group": "Sectors",
                      "description": "Bands of y each bar is split into, of equal width. Ignored without y."},
            "percent": {"default": True, "type": bool, "group": "Sectors",
                        "description": "Bars as a share of all the observations, rather than counts."},
            "cmap": {"default": "viridis", "type": str, "group": "Appearance",
                     "description": "Colormap the bands are taken from."},
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="a second rose would be drawn over the first")
        if sd is None:
            return
        if "x" not in sd.df.columns:
            applogger.warning("Series '%s' skipped: Wind Rose needs an x role (the direction).", sd.name)
            return
        if not hasattr(ax, "set_theta_zero_location"):
            applogger.warning("Wind rose: the axis is not polar; set the axis to polar and redraw.")
            return

        direction = to_numbers(sd.df["x"]).to_numpy(dtype=float)
        value = to_numbers(sd.df["y"]).to_numpy(dtype=float) if "y" in sd.df.columns else None
        keep = np.isfinite(direction) & (np.isfinite(value) if value is not None else True)
        direction = direction[keep]
        value = value[keep] if value is not None else None
        if direction.size == 0:
            applogger.info("Series '%s' skipped: no finite directions.", sd.name)
            return

        style = self.merge_style(options, sd.style or {})
        kwargs = self.get_kwargs(style)
        kwargs = {key: item for key, item in kwargs.items() if item is not None}
        sectors = int(self.opt_typed("sectors", style) or 16)
        period = _PERIOD.get(str(self.opt("angle_unit", style) or "degrees"), 360.0)
        width = period / sectors
        # Sector 0 is centred on 0, so north is the middle of a bar and not an edge between two.
        index = np.floor(((direction + width / 2) % period) / width).astype(int) % sectors
        centres = np.arange(sectors) * (2 * np.pi / sectors)
        total = float(direction.size) if bool(self.opt("percent", style)) else 1.0
        scale = 100.0 / total if bool(self.opt("percent", style)) else 1.0

        bar_width = 2 * np.pi / sectors
        if value is None or value.size == 0:
            counts = np.bincount(index, minlength=sectors) * scale
            ax.bar(centres, counts, width=bar_width, color=self.series_color(sd.style or {}, 0), **kwargs)
        else:
            count = int(self.opt_typed("bands", style) or 5)
            edges = np.linspace(float(value.min()), float(value.max()), count + 1)
            if edges[0] == edges[-1]:
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
                count = 1
            band = np.clip(np.digitize(value, edges[1:-1]), 0, count - 1)
            import matplotlib  # noqa: PLC0415 - only for the colormap

            cmap = matplotlib.colormaps[str(self.opt("cmap", style) or "viridis")]
            bottom = np.zeros(sectors)
            for number in range(count):
                counts = np.bincount(index[band == number], minlength=sectors) * scale
                ax.bar(centres, counts, width=bar_width, bottom=bottom, color=cmap(number / max(count - 1, 1)),
                       label=f"{edges[number]:.3g} - {edges[number + 1]:.3g}", **kwargs)
                bottom += counts
            if bool(self.opt("show_legend", options)):
                ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.1), borderaxespad=0.0, title=sd.roles.get("y") if isinstance(sd.roles, dict) else None)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if period == 24.0:
            ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False), [f"{hour:02d}:00" for hour in range(0, 24, 3)])
        elif period == 360.0:
            ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_rlabel_position(22.5)
        if bool(self.opt("percent", style)):
            ax.yaxis.set_major_formatter(lambda tick, _pos: f"{tick:g}%")
        self.apply_annotations(ax, options)
