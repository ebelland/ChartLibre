"""Strip plot: every value of each group as a point, jittered or packed as a beeswarm.

A box plot summarises a group in five numbers; a strip plot shows the values
themselves, so a small sample, a gap or two clusters in one group are seen
rather than guessed. Jitter spreads the points sideways at random; the
beeswarm packs them side by side so none overlaps, and the width of the swarm
then reads as a density, like a violin made of the data.

Roles: y the values, x optional the group (several series side by side when there is none).

  https://seaborn.pydata.org/generated/seaborn.swarmplot.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from app.plugin_api.charts.base import (
    LEGEND_OPTIONS,
    BaseAxisRenderer,
    SeriesData,
    merge,
)
from app.utils.coercion import column_of, to_numbers

#: Above this many points in one group the beeswarm's packing (quadratic in
#: the worst case) gives way to jitter, which reads the same at that density.
_SWARM_LIMIT = 2_000


def swarm_offsets(values: np.ndarray, diameter_y: float, diameter_x: float, half_width: float) -> np.ndarray:
    """Sideways offsets that keep the points of *values* from overlapping.

    Each point, in increasing order, takes the slot nearest the centre line -
    0, then half a diameter either side, then a whole one, ... - where no
    point already placed lies within one diameter of it. Distances are
    measured in diameters on both axes, so the swarm is round on screen
    whatever the axes' scale. Offsets beyond *half_width* are clipped: a
    crowded group overlaps at its edges rather than spilling into the next.
    """
    order = np.argsort(values, kind="stable")
    offsets = np.zeros(values.size)
    placed: list[tuple[float, float]] = []
    steps = [0.0]
    for step in range(1, 2 * values.size + 2):
        steps.extend((step * diameter_x / 2.0, -step * diameter_x / 2.0))
    for index in order:
        y = float(values[index])
        # Points further below than one diameter can no longer collide.
        placed = [(py, px) for py, px in placed if y - py < diameter_y]
        chosen = 0.0
        for candidate in steps:
            if all(((candidate - px) / diameter_x) ** 2 + ((y - py) / diameter_y) ** 2 >= 0.998
                   for py, px in placed):
                chosen = candidate
                break
        placed.append((y, chosen))
        offsets[index] = chosen
    return np.clip(offsets, -half_width, half_width)


class StripAxisRenderer(BaseAxisRenderer):
    """Each value as a point, grouped along x."""

    Name: str = "Strip Plot"
    Category: str = "Statistical distributions"
    Description: str = (
        "Every value of each group as a point, jittered or packed as a beeswarm: the data itself "
        "beside or instead of a box plot."
    )
    Keywords: str = "strip plot beeswarm swarm jitter dot plot punti gruppi distribuzione sciame"
    Goals: ClassVar[list[str]] = ["compare", "describe"]
    Link: str = "https://seaborn.pydata.org/generated/seaborn.swarmplot.html"

    RequiredRoles: ClassVar[list[str]] = ["y"]
    OptionalRoles: ClassVar[list[str]] = ["x"]

    Kwargs: ClassVar[dict[str, Any]] = {
        "s": {"default": 18.0, "type": float, "min": 1.0, "max": 400.0, "group": "Points",
              "description": "Area of each point, in points squared."},
        "alpha": {"default": 0.85, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Points",
                  "description": "Opacity of the points."},
        "marker": {"default": "o", "type": str, "kind": "marker", "group": "Points",
                   "description": "Marker of each point."},
    }

    Options: ClassVar[dict[str, Any]] = merge(
        LEGEND_OPTIONS,
        {
            "layout": {"default": "beeswarm", "type": ["beeswarm", "jitter"], "group": "Layout",
                       "description": "Pack the points side by side without overlap, or scatter them at random."},
            "width": {"default": 0.7, "type": float, "min": 0.1, "max": 1.0, "step": 0.05, "group": "Layout",
                      "description": "Width each group may take, as a fraction of the space between groups."},
            "summary": {"default": "median", "type": ["none", "median", "mean"], "group": "Layout",
                        "description": "A short line across each group at its median or mean."},
            "horizontal": {"default": False, "type": bool, "group": "Layout",
                           "description": "Groups down the y axis, values along x."},
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series)
        if not valid:
            return
        horizontal = bool(self.opt("horizontal", options))
        layout = str(self.opt("layout", options) or "beeswarm")
        width = float(self.opt("width", options) or 0.7)
        summary = str(self.opt("summary", options) or "none")

        # One frame of (group, series, value): the groups come from x, or are the series.
        frames = []
        for index, sd in enumerate(valid):
            values = to_numbers(column_of(sd.df, "y"))
            groups = (pd.Series(np.asarray(sd.df["x"]).astype(str)) if "x" in sd.df.columns
                      else pd.Series([sd.name] * len(values)))
            frames.append(pd.DataFrame({"group": groups.to_numpy(), "series": index, "value": values.to_numpy()}))
        data = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
        if data.empty:
            return
        groups = list(dict.fromkeys(data["group"]))
        by_x = any("x" in sd.df.columns for sd in valid)
        dodge = len(valid) if by_x and len(valid) > 1 else 1
        slot = width / dodge

        # A point's diameter in data units, from the axes' size in pixels.
        every_value = data["value"].to_numpy(dtype=float)
        low, high = float(every_value.min()), float(every_value.max())
        span = (high - low) or 1.0
        bbox = ax.get_window_extent()
        across_px, along_px = (bbox.height, bbox.width) if horizontal else (bbox.width, bbox.height)
        rng = np.random.default_rng(0)
        handles = []
        for index, sd in enumerate(valid):
            style = self.merge_style(options, sd.style or {})
            kwargs = {key: value for key, value in self.get_kwargs(style).items() if value is not None}
            size = float(kwargs.get("s", 18.0))
            diameter_px = np.sqrt(size) * ax.figure.dpi / 72.0
            diameter_v = diameter_px * span * 1.1 / max(along_px, 1.0)
            diameter_g = diameter_px * (len(groups) + 0.0) / max(across_px, 1.0)
            colour = self.series_color(style, index)
            first = True
            for group_index, group in enumerate(groups):
                values = data.loc[(data["group"] == group) & (data["series"] == index), "value"].to_numpy(float)
                if not values.size:
                    continue
                centre = group_index + ((index - (dodge - 1) / 2.0) * slot if dodge > 1 else 0.0)
                if layout == "beeswarm" and values.size <= _SWARM_LIMIT:
                    offsets = swarm_offsets(values, diameter_v, diameter_g, slot / 2.0)
                else:
                    offsets = rng.uniform(-slot / 2.0 * 0.8, slot / 2.0 * 0.8, values.size)
                positions = centre + offsets
                colours = colour if (dodge > 1 or len(valid) > 1 or not by_x) else f"C{group_index % 10}"
                xy = (values, positions) if horizontal else (positions, values)
                artist = ax.scatter(*xy, color=colours, label=sd.name if first else "_nolegend_",
                                    edgecolors="white", linewidths=0.4, zorder=3, **kwargs)
                if first:
                    handles.append(artist)
                    first = False
                if summary != "none":
                    level = float(np.median(values) if summary == "median" else np.mean(values))
                    edge = (centre - slot / 2.0, centre + slot / 2.0)
                    if horizontal:
                        ax.vlines(level, *edge, colors="0.15", linewidth=1.8, zorder=4)
                    else:
                        ax.hlines(level, *edge, colors="0.15", linewidth=1.8, zorder=4)

        ticks = list(range(len(groups)))
        if horizontal:
            ax.set_yticks(ticks, groups)
            ax.set_ylim(-0.6, len(groups) - 0.4)
        else:
            ax.set_xticks(ticks, groups)
            ax.set_xlim(-0.6, len(groups) - 0.4)
            if len(groups) > 8:
                ax.tick_params(axis="x", rotation=45)
        if len(valid) > 1 and bool(self.opt("show_legend", options)):
            ax.legend(handles=handles)
        self.apply_annotations(ax, options)
