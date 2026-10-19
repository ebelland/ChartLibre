"""Ternary plot: three parts of a whole as a point in a triangle (todo C-02).

As JMP's Graph > Ternary Plot: a mixture's three components (A, B, C, in
any units - each row is divided by its own sum), each corner one pure
component, the grid lines every 10%. For mixture designs, alloys, soils,
compositions.

Roles: a, b, c (the components); group, optional, colours the points.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.plugin_api.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.utils.coercion import to_numbers

_HEIGHT = np.sqrt(3.0) / 2.0


def ternary_xy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where (a, b, c), shares of their sum, falls: A at the bottom left, B bottom right, C on top."""
    total = a + b + c
    with np.errstate(all="ignore"):
        b_share, c_share = b / total, c / total
    return b_share + c_share / 2.0, c_share * _HEIGHT


class TernaryAxisRenderer(BaseAxisRenderer):
    """Points in a triangle whose corners are three pure components."""

    Name: str = "Ternary Plot"
    Category: str = "Pairwise data"
    Description: str = (
        "Three parts of a whole - a mixture's components - as points in a triangle, each corner one pure "
        "component, grid lines every 10%."
    )
    Keywords: str = "ternary ternario diagramma ternario mixture miscela composition composizione triangle triangolo"
    Goals: list[str] = ["relate", "describe"]
    Link: str = "https://en.wikipedia.org/wiki/Ternary_plot"

    RequiredRoles: list[str] = ["a", "b", "c"]
    OptionalRoles: list[str] = ["group"]

    Kwargs: dict[str, Any] = {
        "s": {"default": 20.0, "type": float, "min": 0.5, "max": 500.0, "group": "Marker",
              "description": "Size of the points, in points squared."},
        "alpha": {"default": 0.85, "type": float, "min": 0.0, "max": 1.0, "group": "Marker",
                  "description": "Opacity of the points."},
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "grid_step": {
                "default": 10, "type": int, "min": 0, "max": 50, "group": "Triangle",
                "description": "Grid lines every this many percent; 0 for none.",
            },
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series)
        if not valid:
            return
        roles = valid[0].roles if isinstance(valid[0].roles, dict) else {}
        names = [str(roles.get(key) or key.upper()) for key in ("a", "b", "c")]

        ax.set_axis_off()
        ax._dhub_axis_off = True  # pyright: ignore[reportAttributeAccessIssue]
        ax.set_aspect("equal")
        ax.plot([0, 1, 0.5, 0], [0, 0, _HEIGHT, 0], color="0.2", linewidth=1.0)
        step = int(self.opt_typed("grid_step", options) or 0)
        if 0 < step < 100:
            for share in np.arange(step, 100, step) / 100.0:
                for start, end in (((share, 0, 1 - share), (share, 1 - share, 0)),   # A constant
                                   ((0, share, 1 - share), (1 - share, share, 0)),   # B constant
                                   ((0, 1 - share, share), (1 - share, 0, share))):  # C constant
                    xs, ys = ternary_xy(np.array([start[0], end[0]]), np.array([start[1], end[1]]),
                                        np.array([start[2], end[2]]))
                    ax.plot(xs, ys, color="0.85", linewidth=0.6, zorder=0)
                ax.text(*[float(v[0]) for v in ternary_xy(np.array([1 - share]), np.array([share]), np.array([0.0]))],
                        f"{share:.0%}", ha="center", va="top", fontsize="x-small", color="0.4")
        offset = 0.04
        ax.text(-offset, -offset, names[0], ha="right", va="top", fontweight="bold")
        ax.text(1 + offset, -offset, names[1], ha="left", va="top", fontweight="bold")
        ax.text(0.5, _HEIGHT + offset, names[2], ha="center", va="bottom", fontweight="bold")

        kwargs = self.get_kwargs(options)
        drawn = 0
        for sd in valid:
            a, b, c = (to_numbers(sd.df[key]).to_numpy(dtype=float) for key in ("a", "b", "c"))
            x, y = ternary_xy(a, b, c)
            if "group" in sd.df.columns:
                groups = np.asarray(sd.df["group"], dtype=object)
                for value in dict.fromkeys(groups.tolist()):
                    mask = np.array([item == value for item in groups.tolist()], dtype=bool)
                    ax.scatter(x[mask], y[mask], s=kwargs.get("s", 20.0), alpha=kwargs.get("alpha", 0.85),
                               color=self.series_color({}, drawn), label=str(value), zorder=3)
                    drawn += 1
            else:
                ax.scatter(x, y, s=kwargs.get("s", 20.0), alpha=kwargs.get("alpha", 0.85),
                           color=self.series_color(sd.style or {}, drawn), label=sd.name, zorder=3)
                drawn += 1
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.12, _HEIGHT + 0.12)
        if bool(self.opt("show_legend", options)) and drawn > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
