"""Waterfall chart: how a start value became an end value, one step at a time.

Each bar floats from where the running total was to where it is after the
step, green for a rise and red for a fall, so what carried a profit from last
year to this year - or a mass from the input to the yield - reads as a
staircase. A row whose Total role is non-zero is not a step but a total: a bar
from zero to the running total - or, when its y is not zero, to that value,
which then becomes the running total (the opening figure).

Roles: x the step's name, y its change, total optional (non-zero marks a total; with a y, an opening figure).

  https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers


class WaterfallAxisRenderer(BaseAxisRenderer):
    """Floating bars of a running total."""

    Name: str = "Waterfall Chart"
    Category: str = "Pairwise data"
    Description: str = (
        "A running total as floating bars: what each step added or took away between a start and an "
        "end, with subtotals where marked."
    )
    Keywords: str = "waterfall bridge cascade running total variance cascata ponte profit"
    Goals: list[str] = ["describe"]
    Link: str = "https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html"

    RequiredRoles: list[str] = ["x", "y"]
    OptionalRoles: list[str] = ["total"]

    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "width": {"default": 0.7, "type": float, "min": 0.1, "max": 1.0, "step": 0.05, "group": "Appearance",
                  "description": "Width of each bar."},
        "edgecolor": {"default": None, "type": str, "kind": "color", "group": "Appearance",
                      "description": "Outline of each bar."},
    }

    Options: dict[str, Any] = {
        "up_color": {"default": "#2E9E6B", "type": str, "kind": "color", "group": "Colours",
                     "description": "Steps that add."},
        "down_color": {"default": "#D6453D", "type": str, "kind": "color", "group": "Colours",
                       "description": "Steps that take away."},
        "total_color": {"default": "#5B6B7C", "type": str, "kind": "color", "group": "Colours",
                        "description": "Subtotal bars and the final total."},
        "final_total": {"default": True, "type": bool, "group": "Steps",
                        "description": "Close with a bar for the end value."},
        "connectors": {"default": True, "type": bool, "group": "Steps",
                       "description": "A thin line from each bar's end to the next bar."},
        "value_labels": {"default": True, "type": bool, "group": "Steps",
                         "description": "Write each step's change on its bar."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="two waterfalls on one axis would overlap")
        if sd is None:
            return
        if not self.ensure_required_roles(sd.df):
            return
        steps = to_numbers(sd.df["y"]).to_numpy(dtype=float)
        names = sd.df["x"].astype(str).to_numpy()
        flags = (
            np.nan_to_num(to_numbers(sd.df["total"]).to_numpy(dtype=float)) != 0
            if "total" in sd.df.columns
            else np.zeros(len(steps), dtype=bool)
        )
        keep = np.isfinite(steps) | flags
        steps, names, flags = np.nan_to_num(steps[keep]), names[keep], flags[keep]
        if steps.size == 0:
            applogger.info("Series '%s' skipped: no steps.", sd.name)
            return

        style = self.merge_style(options, sd.style or {})
        kwargs = {key: value for key, value in self.get_kwargs(style).items() if value is not None}
        up, down, total_colour = (str(self.opt(name, style)) for name in ("up_color", "down_color", "total_color"))

        bottoms, heights, colours, labels = [], [], [], []
        running = 0.0
        for step, is_total in zip(steps, flags):
            if is_total:
                running = float(step) if step != 0 else running
                bottoms.append(0.0)
                heights.append(running)
                colours.append(total_colour)
                labels.append(f"{running:,.4g}")
            else:
                bottoms.append(running)
                heights.append(step)
                colours.append(up if step >= 0 else down)
                labels.append(f"{step:+,.4g}")
                running += step
        names = list(names)
        if bool(self.opt("final_total", style)):
            bottoms.append(0.0)
            heights.append(running)
            colours.append(total_colour)
            labels.append(f"{running:,.4g}")
            names.append("Total")

        positions = np.arange(len(heights))
        bars = ax.bar(positions, heights, bottom=bottoms, color=colours, **kwargs)
        ax.set_xticks(positions, names)
        if bool(self.opt("connectors", style)):
            ends = [bottom + height for bottom, height in zip(bottoms, heights)]
            for index in range(len(ends) - 1):
                ax.plot([index, index + 1], [ends[index]] * 2, color="0.4", linewidth=0.7, zorder=0)
        if bool(self.opt("value_labels", style)):
            ax.bar_label(bars, labels=labels, label_type="edge", fontsize="small", padding=2)
        ax.axhline(0, color="0.3", linewidth=0.8)
        ax.margins(y=0.1)
        if len(names) > 8:
            ax.tick_params(axis="x", rotation=45)
        self.apply_annotations(ax, options)
