"""Bullet chart: a measure against its target, over bands of poor, fair and good.

Stephen Few's replacement for the dashboard gauge, one row per measure: a
dark bar for the value, a short vertical line for the target, and behind them
up to three grey bands for the qualitative ranges. It takes a fraction of a
gauge's space, and a column of them compares at a glance how far each
measure is from where it should be.

Roles: value required; label, target, and poor / fair / good (the upper limit of each band) optional.

  https://en.wikipedia.org/wiki/Bullet_graph
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import column_of, to_numbers

#: The bands' roles, from the innermost (darkest) to the outermost.
_BANDS: tuple[str, ...] = ("poor", "fair", "good")


class BulletAxisRenderer(BaseAxisRenderer):
    """A value bar, a target line and qualitative bands, one row per measure."""

    Name: str = "Bullet Chart"
    Category: str = "Pairwise data"
    Description: str = (
        "A measure against its target, over grey bands for poor, fair and good: a compact gauge, one "
        "row per measure."
    )
    Keywords: str = "bullet chart graph kpi target obiettivo gauge indicatore dashboard"
    Goals: ClassVar[list[str]] = ["compare", "process"]
    Link: str = "https://en.wikipedia.org/wiki/Bullet_graph"

    RequiredRoles: ClassVar[list[str]] = ["value"]
    OptionalRoles: ClassVar[list[str]] = ["label", "target", *_BANDS]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {}

    Options: ClassVar[dict[str, Any]] = {
        "bar_color": {"default": "#2B313B", "type": str, "kind": "color", "group": "Colours",
                      "description": "The value's bar."},
        "target_color": {"default": "#E5484D", "type": str, "kind": "color", "group": "Colours",
                         "description": "The target's line."},
        "band_color": {"default": "#7A8797", "type": str, "kind": "color", "group": "Colours",
                       "description": "The bands, from dark (poor) to light (good)."},
        "relative": {"default": False, "type": bool, "group": "Scale",
                     "description": "Show each row as a percentage of its own outer band (or target), so "
                                    "measures in different units share one scale."},
        "value_labels": {"default": True, "type": bool, "group": "Labels",
                         "description": "Write each value at the end of its bar."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="the rows of two series would overlap")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        value = to_numbers(column_of(sd.df, "value")).to_numpy(dtype=float)
        rows = value.size

        def column(name: str) -> np.ndarray:
            if name in sd.df.columns:
                return to_numbers(column_of(sd.df, name)).to_numpy(dtype=float)
            return np.full(rows, np.nan)

        labels = np.asarray(sd.df["label"]).astype(str) if "label" in sd.df.columns else np.array(
            [str(index + 1) for index in range(rows)])
        target = column("target")
        bands = np.vstack([column(name) for name in _BANDS])
        keep = np.isfinite(value)
        value, labels, target, bands = value[keep], labels[keep], target[keep], bands[:, keep]
        if not value.size:
            return
        shown = value.copy()
        if bool(self.opt("relative", style)):
            outer = np.nanmax(np.vstack([bands, target[None, :]]), axis=0)
            outer = np.where(np.isfinite(outer) & (outer != 0), outer, np.abs(value))
            outer = np.where(outer == 0, 1.0, outer)
            shown, target, bands = value / outer * 100.0, target / outer * 100.0, bands / outer[None, :] * 100.0

        band_colour = str(self.opt("band_color", style))
        bar_colour = str(self.opt("bar_color", style))
        target_colour = str(self.opt("target_color", style))
        positions = np.arange(value.size)[::-1].astype(float)
        for level in range(len(_BANDS) - 1, -1, -1):
            limits = bands[level]
            drawn = np.isfinite(limits)
            if drawn.any():
                ax.barh(positions[drawn], limits[drawn], height=0.8, color=band_colour,
                        alpha=0.18 + 0.16 * (len(_BANDS) - 1 - level), linewidth=0, zorder=1)
        bars = ax.barh(positions, shown, height=0.3, color=bar_colour, linewidth=0, zorder=3)
        drawn = np.isfinite(target)
        if drawn.any():
            ax.vlines(target[drawn], positions[drawn] - 0.28, positions[drawn] + 0.28, colors=target_colour,
                      linewidth=2.6, zorder=4)
        if bool(self.opt("value_labels", style)):
            texts = [f"{number:,.4g}" for number in value]
            ax.bar_label(bars, labels=texts, padding=4, fontsize="small")
        ax.set_yticks(positions, labels)
        ax.set_ylim(-0.6, value.size - 0.4)
        ax.margins(x=0.06)
        ax.set_xlim(left=min(0.0, float(np.nanmin(shown))))
        ax.grid(False, axis="y")
        self.apply_annotations(ax, options)
