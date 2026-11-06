"""Treemap: parts of a whole as rectangles whose areas are their values.

A pie past six wedges is a guessing game; a treemap fills a rectangle with
one tile per item, its area proportional to the value, laid out by the
squarified algorithm (Bruls, Huizing and van Wijk, 2000) so the tiles stay
close to square and comparable. With a group, the tiles of one group sit
together in one colour, and the whole reads as a two-level breakdown -
budget by department and line, disk space by folder and file.

Roles: value, label optional, group optional.

  https://en.wikipedia.org/wiki/Treemapping
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from matplotlib.patches import Rectangle

from app.plugin_api.charts.base import (
    LEGEND_OPTIONS,
    BaseAxisRenderer,
    SeriesData,
    merge,
)
from app.utils.coercion import column_of, to_numbers

Rect = tuple[float, float, float, float]


def _worst(row: list[float], side: float) -> float:
    """The worst aspect ratio of *row*'s tiles laid along a side of length *side*."""
    total = sum(row)
    if total <= 0 or side <= 0:
        return float("inf")
    return max(max(side * side * area / (total * total), (total * total) / (side * side * area))
               for area in row if area > 0)


def squarify(areas: list[float], x: float, y: float, width: float, height: float) -> list[Rect]:
    """Lay *areas* (already scaled to sum to width × height) out as near-square tiles.

    The squarified algorithm: tiles are added to the current row along the
    shorter side while that does not make the row's worst aspect ratio worse;
    then the row is fixed and the rest of the rectangle is filled the same way.
    Sorting the areas largest first, as the caller does, is what keeps it good.
    """
    rects: list[Rect] = []
    remaining = list(areas)
    while remaining:
        side = min(width, height)
        row = [remaining.pop(0)]
        while remaining and _worst([*row, remaining[0]], side) <= _worst(row, side):
            row.append(remaining.pop(0))
        total = sum(row)
        if width >= height:
            # A column on the left, as wide as the row's total over the height.
            depth = total / height if height else 0.0
            cursor = y
            for area in row:
                tall = area / depth if depth else 0.0
                rects.append((x, cursor, depth, tall))
                cursor += tall
            x, width = x + depth, width - depth
        else:
            depth = total / width if width else 0.0
            cursor = x
            for area in row:
                wide = area / depth if depth else 0.0
                rects.append((cursor, y, wide, depth))
                cursor += wide
            y, height = y + depth, height - depth
    return rects


def _fitting_label(lines: list[str], width_pt: float, height_pt: float) -> tuple[float, list[str]] | None:
    """The largest font size, 7 to 13 points, at which *lines* fit a tile, and the lines that fit.

    A tile too small for the value and the share keeps just the label; one too
    small even for that gets none.
    """
    longest = max(len(line) for line in lines) or 1
    for size in (13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0):
        if longest * size * 0.58 + 4.0 <= width_pt and len(lines) * size * 1.2 + 4.0 <= height_pt:
            return size, lines
    if len(lines) > 1:
        return _fitting_label(lines[:1], width_pt, height_pt)
    return None


class TreemapAxisRenderer(BaseAxisRenderer):
    """Tiles whose areas are the values, grouped and coloured by group."""

    Name: str = "Treemap"
    Category: str = "Pairwise data"
    Description: str = (
        "Parts of a whole as tiles whose areas are their values, grouped by colour: a breakdown with "
        "too many parts for a pie."
    )
    Keywords: str = "treemap tiles parts whole proportion gerarchia parti tutto mappa ad albero budget"
    Goals: ClassVar[list[str]] = ["compare", "describe"]
    Link: str = "https://en.wikipedia.org/wiki/Treemapping"

    RequiredRoles: ClassVar[list[str]] = ["value"]
    OptionalRoles: ClassVar[list[str]] = ["label", "group"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {}

    Options: ClassVar[dict[str, Any]] = merge(
        LEGEND_OPTIONS,
        {
            "gap_color": {"default": "white", "type": str, "kind": "color", "group": "Appearance",
                          "description": "Lines between the tiles."},
            "gap_width": {"default": 1.5, "type": float, "min": 0.0, "max": 10.0, "step": 0.5,
                          "group": "Appearance", "description": "Width of the lines between the tiles."},
            "show_values": {"default": True, "type": bool, "group": "Labels",
                            "description": "Write each tile's value under its label."},
            "show_percent": {"default": False, "type": bool, "group": "Labels",
                             "description": "Write each tile's share of the whole."},
            "min_label_share": {"default": 0.015, "type": float, "min": 0.0, "max": 0.2, "step": 0.005,
                                "group": "Labels",
                                "description": "Tiles smaller than this share of the whole get no label."},
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="one whole is divided at a time")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        values = to_numbers(column_of(sd.df, "value")).to_numpy(dtype=float)
        rows = values.size
        labels = (np.asarray(sd.df["label"]).astype(str) if "label" in sd.df.columns
                  else np.array([str(index + 1) for index in range(rows)]))
        groups = np.asarray(sd.df["group"]).astype(str) if "group" in sd.df.columns else np.full(rows, "")
        # Rows with the same group and label are one tile; the groups largest first.
        sums: dict[str, dict[str, float]] = {}
        for label, group, value in zip(labels, groups, values):
            if np.isfinite(value) and value > 0:
                tiles_of = sums.setdefault(str(group), {})
                tiles_of[str(label)] = tiles_of.get(str(label), 0.0) + float(value)
        if not sums:
            return
        totals = {name: sum(parts.values()) for name, parts in sums.items()}
        names = sorted(totals, key=lambda name: -totals[name])
        whole = sum(totals.values())
        width, height = 100.0, 100.0 * 0.62
        colour_of = {name: f"C{index % 10}" for index, name in enumerate(names)}
        gap_colour = str(self.opt("gap_color", style) or "white")
        gap_width = float(self.opt("gap_width", style) or 0.0)
        min_share = float(self.opt("min_label_share", style) or 0.0)
        show_values = bool(self.opt("show_values", style))
        show_percent = bool(self.opt("show_percent", style))
        grouped = names != [""]

        scale = width * height / whole
        # Points per data unit, for fitting the labels: the axes' width in
        # points over the 100 units it spans (the aspect is equal).
        bbox = ax.get_window_extent()
        points_per_unit = min(bbox.width / width, bbox.height / height) * 72.0 / ax.figure.dpi
        outer = squarify([totals[name] * scale for name in names], 0.0, 0.0, width, height)
        for name, (gx, gy, gw, gh) in zip(names, outer):
            part = sorted(sums[name].items(), key=lambda item: -item[1])
            inner_scale = gw * gh / totals[name]
            tiles = squarify([value * inner_scale for _, value in part], gx, gy, gw, gh)
            for index, ((tx, ty, tw, th), (label, value)) in enumerate(zip(tiles, part)):
                colour = colour_of[name] if grouped else f"C{index % 10}"
                ax.add_patch(Rectangle((tx, ty), tw, th, facecolor=colour, edgecolor=gap_colour,
                                       linewidth=gap_width, alpha=0.9 if grouped else 1.0))
                share = value / whole
                if share < min_share:
                    continue
                lines = [label]
                if show_values:
                    lines.append(f"{value:,.4g}")
                if show_percent:
                    lines.append(f"{share:.1%}")
                fitted = _fitting_label(lines, tw * points_per_unit, th * points_per_unit)
                if fitted is None:
                    continue
                size, lines = fitted
                ax.text(tx + 1.0, ty + th - 1.0, "\n".join(lines), ha="left", va="top", fontsize=size,
                        color="white", clip_on=True, linespacing=1.15)
            if grouped:
                ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor="none", edgecolor=gap_colour,
                                       linewidth=gap_width * 2.5))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal", adjustable="box")
        # A drawing, not a plot: the axis settings applied afterwards keep the frame off.
        ax.set_axis_off()
        ax._chartlibre_axis_off = True
        if grouped and bool(self.opt("show_legend", style)):
            handles = [Rectangle((0, 0), 1, 1, facecolor=colour_of[name]) for name in names]
            ax.legend(handles, names, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        self.apply_annotations(ax, options)
