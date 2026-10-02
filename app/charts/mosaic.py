"""Mosaic plot: two categorical variables as one square divided into tiles.

Each column is as wide as its category's share of the rows, and is split
into tiles as tall as the second variable's share within it - so every
tile's area is the share of rows in that pair of categories, and unequal
splits from one column to the next are the association between the two.
Shading by the Pearson residual (Friendly's mosaic) colours the tiles that
hold more rows than independence would give in blue and fewer in red; the
chi-squared test of independence is written in the log.

The tiles are computed in app/analysis/diagnostics.py (todo R-06).

Roles:
    x        required, the first variable: the columns
    y        required, the second variable: the tiles within each column
    weight   optional, a count per row, for a table already counted
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps  # pyright: ignore[reportAttributeAccessIssue]
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch, Rectangle

from app.analysis import NUMERICAL_FAILURES
from app.analysis.diagnostics import mosaic
from app.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.logs.logger import applogger
from app.utils.coercion import to_numbers


class MosaicAxisRenderer(BaseAxisRenderer):
    """A two-way contingency table drawn as tiles whose areas are the proportions."""

    Name: str = "Mosaic Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Two categorical variables as tiles whose areas are the share of rows in each pair, "
        "optionally shaded by how far each is from independence."
    )
    Link: str = "https://www.statsmodels.org/stable/generated/statsmodels.graphics.mosaicplot.mosaic.html"

    RequiredRoles: list[str] = ["x", "y"]
    OptionalRoles: list[str] = ["weight"]

    #: One square to divide; a second table would be drawn over the first.
    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "edgecolor": {
            "default": "white",
            "type": str,
            "kind": "color",
            "group": "Tiles",
            "description": "Colour of the tile outlines.",
        },
        "linewidth": {
            "default": 0.8,
            "type": float,
            "min": 0.0,
            "max": 5.0,
            "group": "Tiles",
            "description": "Width of the tile outlines.",
        },
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "colour_by": {
                "default": "category",
                "type": ["category", "residual"],
                "group": "Tiles",
                "description": (
                    "Colour each tile by its category on the y variable, or by its Pearson "
                    "residual: blue where there are more rows than independence predicts, red "
                    "where there are fewer."
                ),
            },
            "gap": {
                "default": 0.01,
                "type": float,
                "min": 0.0,
                "max": 0.1,
                "group": "Tiles",
                "description": "Space between columns and between tiles, as a fraction of the square.",
            },
            "labels": {
                "default": "percent",
                "type": ["percent", "count", "none"],
                "group": "Tiles",
                "description": "Write each tile's share of all rows, its count, or nothing.",
            },
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="a mosaic divides one square")
        if sd is None:
            return
        weight = to_numbers(sd.df["weight"]).to_numpy(dtype=float) if "weight" in sd.df.columns else None
        try:
            result = mosaic(
                np.asarray(sd.df["x"], dtype=object), np.asarray(sd.df["y"], dtype=object), weight,
                gap=float(self.opt_typed("gap", options) or 0.0),
            )
        except NUMERICAL_FAILURES as exc:
            applogger.info("Mosaic: series '%s' not drawn - %s", sd.name, exc)
            return

        kwargs = self.get_kwargs(options)
        by_residual = str(self.opt("colour_by", options) or "category") == "residual"
        palette = self.palette_colors()
        colours = {row: palette[index % len(palette)] for index, row in enumerate(result.rows)}
        cmap = colormaps["RdBu"]
        norm = Normalize(vmin=-4.0, vmax=4.0)
        total = sum(tile.count for tile in result.tiles)
        label_kind = str(self.opt("labels", options) or "percent")

        for tile in result.tiles:
            if tile.height <= 0 or tile.width <= 0:
                continue
            face = cmap(norm(tile.residual)) if by_residual else colours[tile.row]
            ax.add_patch(Rectangle((tile.x, tile.y), tile.width, tile.height, facecolor=face, **kwargs))
            if label_kind != "none" and tile.width > 0.06 and tile.height > 0.05:
                text = f"{tile.count / total:.0%}" if label_kind == "percent" else f"{tile.count:g}"
                if by_residual:  # the colour no longer names the category, so the tile does
                    text = f"{tile.row}\n{text}"
                ax.text(tile.x + tile.width / 2, tile.y + tile.height / 2, text,
                        ha="center", va="center", fontsize="small", color=_ink_on(face))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(result.column_centres)
        ax.set_xticklabels([str(column) for column in result.columns])
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        if bool(self.opt("show_legend", options)):
            if by_residual:
                handles = [Patch(facecolor=cmap(norm(value)), label=label)
                           for value, label in ((3, "more than expected"), (0, "as expected"), (-3, "fewer than expected"))]
            else:
                handles = [Patch(facecolor=colours[row], label=str(row)) for row in result.rows]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        p_text = "n/a" if not np.isfinite(result.pvalue) else f"{result.pvalue:.4g}"
        applogger.info(
            "Mosaic '%s': chi-squared test of independence %.4g on %d df, p = %s.",
            sd.name, result.chi_square, result.dof, p_text,
        )
        self.apply_annotations(ax, options)


def _ink_on(face: Any) -> str:
    """Black or white text, whichever reads on *face*."""
    red, green, blue = to_rgb(face)
    return "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.55 else "white"
