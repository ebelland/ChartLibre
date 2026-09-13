"""Heatmap: the raw per-cell value over a regular x/y grid.

The gridded sibling Contour Plot (contour.py) already covers this data shape,
but a contour map always interpolates: it bins z into levels and draws bands
between them, smoothing over the very thing a heatmap is for - reading the
actual value that was measured in each cell, cell by cell, the way a
correlation matrix or a raw instrument grid is normally read. ``pcolormesh``
draws exactly the cells that were given it, nothing between them.

  https://matplotlib.org/stable/plot_types/arrays/pcolormesh.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.charts.base import BaseAxisRenderer, SeriesData
from app.charts.grids import pivot_to_grid
from app.logs.logger import applogger
from app.utils.config import get_constant

#: Options this renderer consumes itself, stripped before the rest reaches
#: ``pcolormesh`` - which rejects anything it does not recognise.
_RENDERER_ONLY_KWARGS: tuple[str, ...] = (
    "colorbar",
    "colorbar_label",
    "annotate",
    "annotate_format",
    "annotate_fontsize",
    "annotate_color",
)

#: Above this many cells, values are not written into them - one ax.text per
#: cell is fine for the 10x10 correlation matrix this feature exists for and
#: not for a 500x500 measurement grid, where the text would be an unreadable
#: black smear over the colours it is meant to label. Same principle as
#: contour.py's MAX_POINT_MARKERS.
MAX_ANNOTATED_CELLS: int = get_constant("max_annotated_heatmap_cells", 2_500)


class HeatmapAxisRenderer(BaseAxisRenderer):
    """Renderer for a flat, per-cell colour map over a regular x/y grid.

    Role columns:
        x, y, z   required.  Every (x, y) combination present must appear
                  exactly once - a complete Cartesian product of the distinct
                  x and y values, at any spacing.  Exactly Contour Plot's own
                  requirement; see grids.pivot_to_grid.
    """

    Name: str = "Heatmap"
    Category: str = "Gridded data"
    Description: str = (
        "Raw per-cell colour over a regular x/y grid, no interpolation "
        "between cells - the value actually measured in each one."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/arrays/pcolormesh.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    #: A second scalar field would cover the first, not compare with it -
    #: same reasoning as Contour Plot.
    MaxSeries: int | None = 1

    Kwargs: dict[str, object] = {
        "cmap": {
            "default": "viridis",
            "type": str,
            "group": "Appearance",
            "description": "Colormap the value is mapped through, e.g. 'viridis' or 'coolwarm'.",
        },
        "vmin": {
            "default": None,
            "type": float,
            "group": "Appearance",
            "description": "Value mapped to the colormap's low end. Empty lets Matplotlib pick the data minimum.",
        },
        "vmax": {
            "default": None,
            "type": float,
            "group": "Appearance",
            "description": "Value mapped to the colormap's high end. Empty lets Matplotlib pick the data maximum.",
        },
        "alpha": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "group": "Appearance",
            "description": "Opacity of the cells.",
        },
        "shading": {
            "default": "auto",
            "type": ["auto", "flat", "nearest", "gouraud"],
            "group": "Appearance",
            "description": (
                "'flat'/'nearest' keep each cell one flat colour - the honest "
                "reading for measured data; 'gouraud' interpolates between "
                "cell centres, which is a smoothing choice like Contour "
                "Plot's, not a heatmap's."
            ),
        },
        "edgecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Cell border colour. Empty draws no border, which is invisible on a dense grid anyway.",
        },
        "linewidth": {
            "default": 0.0,
            "type": float,
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
            "group": "Appearance",
            "description": "Cell border width. Only visible with Edgecolor set.",
        },
        "annotate": {
            "default": False,
            "type": bool,
            "group": "Labels",
            "description": (
                "Write each cell's value into it - readable on a small grid "
                "(a correlation matrix), a black smear past a few thousand "
                "cells, where it is skipped regardless of this setting."
            ),
        },
        "annotate_format": {
            "default": "%.2g",
            "type": str,
            "group": "Labels",
            "description": "printf-style format for the cell labels, e.g. '%.2f' or '%d'.",
        },
        "annotate_fontsize": {
            "default": 8.0,
            "type": float,
            "min": 1.0,
            "max": 72.0,
            "step": 0.5,
            "group": "Labels",
            "description": "Cell label font size, in points.",
        },
        "annotate_color": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Labels",
            "description": "Cell label colour. Empty means black, which reads on most colormaps.",
        },
        "colorbar": {
            "default": False,
            "type": bool,
            "group": "Colorbar",
            "description": (
                "Add a colorbar for the value scale. It is a second Axes "
                "taken out of this one's space, so a constrained or "
                "compressed figure layout places it best."
            ),
        },
        "colorbar_label": {
            "default": None,
            "type": str,
            "group": "Colorbar",
            "description": "Label written alongside the colorbar. Empty leaves it unlabelled.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self.single_series(
            series, reason="the second field would cover the first"
        )
        if sd is None:
            return

        grid = pivot_to_grid(sd.df)
        if grid is None:
            applogger.error(
                "Heatmap needs a complete grid: every x value paired with "
                "every y value exactly once. Use 'Contour Plot (Scattered)' "
                "for data that is not on a regular grid.",
                show_dialog=False,
                raise_error=False,
            )
            return

        x_grid, y_grid, z_grid = grid
        z_grid = np.where(np.isfinite(z_grid), z_grid, np.nan)

        style = self.merge_style(axis_options, sd.style or {})
        kwargs = self.get_kwargs(style)
        for key in _RENDERER_ONLY_KWARGS:
            kwargs.pop(key, None)

        mesh = ax.pcolormesh(x_grid, y_grid, z_grid, **kwargs)

        if bool(self.opt("annotate", style)):
            self._annotate_cells(ax, x_grid, y_grid, z_grid, style)

        if bool(self.opt("colorbar", style)):
            self._colorbar(ax, mesh, style)

        self.apply_annotations(ax, axis_options)

    def _annotate_cells(
        self,
        ax: Any,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        options: dict[str, Any],
    ) -> None:
        if z_grid.size > MAX_ANNOTATED_CELLS:
            applogger.info(
                "Heatmap cell values were not written in: %s cells, past "
                "the %s this labels without turning into a smear.",
                f"{z_grid.size:,}",
                f"{MAX_ANNOTATED_CELLS:,}",
            )
            return

        fmt = str(self.opt("annotate_format", options) or "%.2g")
        color = str(self.opt("annotate_color", options) or "black")
        try:
            fontsize = float(str(self.opt("annotate_fontsize", options) or 8.0))
        except (TypeError, ValueError):
            fontsize = 8.0

        for row in range(z_grid.shape[0]):
            for col in range(z_grid.shape[1]):
                value = z_grid[row, col]
                if not np.isfinite(value):
                    continue
                try:
                    text = fmt % value
                except (TypeError, ValueError):
                    text = str(value)
                ax.text(
                    x_grid[row, col],
                    y_grid[row, col],
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=color,
                    zorder=4,
                )

    def _colorbar(self, ax: Any, mappable: Any, options: dict[str, Any]) -> None:
        """Add a colorbar for *mappable* beside *ax*.

        ``use_gridspec=False`` for the same reason contour.py's own
        ``_colorbar`` uses it: renderers draw before the figure's layout
        engine is applied, and the gridspec path Matplotlib takes by
        default in that state divides by a zero-height padding row a
        constrained/compressed engine then trips over.
        """
        figure = ax.get_figure()
        if figure is None:
            return
        colorbar = figure.colorbar(mappable, ax=ax, use_gridspec=False)
        label = self.opt("colorbar_label", options)
        if label:
            colorbar.set_label(str(label))
