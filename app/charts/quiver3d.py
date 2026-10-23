"""Quiver 3D (I-04): an arrow per sample of a field in space - Axes3D.quiver.

The x, y, z position and the u, v, w components at it; an optional fourth
column colours the arrows through the colormap (a field's magnitude, its
divergence). Arrows are drawn to scale by default (``length`` times the
vector), or all of one length with ``normalize``.

  https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps

from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers

_ROLES = ("x", "y", "z", "u", "v", "w")


class Quiver3DAxisRenderer(BaseAxisRenderer):
    """An arrow at each point of a 3D vector field."""

    Name: str = "Quiver 3D"
    Category: str = "3D"
    Description: str = "An arrow at each point in space showing a 3D vector field's direction and size (u, v, w)."
    Keywords: str = "quiver 3d vector field campo vettoriale frecce arrows flusso flow"
    Goals: list[str] = ["relate"]
    Link: str = "https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html"
    Projection: str = "3d"

    RequiredRoles: list[str] = list(_ROLES)
    OptionalRoles: list[str] = ["color"]

    Kwargs: dict[str, Any] = {
        "length": {
            "default": 1.0, "type": float, "min": 0.0, "group": "Arrows",
            "description": "How long an arrow is per unit of the vector (or every arrow, with Normalize).",
        },
        "normalize": {
            "default": False, "type": bool, "group": "Arrows",
            "description": "Every arrow the same length: the direction alone, the size left to the colour.",
        },
        "arrow_length_ratio": {
            "default": 0.3, "type": float, "min": 0.0, "max": 1.0, "group": "Arrows",
            "description": "The head's length as a fraction of the arrow's.",
        },
        "linewidth": {
            "default": 1.0, "type": float, "min": 0.1, "max": 10.0, "group": "Arrows",
            "description": "Width of the arrows' lines.",
        },
        "cmap": {
            "default": "viridis", "type": str, "kind": "colormap", "group": "Colour",
            "description": "Colormap of the colour column, when there is one.",
        },
        "color": {
            "default": None, "type": str, "kind": "color", "group": "Colour",
            "description": "One colour for every arrow, without a colour column.",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        if not hasattr(ax, "get_zlim"):
            return
        kwargs = self.get_kwargs(options)
        cmap_name = str(kwargs.pop("cmap", None) or "viridis")
        for sd in self.valid_series(series):
            values = np.column_stack([to_numbers(sd.df[role]).to_numpy(float) for role in _ROLES])
            keep = np.all(np.isfinite(values), axis=1)
            colour_values = None
            if "color" in sd.df.columns:
                colour_values = to_numbers(sd.df["color"]).to_numpy(float)
                keep &= np.isfinite(colour_values)
            values = values[keep]
            if not len(values):
                continue
            draw = dict(kwargs)
            if colour_values is not None:
                shown = colour_values[keep]
                span = float(np.ptp(shown)) or 1.0
                rgba = colormaps[cmap_name]((shown - shown.min()) / span)
                # Axes3D.quiver draws every shaft, then two strokes per head: colours in that order.
                draw["colors"] = np.concatenate([rgba, np.repeat(rgba, 2, axis=0)])
                draw.pop("color", None)
            elif not draw.get("color"):
                draw.pop("color", None)
            ax.quiver(*values.T, label=sd.name, **draw)
        self.apply_annotations(ax, options)
