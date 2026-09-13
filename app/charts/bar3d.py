"""3D Bar Chart: categorical bars standing on the x/y plane.

The three-dimensional counterpart Bar Chart (app/charts/bar.py) already has
in two - each row is one bar, planted at (x, y) and rising to z, instead of
along a single category axis. The natural fit is a small table read both
ways at once: a value per (row category, column category) pair, read as a
bar forest rather than as a Heatmap's flat colour - useful exactly where a
heatmap's colour-only encoding is not enough to compare heights by eye.

  https://matplotlib.org/stable/gallery/mplot3d/bars3d.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.charts.base import (
    ARTIST_KWARGS,
    BaseAxisRenderer,
    SeriesData,
    VIEW_OPTIONS,
    merge,
    pick,
)
from app.charts.grids import finite_xyz
from app.logs.logger import applogger


def _view_kwargs(options: dict[str, Any], renderer: BaseAxisRenderer) -> dict[str, float]:
    """Camera angles read from Options and applied to the axes.

    Duplicated from surface.py rather than imported: a private module-level
    helper there, kept private here for the same reason - each 3D renderer
    file owns its own copy rather than the two importing from each other for
    three lines.
    """
    view: dict[str, float] = {}
    for name in ("elev", "azim", "roll"):
        value = renderer.opt(name, options)
        if value is not None and value != "":
            view[name] = float(str(value))
    return view


class Bar3DAxisRenderer(BaseAxisRenderer):
    """Renderer for bars rising from the x/y plane to a mapped height.

    Role columns:
        x, y   required base position of each bar, one row per bar.
        z      required bar height - the value each (x, y) pair measures.
    """

    Name: str = "3D Bar Chart"
    Category: str = "3D and volumetric data"
    Description: str = (
        "Bars rising from the x/y plane to z - categorical data in three "
        "dimensions, where a Heatmap's flat colour alone is not enough."
    )
    Link: str = "https://matplotlib.org/stable/gallery/mplot3d/bars3d.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    #: A second series of bars sharing the same (x, y) footprint would grow
    #: out of the first one rather than beside it - occlusion, not
    #: comparison, exactly like the surfaces' MaxSeries.
    MaxSeries: int | None = 1

    Kwargs: dict[str, object] = merge(
        pick(ARTIST_KWARGS, "alpha", "label", "zorder", "visible"),
        {
            "width": {
                "default": 0.8,
                "type": float,
                "min": 0.01,
                "max": 10.0,
                "step": 0.05,
                "group": "Shape",
                "description": "Bar footprint along x.",
            },
            "depth": {
                "default": 0.8,
                "type": float,
                "min": 0.01,
                "max": 10.0,
                "step": 0.05,
                "group": "Shape",
                "description": "Bar footprint along y.",
            },
            "bottom": {
                "default": 0.0,
                "type": float,
                "group": "Shape",
                "description": "z each bar starts from. Only the height above this is drawn.",
            },
            "color": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": "One colour for every bar. Empty uses this series' own colour from the style cycle.",
            },
            "edgecolor": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": "Bar outline colour.",
            },
            "linewidth": {
                "default": 0.5,
                "type": float,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "group": "Appearance",
                "description": "Bar outline width.",
            },
            "shade": {
                "default": True,
                "type": bool,
                "group": "Appearance",
                "description": "Light each bar's faces differently by orientation - the main depth cue a still 3D image has.",
            },
        },
    )

    #: The camera, read here and applied to the axes rather than forwarded.
    Options: dict[str, object] = dict(VIEW_OPTIONS)

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self.single_series(
            series, reason="a second forest of bars would grow out of the first"
        )
        if sd is None:
            return

        x, y, z = finite_xyz(sd.df)
        if x.size == 0:
            applogger.info("Series '%s' skipped: no finite x/y/z rows.", sd.name)
            return

        merged = self.merge_style(axis_options, sd.style or {})
        kwargs = {
            key: value
            for key, value in self.get_kwargs(merged).items()
            if key not in ("width", "depth", "bottom")
            and value is not None
            and value != ""
        }
        kwargs.setdefault("color", self.series_color(sd.style or {}, 0))

        width = float(self.opt("width", merged) or 0.8)
        depth = float(self.opt("depth", merged) or 0.8)
        bottom = float(self.opt("bottom", merged) or 0.0)

        try:
            ax.bar3d(
                x - width / 2.0,
                y - depth / 2.0,
                np.full_like(z, bottom),
                width,
                depth,
                z - bottom,
                **kwargs,
            )
        except Exception:
            applogger.exception("3D Bar Chart failed to draw series '%s'.", sd.name)

        view = _view_kwargs(axis_options, self)
        if view:
            ax.view_init(**view)

        self.apply_annotations(ax, axis_options)
