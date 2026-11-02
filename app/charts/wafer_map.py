"""Wafer Map: points, a contour or the dies on a wafer drawn to scale, its notch or flat and its die grid.

The wafer is drawn in millimetres from its centre: the edge with its orientation mark (a notch or a flat,
where Mark angle says), the edge exclusion as a dashed ring, and - with a die size - the whole dies' grid.
On it, by Display: the points themselves (defect positions, measurement sites coloured by their value), a
contour of a value on a grid (NaN off the wafer is left blank), or the dies filled by their value (a mean,
a defect count, a bin). The Wafer Map operation computes those tables; any table of x, y and a value in mm
draws the same way.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from app.analysis import wafer as wf
from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers

POINTS, CONTOUR, DIES = "points", "contour", "dies"
#: The bins' colours: no data, pass, low, high.
BIN_COLOURS = {wf.BIN_EMPTY: "#e5e7eb", wf.BIN_PASS: "#16a34a", wf.BIN_LOW: "#2563eb", wf.BIN_HIGH: "#dc2626"}


class WaferMapAxisRenderer(BaseAxisRenderer):
    """Renderer for a wafer map.

    Role columns:
        x, y      millimetres from the wafer's centre: a point, a grid node or a die's centre.
        value     optional: the colour - a measurement, a count, or (Bins) a die bin 0-3.
    """

    Name: str = "Wafer Map"
    Category: str = "Quality and process"
    Description: str = (
        "A wafer to scale - its notch or flat, edge exclusion and die grid - with points, a contour "
        "or dies coloured by a value: thickness maps, die bins, defect maps."
    )
    Keywords: str = "wafer fetta semiconductor semiconduttori die notch flat defect difetti particle particelle"
    Goals: ClassVar[list[str]] = ["describe"]

    RequiredRoles: ClassVar[list[str]] = ["x", "y"]
    OptionalRoles: ClassVar[list[str]] = ["value"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {
        "display": {"default": POINTS, "type": [POINTS, CONTOUR, DIES], "group": "Wafer",
                    "description": "Points as they are, a contour of the value on a grid, or the dies filled."},
        "diameter": {"default": 300.0, "type": float, "min": 10.0, "max": 1000.0, "group": "Wafer",
                     "description": "The wafer's diameter, mm (100, 150, 200, 300...)."},
        "mark": {"default": wf.NOTCH, "type": [wf.NOTCH, wf.FLAT, wf.NO_MARK], "group": "Wafer",
                 "description": "The orientation mark: a notch (200 mm and up) or a flat."},
        "mark_angle": {"default": 270.0, "type": float, "min": 0.0, "max": 360.0, "group": "Wafer",
                       "description": "Where the mark is, degrees counterclockwise from the right: 270 at the bottom."},
        "edge_exclusion": {"default": 3.0, "type": float, "min": 0.0, "max": 100.0, "group": "Wafer",
                           "description": "The ring at the edge left out, mm: drawn dashed."},
        "die_width": {"default": 0.0, "type": float, "min": 0.0, "max": 500.0, "group": "Dies",
                      "description": "The die's width, mm; 0 draws no die grid."},
        "die_height": {"default": 0.0, "type": float, "min": 0.0, "max": 500.0, "group": "Dies",
                       "description": "The die's height, mm; 0 takes the width."},
        "die_offset_x": {"default": 0.0, "type": float, "group": "Dies",
                         "description": "The die grid's shift from the centre in x, mm."},
        "die_offset_y": {"default": 0.0, "type": float, "group": "Dies",
                         "description": "The die grid's shift from the centre in y, mm."},
        "bins": {"default": False, "type": bool, "group": "Dies",
                 "description": "The value is a die bin (0 no data, 1 pass, 2 low, 3 high): grey, green, blue, red."},
        "cmap": {"default": "viridis", "type": str, "group": "Appearance",
                 "description": "Colormap of the value."},
        "markersize": {"default": 4.0, "type": float, "min": 0.1, "max": 50.0, "group": "Appearance",
                       "description": "The points' size."},
        "colorbar": {"default": True, "type": bool, "group": "Colorbar",
                     "description": "A colorbar for the value."},
        "colorbar_label": {"default": None, "type": str, "group": "Colorbar",
                           "description": "The colorbar's label."},
    }

    def _wafer(self, style: dict[str, Any]) -> wf.Wafer:
        return wf.Wafer(diameter=float(self.opt("diameter", style) or 300.0),
                        mark=str(self.opt("mark", style) or wf.NOTCH),
                        mark_angle=float(self.opt("mark_angle", style) or 0.0),
                        edge_exclusion=float(self.opt("edge_exclusion", style) or 0.0))

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict[str, Any] | None = None) -> None:
        axis_options = options or {}
        sd = self.single_series(series, reason="one wafer is drawn per axis")
        style = self.merge_style(axis_options, (sd.style if sd is not None else None) or {})
        wafer = self._wafer(style)
        width = float(self.opt("die_width", style) or 0.0)
        height = float(self.opt("die_height", style) or 0.0) or width
        offset = (float(self.opt("die_offset_x", style) or 0.0), float(self.opt("die_offset_y", style) or 0.0))
        if sd is not None and {"x", "y"} <= set(sd.df.columns):
            x = to_numbers(sd.df["x"]).to_numpy(dtype=float)
            y = to_numbers(sd.df["y"]).to_numpy(dtype=float)
            value = to_numbers(sd.df["value"]).to_numpy(dtype=float) if "value" in sd.df.columns else None
            display = str(self.opt("display", style) or POINTS)
            if display == CONTOUR and value is not None:
                self._contour(ax, x, y, value, style)
            elif display == DIES and width > 0:
                self._dies(ax, x, y, value, width, height, style)
            else:
                self._points(ax, x, y, value, style)
        elif sd is not None:
            applogger.warning("Series '%s' skipped: Wafer Map needs x and y roles.", sd.name)
        self._wafer_frame(ax, wafer, width, height, offset)
        self.apply_annotations(ax, axis_options)

    def _points(self, ax: Any, x: np.ndarray, y: np.ndarray, value: np.ndarray | None, style: dict) -> None:
        size = float(self.opt("markersize", style) or 4.0) ** 2
        if value is None or not np.isfinite(value).any():
            ax.scatter(x, y, s=size, color=self.series_color(style), linewidths=0, zorder=3)
            return
        points = ax.scatter(x, y, c=value, s=size, cmap=str(self.opt("cmap", style) or "viridis"), linewidths=0,
                            zorder=3)
        if bool(self.opt("colorbar", style)):
            self._colorbar(ax, points, style)

    def _contour(self, ax: Any, x: np.ndarray, y: np.ndarray, value: np.ndarray, style: dict) -> None:
        xs, ys = np.unique(x), np.unique(y)
        # Off the wafer the grid has no rows (their value is empty): those cells stay blank. More rows than
        # cells is not a grid.
        if xs.size * ys.size < x.size or xs.size < 2 or ys.size < 2:
            applogger.error("Wafer Map contour needs the values on a grid of x and y: draw the points instead.")
            return
        grid = np.full((ys.size, xs.size), np.nan)
        grid[np.searchsorted(ys, y), np.searchsorted(xs, x)] = value
        filled = ax.contourf(xs, ys, np.ma.masked_invalid(grid), levels=14,
                             cmap=str(self.opt("cmap", style) or "viridis"), zorder=1.1)
        ax.contour(xs, ys, np.ma.masked_invalid(grid), levels=filled.levels, colors="k", linewidths=0.3,
                   alpha=0.4, zorder=1.2)
        if bool(self.opt("colorbar", style)):
            self._colorbar(ax, filled, style)

    def _dies(self, ax: Any, x: np.ndarray, y: np.ndarray, value: np.ndarray | None, width: float, height: float,
              style: dict) -> None:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        rectangles = [Rectangle((cx - width / 2, cy - height / 2), width, height) for cx, cy in zip(x, y, strict=True)]
        if bool(self.opt("bins", style)) and value is not None:
            colours = [BIN_COLOURS.get(int(v) if np.isfinite(v) else wf.BIN_EMPTY, "#e5e7eb") for v in value]
            ax.add_collection(PatchCollection(rectangles, facecolors=colours, edgecolors="white", linewidths=0.4,
                                              zorder=1.1))
            from matplotlib.patches import Patch

            present = sorted({int(v) for v in value if np.isfinite(v)})
            ax.legend(handles=[Patch(color=BIN_COLOURS[b], label=wf.BIN_NAMES[b]) for b in present if b in BIN_COLOURS],
                      loc="upper right", fontsize="small", frameon=False)
            return
        collection = PatchCollection(rectangles, cmap=str(self.opt("cmap", style) or "viridis"), edgecolors="white",
                                     linewidths=0.4, zorder=1.1)
        shown = np.asarray(value if value is not None else np.ones(len(rectangles)), float)
        collection.set_array(np.ma.masked_invalid(shown))
        ax.add_collection(collection)
        if value is not None and bool(self.opt("colorbar", style)):
            self._colorbar(ax, collection, style)

    def _colorbar(self, ax: Any, mappable: Any, style: dict) -> None:
        """Beside the disc, in an inset of its axes: it follows the axes wherever the figure's layout puts them
        (a colorbar taking its space from them was left behind, over the wafer)."""
        bar = ax.figure.colorbar(mappable, cax=ax.inset_axes((1.03, 0.0, 0.04, 1.0)))
        label = self.opt("colorbar_label", style)
        if label is not None and str(label).strip():
            bar.set_label(str(label))

    @staticmethod
    def _wafer_frame(ax: Any, wafer: wf.Wafer, width: float, height: float, offset: tuple[float, float]) -> None:
        """The edge with its mark, the edge exclusion, the dies' grid; equal scale, a margin round the disc."""
        edge_x, edge_y = wafer.outline()
        ax.plot(edge_x, edge_y, color="0.2", linewidth=1.2, zorder=4)
        if wafer.mark == wf.NOTCH:
            # A real notch is a millimetre: on a 300 mm disc it needs a sign to be found.
            angle = np.radians(wafer.mark_angle)
            ax.plot([wafer.radius * np.cos(angle)], [wafer.radius * np.sin(angle)], marker=(3, 0, wafer.mark_angle + 90),
                    markersize=7, color="0.2", zorder=5, clip_on=False)
        if wafer.edge_exclusion > 0:
            ring = wf.Wafer(wafer.diameter - 2 * wafer.edge_exclusion, wf.NO_MARK)
            ring_x, ring_y = ring.outline()
            ax.plot(ring_x, ring_y, color="0.5", linewidth=0.7, linestyle="--", zorder=4)
        if width > 0:
            dies = wf.die_grid(wafer, width, height, *offset)
            from matplotlib.collections import LineCollection

            # One collection, not a line per die: a small die makes thousands of them.
            outlines = [[(cx - width / 2, cy - height / 2), (cx + width / 2, cy - height / 2),
                         (cx + width / 2, cy + height / 2), (cx - width / 2, cy + height / 2),
                         (cx - width / 2, cy - height / 2)] for cx, cy in zip(dies["X"], dies["Y"], strict=True)]
            ax.add_collection(LineCollection(outlines, colors="0.6", linewidths=0.3, zorder=2))
        reach = wafer.radius * 1.05
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        # Square: a wafer is a circle.
        ax.set_aspect("equal", adjustable="box")
        ax._chartlibre_keep_aspect_box = True  # render_figure turns "box" into "datalim" otherwise: x would widen
