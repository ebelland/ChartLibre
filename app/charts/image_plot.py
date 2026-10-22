"""Image plot: a table of numbers, or a picture file, drawn as an image (todo G-16).

``imshow`` / ``matshow`` read a matrix as it is laid out: each row of the
table is a row of the picture and each numeric column a column, coloured by
its value. For a correlation matrix, a plate reader's wells, a spectrogram
computed elsewhere or a camera frame stored as a grid. Not Heatmap, which
takes x, y, z triples and pivots them onto a grid; here the table already is
the grid. Alternatively an image file (png, jpg, tif...), shown as it is, or as
grey values through a colormap.

Roles: label (optional, the name of each row), and no column roles at all -
every numeric column is a column of the image, by its own name.

  https://matplotlib.org/stable/plot_types/arrays/imshow.html
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers

_NOT_DATA = ("label", "Hide", "Selected", "__rowid__")


class ImagePlotAxisRenderer(BaseAxisRenderer):
    """A matrix (a table of numbers) or an image file, as pixels."""

    Name: str = "Image Plot"
    Category: str = "Pairwise data"
    Description: str = (
        "A table of numbers as an image - each row a row, each numeric column a column, coloured by "
        "value - or an image file, with a colorbar."
    )
    Keywords: str = "imshow matshow image matrix matrice immagine picture pixel grid camera"
    Goals: list[str] = ["describe"]
    Link: str = "https://matplotlib.org/stable/plot_types/arrays/imshow.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = ["label"]

    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "cmap": {"default": "viridis", "type": str, "group": "Appearance",
                 "description": "Colormap the values are mapped through. Ignored for a colour image file."},
        "vmin": {"default": None, "type": float, "group": "Appearance",
                 "description": "Value at the low end of the colormap. Empty: the smallest value."},
        "vmax": {"default": None, "type": float, "group": "Appearance",
                 "description": "Value at the high end of the colormap. Empty: the largest value."},
        "interpolation": {
            "default": "nearest",
            "type": ["nearest", "bilinear", "bicubic", "antialiased", "none"],
            "group": "Appearance",
            "description": "How a pixel is drawn larger than it is: nearest keeps every cell a square.",
        },
        "aspect": {"default": "auto", "type": ["auto", "equal"], "group": "Appearance",
                   "description": "equal keeps cells square, as a picture needs; auto fills the axes."},
        "alpha": {"default": None, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Appearance",
                  "description": "Opacity."},
    }

    Options: dict[str, Any] = {
        "image_file": {"default": None, "type": str, "group": "Source",
                       "description": "Path of an image file to show instead of the table's numbers."},
        "colorbar": {"default": True, "type": bool, "group": "Colorbar",
                     "description": "Add a colorbar for the values. Left out for a colour image file."},
        "colorbar_label": {"default": None, "type": str, "group": "Colorbar",
                           "description": "Label written alongside the colorbar."},
        "annotate": {"default": False, "type": bool, "group": "Cells",
                     "description": "Write each cell's value in it, for a small matrix."},
        "annotate_format": {"default": ".2g", "type": str, "group": "Cells",
                            "description": "Format of the written value, e.g. .2f or .0f."},
        "origin": {"default": "upper", "type": ["upper", "lower"], "group": "Cells",
                   "description": "Where the first row is: at the top, as a table, or at the bottom, as a graph."},
    }

    #: Past this many cells no value is written into them.
    MAX_ANNOTATED_CELLS: int = 2500

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="two images on one axis would cover each other")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        kwargs = {key: value for key, value in self.get_kwargs(style).items() if value is not None}
        origin = str(self.opt("origin", style) or "upper")

        path = str(self.opt("image_file", style) or "").strip()
        names: list[str] = []
        rows: list[str] = []
        if path:
            import matplotlib.image as mpimg  # noqa: PLC0415 - only for a file

            try:
                data = mpimg.imread(Path(path))
            except Exception:  # noqa: BLE001 - a missing or unreadable file costs this chart, not the figure
                applogger.exception("Image Plot: cannot read the image file %r.", path)
                return
            if data.ndim == 3 and data.shape[2] in (3, 4):
                kwargs.pop("cmap", None)
                style = {**style, "colorbar": False}
            elif data.ndim == 3:
                data = data[:, :, 0]
            kwargs["aspect"] = "equal"  # a picture's pixels are square
        else:
            columns = [c for c in sd.df.columns if c not in _NOT_DATA]
            numeric = [
                (column, to_numbers(sd.df[column]).to_numpy(dtype=float))
                for column in columns
            ]
            numeric = [(column, values) for column, values in numeric if np.isfinite(values).any()]
            if not numeric:
                applogger.info("Image Plot: '%s' has no numeric column.", sd.name)
                return
            data = np.column_stack([values for _column, values in numeric])
            names = [str(column) for column, _values in numeric]
            rows = sd.df["label"].astype(str).tolist() if "label" in sd.df.columns else []

        # Always drawn with the first row at the bottom of the array's own frame: the axis options
        # set the direction of y afterwards (Invert off means upward), so imshow's own "upper" origin
        # - which is an inverted axis - would be undone. The rows are turned over instead.
        if origin == "upper":
            data = data[::-1]
            rows = rows[::-1]
        image = ax.imshow(data, origin="lower", **kwargs)

        if names:
            ax.set_xticks(np.arange(len(names)), names)
            if len(names) > 8:
                ax.tick_params(axis="x", rotation=45)
        if rows and len(rows) <= 60:
            ax.set_yticks(np.arange(len(rows)), rows)
        ax.grid(False)

        if bool(self.opt("annotate", style)) and data.ndim == 2 and data.size <= self.MAX_ANNOTATED_CELLS:
            pattern = str(self.opt("annotate_format", style) or ".2g")
            for (row, column), value in np.ndenumerate(data):
                if not np.isfinite(value):
                    continue
                try:
                    text = format(value, pattern)
                except ValueError:
                    text = f"{value:.2g}"
                red, green, blue, _alpha = image.cmap(image.norm(value))
                dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.5
                ax.text(column, row, text, ha="center", va="center", fontsize="x-small",
                        color="white" if dark else "black")

        if bool(self.opt("colorbar", style)) and data.ndim == 2:
            self.add_colorbar(ax, image, style)
        self.apply_annotations(ax, options)
