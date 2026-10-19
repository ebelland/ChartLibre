"""Cell plot: the table itself, a coloured cell per value (todo C-02).

As JMP's Graph > Cell Plot: one column of cells per variable, one row of
cells per row of the table, each coloured by its value on its own column's
scale - sorted runs, clusters of high values and missing cells (left
blank) are seen at a glance across the whole table.

Roles: as Pair Plot - column_1 .. column_6 (none mapped: every numeric
column); label, optional, names the rows on the left.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.charts.pair_plot import PAIR_COLUMN_ROLES, PairPlotAxisRenderer
from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData


class CellPlotAxisRenderer(BaseAxisRenderer):
    """A cell per value, coloured on its column's scale."""

    Name: str = "Cell Plot"
    Category: str = "Gridded data"
    Description: str = (
        "The table as coloured cells, one column per variable and one row per row, each column on its "
        "own colour scale: runs, clusters and missing values across the table."
    )
    Keywords: str = "cell plot grafico a celle heat map mappa di calore table tabella colori"
    Goals: list[str] = ["describe", "anomalies"]
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = [*PAIR_COLUMN_ROLES, "label"]

    Kwargs: dict[str, Any] = {
        "cmap": {"default": "viridis", "type": str, "group": "Appearance",
                 "description": "The colour map: low values at its start, high at its end."},
    }

    Options: dict[str, Any] = {
        "scale": {
            "default": "column",
            "type": ["column", "common"],
            "group": "Cells",
            "description": "Each column coloured from its own minimum to its maximum, or every column on one scale.",
        },
        "max_rows": {
            "default": 500, "type": int, "min": 10, "max": 100_000, "group": "Cells",
            "description": "Show at most this many rows, the first ones.",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series, require_roles=False)
        if not valid:
            return
        sd = valid[0]
        variables = [(column, name) for column, name in PairPlotAxisRenderer._variables(sd) if column != "label"]
        if not variables:
            applogger.info("Cell plot: '%s' has no numeric column.", sd.name)
            return
        limit = int(self.opt_typed("max_rows", options) or 500)
        from app.utils.coercion import to_numbers

        matrix = np.column_stack([to_numbers(sd.df[column]).to_numpy(dtype=float) for column, _n in variables])[:limit]
        with np.errstate(all="ignore"):
            if str(self.opt("scale", options) or "column") == "column":
                low, high = np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0)
                matrix = (matrix - low) / np.where(high > low, high - low, 1.0)
        kwargs = self.get_kwargs(options)
        # Drawn upside down, the first row at the top as in the table: the
        # figure's own limits would undo an inverted axis.
        rows = len(matrix)
        mesh = ax.pcolormesh(np.ma.masked_invalid(matrix[::-1]), cmap=kwargs.get("cmap", "viridis"),
                             edgecolors="white", linewidth=0.4 if rows <= 40 else 0.0)
        ax.set_xticks(np.arange(len(variables)) + 0.5, [name for _column, name in variables])
        ax.set_xlim(0, len(variables))
        ax.set_ylim(0, rows)
        if "label" in sd.df.columns and rows <= 60:
            shown = list(range(rows))
            names = [str(v) for v in sd.df["label"].tolist()[:limit]]
        else:
            step = max(1, int(np.ceil(rows / 10)))
            shown = [0, *range(step - 1, rows, step)] if rows > 1 else [0]
            names = [str(i + 1) for i in range(rows)]
        ax.set_yticks([rows - i - 0.5 for i in shown], [names[i] for i in shown])
        ax.grid(False)
        ax.tick_params(length=0)
        if str(self.opt("scale", options) or "column") == "common":
            ax.figure.colorbar(mesh, ax=ax)
