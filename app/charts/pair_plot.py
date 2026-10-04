"""Pair plot: every numeric variable against every other, in one grid.

The first look at a table with several measurements: a scatter for each pair
below the diagonal, each variable's own distribution on it, and above it
either the mirror scatter or the correlation coefficient. Which variables go
together, which are skewed, which group separates where - all before
choosing what to fit. A series per group, or a ``group`` column, colours the
points; the grid is drawn inside the one plot area the axis has, so it sits
in a figure beside other charts like any of them.

Roles:
    column_1 .. column_6   optional, the variables, in order. Mapping none
                           takes every numeric column the query returns.
    group                  optional, colours the rows by its value
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.lines import Line2D

from app.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.logs.logger import applogger
from app.utils.coercion import to_numbers

#: The variable slots the role panel offers.
PAIR_COLUMN_ROLES: list[str] = [f"column_{index}" for index in range(1, 7)]


class PairPlotAxisRenderer(BaseAxisRenderer):
    """A scatter-plot matrix with the distributions on the diagonal."""

    Name: str = "Pair Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Every numeric variable against every other in one grid, with each one's distribution "
        "on the diagonal and the correlations above it."
    )
    Link: str = "https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = [*PAIR_COLUMN_ROLES, "group"]

    Kwargs: dict[str, Any] = {
        "s": {
            "default": 6.0,
            "type": float,
            "min": 0.5,
            "max": 100.0,
            "group": "Points",
            "description": "Size of the points, in points squared.",
        },
        "alpha": {
            "default": 0.6,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "group": "Points",
            "description": "Opacity of the points.",
        },
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "diagonal": {
                "default": "histogram",
                "type": ["histogram", "density", "none"],
                "group": "Grid",
                "description": "What each variable's own cell shows: a histogram, a smooth density, or only its name.",
            },
            "upper": {
                "default": "correlation",
                "type": ["correlation", "scatter", "none"],
                "group": "Grid",
                "description": "Above the diagonal: Pearson's r for each pair, the mirrored scatter, or nothing.",
            },
            "bins": {
                "default": 20,
                "type": int,
                "min": 2,
                "max": 200,
                "group": "Grid",
                "description": "Histogram bins on the diagonal.",
            },
            "max_points": {
                "default": 5000,
                "type": int,
                "min": 100,
                "max": 1_000_000,
                "group": "Grid",
                "description": (
                    "Draw at most this many rows in the scatters, a random sample of them - "
                    "a grid of 36 scatters is slow long before it is unreadable. The "
                    "histograms and correlations use every row."
                ),
            },
        },
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    @staticmethod
    def _variables(sd: SeriesData) -> list[tuple[str, str]]:
        """(column in the frame, name to show) for each variable of the grid."""
        roles = sd.roles if isinstance(sd.roles, dict) else {}
        mapped = [(role, str(roles.get(role) or role)) for role in PAIR_COLUMN_ROLES if role in sd.df.columns]
        if mapped:
            return mapped
        numeric = []
        for column in sd.df.columns:
            if column == "group" or column in ("Hide", "Selected", "__rowid__"):
                continue
            values = to_numbers(sd.df[column]).to_numpy(dtype=float)
            if np.isfinite(values).sum() >= 2:
                numeric.append((column, column))
        return numeric[: len(PAIR_COLUMN_ROLES)]

    def _layers(self, series: list[SeriesData], variables: list[tuple[str, str]]) -> list[tuple[str, Any, np.ndarray]]:
        """(label, colour, rows x variables) for each colour drawn: per series, per group."""
        layers = []
        for sd in series:
            if not all(column in sd.df.columns for column, _name in variables):
                applogger.info("Pair plot: series '%s' lacks some of the variables and was not drawn.", sd.name)
                continue
            matrix = np.column_stack([to_numbers(sd.df[column]).to_numpy(dtype=float) for column, _name in variables])
            if "group" in sd.df.columns:
                groups = np.asarray(sd.df["group"], dtype=object)
                for value in dict.fromkeys(groups.tolist()):
                    if value is None or value != value:
                        continue
                    mask = np.array([item == value for item in groups.tolist()], dtype=bool)
                    label = f"{sd.name}: {value}" if len(series) > 1 else str(value)
                    layers.append((label, self.series_color({}, len(layers)), matrix[mask]))
            else:
                layers.append((sd.name, self.series_color(sd.style or {}, len(layers)), matrix))
        return layers

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        valid = self.valid_series(series, require_roles=False)
        if not valid:
            return
        variables = self._variables(valid[0])
        if len(variables) < 2:
            applogger.info("Pair plot: it needs at least two numeric columns; '%s' has %d.", valid[0].name, len(variables))
            return
        layers = self._layers(valid, variables)
        if not layers:
            return

        kwargs = self.get_kwargs(options)
        diagonal = str(self.opt("diagonal", options) or "histogram")
        upper = str(self.opt("upper", options) or "correlation")
        bins = int(self.opt_typed("bins", options) or 20)
        limit = int(self.opt_typed("max_points", options) or 5000)
        rng = np.random.default_rng(0)
        shown = [matrix if len(matrix) <= limit else matrix[rng.choice(len(matrix), limit, replace=False)]
                 for _label, _colour, matrix in layers]

        # The grid lives inside the axis' own area: the axis itself is hidden.
        ax.set_axis_off()
        ax._dhub_axis_off = True  # pyright: ignore[reportAttributeAccessIssue]
        n = len(variables)
        gap = 0.025
        size = (1.0 - gap * (n - 1)) / n
        everything = np.vstack([matrix for _label, _colour, matrix in layers])
        limits = []
        for column in range(n):
            values = everything[:, column][np.isfinite(everything[:, column])]
            low, high = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
            pad = (high - low) * 0.05 or 0.5
            limits.append((low - pad, high + pad))

        for row in range(n):
            for column in range(n):
                cell = ax.inset_axes([column * (size + gap), 1.0 - (row + 1) * size - row * gap, size, size])
                if row == column:
                    self._diagonal(cell, layers, column, diagonal, bins, variables[column][1])
                elif row > column or upper == "scatter":
                    for (_label, colour, _matrix), points in zip(layers, shown):
                        cell.scatter(points[:, column], points[:, row], s=kwargs.get("s", 6.0),
                                     alpha=kwargs.get("alpha", 0.6), color=colour, linewidths=0)
                    cell.set_ylim(*limits[row])
                elif upper == "correlation":
                    self._correlations(cell, layers, row, column)
                else:
                    cell.set_axis_off()
                cell.set_xlim(*limits[column])
                cell.tick_params(labelsize="x-small", length=2)
                if row < n - 1:
                    cell.set_xticklabels([])
                else:
                    cell.set_xlabel(variables[column][1], fontsize="small")
                if column > 0 or row == column:
                    cell.set_yticklabels([])
                if column == 0 and row > 0:
                    cell.set_ylabel(variables[row][1], fontsize="small")

        if bool(self.opt("show_legend", options)) and len(layers) > 1:
            handles = [Line2D([], [], marker="o", linestyle="", color=colour, label=label) for label, colour, _m in layers]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    @staticmethod
    def _diagonal(cell: Any, layers: list, column: int, kind: str, bins: int, name: str) -> None:
        values_by_layer = [(colour, matrix[:, column][np.isfinite(matrix[:, column])]) for _l, colour, matrix in layers]
        everything = np.concatenate([values for _c, values in values_by_layer]) if values_by_layer else np.empty(0)
        if kind == "histogram" and everything.size:
            edges = np.histogram_bin_edges(everything, bins=bins)
            for colour, values in values_by_layer:
                cell.hist(values, bins=edges, color=colour, alpha=0.55 if len(layers) > 1 else 0.85)
        elif kind == "density" and everything.size > 1:
            from scipy.stats import gaussian_kde

            grid = np.linspace(everything.min(), everything.max(), 200)
            for colour, values in values_by_layer:
                if values.size > 1 and np.ptp(values) > 0:
                    cell.plot(grid, gaussian_kde(values)(grid), color=colour)
        cell.set_yticks([])
        cell.text(0.04, 0.92, name, transform=cell.transAxes, ha="left", va="top", fontsize="small", fontweight="bold")

    @staticmethod
    def _correlations(cell: Any, layers: list, row: int, column: int) -> None:
        """Pearson's r of the pair: of all the rows, then of each colour.

        Both, because they can disagree - pooled groups can show no relation,
        or the opposite one, where every group has a strong one (Simpson's
        paradox) - and that disagreement is worth seeing at a glance.
        """
        def pearson(matrix: np.ndarray) -> float | None:
            pair = matrix[:, [column, row]]
            pair = pair[np.all(np.isfinite(pair), axis=1)]
            if len(pair) < 3 or np.ptp(pair[:, 0]) == 0 or np.ptp(pair[:, 1]) == 0:
                return None
            return float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1])

        lines: list[tuple[str, Any, float]] = []
        if len(layers) > 1:
            pooled = pearson(np.vstack([matrix for _label, _colour, matrix in layers]))
            if pooled is not None:
                lines.append(("all: ", "black", pooled))
        for _label, colour, matrix in layers:
            r = pearson(matrix)
            if r is not None:
                lines.append(("", colour if len(layers) > 1 else "black", r))
        for position, (prefix, colour, r) in enumerate(lines):
            y = 0.5 + (len(lines) - 1) * 0.08 - position * 0.16
            cell.text(0.5, y, f"{prefix}r = {r:.2f}", transform=cell.transAxes, ha="center", va="center",
                      color=colour, fontsize="medium" if len(lines) == 1 else "small",
                      fontweight="bold" if prefix else "normal")
        cell.set_xticks([])
        cell.set_yticks([])
