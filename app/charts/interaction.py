"""Interaction plot: does the effect of one factor depend on another?

The mean response at each level of one factor (on the x axis), one line per
level of a second factor (the trace). Parallel lines mean the two factors
act independently - the effect of x is the same whatever the trace; lines
that converge, diverge or cross are an interaction, the thing a two-way
ANOVA tests and this chart shows. Error bars give each mean's standard
error, confidence interval or standard deviation.

The cell means are computed in app/analysis/diagnostics.py (todo R-06).

Roles:
    x        required, the factor on the horizontal axis (categories)
    y        required, the numeric response
    trace    optional, the second factor: one line per value
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from app.analysis.diagnostics import cell_means
from app.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.logs.logger import applogger
from app.utils.coercion import to_numbers


class InteractionAxisRenderer(BaseAxisRenderer):
    """Cell means of a response, one line per level of a second factor."""

    Name: str = "Interaction Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Mean response at each level of one factor, one line per level of another: parallel "
        "lines mean no interaction."
    )
    Link: str = "https://www.statsmodels.org/stable/generated/statsmodels.graphics.factorplots.interaction_plot.html"

    RequiredRoles: list[str] = ["x", "y"]
    OptionalRoles: list[str] = ["trace"]

    Kwargs: dict[str, Any] = {
        "marker": {
            "default": "o",
            "type": str,
            "kind": "marker",
            "group": "Lines",
            "description": "Marker at each cell mean.",
        },
        "linewidth": {
            "default": 1.6,
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "group": "Lines",
            "description": "Width of the lines joining the means.",
        },
        "capsize": {
            "default": 3.0,
            "type": float,
            "min": 0.0,
            "max": 20.0,
            "group": "Error bars",
            "description": "Length of the error bar caps, in points.",
        },
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "error_bars": {
                "default": "se",
                "type": ["se", "ci", "sd", "none"],
                "group": "Error bars",
                "description": (
                    "What the bar around each mean shows: its standard error, its confidence "
                    "interval (Student's t), the standard deviation of the cell, or nothing."
                ),
            },
            "confidence": {
                "default": 0.95,
                "type": float,
                "min": 0.5,
                "max": 0.999,
                "group": "Error bars",
                "description": "Confidence level of the interval, when the error bars show one.",
            },
            "dodge": {
                "default": 0.06,
                "type": float,
                "min": 0.0,
                "max": 0.4,
                "group": "Lines",
                "description": (
                    "Shift each line sideways by this fraction of a level, so error bars "
                    "at the same level do not cover one another."
                ),
            },
        },
    )

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        kwargs = self.get_kwargs(options)
        error_kind = str(self.opt("error_bars", options) or "se")
        confidence = float(self.opt_typed("confidence", options) or 0.95)
        dodge = float(self.opt_typed("dodge", options) or 0.0)

        tables = []
        for sd in self.valid_series(series):
            trace = np.asarray(sd.df["trace"], dtype=object) if "trace" in sd.df.columns else None
            try:
                means = cell_means(
                    np.asarray(sd.df["x"], dtype=object),
                    to_numbers(sd.df["y"]).to_numpy(dtype=float),
                    trace,
                )
            except NUMERICAL_FAILURES as exc:
                applogger.info("Interaction plot: series '%s' not drawn - %s", sd.name, exc)
                continue
            tables.append((sd, means))
        if not tables:
            return

        # One x axis for every series: the levels in the order first met.
        levels: list[Any] = list(dict.fromkeys(level for _sd, means in tables for level in means.levels))
        lines = [(sd, means, row) for sd, means in tables for row in range(len(means.traces))]
        for number, (sd, means, row) in enumerate(lines):
            style = sd.style or {}
            trace_name = means.traces[row]
            if trace_name in ("", None):
                label = sd.name
            else:
                label = f"{sd.name}: {trace_name}" if len(tables) > 1 else str(trace_name)
            shift = (number - (len(lines) - 1) / 2.0) * dodge
            positions = np.array([levels.index(level) for level in means.levels], dtype=float) + shift
            colour = style.get("color") if len(means.traces) == 1 and style.get("color") else self.series_color({}, number)
            errors = means.half_width(error_kind, confidence)[row] if error_kind != "none" else None
            ax.errorbar(
                positions, means.mean[row], yerr=errors,
                color=colour,
                marker=str(style.get("marker") or kwargs.get("marker", "o")),
                linestyle=str(style.get("linestyle") or "-") or "-",
                linewidth=float(style.get("linewidth") or kwargs.get("linewidth", 1.6)),
                capsize=float(kwargs.get("capsize", 3.0)),
                label=label if bool(style.get("show_in_legend", True)) else "_nolegend_",
                zorder=3,
            )

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(level) for level in levels])
        ax.set_xlim(-0.5, len(levels) - 0.5)
        if not str(ax.get_ylabel() or "").strip():
            ax.set_ylabel({"se": "Mean ± SE", "sd": "Mean ± SD", "ci": f"Mean ± {confidence:.0%} CI"}.get(error_kind, "Mean"))
        if bool(self.opt("show_legend", options)) and len(lines) > 1:
            ax.legend()
        self.apply_annotations(ax, options)
