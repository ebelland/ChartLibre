"""Forest plot: many estimates with their confidence intervals, one per row.

The chart of a meta-analysis, and of any table of effects - one row per
study, subgroup or coefficient, a square at the estimate and a line across
its interval, against a vertical line at "no effect". A square's area can
follow the row's weight, so the precise studies catch the eye; a summary
row - the pooled estimate - is drawn as a diamond as wide as its interval.
For ratios (odds, hazard, risk) the axis is logarithmic and no effect is 1.

Roles:
    label      required, the name of each row
    estimate   required, the point estimate
    lower      required, the lower confidence limit
    upper      required, the upper confidence limit
    weight     optional, sizes each square (its area is proportional)
    summary    optional, nonzero for a summary row, drawn as a diamond
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger
from app.utils.coercion import to_numbers


class ForestAxisRenderer(BaseAxisRenderer):
    """Point estimates and confidence intervals, one row each, against a line of no effect."""

    Name: str = "Forest Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Estimates and their confidence intervals, one row per study or effect, against a line "
        "of no effect; summary rows as diamonds."
    )
    Link: str = "https://en.wikipedia.org/wiki/Forest_plot"

    RequiredRoles: list[str] = ["label", "estimate", "lower", "upper"]
    OptionalRoles: list[str] = ["weight", "summary"]

    Kwargs: dict[str, Any] = {
        "linewidth": {
            "default": 1.4,
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "group": "Rows",
            "description": "Width of the interval lines.",
        },
    }

    Options: dict[str, Any] = {
        "null_value": {
            "default": 0.0,
            "type": float,
            "group": "Reference",
            "description": "Where the line of no effect is drawn: 0 for differences, 1 for ratios.",
        },
        "log_scale": {
            "default": False,
            "type": bool,
            "group": "Reference",
            "description": "A logarithmic axis, for ratios: an interval from 0.5 to 2 is then symmetric about 1.",
        },
        "marker_size": {
            "default": 8.0,
            "type": float,
            "min": 2.0,
            "max": 30.0,
            "group": "Rows",
            "description": "Size of the squares, in points - of the heaviest one when a weight is given.",
        },
        "show_values": {
            "default": True,
            "type": bool,
            "group": "Rows",
            "description": "Write each estimate and its interval at the right of the chart.",
        },
        "value_format": {
            "default": ".2f",
            "type": str,
            "group": "Rows",
            "description": "How the numbers on the right are written (a Python format, e.g. .2f or .3g).",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        kwargs = self.get_kwargs(options)
        size = float(self.opt_typed("marker_size", options) or 8.0)
        number_format = str(self.opt("value_format", options) or ".2f")

        rows: list[tuple[str, float, float, float, float, bool, Any]] = []
        for index, sd in enumerate(self.valid_series(series)):
            colour = self.series_color(sd.style or {}, index)
            frame = sd.df
            estimate = to_numbers(frame["estimate"]).to_numpy(dtype=float)
            lower = to_numbers(frame["lower"]).to_numpy(dtype=float)
            upper = to_numbers(frame["upper"]).to_numpy(dtype=float)
            weight = to_numbers(frame["weight"]).to_numpy(dtype=float) if "weight" in frame.columns else np.ones(estimate.size)
            summary = (
                np.nan_to_num(to_numbers(frame["summary"]).to_numpy(dtype=float)) != 0
                if "summary" in frame.columns else np.zeros(estimate.size, dtype=bool)
            )
            if rows:
                rows.append(("", np.nan, np.nan, np.nan, np.nan, False, colour))  # a gap between series
            for label, e, lo, hi, w, s in zip(frame["label"], estimate, lower, upper, weight, summary):
                rows.append((str(label), float(e), float(lo), float(hi), float(w), bool(s), colour))
        if not rows:
            return

        finite_weights = [w for _l, e, _lo, _hi, w, s, _c in rows if np.isfinite(e) and not s and np.isfinite(w) and w > 0]
        heaviest = max(finite_weights) if finite_weights else 1.0
        positions = np.arange(len(rows))[::-1].astype(float)
        value_transform = blended_transform_factory(ax.transAxes, ax.transData)
        skipped = 0
        for y, (label, e, lo, hi, w, is_summary, colour) in zip(positions, rows):
            if not label and not np.isfinite(e):
                continue
            if not (np.isfinite(e) and np.isfinite(lo) and np.isfinite(hi)):
                skipped += 1
                continue
            if is_summary:
                # cast: the stubs type xy as a flat Sequence[float]; it is (N, 2).
                ax.add_patch(Polygon(cast(Any, [(lo, y), (e, y + 0.3), (hi, y), (e, y - 0.3)]), closed=True,
                                     facecolor=colour, edgecolor=colour, zorder=3))
            else:
                ax.plot([lo, hi], [y, y], color=colour, linewidth=float(kwargs.get("linewidth", 1.4)), zorder=2)
                scale = np.sqrt(w / heaviest) if np.isfinite(w) and w > 0 else 1.0
                ax.plot([e], [y], marker="s", linestyle="", markersize=max(2.0, size * scale),
                        color=colour, zorder=3)
            if bool(self.opt("show_values", options)):
                text = f"{e:{number_format}} [{lo:{number_format}}, {hi:{number_format}}]"
                ax.text(1.02, y, text, transform=value_transform, ha="left", va="center",
                        fontsize="small", fontweight="bold" if is_summary else "normal")
        if skipped:
            applogger.info("Forest plot: %d row(s) without an estimate and both limits were left out.", skipped)

        null_value = float(self.opt_typed("null_value", options) or 0.0)
        log_scale = bool(self.opt("log_scale", options))
        if log_scale:
            ax.set_xscale("log")
            # 0.2, 0.5, 1, 2 rather than 10^-1, 10^0: ratios are read as plain numbers.
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if null_value <= 0:
                null_value = 1.0
        ax.axvline(null_value, color="0.4", linestyle="--", linewidth=1.0, zorder=1)
        ax.set_yticks(positions)
        ax.set_yticklabels([label for label, *_rest in rows])
        for tick, (_label, *_middle, is_summary, _colour) in zip(ax.get_yticklabels(), rows):
            if is_summary:
                tick.set_fontweight("bold")
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        self.apply_annotations(ax, options)
