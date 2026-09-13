"""Event Plot: a raster of discrete events along one axis.

A line chart implies a value that varies continuously between samples; a
scatter implies two measured coordinates. Neither fits a channel of
timestamped events - a digital signal's edges, a log's error lines, a
spike train - where the only thing that happened is "an event, here,
nothing else". Each series is drawn as its own row of tick marks, offset
from the next, which is what makes several channels comparable at a glance
without the events themselves reading as noise on a line or scatter chart.

  https://matplotlib.org/stable/plot_types/basic/eventplot.html
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger

#: Options this renderer consumes itself, stripped before the rest reaches
#: ``ax.eventplot`` - which rejects anything it does not recognise.
_RENDERER_ONLY_KWARGS: tuple[str, ...] = ("row_labels",)


class EventPlotAxisRenderer(BaseAxisRenderer):
    """Renderer for a raster of events, one row per series.

    Role columns:
        x   required.  Every finite value is one event; row order and
            colour follow the axis' own series order and style, exactly
            like every other renderer.
    """

    Name: str = "Event Plot"
    Category: str = "Statistical distributions"
    Description: str = (
        "A row of tick marks per series at each x where an event happened - "
        "a digital signal's edges, an error log, a spike train."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/basic/eventplot.html"

    RequiredRoles: list[str] = ["x"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, object] = {
        "orientation": {
            "default": "horizontal",
            "type": ["horizontal", "vertical"],
            "group": "Layout",
            "description": "'horizontal' draws events along x with rows stacked in y; 'vertical' is the transpose of that.",
        },
        "lineoffsets": {
            "default": 1.0,
            "type": float,
            "min": 0.1,
            "step": 0.1,
            "group": "Layout",
            "description": "Spacing between one series' row and the next.",
        },
        "linelengths": {
            "default": 0.8,
            "type": float,
            "min": 0.05,
            "max": 5.0,
            "step": 0.05,
            "group": "Layout",
            "description": "How tall each tick mark is, as a fraction of the row spacing - short of Line offset avoids one row's marks touching the next.",
        },
        "linewidths": {
            "default": 1.5,
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1,
            "group": "Appearance",
            "description": "Tick mark width.",
        },
        "linestyles": {
            "default": "solid",
            "type": ["solid", "dashed", "dashdot", "dotted"],
            "group": "Appearance",
            "description": "Tick mark line style.",
        },
        "alpha": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "group": "Appearance",
            "description": "Tick mark opacity.",
        },
        "row_labels": {
            "default": True,
            "type": bool,
            "group": "Layout",
            "description": "Label each row with its series name on the perpendicular axis, instead of leaving it a bare number.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        usable = self.valid_series(series)
        if not usable:
            return

        style = self.merge_style(axis_options, {})
        kwargs = self.get_kwargs(style)
        for key in _RENDERER_ONLY_KWARGS:
            kwargs.pop(key, None)

        orientation = str(kwargs.pop("orientation", "horizontal") or "horizontal")
        lineoffsets_step = float(kwargs.pop("lineoffsets", 1.0) or 1.0)
        row_labels = bool(self.opt("row_labels", style))

        offsets: list[float] = []
        names: list[str] = []
        drew_any = False

        for index, sd in enumerate(usable):
            if "x" not in sd.df.columns:
                applogger.warning(
                    "Series '%s' skipped: Event Plot needs an x role.",
                    sd.name,
                    show_dialog=False,
                    raise_error=False,
                )
                continue

            positions = pd.to_numeric(sd.df["x"], errors="coerce").to_numpy(dtype=float)
            positions = positions[np.isfinite(positions)]
            if positions.size == 0:
                applogger.info("Series '%s' skipped: no finite events.", sd.name)
                continue

            row_style = self.merge_style(axis_options, sd.style or {})
            color = self.series_color(row_style, index)
            offset = (index + 1) * lineoffsets_step

            ax.eventplot(
                positions,
                orientation=orientation,
                lineoffsets=offset,
                colors=color,
                label=sd.name if bool(row_style.get("show_in_legend", True)) else None,
                **kwargs,
            )
            offsets.append(offset)
            names.append(sd.name)
            drew_any = True

        if drew_any and row_labels:
            set_ticks = ax.set_yticks if orientation == "horizontal" else ax.set_xticks
            set_labels = (
                ax.set_yticklabels if orientation == "horizontal" else ax.set_xticklabels
            )
            set_ticks(offsets)
            set_labels(names)

        self.apply_annotations(ax, axis_options)
