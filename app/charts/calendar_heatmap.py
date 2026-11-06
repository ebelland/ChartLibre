"""Calendar heatmap: one coloured square per day, weeks as columns, one block per year.

A daily measure over months or years - sales, rainfall, commits, admissions -
laid out like a wall calendar: weekdays down, weeks across. The weekly rhythm
(quiet weekends), the seasons and the odd day that stands out are all visible
at once, which a long time series compresses into a line.

Roles: date, value (several rows on one day are summed, averaged or counted).

  https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import column_of, to_numbers

_WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS: tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: Rows a year takes: seven days and one empty row between years.
_YEAR_ROWS = 8


class CalendarHeatmapAxisRenderer(BaseAxisRenderer):
    """Days as squares in a calendar grid, coloured by value."""

    Name: str = "Calendar Heatmap"
    WhenToUse: str = "Use it to spot weekly and seasonal patterns in daily values."
    Category: str = "Gridded data"
    Description: str = (
        "A daily value as one coloured square per day, laid out like a calendar - weekdays down, weeks "
        "across, a block per year: weekly rhythm, seasons and odd days at a glance."
    )
    Keywords: str = "calendar heatmap calendario giorni daily days github activity stagionalità settimana"
    Goals: ClassVar[list[str]] = ["describe", "anomalies"]
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html"

    RequiredRoles: ClassVar[list[str]] = ["date", "value"]
    MaxSeries: int | None = 1

    Kwargs: ClassVar[dict[str, Any]] = {
        "cmap": {"default": "YlGnBu", "type": str, "kind": "colormap", "group": "Appearance",
                 "description": "Colormap of the daily values."},
    }

    Options: ClassVar[dict[str, Any]] = {
        "aggregate": {"default": "sum", "type": ["sum", "mean", "max", "min", "count"], "group": "Days",
                      "description": "How rows that fall on the same day are combined."},
        "empty_color": {"default": "#EEF0F3", "type": str, "kind": "color", "group": "Appearance",
                        "description": "Days with no data."},
        "gap_color": {"default": "white", "type": str, "kind": "color", "group": "Appearance",
                      "description": "Lines between the days."},
        "colorbar": {"default": False, "type": bool, "group": "Colorbar",
                     "description": "Add a colorbar for the values."},
        "colorbar_label": {"default": None, "type": str, "group": "Colorbar",
                           "description": "Label beside the colorbar."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        sd = self.single_series(series, reason="two calendars would cover each other")
        if sd is None:
            return
        style = self.merge_style(options, sd.style or {})
        dates = pd.to_datetime(pd.Series(np.asarray(sd.df["date"])), errors="coerce")
        values = to_numbers(column_of(sd.df, "value")).reset_index(drop=True)
        frame = pd.DataFrame({"day": dates.dt.normalize(), "value": values}).dropna(subset=["day"])
        how = str(self.opt("aggregate", style) or "sum")
        if how != "count":
            frame = frame.dropna(subset=["value"])
        if frame.empty:
            return
        combined = frame.groupby("day")["value"].agg(how)
        daily: dict[pd.Timestamp, float] = {}
        for day, value in zip(combined.index, combined.to_numpy(dtype=float)):
            stamp = pd.Timestamp(day)
            if isinstance(stamp, pd.Timestamp):
                daily[stamp] = float(value)
        levels = np.fromiter(daily.values(), dtype=float)

        years = sorted({day.year for day in daily})
        kwargs = self.get_kwargs(style)
        cmap = str(kwargs.get("cmap") or "YlGnBu")
        low, high = float(levels.min()), float(levels.max())
        norm = Normalize(vmin=low, vmax=high if high > low else low + 1.0)
        gap = str(self.opt("gap_color", style) or "white")
        mesh = None
        count = len(years)
        for block, year in enumerate(years):
            start = pd.Timestamp(year=year, month=1, day=1)
            days = pd.date_range(start, pd.Timestamp(year=year, month=12, day=31), freq="D")
            grid = np.full((7, 54), np.nan)
            present = np.zeros((7, 54), dtype=bool)
            offset = start.weekday()
            for day in days:
                column = (day.dayofyear - 1 + offset) // 7
                # Monday on the top row: rows count up from Sunday.
                row = 6 - day.weekday()
                present[row, column] = True
                if day in daily:
                    grid[row, column] = daily[day]
            # The first year on top. The y axis is not inverted - the axis
            # settings applied after the renderer reset the direction - so
            # the blocks are stacked downwards from the highest one instead.
            bottom = (count - 1 - block) * _YEAR_ROWS
            x_edges = np.arange(55)
            y_edges = bottom + np.arange(8)
            # The year's days without data first, then the coloured ones over them.
            ax.pcolormesh(x_edges, y_edges, np.where(present & np.isnan(grid), 1.0, np.nan),
                          cmap=_single_colour(str(self.opt("empty_color", style))), edgecolors=gap,
                          linewidth=1.2, vmin=0, vmax=1)
            mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                                 edgecolors=gap, linewidth=1.2)
            ax.text(-3.6, bottom + 3.5, str(year), rotation=90, ha="center", va="center", fontweight="bold")
            for weekday in (0, 2, 4):
                ax.text(-0.4, bottom + (6 - weekday) + 0.5, _WEEKDAYS[weekday], ha="right", va="center",
                        fontsize="x-small", color="0.35")

        first_year = pd.Timestamp(year=years[0], month=1, day=1)
        month_columns = [((pd.Timestamp(year=years[0], month=month, day=1).dayofyear - 1 + first_year.weekday())
                          // 7) + 0.5 for month in range(1, 13)]
        ax.set_xticks(month_columns, _MONTHS)
        ax.xaxis.tick_top()
        ax.set_yticks([])
        ax.set_xlim(-4.4, 54)
        ax.set_ylim(0, count * _YEAR_ROWS - 1)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="x", length=0, labelsize="small")
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        if mesh is not None and bool(self.opt("colorbar", style)):
            self.add_colorbar(ax, mesh, style)
        self.apply_annotations(ax, options)


def _single_colour(colour: str) -> Any:
    """A colormap that paints everything *colour*, for the days without data."""
    from matplotlib.colors import ListedColormap

    return ListedColormap([colour])
