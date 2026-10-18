"""Thumbnails of the guided wizard's chart ideas, drawn from the table's own rows (G-02).

Small and quick, with plain Matplotlib rather than the chart renderers: a
picture of what the chart will show - the shape of the data - not the chart
itself, which Create chart then draws with every option. A chart that
cannot be sketched from the rows gives an empty picture, never an error.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtGui import QPixmap

from app.logs.logger import applogger
from app.plugin_api.table_operations.guided_plan import COUNT, ChartIdea

#: The thumbnail's size, in pixels.
WIDTH, HEIGHT = 176, 120
_COLOR = "#2563EB"


def _numbers(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _draw(ax, idea: ChartIdea, frame: pd.DataFrame) -> None:
    roles = idea.roles
    kind = idea.chart_type
    if kind == "Histogram":
        ax.hist(_numbers(frame[roles["value"]]), bins=20, color=_COLOR, alpha=0.85)
    elif kind in ("Box Plot", "Violin Plot"):
        value, group = roles["value"], roles.get("group")
        if group:
            levels = list(pd.unique(frame[group].dropna()))[:8]
            data = [_numbers(frame.loc[frame[group] == level, value]) for level in levels]
        else:
            data = [_numbers(frame[value])]
        data = [d for d in data if d.size]
        if kind == "Box Plot":
            ax.boxplot(data, patch_artist=True, boxprops={"facecolor": _COLOR, "alpha": 0.5})
        elif data:
            parts = ax.violinplot(data, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(_COLOR)
    elif kind == "Q-Q Plot":
        from scipy import stats

        values = np.sort(_numbers(frame[roles["value"]]))
        if values.size >= 3:
            theory = stats.norm.ppf((np.arange(1, values.size + 1) - 0.5) / values.size)
            ax.plot(theory, values, ".", color=_COLOR, markersize=3)
            slope, intercept = np.polyfit(theory, values, 1)
            ax.plot(theory, slope * theory + intercept, color="0.4", linewidth=1)
    elif kind in ("Pareto Chart", "Bar Chart"):
        x = roles["X"]
        if roles.get("Y") == COUNT:
            heights = frame[x].dropna().astype(str).value_counts()
        else:
            function, of = idea.aggregates.get("Y", ("AVG", x))
            heights = pd.to_numeric(frame[of], errors="coerce").groupby(frame[x].astype(str)).mean().dropna()
        if kind == "Pareto Chart":
            heights = heights.sort_values(ascending=False)
        heights = heights.head(12)
        ax.bar(range(len(heights)), heights.to_numpy(), color=_COLOR, alpha=0.85)
        if kind == "Pareto Chart" and heights.sum() > 0:
            twin = ax.twinx()
            twin.plot(range(len(heights)), 100 * heights.cumsum() / heights.sum(), color="#DC2626", marker=".")
            twin.set_axis_off()
    elif kind in ("Scatter Plot", "Hexbin", "Time Series"):
        x_name, y_name = roles["x"], roles["y"]
        both = frame[[x_name, y_name]].copy()
        if kind == "Time Series":
            both[x_name] = pd.to_datetime(both[x_name], errors="coerce", format="mixed")
        else:
            both[x_name] = pd.to_numeric(both[x_name], errors="coerce")
        both[y_name] = pd.to_numeric(both[y_name], errors="coerce")
        both = both.dropna()
        if kind == "Hexbin" and len(both):
            ax.hexbin(both[x_name], both[y_name], gridsize=14, cmap="Blues", mincnt=1)
        elif kind == "Time Series":
            both = both.sort_values(x_name)
            ax.plot(both[x_name], both[y_name], color=_COLOR, linewidth=1)
        else:
            colors = None
            group = roles.get("color")
            if group:
                codes = pd.Categorical(frame.loc[both.index, group]).codes
                colors = [f"C{code % 10}" for code in codes]
            ax.scatter(both[x_name], both[y_name], s=5, color=colors or _COLOR, alpha=0.7)
    elif kind == "Mosaic Plot":
        table = pd.crosstab(frame[roles["y"]].astype(str), frame[roles["x"]].astype(str))
        shares = table / table.sum(axis=0)
        bottom = np.zeros(shares.shape[1])
        for index, (_level, row) in enumerate(shares.iterrows()):
            ax.bar(range(shares.shape[1]), row.to_numpy(), bottom=bottom, color=f"C{index % 10}", width=0.95)
            bottom += row.to_numpy()
    elif kind == "Pair Plot":
        columns = [roles[r] for r in sorted(roles) if r.startswith("column_")][:2]
        if len(columns) == 2:
            both = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
            ax.scatter(both[columns[0]], both[columns[1]], s=4, color=_COLOR, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def thumbnail(idea: ChartIdea, frame: pd.DataFrame) -> QPixmap:
    """A small picture of *idea* drawn from *frame*; an empty one when it cannot be sketched."""
    figure = Figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = figure.add_axes((0.04, 0.06, 0.92, 0.9))
    try:
        _draw(ax, idea, frame)
    except Exception:  # noqa: BLE001 - a sketch of the user's data: an empty picture, never an error
        applogger.exception("Guided wizard: no thumbnail for %s", idea.chart_type)
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.8")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, transparent=True)
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap
