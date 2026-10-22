"""Sankey diagram: quantities flowing through stages, each ribbon as wide as its flow.

Where the budget went, which channel the customers came through and where they
ended up, how the energy of a plant divides into work and losses. Every row is
one flow from a source to a target; a node is as tall as the larger of what
enters it and what leaves it, and the nodes sit in columns by how many steps
they are from a source. Flows that would close a loop are left out, with a
note in the log.

Roles: source and target (names), value (the flow, positive). Not
``matplotlib.sankey``, which draws one hand-built stage; this takes a table of
flows and lays out as many stages as it has.

  https://matplotlib.org/stable/api/sankey_api.html
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers


def _layers(flows: list[tuple[str, str, float]]) -> tuple[dict[str, int], list[tuple[str, str, float]]]:
    """Each node's column (longest path from a source), and the flows that do not close a loop."""
    kept: list[tuple[str, str, float]] = []
    layer: dict[str, int] = {}
    nodes = list(dict.fromkeys(name for source, target, _v in flows for name in (source, target)))
    for name in nodes:
        layer[name] = 0
    # Relax edge by edge; an edge that still moves after len(nodes) rounds is on a cycle.
    for _round in range(len(nodes)):
        moved = False
        for source, target, _value in flows:
            if source != target and layer[target] < layer[source] + 1 and layer[source] < len(nodes):
                layer[target] = layer[source] + 1
                moved = True
        if not moved:
            break
    for source, target, value in flows:
        if layer[target] > layer[source]:
            kept.append((source, target, value))
        else:
            applogger.info("Sankey: the flow %s -> %s closes a loop and is left out.", source, target)
    return layer, kept


class SankeyAxisRenderer(BaseAxisRenderer):
    """Nodes in columns joined by ribbons of width proportional to the flow."""

    Name: str = "Sankey Diagram"
    Category: str = "Pairwise data"
    Description: str = (
        "Flows between stages as ribbons as wide as the quantity: where something came from, where "
        "it went, and how it divided on the way."
    )
    Keywords: str = "sankey flow flows alluvial energy budget funnel flusso diagramma di flusso"
    Goals: list[str] = ["describe"]
    Link: str = "https://matplotlib.org/stable/api/sankey_api.html"

    RequiredRoles: list[str] = ["source", "target", "value"]
    OptionalRoles: list[str] = []

    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "alpha": {"default": 0.45, "type": float, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Appearance",
                  "description": "Opacity of the ribbons."},
    }

    Options: dict[str, Any] = {
        "node_width": {"default": 0.04, "type": float, "min": 0.005, "max": 0.2, "step": 0.005, "group": "Layout",
                       "description": "Width of a node, as a share of the picture's width."},
        "node_gap": {"default": 0.04, "type": float, "min": 0.0, "max": 0.3, "step": 0.01, "group": "Layout",
                     "description": "Space between nodes of one column, as a share of the tallest column."},
        "color_by": {"default": "source", "type": ["source", "target"], "group": "Appearance",
                     "description": "Colour each ribbon as its source node or its target node."},
        "cmap": {"default": "tab10", "type": str, "group": "Appearance",
                 "description": "Colormap the nodes' colours are taken from."},
        "value_labels": {"default": False, "type": bool, "group": "Appearance",
                         "description": "Write each node's total after its name."},
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        import matplotlib  # noqa: PLC0415 - only for the colormap

        options = options or {}
        sd = self.single_series(series, reason="two diagrams on one axis would overlap")
        if sd is None:
            return
        if not self.ensure_required_roles(sd.df):
            return
        values = to_numbers(sd.df["value"]).to_numpy(dtype=float)
        sources = sd.df["source"].astype(str).to_numpy()
        targets = sd.df["target"].astype(str).to_numpy()
        flows = [(s, t, float(v)) for s, t, v in zip(sources, targets, values) if np.isfinite(v) and v > 0]
        if not flows:
            applogger.info("Series '%s' skipped: no positive flows.", sd.name)
            return

        layer, flows = _layers(flows)
        if not flows:
            return
        nodes = list(dict.fromkeys(name for s, t, _v in flows for name in (s, t)))
        inflow = {name: 0.0 for name in nodes}
        outflow = {name: 0.0 for name in nodes}
        for source, target, value in flows:
            outflow[source] += value
            inflow[target] += value
        size = {name: max(inflow[name], outflow[name]) for name in nodes}

        style = self.merge_style(options, sd.style or {})
        gap_share = float(self.opt_typed("node_gap", style) or 0.0)
        width = float(self.opt_typed("node_width", style) or 0.04)
        columns = max(layer[name] for name in nodes) + 1
        by_column: dict[int, list[str]] = {}
        for name in nodes:
            by_column.setdefault(layer[name], []).append(name)
        tallest = max(sum(size[n] for n in names) for names in by_column.values())
        gap = gap_share * tallest

        top: dict[str, float] = {}
        span = max(sum(size[n] for n in names) + gap * (len(names) - 1) for names in by_column.values())
        for names in by_column.values():
            names.sort(key=lambda n: -size[n])
            used = sum(size[n] for n in names) + gap * (len(names) - 1)
            cursor = (span + used) / 2  # columns centred on each other
            for name in names:
                top[name] = cursor
                cursor -= size[name] + gap

        palette = matplotlib.colormaps[str(self.opt("cmap", style) or "tab10")]
        colour = {name: palette(index % palette.N if palette.N < 256 else index) for index, name in enumerate(nodes)}
        x_of = {name: layer[name] / max(columns - 1, 1) * (1 - width) for name in nodes}

        out_cursor, in_cursor = dict(top), dict(top)
        alpha = float(self.opt_typed("alpha", style) or 0.45)
        by = str(self.opt("color_by", style) or "source")
        for source, target, value in sorted(flows, key=lambda f: (-size[f[1]], -f[2])):
            x0, x1 = x_of[source] + width, x_of[target]
            y0, y1 = out_cursor[source], in_cursor[target]
            out_cursor[source] -= value
            in_cursor[target] -= value
            mid = (x0 + x1) / 2
            path = Path(
                [(x0, y0), (mid, y0), (mid, y1), (x1, y1), (x1, y1 - value), (mid, y1 - value), (mid, y0 - value),
                 (x0, y0 - value), (x0, y0)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CLOSEPOLY],
            )
            ax.add_patch(PathPatch(path, facecolor=colour[source if by == "source" else target], alpha=alpha,
                                   edgecolor="none", zorder=1))

        show_values = bool(self.opt("value_labels", style))
        for name in nodes:
            ax.add_patch(Rectangle((x_of[name], top[name] - size[name]), width, size[name], facecolor=colour[name],
                                   edgecolor="white", linewidth=0.6, zorder=2))
            label = f"{name} ({size[name]:,.4g})" if show_values else name
            left_half = layer[name] < columns / 2 or columns == 1
            ax.text(x_of[name] + (width * 1.3 if left_half else -width * 0.3), top[name] - size[name] / 2, label,
                    ha="left" if left_half else "right", va="center", fontsize="small", zorder=3)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02 * span, span * 1.02)
        # Not set_axis_off(): the axis options turn the axis back on after a render.
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.grid(False)
        self.apply_annotations(ax, options)
