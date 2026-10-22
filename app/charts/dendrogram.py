"""Dendrogram: the tree a hierarchical clustering builds, drawn as nested joins.

Cluster Analysis (Ward) says which rows share a group; the tree says how
they came together - which pairs merged first, how far apart the last two
groups were, and so where a cut gives two, three or six clusters. The height
of a join is the distance at which its two branches were merged: a long
vertical run before the next join is a natural place to cut.

Roles: as Pair Plot - column_1 .. column_6 (none mapped: every numeric
column) - and label, the leaf names (the row number when absent).

  https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.charts.pair_plot import PAIR_COLUMN_ROLES, PairPlotAxisRenderer
from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers


class DendrogramAxisRenderer(BaseAxisRenderer):
    """Hierarchical clustering of the rows as a tree."""

    Name: str = "Dendrogram"
    Category: str = "Diagnostic plots"
    Description: str = (
        "The tree of a hierarchical clustering: which rows join first, at what distance, "
        "and where a cut gives a sensible number of groups."
    )
    Keywords: str = "hierarchical clustering ward linkage tree dendrogramma cluster albero gerarchico"
    Goals: list[str] = ["groups"]
    Link: str = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = [*PAIR_COLUMN_ROLES, "label"]

    #: Two trees on one axis would be drawn over each other.
    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {}

    Options: dict[str, Any] = {
        "method": {
            "default": "ward",
            "type": ["ward", "average", "complete", "single"],
            "group": "Clustering",
            "description": "How the distance between two groups is measured. Ward minimises the growth of the "
                           "within-group variance, and the usual choice.",
        },
        "metric": {
            "default": "euclidean",
            "type": ["euclidean", "cityblock", "cosine", "correlation"],
            "group": "Clustering",
            "description": "Distance between two rows. Ward needs euclidean.",
        },
        "standardize": {
            "default": True,
            "type": bool,
            "group": "Clustering",
            "description": "Put every column in standard deviations first, so a column in large units "
                           "does not decide the tree alone.",
        },
        "cut_height": {
            "default": None,
            "type": float,
            "min": 0.0,
            "group": "Clustering",
            "description": "Draw a line at this distance and colour the groups it cuts apart. "
                           "Empty: a tenth below the last join.",
        },
        "orientation": {
            "default": "top",
            "type": ["top", "bottom", "left", "right"],
            "group": "Layout",
            "description": "Where the root is.",
        },
        "max_rows": {
            "default": 200,
            "type": int,
            "min": 3,
            "max": 5000,
            "group": "Layout",
            "description": "Cluster at most this many rows, a random sample: a tree of thousands of leaves cannot be read.",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        from scipy.cluster import hierarchy  # noqa: PLC0415 - SciPy loads on use
        from scipy.spatial.distance import pdist  # noqa: PLC0415

        options = options or {}
        sd = self.single_series(series, reason="two trees on one axis would be drawn over each other")
        if sd is None:
            return
        variables = PairPlotAxisRenderer._variables(sd)
        if len(variables) < 1:
            applogger.info("Dendrogram: '%s' has no numeric column to cluster the rows on.", sd.name)
            return

        data = np.column_stack([to_numbers(sd.df[column]).to_numpy(dtype=float) for column, _name in variables])
        labels = sd.df["label"].astype(str).to_numpy() if "label" in sd.df.columns else np.arange(1, len(data) + 1).astype(str)
        keep = np.isfinite(data).all(axis=1)
        data, labels = data[keep], labels[keep]

        limit = int(self.opt_typed("max_rows", options) or 200)
        if len(data) > limit:
            chosen = np.sort(np.random.default_rng(0).choice(len(data), limit, replace=False))
            applogger.info("Dendrogram: %d of %d rows drawn (Max rows).", limit, len(data))
            data, labels = data[chosen], labels[chosen]
        if len(data) < 3:
            applogger.info("Dendrogram: it needs at least three complete rows; '%s' has %d.", sd.name, len(data))
            return

        if bool(self.opt("standardize", options)):
            spread = data.std(axis=0, ddof=1)
            data = (data - data.mean(axis=0)) / np.where(spread > 0, spread, 1.0)

        method = str(self.opt("method", options) or "ward")
        metric = "euclidean" if method == "ward" else str(self.opt("metric", options) or "euclidean")
        try:
            linkage = hierarchy.linkage(pdist(data, metric=metric), method=method)
            cut = self.opt_typed("cut_height", options)
            threshold = float(cut) if cut else 0.9 * float(linkage[-1, 2])
            hierarchy.dendrogram(
                linkage,
                labels=labels,
                orientation=str(self.opt("orientation", options) or "top"),
                color_threshold=threshold,
                above_threshold_color="0.35",
                ax=ax,
            )
        except Exception:  # noqa: BLE001 - SciPy with the user's settings: a bad one costs this element, not the chart
            applogger.exception("Dendrogram failed to draw series '%s'.", sd.name)
            return

        horizontal = str(self.opt("orientation", options)) in ("left", "right")
        (ax.axvline if horizontal else ax.axhline)(threshold, color="0.5", linestyle="--", linewidth=0.8)
        (ax.set_xlabel if horizontal else ax.set_ylabel)("Distance")
        if len(labels) > 40:
            (ax.set_yticks if horizontal else ax.set_xticks)([])
        ax.grid(False)
        self.apply_annotations(ax, options)
