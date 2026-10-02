"""Q-Q and P-P plots: does this sample follow that distribution?

A histogram answers the question with a bin width chosen for it, an ECDF
with two curves to compare by eye; a probability plot turns it into "do the
points lie on a straight line", which the eye judges best of all. The Q-Q
plot puts the sample's quantiles against the fitted distribution's, and is
sharp in the tails - where the normal assumption usually fails; the P-P plot
compares probabilities instead, and is sharp in the middle.

The points are drawn by :class:`ScatterAxisRenderer`, the way the ECDF is,
so a series' colour, marker and size apply unchanged; this module adds the
reference line and the confidence band. The arithmetic is in
app/analysis/diagnostics.py (todo R-06).

Roles:
    value    required numeric sample
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from app.analysis.diagnostics import PROBABILITY_DISTRIBUTIONS, ProbabilityPlot, pp_plot, qq_plot
from app.charts.base import BaseAxisRenderer, SeriesData, merge
from app.charts.scatter import ScatterAxisRenderer
from app.data.series_frame import SeriesFrame
from app.logs.logger import applogger
from app.utils.coercion import to_numbers

#: The options both plots share.
_SHARED_OPTIONS: dict[str, Any] = {
    "distribution": {
        "default": "norm",
        "type": list(PROBABILITY_DISTRIBUTIONS),
        "group": "Distribution",
        "description": (
            "The distribution the sample is compared with, fitted to each series: "
            "the normal by its mean and standard deviation, the others by maximum likelihood."
        ),
    },
    "confidence": {
        "default": 0.95,
        "type": float,
        "min": 0.0,
        "max": 0.999,
        "group": "Distribution",
        "description": (
            "Shade the pointwise band a sample from the fitted distribution stays inside "
            "with this probability (from the order statistics). 0 draws no band."
        ),
    },
    "band_alpha": {
        "default": 0.15,
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "group": "Distribution",
        "description": "Opacity of the confidence band.",
    },
}


class QQPlotAxisRenderer(ScatterAxisRenderer, BaseAxisRenderer):
    """Sample quantiles against the quantiles of a fitted distribution."""

    Name: str = "Q-Q Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Sample quantiles against those of a fitted distribution: a straight line means the "
        "distribution fits. Sharpest in the tails."
    )
    Link: str = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html"

    RequiredRoles: list[str] = ["value"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, Any] = dict(ScatterAxisRenderer.Kwargs)

    Options: dict[str, Any] = merge(
        ScatterAxisRenderer.Options,
        _SHARED_OPTIONS,
        {
            "standardized": {
                "default": True,
                "type": bool,
                "group": "Distribution",
                "description": (
                    "Theoretical quantiles in the distribution's standard form - z-scores for "
                    "the normal - as in the classic normal plot. Off: in the data's units, where "
                    "a perfect fit lies on y = x."
                ),
            },
            "reference_line": {
                "default": "quartiles",
                "type": ["quartiles", "least squares", "identity", "none"],
                "group": "Distribution",
                "description": (
                    "The line the points are judged against: through the first and third "
                    "quartiles (R's qqline, not pulled by outliers), the least-squares line, "
                    "the fitted distribution itself, or none."
                ),
            },
        },
    )

    _scatter = ScatterAxisRenderer()

    def compute(self, values: np.ndarray, options: dict[str, Any]) -> ProbabilityPlot:
        return qq_plot(
            values,
            str(self.opt("distribution", options) or "norm"),
            standardized=bool(self.opt("standardized", options)),
            line=str(self.opt("reference_line", options) or "quartiles"),
            confidence=float(self.opt_typed("confidence", options) or 0.0),
        )

    def draw_band(self, ax: Any, plot: ProbabilityPlot, **style: Any) -> None:
        """The band is in the sample's units, at each theoretical quantile."""
        ax.fill_between(plot.x, plot.lower, plot.upper, **style)

    def axis_labels(self, plot: ProbabilityPlot, options: dict[str, Any]) -> tuple[str, str]:
        units = "standardized" if bool(self.opt("standardized", options)) else "fitted"
        return f"Theoretical quantiles ({plot.distribution}, {units})", "Sample quantiles"

    def _scatter_options(self, options: dict[str, Any]) -> dict[str, Any]:
        own = set(self.Options) - set(ScatterAxisRenderer.Options)
        cleaned = {key: value for key, value in options.items() if key not in own}
        for name in self.VALUE_SOURCES:
            nested = cleaned.get(name)
            if isinstance(nested, dict):
                cleaned[name] = {key: value for key, value in nested.items() if key not in own}
        return cleaned

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        plots: list[tuple[SeriesData, ProbabilityPlot]] = []
        for sd in self.valid_series(series):
            merged = self.merge_style(options, sd.style or {})
            values = to_numbers(sd.df["value"]).to_numpy(dtype=float)
            try:
                plots.append((sd, self.compute(values, merged)))
            except NUMERICAL_FAILURES as exc:
                applogger.info("%s: series '%s' not drawn - %s", self.Name, sd.name, exc)
        if not plots:
            return

        points = [
            SeriesData(name=sd.name, df=SeriesFrame({"x": plot.x, "y": plot.y}), style=sd.style)
            for sd, plot in plots
        ]
        self._scatter.render_axis(ax, points, self._scatter_options(options))

        band_alpha = float(self.opt_typed("band_alpha", options) or 0.0)
        for index, (sd, plot) in enumerate(plots):
            colour = self.series_color(sd.style or {}, index)
            if plot.lower.size and band_alpha > 0:
                self.draw_band(ax, plot, color=colour, alpha=band_alpha, linewidth=0, zorder=1, label="_nolegend_")
            if plot.slope is not None and plot.intercept is not None:
                ends = np.array([np.min(plot.x), np.max(plot.x)])
                ax.plot(ends, plot.slope * ends + plot.intercept, color=colour, linewidth=1.2,
                        linestyle="--", zorder=2, label="_nolegend_")
            applogger.info(
                "%s of '%s' against %s %s: Kolmogorov-Smirnov distance %.4g.",
                self.Name, sd.name, plot.distribution,
                tuple(round(value, 6) for value in plot.params), plot.ks_statistic,
            )

        x_label, y_label = self.axis_labels(plots[0][1], options)
        if not str(ax.get_xlabel() or "").strip():
            ax.set_xlabel(x_label)
        if not str(ax.get_ylabel() or "").strip():
            ax.set_ylabel(y_label)


class PPPlotAxisRenderer(QQPlotAxisRenderer, BaseAxisRenderer):
    """Sample probabilities against the fitted distribution's, on the unit square."""

    Name: str = "P-P Plot"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Sample probabilities against those of a fitted distribution: points on the diagonal "
        "mean the distribution fits. Sharpest in the middle."
    )
    Link: str = "https://en.wikipedia.org/wiki/P%E2%80%93P_plot"

    RequiredRoles: list[str] = ["value"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, Any] = dict(ScatterAxisRenderer.Kwargs)
    Options: dict[str, Any] = merge(ScatterAxisRenderer.Options, _SHARED_OPTIONS)

    def compute(self, values: np.ndarray, options: dict[str, Any]) -> ProbabilityPlot:
        return pp_plot(
            values,
            str(self.opt("distribution", options) or "norm"),
            confidence=float(self.opt_typed("confidence", options) or 0.0),
        )

    def draw_band(self, ax: Any, plot: ProbabilityPlot, **style: Any) -> None:
        """Here the band bounds F(x(i)) - the x - at each sample probability."""
        ax.fill_betweenx(plot.y, plot.lower, plot.upper, **style)

    def axis_labels(self, plot: ProbabilityPlot, options: dict[str, Any]) -> tuple[str, str]:
        return f"Theoretical probability ({plot.distribution})", "Sample probability"

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        super().render_axis(ax, series, options)
        if ax.collections or ax.lines:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
