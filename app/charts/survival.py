"""Kaplan-Meier survival curves: how long until the event, censoring included.

Survival data - time to failure, to relapse, to a customer leaving - has
rows whose event has not happened yet when the study ends. Dropping them
biases every average down, and treating their time as an event time biases
it again; the Kaplan-Meier estimate uses each one for exactly as long as it
was watched. The curve steps down at every observed event, a tick marks
every censored row, and the shaded band is the pointwise confidence
interval. With two or more curves the log-rank test says whether they
differ by more than chance.

The arithmetic is in app/analysis/diagnostics.py (todo R-06).

Roles:
    time     required, time to the event or to the end of observation (>= 0)
    event    required, 1 where the event was observed, 0 where censored
    group    optional, one curve per value, within each series
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from app.analysis.diagnostics import SurvivalCurve, kaplan_meier, log_rank
from app.charts.base import LEGEND_OPTIONS, BaseAxisRenderer, SeriesData, merge
from app.logs.logger import applogger
from app.utils.coercion import to_numbers


class KaplanMeierAxisRenderer(BaseAxisRenderer):
    """Step survival curves with censoring ticks, confidence bands and the log-rank test."""

    Name: str = "Kaplan-Meier"
    Category: str = "Diagnostic plots"
    Description: str = (
        "Survival curves from time-to-event data with censored rows, their confidence bands, "
        "and the log-rank test between groups."
    )
    Link: str = "https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator"

    RequiredRoles: list[str] = ["time", "event"]
    OptionalRoles: list[str] = ["group"]

    Kwargs: dict[str, Any] = {
        "linewidth": {
            "default": 1.6,
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "group": "Lines",
            "description": "Width of the survival curves.",
        },
    }

    Options: dict[str, Any] = merge(
        LEGEND_OPTIONS,
        {
            "confidence": {
                "default": 0.95,
                "type": float,
                "min": 0.0,
                "max": 0.999,
                "group": "Survival",
                "description": (
                    "Shade the pointwise confidence interval (Greenwood, log-log scale). "
                    "0 draws no band."
                ),
            },
            "band_alpha": {
                "default": 0.15,
                "type": float,
                "min": 0.0,
                "max": 1.0,
                "group": "Survival",
                "description": "Opacity of the confidence band.",
            },
            "censor_marks": {
                "default": True,
                "type": bool,
                "group": "Survival",
                "description": "Mark every censored row with a tick on its curve.",
            },
            "median_line": {
                "default": False,
                "type": bool,
                "group": "Survival",
                "description": "Draw dotted guides to where each curve crosses 50%: the median survival time.",
            },
            "log_rank": {
                "default": True,
                "type": bool,
                "group": "Survival",
                "description": "With two or more curves, write the log-rank test's p-value on the chart.",
            },
            "as_percent": {
                "default": False,
                "type": bool,
                "group": "Survival",
                "description": "Label the survival axis 0-100 instead of 0-1.",
            },
        },
    )

    def _curves(self, series: list[SeriesData]) -> list[tuple[str, dict[str, Any], np.ndarray, np.ndarray]]:
        """(label, style, time, event) per curve: one per series, or per group within it."""
        curves = []
        for sd in self.valid_series(series):
            time = to_numbers(sd.df["time"]).to_numpy(dtype=float)
            event = to_numbers(sd.df["event"]).to_numpy(dtype=float)
            if "group" not in sd.df.columns:
                curves.append((sd.name, dict(sd.style or {}), time, event))
                continue
            groups = np.asarray(sd.df["group"], dtype=object)
            for value in dict.fromkeys(groups.tolist()):
                if value is None or value != value:
                    continue
                mask = np.array([item == value for item in groups.tolist()], dtype=bool)
                label = f"{sd.name}: {value}" if len(series) > 1 else str(value)
                style = {key: item for key, item in (sd.style or {}).items() if key != "color"}
                curves.append((label, style, time[mask], event[mask]))
        return curves

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        options = options or {}
        kwargs = self.get_kwargs(options)
        confidence = float(self.opt_typed("confidence", options) or 0.0)
        band_alpha = float(self.opt_typed("band_alpha", options) or 0.0)
        scale = 100.0 if bool(self.opt("as_percent", options)) else 1.0

        drawn: list[tuple[str, np.ndarray, np.ndarray]] = []
        for index, (label, style, time, event) in enumerate(self._curves(series)):
            try:
                curve = kaplan_meier(time, event, confidence=confidence or 0.95)
            except NUMERICAL_FAILURES as exc:
                applogger.info("Kaplan-Meier: '%s' not drawn - %s", label, exc)
                continue
            colour = self.series_color(style, index)
            end = float(np.nanmax(time[np.isfinite(time)]))
            steps_x = np.append(curve.time, max(end, float(curve.time[-1])))
            steps_y = np.append(curve.survival, curve.survival[-1]) * scale
            show = bool(style.get("show_in_legend", True))
            ax.step(steps_x, steps_y, where="post", color=colour,
                    linestyle=str(style.get("linestyle") or "-") or "-",
                    linewidth=float(style.get("linewidth") or kwargs.get("linewidth", 1.6)),
                    label=label if show else "_nolegend_", zorder=3)
            if confidence > 0 and band_alpha > 0:
                ax.fill_between(steps_x, np.append(curve.lower, curve.lower[-1]) * scale,
                                np.append(curve.upper, curve.upper[-1]) * scale,
                                step="post", color=colour, alpha=band_alpha, linewidth=0, zorder=1)
            if bool(self.opt("censor_marks", options)) and curve.censored_time.size:
                ax.plot(curve.censored_time, curve.censored_survival * scale, linestyle="",
                        marker="|", markersize=8, markeredgewidth=1.2, color=colour,
                        label="_nolegend_", zorder=4)
            if bool(self.opt("median_line", options)) and np.isfinite(curve.median):
                ax.plot([0, curve.median, curve.median], [0.5 * scale, 0.5 * scale, 0],
                        linestyle=":", linewidth=1.0, color=colour, label="_nolegend_", zorder=2)
            self._log_curve(label, curve)
            drawn.append((label, time, event))

        if not drawn:
            return
        ax.set_ylim(0, 1.05 * scale)
        ax.set_xlim(left=0)
        if len(drawn) > 1 and bool(self.opt("log_rank", options)):
            self._write_log_rank(ax, drawn)
        if not str(ax.get_xlabel() or "").strip():
            ax.set_xlabel("Time")
        if not str(ax.get_ylabel() or "").strip():
            ax.set_ylabel("Survival probability" + (" [%]" if scale == 100.0 else ""))
        if bool(self.opt("show_legend", options)) and ax.get_legend_handles_labels()[0]:
            ax.legend()
        self.apply_annotations(ax, options)

    @staticmethod
    def _log_curve(label: str, curve: SurvivalCurve) -> None:
        median = f"{curve.median:g}" if np.isfinite(curve.median) else "not reached"
        applogger.info(
            "Kaplan-Meier '%s': %d rows, %d events, %d censored, median survival %s.",
            label, curve.n, int(curve.events.sum()), curve.censored_time.size, median,
        )

    @staticmethod
    def _write_log_rank(ax: Any, drawn: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
        try:
            test = log_rank([(time, event) for _label, time, event in drawn])
        except NUMERICAL_FAILURES as exc:
            applogger.info("Log-rank test not computed - %s", exc)
            return
        p_text = "p < 0.0001" if test.pvalue < 1e-4 else f"p = {test.pvalue:.4f}"
        ax.text(0.02, 0.03, f"Log-rank: χ² = {test.statistic:.3g}, df = {test.dof}, {p_text}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize="small")
        applogger.info("Log-rank test: chi-squared %.4g on %d df, p = %.4g.", test.statistic, test.dof, test.pvalue)
