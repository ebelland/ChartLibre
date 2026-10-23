"""A broken axis (G-17): the values between two ends left out, the axis drawn on as one.

Matplotlib's own example (gallery "Broken axis") draws two axes and hides the
spines between them. Here it is one axes and a function scale: everything
past the break's upper end is moved down by the gap's width, so the gap takes
no room; the ticks that would fall in it are dropped, and the usual pair of
slanted marks is drawn across the spines where it was.

The option is "<axis>_break": "a, b" (a < b), on an axis whose scale is linear.

  https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from matplotlib.ticker import Locator, MaxNLocator

from app.logs.logger import applogger


def parse_break(text: Any) -> tuple[float, float] | None:
    """(a, b) from "a, b" or "a; b" (a comma may also be the decimal point with ";"), a < b; None otherwise."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)) and len(text) == 2:
        parts = [str(text[0]), str(text[1])]
    else:
        raw = str(text).strip()
        if not raw:
            return None
        parts = raw.split(";") if ";" in raw else re.split(r"\s*,\s*|\s+", raw)
    try:
        low, high = (float(part.strip().replace(",", ".")) for part in parts if part.strip())
    except ValueError:
        return None
    return (low, high) if low < high else None


class _SkipGap(Locator):
    """Ticks for each side of the gap on its own, so the short side gets ticks of its size.

    One locator over the whole range picked a step fit for the long side, and
    the short side was left with none.
    """

    #: Ticks for the whole axis, shared between the sides by their length on screen.
    TICKS = 7

    def __init__(self, low: float, high: float) -> None:
        self._low, self._high = low, high

    def __call__(self) -> Any:
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin: float, vmax: float) -> Any:
        vmin, vmax = min(vmin, vmax), max(vmin, vmax)
        below = max(0.0, min(vmax, self._low) - vmin)
        above = max(0.0, vmax - max(vmin, self._high))
        shown = below + above or 1.0
        ticks: list[float] = []
        for start, end, length, keep in ((vmin, min(vmax, self._low), below, lambda v: v <= self._low),
                                         (max(vmin, self._high), vmax, above, lambda v: v >= self._high)):
            if length <= 0:
                continue
            bins = max(1, round(self.TICKS * length / shown))
            ticks.extend(v for v in MaxNLocator(nbins=bins).tick_values(start, end) if keep(v) and start <= v <= end)
        return np.asarray(ticks, dtype=float)


def apply_breaks(ax: Any, options: dict[str, Any]) -> None:
    """Break the x and/or y axis of *ax* where its options ask, on a linear scale only."""
    for axis_name in ("x", "y"):
        gap = parse_break(options.get(f"{axis_name}_break"))
        if gap is None:
            continue
        if str(options.get(f"{axis_name}_scale") or "linear").lower() != "linear":
            applogger.warning("A broken %s axis needs a linear scale: the break is not drawn.", axis_name)
            continue
        low, high = gap
        width = high - low

        def forward(values: Any, low: float = low, high: float = high, width: float = width) -> Any:
            v = np.asarray(values, dtype=float)
            return np.where(v <= low, v, np.where(v >= high, v - width, low))

        def inverse(values: Any, low: float = low, width: float = width) -> Any:
            u = np.asarray(values, dtype=float)
            return np.where(u <= low, u, u + width)

        axis = ax.xaxis if axis_name == "x" else ax.yaxis
        (ax.set_xscale if axis_name == "x" else ax.set_yscale)("function", functions=(forward, inverse))
        axis.set_major_locator(_SkipGap(low, high))
        _draw_marks(ax, axis_name, low)


def _draw_marks(ax: Any, axis_name: str, at: float) -> None:
    """The slanted pair across both spines at the break, as in Matplotlib's example."""
    slant = 0.5
    kwargs = dict(marker=[(-1, -slant), (1, slant)], markersize=12, linestyle="none",
                  color=ax.spines["left"].get_edgecolor(), mec=ax.spines["left"].get_edgecolor(),
                  mew=1, clip_on=False, zorder=5)
    if axis_name == "y":
        ax.plot([0, 1], [at, at], transform=ax.get_yaxis_transform(), **kwargs)
    else:
        ax.plot([at, at], [0, 1], transform=ax.get_xaxis_transform(), **kwargs)
