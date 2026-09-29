"""Peak finding: the arithmetic behind the Peaks series operation.

A curve's peaks come from ``scipy.signal.find_peaks`` filtered by prominence
(default) or height, and are measured by their width at half prominence and
the x bounds of that width - what the Calculus operation's integral wants. A
surface's local maxima come from a neighbourhood filter instead. Plain
arrays in, no Qt, nothing logged (todo R-01).

**Prominence is the default filter, not height.** Height compares a peak to
zero, which is only meaningful when the baseline is at zero - on a sloping
or raised background it selects whichever part of the signal happens to sit
highest, not the peaks. Prominence measures how far a peak stands above the
higher of the two saddles flanking it, so it asks "how much of a peak is
this, locally", which is the question that survives a moving baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths

MAXIMA = "maxima"
MINIMA = "minima"
BOTH = "both"


@dataclass(slots=True)
class Peak:
    """One located peak, with the measurements that describe it.

    ``z`` is None for an ordinary 1D peak on a curve. A peak found on a
    surface (``z`` role present, see ``_find_one_3d``) sets it, and leaves
    ``width``/``left_x``/``right_x`` as NaN - those describe a 1D half-
    prominence interval, which a 2D local maximum simply does not have.
    """

    x: float
    y: float
    prominence: float
    width: float
    left_x: float
    right_x: float
    is_minimum: bool = False
    z: float | None = None


@dataclass(frozen=True, slots=True)
class PeakSettings:
    """The filters. The threshold is a fraction of the signal's range."""

    #: "prominence" (default) or "height".
    filter_by: str = "prominence"
    #: Thresholds are fractions of the signal's range so one setting means the
    #: same on a millivolt trace and on a count rate.
    threshold: float = 0.05
    #: Minimum separation in samples (1D) or neighbourhood radius (2D).
    distance: int = 1
    #: Minimum width in samples (1D only); 0 keeps every width.
    min_width: float = 0.0
    #: Keep at most this many, the most prominent.
    limit: int = 50


def find_peaks_1d(x: np.ndarray, y: np.ndarray, mode: str, settings: PeakSettings | None = None) -> list[Peak]:
    """The maxima and/or minima of ``(x, y)``, in x order."""
    settings = settings or PeakSettings()
    peaks: list[Peak] = []
    if mode in (MAXIMA, BOTH):
        peaks.extend(_search_1d(x, y, settings, minimum=False))
    if mode in (MINIMA, BOTH):
        # A minimum is a maximum of the inverted signal. Inverting rather
        # than writing a second search keeps one implementation of the
        # prominence and width logic, which is where the subtlety is.
        peaks.extend(_search_1d(x, y, settings, minimum=True))
    peaks.sort(key=lambda peak: peak.x)
    return peaks


def find_peaks_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    mode: str,
    settings: PeakSettings | None = None,
) -> list[Peak]:
    """The local maxima and/or minima of a gridded surface, in (x, y) order."""
    settings = settings or PeakSettings()
    peaks: list[Peak] = []
    if mode in (MAXIMA, BOTH):
        peaks.extend(_search_2d(x_grid, y_grid, z_grid, settings, minimum=False))
    if mode in (MINIMA, BOTH):
        peaks.extend(_search_2d(x_grid, y_grid, z_grid, settings, minimum=True))
    peaks.sort(key=lambda peak: (peak.x, peak.y))
    return peaks


def _search_1d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    settings: PeakSettings,
    *,
    minimum: bool,
) -> list[Peak]:
    """Run find_peaks once, on the signal or on its inverse."""
    signal = -y_values if minimum else y_values

    # Thresholds are given as a fraction of the signal's range so that one
    # setting means the same thing on a millivolt trace and on a count
    # rate. An absolute default would be meaningless on arrival.
    span = float(np.ptp(y_values))
    if span <= 0.0:
        return []

    threshold = float(settings.threshold) * span
    distance = max(1, int(settings.distance))
    min_width = float(settings.min_width)

    kwargs: dict[str, Any] = {"distance": distance}
    if settings.filter_by == "height":
        kwargs["height"] = float(np.min(signal)) + threshold
    else:
        kwargs["prominence"] = max(threshold, 1e-12)
    if min_width > 0.0:
        kwargs["width"] = min_width

    indices, _properties = find_peaks(signal, **kwargs)
    if indices.size == 0:
        return []

    prominences = peak_prominences(signal, indices)[0]
    # rel_height=0.5 is the width at half prominence, not at half height:
    # on a raised baseline those differ, and the half-prominence width is
    # the one that describes the peak rather than the background.
    _widths, _heights, left_ips, right_ips = peak_widths(
        signal, indices, rel_height=0.5
    )

    # peak_widths returns fractional sample positions, so the x bounds have
    # to be interpolated back onto the real axis rather than indexed.
    sample_positions = np.arange(x_values.size, dtype=float)
    left_x = np.interp(left_ips, sample_positions, x_values)
    right_x = np.interp(right_ips, sample_positions, x_values)
    width_in_x = right_x - left_x

    found = [
        Peak(
            x=float(x_values[index]),
            y=float(y_values[index]),
            prominence=float(prominence),
            width=float(width),
            left_x=float(left),
            right_x=float(right),
            is_minimum=minimum,
        )
        for index, prominence, width, left, right in zip(
            indices, prominences, width_in_x, left_x, right_x
        )
    ]

    limit = int(settings.limit)
    if len(found) > limit:
        # Keep the most prominent, not the first: truncating in x order
        # would discard the strongest peaks whenever they are late in the
        # series.
        found.sort(key=lambda peak: peak.prominence, reverse=True)
        found = found[:limit]

    return found


def _search_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    settings: PeakSettings,
    *,
    minimum: bool,
) -> list[Peak]:
    """2D analogue of ``_search``: a local-maximum filter, not find_peaks.

    ``scipy.signal.find_peaks`` is a 1D algorithm; a surface's local
    maxima are found with ``scipy.ndimage.maximum_filter`` instead - a
    cell is a candidate when it equals the maximum of its own
    neighbourhood, and its "prominence" is how far it stands above the
    *minimum* of that same neighbourhood (the 2D reading of the 1D
    prominence idea: height above the nearest lower ground).

    NaN cells - present only in an interpolated grid, outside the convex
    hull of the original points (see ``series_grid_xyz``) - are pushed to
    -inf before filtering so they can never win a maximum comparison, and
    any candidate whose own cell was NaN, or whose neighbourhood touches
    a NaN (making its "minimum" -inf and its prominence infinite), is
    discarded rather than reported as a peak.
    """
    signal = -z_grid if minimum else z_grid
    finite_mask = np.isfinite(signal)
    if not np.any(finite_mask):
        return []

    finite_values = signal[finite_mask]
    span = float(np.ptp(finite_values))
    if span <= 0.0:
        return []

    threshold = float(settings.threshold) * span
    distance = max(1, int(settings.distance))
    neighborhood = 2 * distance + 1

    masked = np.where(finite_mask, signal, -np.inf)
    local_max = maximum_filter(masked, size=neighborhood, mode="nearest")
    local_min = minimum_filter(masked, size=neighborhood, mode="nearest")
    with np.errstate(invalid="ignore"):
        prominence = masked - local_min

    is_candidate = finite_mask & (masked == local_max) & np.isfinite(prominence)

    if settings.filter_by == "height":
        base = float(np.min(finite_values))
        keep = is_candidate & (masked >= base + threshold)
    else:
        keep = is_candidate & (prominence >= max(threshold, 1e-12))

    rows_idx, cols_idx = np.nonzero(keep)
    found = [
        Peak(
            x=float(x_grid[r, c]),
            y=float(y_grid[r, c]),
            z=float(z_grid[r, c]),
            prominence=float(prominence[r, c]),
            width=float("nan"),
            left_x=float("nan"),
            right_x=float("nan"),
            is_minimum=minimum,
        )
        for r, c in zip(rows_idx.tolist(), cols_idx.tolist())
    ]

    limit = int(settings.limit)
    if len(found) > limit:
        found.sort(key=lambda peak: peak.prominence, reverse=True)
        found = found[:limit]

    return found
