"""A wafer's map: the measurements or the defects on a disc, its notch or flat, its dies.

Coordinates are millimetres from the wafer's centre, y up, with the orientation mark (notch or flat) at
*mark_angle* degrees (270: at the bottom, the usual way a wafer map is drawn). Rotating the data turns it
about the centre - a map taken with the notch elsewhere brought to the usual place.

Measurements (a film thickness, a resistance at each site) are interpolated over the disc for the contour
map, averaged per die and binned against limits - the share of good dies is the yield. Defects (particle
positions, no value) are counted per die: the defect density and the Poisson yield exp(-D A) the density
predicts, beside the share of dies with no defect at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: The usual wafer diameters, mm.
DIAMETERS: tuple[int, ...] = (100, 125, 150, 200, 300, 450)
NOTCH = "notch"
FLAT = "flat"
NO_MARK = "none"
#: The primary flat's length by diameter, mm (SEMI M1); a 200 mm wafer has a notch or this flat.
FLAT_LENGTH: dict[int, float] = {100: 32.5, 125: 42.5, 150: 57.5, 200: 57.5, 300: 57.5, 450: 57.5}
#: The notch's depth and its opening, mm (SEMI M1: 1 mm deep, a 90° V).
NOTCH_DEPTH = 1.0
#: Die bins: no data, good, below the lower limit, above the upper one.
BIN_EMPTY, BIN_PASS, BIN_LOW, BIN_HIGH = 0, 1, 2, 3
BIN_NAMES: dict[int, str] = {BIN_EMPTY: "No data", BIN_PASS: "Pass", BIN_LOW: "Low", BIN_HIGH: "High"}


@dataclass(slots=True)
class Wafer:
    diameter: float = 300.0
    mark: str = NOTCH
    #: Where the mark is, degrees counterclockwise from +x: 270 at the bottom.
    mark_angle: float = 270.0
    #: The ring at the edge left out, mm.
    edge_exclusion: float = 3.0

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def usable_radius(self) -> float:
        return max(0.0, self.radius - self.edge_exclusion)

    def flat_distance(self) -> float:
        """How far from the centre the flat cuts the disc."""
        half = FLAT_LENGTH.get(int(round(self.diameter)), 0.19 * self.diameter) / 2.0
        return math.sqrt(max(self.radius ** 2 - half ** 2, 0.0))

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The points on the usable wafer: within the edge exclusion, and on this side of the flat."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        keep = np.hypot(x, y) <= self.usable_radius
        if self.mark == FLAT:
            angle = math.radians(self.mark_angle)
            along = x * math.cos(angle) + y * math.sin(angle)
            keep &= along <= self.flat_distance() - self.edge_exclusion
        return keep

    def outline(self, points: int = 720) -> tuple[np.ndarray, np.ndarray]:
        """The wafer's edge as a closed line, its notch or flat in place."""
        angle = math.radians(self.mark_angle)
        theta = np.linspace(0.0, 2 * math.pi, points, endpoint=False) + angle
        radius = np.full(points, self.radius)
        offset = np.angle(np.exp(1j * (theta - angle)))  # each point's angle from the mark, -pi..pi
        if self.mark == FLAT:
            distance = self.flat_distance()
            # Past the flat's ends the edge is the chord: r = d / cos(offset).
            half = math.acos(distance / self.radius)
            on_flat = np.abs(offset) <= half
            radius[on_flat] = distance / np.cos(offset[on_flat])
        elif self.mark == NOTCH:
            width = NOTCH_DEPTH / self.radius  # a 90° V: as wide either side as it is deep
            in_notch = np.abs(offset) <= width
            radius[in_notch] = self.radius - NOTCH_DEPTH * (1 - np.abs(offset[in_notch]) / width)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        return np.append(x, x[0]), np.append(y, y[0])


def rotate(x: np.ndarray, y: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """The points turned *degrees* counterclockwise about the centre."""
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    x, y = np.asarray(x, float), np.asarray(y, float)
    return c * x - s * y, s * x + c * y


def die_grid(wafer: Wafer, width: float, height: float, offset_x: float = 0.0, offset_y: float = 0.0) -> pd.DataFrame:
    """The whole dies on the usable wafer: Column, Row (0 at the centre), the die's centre X, Y.

    The grid is centred on the wafer, shifted by *offset_x*, *offset_y* (a die corner on the centre is an
    offset of half a die); a die counts when its four corners are on the usable wafer.
    """
    if width <= 0 or height <= 0:
        raise ValueError("A die needs a width and a height above 0 mm.")
    reach = int(math.ceil(wafer.radius / min(width, height))) + 1
    columns, rows = np.meshgrid(np.arange(-reach, reach + 1), np.arange(-reach, reach + 1))
    cx = columns.ravel() * width + offset_x
    cy = rows.ravel() * height + offset_y
    whole = np.ones(cx.size, bool)
    for dx in (-width / 2, width / 2):
        for dy in (-height / 2, height / 2):
            whole &= wafer.inside(cx + dx, cy + dy)
    return pd.DataFrame({"Column": columns.ravel()[whole], "Row": rows.ravel()[whole],
                         "X": cx[whole], "Y": cy[whole]}).reset_index(drop=True)


def assign_dies(x: np.ndarray, y: np.ndarray, dies: pd.DataFrame, width: float, height: float, offset_x: float = 0.0,
                offset_y: float = 0.0) -> np.ndarray:
    """Each point's die (its row in *dies*), -1 for a point on no whole die."""
    column = np.round((np.asarray(x, float) - offset_x) / width).astype(int)
    row = np.round((np.asarray(y, float) - offset_y) / height).astype(int)
    index = {(int(c), int(r)): i for i, (c, r) in enumerate(zip(dies["Column"], dies["Row"], strict=True))}
    return np.array([index.get((int(c), int(r)), -1) for c, r in zip(column, row, strict=True)], dtype=int)


def contour_grid(x: np.ndarray, y: np.ndarray, value: np.ndarray, wafer: Wafer, points: int = 81) -> pd.DataFrame:
    """The values interpolated on a square grid over the wafer (cubic, linear where cubic cannot reach);
    NaN off the usable wafer. Columns x, y, value - x fastest."""
    from scipy.interpolate import griddata

    axis = np.linspace(-wafer.radius, wafer.radius, int(points))
    gx, gy = np.meshgrid(axis, axis)
    known = np.column_stack([x, y])
    method = "cubic" if len(x) >= 16 else "linear"
    grid = griddata(known, value, (gx, gy), method=method)
    near = griddata(known, value, (gx, gy), method="nearest")
    # Between the outermost sites and the edge the hull ends: the nearest site's value fills the ring.
    grid = np.where(np.isfinite(grid), grid, near)
    grid[~wafer.inside(gx, gy)] = np.nan
    return pd.DataFrame({"x": gx.ravel(), "y": gy.ravel(), "value": grid.ravel()})


def uniformity(value: np.ndarray) -> dict[str, float]:
    """Mean, standard deviation, range and the two uniformity figures fabs quote, in %."""
    value = np.asarray(value, float)
    value = value[np.isfinite(value)]
    if not value.size:
        return {}
    mean = float(value.mean())
    sd = float(value.std(ddof=1)) if value.size > 1 else 0.0
    spread = float(value.max() - value.min())
    return {"Sites": int(value.size), "Mean": mean, "Std Dev": sd, "Min": float(value.min()),
            "Max": float(value.max()), "Range": spread,
            "Non-uniformity, 1σ / mean (%)": 100 * sd / abs(mean) if mean else math.nan,
            "Non-uniformity, range / 2 mean (%)": 100 * spread / (2 * abs(mean)) if mean else math.nan}


def radial_zones(x: np.ndarray, y: np.ndarray, value: np.ndarray | None, wafer: Wafer, zones: int = 3) -> pd.DataFrame:
    """The wafer in rings of equal width, centre to edge: each ring's sites and their mean (or count)."""
    r = np.hypot(x, y)
    edges = np.linspace(0.0, wafer.usable_radius, zones + 1)
    rows = []
    for index in range(zones):
        ring = (r >= edges[index]) & (r <= edges[index + 1] if index == zones - 1 else r < edges[index + 1])
        row = {"Zone": f"{edges[index]:.0f}-{edges[index + 1]:.0f} mm", "Points": int(ring.sum())}
        if value is not None:
            row["Mean"] = float(np.mean(value[ring])) if ring.any() else math.nan
        else:
            area = math.pi * (edges[index + 1] ** 2 - edges[index] ** 2) / 100.0  # cm²
            row["Defects per cm²"] = ring.sum() / area if area else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(slots=True)
class DieSummary:
    dies: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)


def die_summary(dies: pd.DataFrame, die: np.ndarray, value: np.ndarray | None, width: float, height: float, *,
                lower: float | None = None, upper: float | None = None) -> DieSummary:
    """Per die: its points and - measurements - their mean and its bin; defects - the density and yields."""
    table = dies.copy()
    on = die >= 0
    counts = np.bincount(die[on], minlength=len(table))
    table["Count"] = counts
    summary: dict[str, float] = {"Dies": len(table)}
    area_cm2 = width * height / 100.0
    if value is not None:
        sums = np.bincount(die[on], weights=np.asarray(value, float)[on], minlength=len(table))
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        table["Mean"] = mean
        bins = np.full(len(table), BIN_PASS)
        if lower is not None:
            bins[mean < lower] = BIN_LOW
        if upper is not None:
            bins[mean > upper] = BIN_HIGH
        bins[counts == 0] = BIN_EMPTY
        table["Bin"] = bins
        table["Bin name"] = [BIN_NAMES[b] for b in bins]
        measured = int((counts > 0).sum())
        summary["Dies measured"] = measured
        if lower is not None or upper is not None:
            summary["Yield (%)"] = 100 * float((bins == BIN_PASS).sum()) / measured if measured else math.nan
    else:
        table["Defects per cm²"] = counts / area_cm2
        density = float(counts.sum()) / (len(table) * area_cm2) if len(table) else math.nan
        summary.update({"Defects on dies": int(counts.sum()), "Defect density (per cm²)": density,
                        "Die area (cm²)": area_cm2,
                        "Dies with no defect (%)": 100 * float((counts == 0).sum()) / len(table) if len(table)
                        else math.nan,
                        "Poisson yield exp(-D A) (%)": 100 * math.exp(-density * area_cm2)})
    return DieSummary(table, summary)
