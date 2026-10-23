"""Divergence and curl of a vector field on a grid (I-04).

A field sampled on a regular grid - x, y (and z) with the components u, v
(and w) at each node, one row per node - differentiated by central
differences (numpy.gradient, second order inside, first at the edges):

* 2D: divergence du/dx + dv/dy, and the curl's one component, dv/dx - du/dy
  (the vorticity: how fast the field turns about z);
* 3D: divergence du/dx + dv/dy + dw/dz, and the curl's three components
  (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy) and its magnitude.

The grid may be unevenly spaced along each axis, but it must be complete:
every combination of the x, y (and z) values present exactly once - the same
rule as the Heatmap's. No Qt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FieldDerivatives:
    """The field's nodes (x, y[, z]), its components (u, v[, w]), divergence and curl; ``dims`` 2 or 3."""

    frame: pd.DataFrame
    dims: int


def _axis_values(column: pd.Series, name: str) -> np.ndarray:
    values = np.sort(pd.to_numeric(column, errors="coerce").dropna().unique())
    if values.size < 2:
        raise ValueError(f"The field needs at least two distinct {name} values.")
    return values


def derivatives(frame: pd.DataFrame, *, x: str, y: str, u: str, v: str, z: str = "", w: str = "") -> FieldDerivatives:
    """Divergence and curl of the field (u, v[, w]) at the nodes (x, y[, z]) of *frame*."""
    three = bool(z and w)
    if bool(z) != bool(w):
        raise ValueError("A 3D field needs both z and w; a 2D one neither.")
    names = [x, y, z] if three else [x, y]
    comps = [u, v, w] if three else [u, v]
    data = frame[names + comps].apply(pd.to_numeric, errors="coerce").dropna()
    axes = [_axis_values(data[name], label) for name, label in zip(names, "xyz")]
    shape = tuple(a.size for a in axes)
    if len(data) != int(np.prod(shape)) or data.duplicated(subset=names).any():
        raise ValueError("The field is not on a complete grid: every combination of the coordinates is needed "
                         "exactly once.")
    # Index every node on the grid, then fill one array per component.
    index = tuple(np.searchsorted(a, data[name].to_numpy(float)) for a, name in zip(axes, names))
    grids = []
    for comp in comps:
        grid = np.empty(shape)
        grid[index] = data[comp].to_numpy(float)
        grids.append(grid)

    def d(field: np.ndarray, along: int) -> np.ndarray:
        return np.gradient(field, axes[along], axis=along, edge_order=1)

    out = {label: data[name].to_numpy(float) for label, name in zip("xyz", names)}
    for label, comp in zip("uvw", comps):
        out[label] = data[comp].to_numpy(float)
    if three:
        fu, fv, fw = grids
        divergence = d(fu, 0) + d(fv, 1) + d(fw, 2)
        curl = (d(fw, 1) - d(fv, 2), d(fu, 2) - d(fw, 0), d(fv, 0) - d(fu, 1))
        out["divergence"] = divergence[index]
        for label, part in zip(("curl_x", "curl_y", "curl_z"), curl):
            out[label] = part[index]
        out["curl_magnitude"] = np.sqrt(sum(part[index] ** 2 for part in curl))
    else:
        fu, fv = grids
        out["divergence"] = (d(fu, 0) + d(fv, 1))[index]
        out["curl"] = (d(fv, 0) - d(fu, 1))[index]
    result = pd.DataFrame(out)
    return FieldDerivatives(frame=result.sort_values(["x", "y"] + (["z"] if three else [])).reset_index(drop=True),
                            dims=3 if three else 2)
