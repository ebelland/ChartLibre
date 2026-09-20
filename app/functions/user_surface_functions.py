"""Your own surface (z = f(x, y)) fit functions.

Same idea as ``user_functions.py``, one variable more: a class here appears
in the fit dialog's "Surfaces" category with no registration of any kind.
See ``user_functions.py`` for why ``expression`` and ``initial_guess`` are
worth writing.
"""
from __future__ import annotations

import numpy as np

from app.functions.base import base_surface_function


class sample_user_surface_function(base_surface_function):
    name = "Sample user plane"
    category = "User surfaces"
    description = "Example user-defined plane. Edit or replace this class."
    expression = "<b>Sample user plane</b><br>z = b0 + bx x + by y"
    p0 = [0.0, 1.0, 1.0]
    params = ["b0", "bx", "by"]

    @staticmethod
    def execute(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x + p[2] * y

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
        """Read the plane's coefficients straight off the data."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        if x.size < 3:
            return None
        design = np.column_stack([np.ones_like(x), x, y])
        try:
            coeffs, *_rest = np.linalg.lstsq(design, z, rcond=None)
        except Exception:
            return None
        if not np.all(np.isfinite(coeffs)):
            return None
        return [float(coeffs[0]), float(coeffs[1]), float(coeffs[2])]
