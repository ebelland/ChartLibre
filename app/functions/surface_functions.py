"""Built-in surface (z = f(x, y)) fit functions.

Same idea as ``functions.py``, one variable more: every class here inherits
directly from ``base_surface_function`` and is discovered by
``SurfaceFunctionScanner`` the same way ``FunctionScanner`` discovers the 1D
library - by AST, not by registration.
"""
from __future__ import annotations

import numpy as np

from app.functions.base import base_surface_function


class plane(base_surface_function):
    name = "Plane"
    category = "Surfaces"
    description = "Flat inclined plane through the data."
    expression = "<b>Plane</b><br>z = a + b x + c y"
    p0 = [0.0, 1.0, 1.0]
    params = ["a", "b", "c"]

    @staticmethod
    def execute(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x + p[2] * y

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
        """Solve the linear least squares directly - a plane always can.

        Same spirit as the 1D library's ``_polynomial_guess`` (``np.polyfit``
        under the hood): a plane is linear in its parameters, so there is no
        need to guess when the exact least-squares answer is one
        ``np.linalg.lstsq`` call away.
        """
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


class paraboloid(base_surface_function):
    name = "Paraboloid"
    category = "Surfaces"
    description = "Elliptic bowl centred at (x0, y0)."
    expression = "<b>Paraboloid</b><br>z = a + b (x-x0)&sup2; + c (y-y0)&sup2;"
    p0 = [0.0, 1.0, 0.0, 1.0, 0.0]
    params = ["a", "b", "x0", "c", "y0"]

    @staticmethod
    def execute(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * (x - p[2]) ** 2 + p[3] * (y - p[4]) ** 2

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
        del x, y, z
        # Non-linear in x0/y0; left to the Monte Carlo search, same as most
        # of the 1D library's own shaped functions (Gaussian, logistic, ...).
        return None


class gaussian2d(base_surface_function):
    name = "Gaussian2D"
    category = "Surfaces"
    description = "Bump or dip on a flat baseline."
    expression = (
        "<b>Gaussian2D</b><br>"
        "z = a + amp exp(-((x-x0)&sup2;/(2 sx&sup2;) + (y-y0)&sup2;/(2 sy&sup2;)))"
    )
    p0 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    params = ["a", "amp", "x0", "sx", "y0", "sy"]

    @staticmethod
    def execute(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        a, amp, x0, sx, y0, sy = p[0], p[1], p[2], p[3], p[4], p[5]
        sx = sx if abs(sx) > 1e-12 else 1e-12
        sy = sy if abs(sy) > 1e-12 else 1e-12
        return a + amp * np.exp(
            -(((x - x0) ** 2) / (2.0 * sx * sx) + ((y - y0) ** 2) / (2.0 * sy * sy))
        )

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
        del x, y, z
        return None


class saddle(base_surface_function):
    name = "Saddle"
    category = "Surfaces"
    description = "Hyperbolic paraboloid: rises along x, falls along y."
    expression = "<b>Saddle</b><br>z = a + b x&sup2; - c y&sup2;"
    p0 = [0.0, 1.0, 1.0]
    params = ["a", "b", "c"]

    @staticmethod
    def execute(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x**2 - p[2] * y**2

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
        del x, y, z
        return None


class ripple2d(base_surface_function):
    name = "Ripple2D"
    category = "Surfaces"
    description = "Separable oscillation in x and y."
    expression = "<b>Ripple2D</b><br>z = a + amp sin(kx x) cos(ky y)"
    p0 = [0.0, 1.0, 1.0, 1.0]
    params = ["a", "amp", "kx", "ky"]

    @staticmethod
    def execute(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * np.sin(p[2] * x) * np.cos(p[3] * y)

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float] | None:
        del x, y, z
        # Frequency-in-two-directions is exactly the kind of many-valleyed
        # search the Monte Carlo starting-point search exists for.
        return None
