"""Rigid and affine motions of 2D and 3D points, and how to spell them in SQL.

The engine behind the Geometry series operation. Every transform here is

    p' = centre + M (p - centre) + t

with M a 3x3 matrix and t a translation; a 2D series is the same with
z left alone. Composing two transforms is multiplying their matrices, and
the SQL is always three expressions - one per coordinate - built from the
nine numbers of M and the three of t, so the moved series can be read
through a query instead of stored.

Rotations use the right-hand rule: a positive angle about an axis turns
counter-clockwise looking down that axis towards the origin. The 3D
rotation applies the x angle first, then y, then z (extrinsic axes), the
convention of scipy's Rotation.from_euler("xyz").
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Below this, a coefficient is floating-point noise rather than a number
#: anyone chose - the midpoint of data symmetric about zero is 1.1e-16.
NOISE = 1e-12


def number(value: float) -> str:
    """A coefficient spelled for SQL, for a person to read."""
    if abs(value) < NOISE:
        return "0"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12g}"


def rotation_2d(angle_degrees: float) -> np.ndarray:
    """Counter-clockwise rotation in the x-y plane, as a 3x3 matrix (z kept)."""
    a = math.radians(angle_degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_3d(about_x: float, about_y: float, about_z: float) -> np.ndarray:
    """Rotation by three angles in degrees: about x, then y, then z."""
    ax, ay, az = (math.radians(v) for v in (about_x, about_y, about_z))
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def scale(sx: float, sy: float, sz: float = 1.0) -> np.ndarray:
    return np.diag([sx, sy, sz]).astype(float)


def shear_xy(kx: float, ky: float) -> np.ndarray:
    """x gains kx per unit of y, y gains ky per unit of x; z kept."""
    return np.array([[1.0, kx, 0.0], [ky, 1.0, 0.0], [0.0, 0.0, 1.0]])


def mirror_xy(line_degrees: float) -> np.ndarray:
    """Reflection in the x-y plane across a line at *line_degrees*; z kept."""
    double = math.radians(2.0 * line_degrees)
    c, s = math.cos(double), math.sin(double)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True, slots=True)
class Motion:
    """``p' = centre + matrix (p - centre) + translation``."""

    matrix: np.ndarray
    centre: np.ndarray
    translation: np.ndarray

    @classmethod
    def of(
        cls,
        matrix: np.ndarray | None = None,
        *,
        centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Motion":
        return cls(
            np.eye(3) if matrix is None else np.asarray(matrix, dtype=float),
            np.asarray(centre, dtype=float),
            np.asarray(translation, dtype=float),
        )

    def apply(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        """Move the points; returns (x', y') for 2D input, (x', y', z') for 3D."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        zz = np.zeros_like(x) if z is None else np.asarray(z, dtype=float)
        points = np.vstack([x, y, zz]) - self.centre[:, None]
        moved = self.matrix @ points + (self.centre + self.translation)[:, None]
        return (moved[0], moved[1]) if z is None else (moved[0], moved[1], moved[2])

    def then(self, other: "Motion") -> "Motion":
        """This motion followed by *other*, as one motion about this centre."""
        matrix = other.matrix @ self.matrix
        # Where this motion sends its own centre, pushed through the other.
        offset = other.matrix @ (self.translation + self.centre - other.centre) + other.centre + other.translation
        return Motion(matrix, self.centre, offset - self.centre)

    def sql(
        self,
        columns: tuple[str, str] | tuple[str, str, str],
        spell: Callable[[int, int], str | None] | None = None,
    ) -> tuple[str, ...]:
        """One SQL expression per coordinate, over quoted *columns*.

        *spell(row, col)* may return a readable spelling of one matrix entry
        - ``cos(radians(30))`` - instead of its decimal; None keeps the number.
        Terms whose coefficient is zero are left out, so a translation reads
        as ``"x" + 5`` rather than as nine multiplications.
        """
        dims = len(columns)
        expressions: list[str] = []
        for row in range(dims):
            constant = float(self.centre[row] + self.translation[row])
            terms: list[str] = []
            for col in range(dims):
                coefficient = float(self.matrix[row, col])
                if abs(coefficient) < NOISE:
                    continue
                shifted = (
                    columns[col]
                    if abs(self.centre[col]) < NOISE
                    else f"({columns[col]} - {number(float(self.centre[col]))})"
                )
                spelled = spell(row, col) if spell is not None else None
                if spelled is None and abs(coefficient - 1.0) < NOISE:
                    terms.append(shifted)
                elif spelled is None and abs(coefficient + 1.0) < NOISE:
                    terms.append(f"-{shifted}")
                else:
                    terms.append(f"{shifted} * {spelled or number(coefficient)}")
            if abs(constant) >= NOISE or not terms:
                terms.insert(0, number(constant))
            expression = " + ".join(terms).replace("+ -", "- ")
            expressions.append(expression)
        return tuple(expressions)
