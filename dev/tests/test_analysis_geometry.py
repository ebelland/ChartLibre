"""The geometry engine: matrices, motions, and the SQL that spells them."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app.analysis import geometry as g


def test_3d_rotation_matches_scipy_xyz_euler() -> None:
    ours = g.rotation_3d(30.0, -45.0, 60.0)
    theirs = Rotation.from_euler("xyz", [30.0, -45.0, 60.0], degrees=True).as_matrix()
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_2d_rotation_turns_counter_clockwise_about_the_centre() -> None:
    motion = g.Motion.of(g.rotation_2d(90.0), centre=(1.0, 1.0, 0.0))
    x, y = motion.apply(np.array([2.0]), np.array([1.0]))
    np.testing.assert_allclose([x[0], y[0]], [1.0, 2.0], atol=1e-12)


def test_roto_translation_rotates_then_moves() -> None:
    motion = g.Motion.of(g.rotation_3d(0, 0, 90), translation=(10.0, 0.0, 5.0))
    x, y, z = motion.apply(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose([x[0], y[0], z[0]], [10.0, 1.0, 5.0], atol=1e-12)


def test_composition_equals_applying_one_after_the_other() -> None:
    rng = np.random.default_rng(2)
    x, y, z = rng.normal(size=(3, 20))
    first = g.Motion.of(g.rotation_3d(10, 20, 30), centre=(1, 2, 3), translation=(0.5, 0, -1))
    second = g.Motion.of(g.scale(2, 1, 0.5), centre=(-1, 0, 4), translation=(3, 3, 3))
    step = second.apply(*first.apply(x, y, z))
    once = first.then(second).apply(x, y, z)
    for a, b in zip(step, once):
        np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("dims", [2, 3])
def test_the_sql_computes_what_numpy_computes(dims: int) -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(dims, 12)) * 10
    matrix = g.rotation_3d(15, -30, 70) if dims == 3 else g.rotation_2d(33) @ g.shear_xy(0.2, 0)
    motion = g.Motion.of(matrix, centre=(1.5, -2.0, 0.5 if dims == 3 else 0.0), translation=(4.0, -1.0, 2.0 if dims == 3 else 0.0))
    names = ('"x"', '"y"', '"z"')[:dims]
    expressions = motion.sql(names)
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE p ({', '.join(names)})")
    con.executemany(f"INSERT INTO p VALUES ({', '.join('?' * dims)})", points.T.tolist())
    rows = np.array(con.execute(f"SELECT {', '.join(expressions)} FROM p").fetchall()).T
    expected = motion.apply(*points)
    np.testing.assert_allclose(rows, np.vstack(expected), atol=1e-8)


def test_a_translation_reads_as_a_translation() -> None:
    x_sql, y_sql = g.Motion.of(translation=(5.0, -2.5, 0.0)).sql(('"x"', '"y"'))
    assert x_sql == '5 + "x"' and y_sql == '-2.5 + "y"'
