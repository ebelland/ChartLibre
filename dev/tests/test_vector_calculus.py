"""Divergence and curl of a gridded field (I-04), against fields whose answers are known in closed form."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.vector_calculus import derivatives


def _grid2(u, v) -> pd.DataFrame:
    xs, ys = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 17), indexing="ij")
    return pd.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "u": u(xs, ys).ravel(), "v": v(xs, ys).ravel()})


def test_a_rotation_has_no_divergence_and_curl_two() -> None:
    field = derivatives(_grid2(lambda x, y: -y, lambda x, y: x), x="x", y="y", u="u", v="v").frame
    assert np.allclose(field["divergence"], 0.0, atol=1e-9) and np.allclose(field["curl"], 2.0)


def test_a_source_has_divergence_two_and_no_curl() -> None:
    field = derivatives(_grid2(lambda x, y: x, lambda x, y: y), x="x", y="y", u="u", v="v").frame
    assert np.allclose(field["divergence"], 2.0) and np.allclose(field["curl"], 0.0, atol=1e-9)


def test_three_dimensions_rows_in_any_order() -> None:
    g = np.linspace(-1, 1, 9)
    xs, ys, zs = np.meshgrid(g, g, g, indexing="ij")
    # u = -y, v = x, w = z: divergence 1, curl (0, 0, 2).
    frame = pd.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "z": zs.ravel(), "u": -ys.ravel(), "v": xs.ravel(),
                          "w": zs.ravel()}).sample(frac=1.0, random_state=1)
    result = derivatives(frame, x="x", y="y", z="z", u="u", v="v", w="w")
    assert result.dims == 3
    assert np.allclose(result.frame["divergence"], 1.0) and np.allclose(result.frame["curl_z"], 2.0)
    assert np.allclose(result.frame[["curl_x", "curl_y"]], 0.0, atol=1e-9)


def test_an_incomplete_grid_is_refused() -> None:
    frame = _grid2(lambda x, y: x, lambda x, y: y).iloc[:-1]
    with pytest.raises(ValueError, match="complete grid"):
        derivatives(frame, x="x", y="y", u="u", v="v")


@pytest.mark.parametrize("dims", [2, 3])
def test_the_dialog_draws_the_field(qapp, repo, dims: int) -> None:
    import time

    from app.table_operations.vector_field_dialog import VectorFieldDialog

    g = np.linspace(-1, 1, 6)
    if dims == 2:
        xs, ys = np.meshgrid(g, g, indexing="ij")
        frame = pd.DataFrame({"px": xs.ravel(), "py": ys.ravel(), "fu": -ys.ravel(), "fv": xs.ravel()})
        roles = {"x": ["px"], "y": ["py"], "u": ["fu"], "v": ["fv"]}
        expected = ["Heatmap", "Heatmap"]
    else:
        xs, ys, zs = np.meshgrid(g, g, g, indexing="ij")
        frame = pd.DataFrame({"px": xs.ravel(), "py": ys.ravel(), "pz": zs.ravel(), "fu": -ys.ravel(),
                              "fv": xs.ravel(), "fw": zs.ravel()})
        roles = {"x": ["px"], "y": ["py"], "z": ["pz"], "u": ["fu"], "v": ["fv"], "w": ["fw"]}
        expected = ["Quiver 3D", "Quiver 3D"]
    repo.import_dataframe(frame, table_name="field", normalize_columns=False)
    dialog = VectorFieldDialog(repo=repo, table="field")
    try:
        dialog.roles_widget.set_casting(roles)
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 30
        while dialog.job_running and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in sorted(figure.axes, key=lambda a: a.index)] == expected
        from matplotlib.figure import Figure

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert all(ax.collections for ax in drawn.axes if ax.get_label() != "<colorbar>")
    finally:
        dialog.close()
