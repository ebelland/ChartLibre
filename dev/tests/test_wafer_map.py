"""Wafer Map: the wafer's geometry (notch, flat, edge exclusion, dies, rotation), its measurements' map, die
means and bins, its defects' density - and the operation drawing them on the Wafer Map chart."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import wafer as wf
from app.data.sqlite_repo import SqliteRepo


def test_the_flat_and_the_edge_exclusion_leave_points_out() -> None:
    wafer = wf.Wafer(150, wf.FLAT, 270, edge_exclusion=0)
    assert wafer.flat_distance() == pytest.approx(np.sqrt(75 ** 2 - 28.75 ** 2))
    assert wafer.inside(np.array([0.0, 0.0, 74.0]), np.array([-60.0, -74.0, 0.0])).tolist() == [True, False, True]
    assert not wf.Wafer(300, edge_exclusion=5).inside(np.array([146.0]), np.array([0.0]))[0]


def test_the_outline_has_its_notch_where_the_mark_is() -> None:
    x, y = wf.Wafer(300, wf.NOTCH, 270).outline()
    radius = np.hypot(x, y)
    deepest = np.argmin(radius)
    assert radius[deepest] == pytest.approx(149.0, abs=0.05) and y[deepest] < -148 and abs(x[deepest]) < 0.5


def test_rotation_turns_counterclockwise() -> None:
    x, y = wf.rotate(np.array([10.0]), np.array([0.0]), 90)
    assert (x[0], y[0]) == pytest.approx((0.0, 10.0), abs=1e-12)


def test_whole_dies_only_and_each_point_on_its_die() -> None:
    wafer = wf.Wafer(100, wf.NO_MARK, edge_exclusion=0)
    dies = wf.die_grid(wafer, 10, 10)
    corners = [(dies["X"] + dx, dies["Y"] + dy) for dx in (-5, 5) for dy in (-5, 5)]
    assert all(wafer.inside(cx.to_numpy(), cy.to_numpy()).all() for cx, cy in corners)
    assert len(dies) == 61  # the whole 10 mm dies on a 100 mm disc, centred (a corner on the edge counts)
    die = wf.assign_dies(np.array([1.0, 12.0, 49.9]), np.array([1.0, -3.0, 0.0]), dies, 10, 10)
    assert dies.loc[die[0], ["Column", "Row"]].tolist() == [0, 0]
    assert dies.loc[die[1], ["Column", "Row"]].tolist() == [1, 0] and die[2] == -1  # on no whole die


def test_die_means_bins_and_the_defects_poisson_yield() -> None:
    wafer = wf.Wafer(100, wf.NO_MARK, edge_exclusion=0)
    dies = wf.die_grid(wafer, 10, 10)
    x, y, value = np.array([0.0, 1.0, 10.0, 20.0]), np.zeros(4), np.array([4.0, 6.0, 20.0, -1.0])
    die = wf.assign_dies(x, y, dies, 10, 10)
    measured = wf.die_summary(dies, die, value, 10, 10, lower=0.0, upper=10.0)
    by_place = measured.dies.set_index(["Column", "Row"])
    assert by_place.loc[(0, 0), "Mean"] == 5.0 and by_place.loc[(0, 0), "Bin name"] == "Pass"
    assert by_place.loc[(1, 0), "Bin name"] == "High" and by_place.loc[(2, 0), "Bin name"] == "Low"
    assert measured.summary["Yield (%)"] == pytest.approx(100 / 3)
    defects = wf.die_summary(dies, die, None, 10, 10)
    density = 4 / (len(dies) * 1.0)  # four defects on dies of 1 cm²
    assert defects.summary["Defect density (per cm²)"] == pytest.approx(density)
    assert defects.summary["Poisson yield exp(-D A) (%)"] == pytest.approx(100 * np.exp(-density))


def test_the_contour_grid_is_blank_off_the_wafer() -> None:
    wafer = wf.Wafer(200, wf.NO_MARK, edge_exclusion=0)
    rng = np.random.default_rng(0)
    r, t = np.sqrt(rng.uniform(0, 1, 60)) * 95, rng.uniform(0, 2 * np.pi, 60)
    x, y = r * np.cos(t), r * np.sin(t)
    grid = wf.contour_grid(x, y, 2.0 * x + 3.0, wafer, points=41)
    off = ~wafer.inside(grid["x"].to_numpy(), grid["y"].to_numpy())
    assert grid.loc[off, "value"].isna().all() and grid.loc[~off, "value"].notna().all()
    near_centre = grid[(grid["x"].abs() < 10) & (grid["y"].abs() < 10)]
    np.testing.assert_allclose(near_centre["value"], 2.0 * near_centre["x"] + 3.0, atol=0.5)


def test_the_operation_draws_the_map_dies_and_bins(qapp, repo: SqliteRepo) -> None:
    from matplotlib.figure import Figure

    from app.data_operations.wafer_map_dialog import WaferMapDialog
    from app.plugin_api.charts.render_figure import render_figure_from_descriptor

    rng = np.random.default_rng(1)
    r, t = np.sqrt(rng.uniform(0, 1, 49)) * 140, rng.uniform(0, 2 * np.pi, 49)
    x, y = r * np.cos(t), r * np.sin(t)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": y, "t": 100 + 0.0004 * (x ** 2 + y ** 2)}), table_name="sites",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Sites"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="Sites", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Thickness", style={},
                                  sql_query='SELECT x, y, t AS z FROM "sites"', roles={"x": "x", "y": "y", "z": "z"})
    dialog = WaferMapDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.source_chart_radio.setChecked(True)
        dialog.series_selector.select_all_series()
        dialog._parameter_form_spec.set_values({"die_width": 20.0, "die_height": 20.0, "upper": "108"})
        dialog.ok()
        qapp.processEvents()
        (made,) = dialog.created_figure_ids
    finally:
        dialog.close()
    descriptor = repo.load_figure_descriptor(made)
    assert [axis.title for axis in descriptor.axes] == ["Thickness: map", "Thickness: die means", "Thickness: die bins"]
    drawn = Figure(figsize=(10, 8))
    render_figure_from_descriptor(figure=drawn, descriptor=descriptor, repo=repo)
    contour = next(c for c in drawn.axes[0].collections if type(c).__name__ == "QuadContourSet")
    assert any(len(path.vertices) for path in contour.get_paths())  # every grid row reached it: not thinned
    assert drawn.axes[0].get_xlim() == pytest.approx((-157.5, 157.5))  # a circle, not stretched
    repo.undo_last()
    assert repo.load_figure_descriptor(made) is None


def test_full_grids_are_never_thinned_and_other_3d_charts_only_past_a_higher_limit() -> None:
    """Downsampling keeps one row in N: a full grid is no grid after it - never; a mesh or a 3D cloud only
    past DENSE_DOWNSAMPLE_FACTOR times the figure's threshold."""
    from app.plugin_api.charts import render_figure as rf

    assert {"Wafer Map", "Contour Plot", "Surface Plot"} <= rf._GRID_REQUIRED_CHART_TYPES
    for chart in ("Surface Plot (Scattered)", "Scatter Plot (3D)", "Heatmap", "Triangular Color Mesh",
                  "3D Line Plot", "Quiver"):
        assert rf._dense_chart(chart), chart
    assert not rf._dense_chart("Scatter Plot") and not rf._dense_chart("Wind Rose")
    assert rf.DENSE_DOWNSAMPLE_FACTOR == 100


def test_a_thinned_series_is_said_in_the_log(repo: SqliteRepo, monkeypatch) -> None:
    from app.plugin_api.charts import render_figure as rf

    said: list[str] = []
    monkeypatch.setattr(rf.applogger, "warning", lambda message, *args: said.append(message % args))
    repo.import_dataframe(pd.DataFrame({"x": np.arange(5000.0), "y": np.zeros(5000)}), table_name="long",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Long"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="Long", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="long", style={},
                                  sql_query='SELECT x, y FROM "long"', roles={"x": "x", "y": "y"})
    series = repo.load_figure_descriptor(figure_id).axes[0].series[0]
    frame = rf._load_series_df(repo=repo, series_desc=series, downsample_threshold=1000)
    assert len(frame) == 1000 and said == ["Series 'long': 5000 rows, about 1000 drawn (one in 5). Raise the "
                                           "figure's downsample threshold (0: none) to draw them all."]
