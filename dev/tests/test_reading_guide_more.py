"""The "How to read this" boxes added for the analyses that had none (dev/plans/explanations_and_line_editor_plan.txt)."""
from __future__ import annotations

from app.analysis.peaks import Peak
from app.data_operations.peaks_dialog import PeakResult, SeriesPeaksDialog


def _peak(x: float, prominence: float, width: float) -> Peak:
    nan = float("nan")
    return Peak(x=x, y=prominence, prominence=prominence, width=width, left_x=x - width / 2, right_x=x + width / 2,
                **{name: nan for name in Peak.__dataclass_fields__ if name not in
                   ("x", "y", "prominence", "width", "left_x", "right_x")})


def test_peaks_says_how_many_and_describes_the_most_prominent() -> None:
    result = PeakResult(source_name="s", result_name="r", model="m",
                        peaks=[_peak(2.0, 1.0, 0.5), _peak(5.0, 3.0, 0.8)])
    lines = SeriesPeaksDialog.how_to_read(None, [result])
    assert len(lines) == 1 and "2 peaks kept" in lines[0] and "x = 5" in lines[0] and "0.8" in lines[0]
    empty = PeakResult(source_name="s", result_name="r", model="m", peaks=[])
    assert "No peak passed" in SeriesPeaksDialog.how_to_read(None, [empty])[0]


def test_the_signal_and_numeric_operations_explain_their_result() -> None:
    import types

    import numpy as np

    from app.data_operations.baseline_dialog import SeriesBaselineDialog
    from app.data_operations.calculus_dialog import SeriesCalculusDialog
    from app.data_operations.filter_dialog import SeriesFilterDialog
    from app.data_operations.roots_dialog import SeriesRootsDialog
    from app.data_operations.smoothing_dialog import SeriesSmoothingDialog
    from app.data_operations.transform_dialog import SeriesTransformDialog

    ns = types.SimpleNamespace
    x = np.linspace(0, 10, 11)
    assert "over 11 points" in SeriesSmoothingDialog.how_to_read(None, [ns(method="Moving average", y=x)])[0]
    assert "at 0.5" in SeriesFilterDialog.how_to_read(None, [ns(model="m", metadata={"response": "Low-pass",
                                                                                      "cutoff": 0.5})])[0]
    line = SeriesBaselineDialog.how_to_read(None, [ns(x=x, corrected=np.ones(11), metadata={"baseline area": 10.0})])[0]
    assert "area of 10" in line and "50%" in line
    root = ns(x=2.5, rising=True)
    assert "first at x = 2.5, going up" in SeriesRootsDialog.how_to_read(None, [ns(roots=[root], level=0.0)])[0]
    assert "no crossing" in SeriesRootsDialog.how_to_read(None, [ns(roots=[], level=1.0)])[0]
    assert "area under the curve is 4.2" in SeriesCalculusDialog.how_to_read(None, [ns(total=4.2, metadata={})])[0]
    assert SeriesCalculusDialog.how_to_read(None, [ns(total=None, metadata={})]) == []
    assert "geometric mean" in SeriesTransformDialog.how_to_read(None, [ns(model="Log")])[0]


def test_function_geometry_gp_and_wafer_explain_their_result() -> None:
    import types

    import numpy as np

    from app.data_operations.function_dialog import SeriesFunctionDialog
    from app.data_operations.geometry_dialog import SeriesGeometryDialog
    from app.data_operations.gp_regression_dialog import SeriesGPRegressionDialog
    from app.data_operations.wafer_map_dialog import WaferMapDialog

    ns = types.SimpleNamespace
    assert "(a = 2)" in SeriesFunctionDialog.how_to_read(None, [ns(function_name="Line", params={"a": 2.0})])[0]
    assert "to 4 points" in SeriesGeometryDialog.how_to_read(None, [ns(model="Rotation", after_x=np.zeros(4))])[0]
    x = np.linspace(0, 4, 5)
    gp = ns(x=x, upper=np.array([1, 1, 3, 1, 1.0]), lower=np.zeros(5))
    assert "spans 3" in SeriesGPRegressionDialog.how_to_read(None, [gp])[0] and "x = 2" in \
        SeriesGPRegressionDialog.how_to_read(None, [gp])[0]
    assert "Yield 90%" in WaferMapDialog.how_to_read(None, [ns(summary={"Yield (%)": 90.0, "Dies measured": 50})])[0]
    assert "Poisson yield 80%" in WaferMapDialog.how_to_read(None, [ns(summary={"Poisson yield exp(-D A) (%)": 80.0,
                                                                             "Dies": 60})])[0]


def test_a_dash_pattern_typed_as_text_is_drawn_as_matplotlib_s_tuple(qapp) -> None:
    from app.plugin_api.charts import kwarg_spec
    from app.plugin_api.charts.dashes import is_custom, to_matplotlib
    from app.widgets.drop_down_controls.line_combo import LineStyleCombo

    assert to_matplotlib("..-") == (0.0, (1.0, 1.6, 1.0, 1.6, 3.7, 1.6))
    assert to_matplotlib("(0, (5, 2, 1, 2))") == (0.0, (5.0, 2.0, 1.0, 2.0))
    assert to_matplotlib("--") == "--" and to_matplotlib("abc") is None and not is_custom("--")
    spec = {"linestyle": {"default": "-", "type": str, "kind": "linestyle"}}
    assert kwarg_spec.resolve(spec, [{"linestyle": "-."}]) == {"linestyle": "-."}
    assert kwarg_spec.resolve(spec, [{"linestyle": ". -"}])["linestyle"][1][0] == 1.0
    assert "linestyle" not in kwarg_spec.resolve(spec, [{"linestyle": "xyz"}])
    combo = LineStyleCombo()
    assert combo.set_current_linestyle("..--") and combo.current_linestyle() == "..--"
    assert combo.set_current_linestyle("--") and combo.current_linestyle() == "--"
    assert combo.findData("..--") >= 0  # kept in the list once added
