"""H-04, the teaching mode: each report with a "How to read this" box - its key numbers, each in two sentences
about this result, under the main result; Settings > Reports switches it off."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.plugin_api import reading_guide as guide
from app.utils import report_html


def test_a_p_value_is_said_about_this_result_on_both_sides_of_the_usual_line() -> None:
    null, shown = "the groups had the same mass", "mass differs between them"
    small = guide.p_value(0.003, null=null, shown=shown)
    assert small.startswith("p = 0.003: if the groups had the same mass") and "about 3 times in 1000" in small
    assert "called real: mass differs between them" in small
    large = guide.p_value(0.21, null=null, shown=shown)
    assert "about 21 times in 100" in large and "not proof that the groups had the same mass" in large
    assert guide.p_value(3e-9, null=null, shown=shown).startswith("p < 0.0001") and "practically never" in \
        guide.p_value(3e-9, null=null, shown=shown)
    assert guide.p_value(math.nan, null=null, shown=shown) == "" and guide.p_value(None, null=null, shown=shown) == ""


def test_the_other_explainers_name_their_numbers() -> None:
    assert "81% of how much mass varies" in guide.r_squared(0.81, y="mass") and "other 19%" in guide.r_squared(
        0.81, y="mass")
    assert "leaves out 0" in guide.slope(2.1, 1.8, 2.4, y="y", x="x")
    assert "includes 0" in guide.slope(0.2, -0.1, 0.5, y="y", x="x")
    assert "large effect" in guide.effect_size(1.9, name="Cohen's d")
    assert "negligible effect" in guide.effect_size(0.005, name="eta²")
    assert "not normal" in guide.normality(0.001, test="Shapiro-Wilk normality", column="mass")
    assert "nothing here says mass is not normal" in guide.normality(0.4, test="Shapiro-Wilk normality",
                                                                     column="mass")
    assert "tends to fall" in guide.correlation(-0.6, x="x", y="y")
    assert "1 signal(s)" in guide.control_limits(10, 7, 13, signals=1)
    assert guide.lookup(pd.DataFrame({"Measure": ["RSquare"], "Value": [0.5]}), "RSquare", by="Measure") == 0.5


def test_the_box_goes_under_the_main_result() -> None:
    box = report_html.reading_guide("How to read this", ["One.", "", "Two."])
    assert box.count("<p") == 2 and report_html.reading_guide("How to read this", ["", ""]) == ""
    body = report_html.note("a note") + report_html.section("Main", report_html.section("Inner", "x")) \
        + report_html.section("Second", "y")
    placed = report_html.with_reading_guide(body, box)
    assert placed.index(box) > placed.index("Inner") and placed.index(box) < placed.index("Second")
    assert report_html.with_reading_guide("<p>answer</p>" + report_html.READING_GUIDE_MARK + "<p>more</p>",
                                          box) == "<p>answer</p>" + box + "<p>more</p>"
    assert report_html.with_reading_guide("<p>only text</p>", box).endswith(box)


def _plant() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    frame = pd.DataFrame({"line": rng.choice(["A", "B", "C"], n), "temp": rng.uniform(0, 100, n),
                          "a": rng.normal(size=n), "b": rng.normal(size=n)})
    frame["c"] = 0.8 * frame["a"] + rng.normal(0, 0.3, n)
    frame["y"] = 0.05 * frame["temp"] + frame["line"].map({"A": 0.0, "B": 2.0, "C": 4.0}) + rng.normal(0, 0.3, n)
    frame["part"] = np.repeat([f"P{i}" for i in range(10)], 12)
    frame["op"] = np.tile(np.repeat(["O1", "O2", "O3"], 4), 10)
    frame["meas"] = frame["part"].str[1:].astype(float) + rng.normal(0, 0.2, n)
    return frame


def _field() -> pd.DataFrame:
    x, y = np.meshgrid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6))
    return pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "u": -y.ravel(), "v": x.ravel()})


def _report(qapp, dialog, casting: dict) -> str:  # noqa: ANN001
    dialog.roles_widget.set_casting(casting)
    if hasattr(dialog, "_add_terms"):  # Fit Model: its effects added, as Add does
        dialog._add_terms([("temp",), ("line",)])
    if type(dialog).__name__ == "TimeSeriesDialog":
        from app.analysis import time_series as ts

        dialog._parameter_form.set_values({"model": ts.AUTO})
    dialog.preview()
    deadline = time.monotonic() + 120
    while dialog.job_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog._result is not None, dialog.problems()
    return dialog.report_document(dialog._result)


@pytest.mark.parametrize("module, name, table, casting, expected", [
    ("differences_dialog", "DifferencesDialog", "plant", {"response": ["y"], "group": ["line"]}, "the groups of line"),
    ("distribution_dialog", "DistributionDialog", "plant", {"response": ["y"]}, "interval for the mean of y"),
    ("fit_y_by_x_dialog", "FitYByXDialog", "plant", {"response": ["y"], "factor": ["temp"]}, "R² = "),
    ("fit_model_dialog", "FitModelDialog", "plant", {"response": ["y"]}, "LogWorth"),
    ("partition_dialog", "PartitionDialog", "plant", {"response": ["y"], "factor": ["line", "temp", "a"]},
     "of what the splits explain"),
    ("pls_dialog", "PLSDialog", "plant", {"response": ["y"], "predictor": ["temp", "a", "b", "c"]}, "Q² = "),
    ("discriminant_dialog", "DiscriminantDialog", "plant", {"response": ["line"], "covariate": ["y", "temp", "a"]},
     "Misclassification"),
    ("multivariate_dialog", "MultivariateDialog", "plant", {"response": ["a", "b", "c", "y"]}, "r = "),
    ("principal_components_dialog", "PrincipalComponentsDialog", "plant", {"response": ["a", "b", "c", "y", "temp"]},
     "The first 2 components"),
    ("factor_analysis_dialog", "FactorAnalysisDialog", "plant", {"response": ["a", "b", "c", "y", "temp"]},
     "factor(s) were enough"),
    ("clustering_dialog", "ClusteringDialog", "plant", {"response": ["a", "c", "y"]}, "Mean silhouette"),
    ("gage_dialog", "GageDialog", "plant", {"response": ["meas"], "part": ["part"], "operator": ["op"]}, "Gage R&amp;R"),
    ("vector_field_dialog", "VectorFieldDialog", "field", {"x": ["x"], "y": ["y"], "u": ["u"], "v": ["v"]},
     "Divergence"),
    ("time_series_dialog", "TimeSeriesDialog", "shop", {"response": ["y"], "time": ["month"]}, "AIC"),
])
def test_a_report_explains_its_own_numbers_unless_switched_off(qapp, repo: SqliteRepo, monkeypatch, module: str,
                                                                 name: str, table: str, casting: dict,
                                                                 expected: str) -> None:
    import importlib

    from dev.tests.test_time_series import _seasonal_walk

    repo.import_dataframe(_plant(), table_name="plant", normalize_columns=False)
    repo.import_dataframe(_field(), table_name="field", normalize_columns=False)
    repo.import_dataframe(_seasonal_walk(), table_name="shop", normalize_columns=False)
    dialog = getattr(importlib.import_module(f"app.table_operations.{module}"), name)(repo=repo, table=table)
    try:
        report = _report(qapp, dialog, casting)
        assert "How to read this" in report and expected in report
        if module == "differences_dialog":  # under the answer, above what it rests on
            assert report.index("How to read this") < report.index("What next?")
        monkeypatch.setattr(guide, "enabled", lambda: False)
        assert "How to read this" not in dialog.report_document(dialog._result)
    finally:
        dialog.close()


@pytest.fixture
def series_figure(repo: SqliteRepo) -> int:
    rng = np.random.default_rng(1)
    n = 128
    x = np.arange(n, dtype=float)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": 2 + 0.5 * x + rng.normal(0, 2, n) + 4 * np.sin(x / 3),
                                        "z": rng.normal(10, 1, n)}), table_name="s", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title="t",
                                              x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="sig", sql_query="SELECT x, y FROM s",
                                  roles={"x": "x", "y": "y"}, style={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=1, name="z", sql_query="SELECT x, z AS y FROM s",
                                  roles={"x": "x", "y": "y"}, style={})
    return figure_id


@pytest.mark.parametrize("module, name, expected", [
    ("regression_dialog", "SeriesRegressionDialog", "R² = "),
    ("control_chart_dialog", "SeriesControlChartDialog", "a stable process keeps"),
    ("outlier_dialog", "SeriesOutlierDialog", "values flagged by"),
    ("statistics_dialog", "SeriesStatisticsDialog", "interval for the mean of"),
    ("fit_dialog", "SeriesFitDialog", "RMSE = "),
])
def test_a_data_operation_explains_its_report(qapp, repo: SqliteRepo, series_figure: int, module: str, name: str,
                                              expected: str) -> None:
    import importlib

    dialog = getattr(importlib.import_module(f"app.data_operations.{module}"), name)(repo=repo,
                                                                                    figure_id=series_figure)
    try:
        dialog.series_selector.reload(select_all_series=True)
        if name == "SeriesFitDialog":
            assert dialog.select_model("Linear")
            dialog.on_estimate_initial_values()
            dialog.evaluate(dialog._after_fit, background=False, optimise=True)
        report = dialog.results_report(dialog.compute_results())
        assert "How to read this" in report and expected in report
    finally:
        dialog.close()
