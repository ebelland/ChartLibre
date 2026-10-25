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
    frame = pd.DataFrame({"line": rng.choice(["A", "B", "C"], n), "temp": rng.uniform(0, 100, n)})
    frame["y"] = 0.05 * frame["temp"] + frame["line"].map({"A": 0.0, "B": 2.0, "C": 4.0}) + rng.normal(0, 0.3, n)
    return frame


def _report(qapp, dialog, casting: dict) -> str:  # noqa: ANN001
    dialog.roles_widget.set_casting(casting)
    if hasattr(dialog, "_add_terms"):  # Fit Model: its effects added, as Add does
        dialog._add_terms([("temp",), ("line",)])
    dialog.preview()
    deadline = time.monotonic() + 60
    while dialog.job_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog._result is not None
    return dialog.report_document(dialog._result)


@pytest.mark.parametrize("module, name, casting, expected", [
    ("differences_dialog", "DifferencesDialog", {"response": ["y"], "group": ["line"]}, "the groups of line"),
    ("distribution_dialog", "DistributionDialog", {"response": ["y"]}, "interval for the mean of y"),
    ("fit_y_by_x_dialog", "FitYByXDialog", {"response": ["y"], "factor": ["temp"]}, "R² = "),
    ("fit_model_dialog", "FitModelDialog", {"response": ["y"]}, "LogWorth"),
])
def test_a_report_explains_its_own_numbers_unless_switched_off(qapp, repo: SqliteRepo, monkeypatch, module: str,
                                                                 name: str, casting: dict, expected: str) -> None:
    import importlib

    repo.import_dataframe(_plant(), table_name="plant", normalize_columns=False)
    dialog = getattr(importlib.import_module(f"app.table_operations.{module}"), name)(repo=repo, table="plant")
    try:
        report = _report(qapp, dialog, casting)
        assert "How to read this" in report and expected in report
        if module == "differences_dialog":  # under the answer, above what it rests on
            assert report.index("How to read this") < report.index("What next?")
        monkeypatch.setattr(guide, "enabled", lambda: False)
        assert "How to read this" not in dialog.report_document(dialog._result)
    finally:
        dialog.close()
