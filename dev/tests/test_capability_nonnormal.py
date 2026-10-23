"""Capability of a process that is not normal (AN-06): the percentile method on a fitted distribution."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from app.analysis.capability import best_distribution, capability

RNG = np.random.default_rng(3)
LOGNORMAL = RNG.lognormal(mean=1.0, sigma=0.5, size=2000)


def test_a_lognormal_process_is_recognised_and_judged_by_its_percentiles() -> None:
    assert best_distribution(LOGNORMAL) == "lognormal"
    result = capability(LOGNORMAL, sigma_within=1.0, lsl=0.5, usl=20.0, distribution="best")
    assert result.distribution == "lognormal" and "Cpk" not in result.indices
    shape, _loc, scale = stats.lognorm.fit(LOGNORMAL, floc=0)
    low, median, high = stats.lognorm.ppf([0.00135, 0.5, 0.99865], shape, 0, scale)
    assert result.indices["Pp"] == pytest.approx((20.0 - 0.5) / (high - low))
    assert result.indices["Ppk"] == pytest.approx(min((median - 0.5) / (median - low), (20.0 - median) / (high - median)))
    above = result.outside.set_index("Portion").loc["Above USL", "Expected % (overall)"]
    assert above == pytest.approx(100 * stats.lognorm.sf(20.0, shape, 0, scale))


def test_normal_stays_the_default_and_a_negative_value_refuses_a_lognormal() -> None:
    normal = RNG.normal(10, 1, 500)
    assert capability(normal, sigma_within=1.0, lsl=7, usl=13).distribution == "normal"
    with pytest.raises(ValueError):
        capability(normal - 20, sigma_within=1.0, lsl=-15, usl=-5, distribution="lognormal")


def test_the_control_chart_reports_a_lognormal_capability(qapp, repo) -> None:
    import time

    import pandas as pd

    from app.data_operations.control_chart_dialog import SeriesControlChartDialog

    repo.import_dataframe(pd.DataFrame({"x": np.arange(200.0), "y": LOGNORMAL[:200]}), table_name="t",
                          normalize_columns=False)
    figure = repo.create_figure_descriptor(name="f")
    axis = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Scatter Plot", title="",
                                       x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis, series_index=0, name="r", sql_query='SELECT x, y FROM "t"',
                                  roles={"x": "x", "y": "y"})
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure, parent=None)
    try:
        dialog.series_selector.select_all_series()
        chosen = {**dialog.parameter_values(), "lsl": "0.5", "usl": "20", "distribution": "lognormal"}
        dialog.parameter_values = lambda: chosen
        result = dialog.compute_results()[0]
        assert result.capability.distribution == "lognormal" and "Ppk" in result.capability.indices
        assert "Percentile method" in dialog._capability_html(result)
    finally:
        dialog.close()
