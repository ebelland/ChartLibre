"""Gage R&R (AN-06): the variance components of a crossed study, recovered from data built with known ones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.gage import gage_rr

#: Built with: part sd 2.0, operator sd 0.5, repeatability sd 0.3, no interaction.
PART_SD, OPERATOR_SD, REPEAT_SD = 2.0, 0.5, 0.3


def _study(parts: int = 30, operators: int = 3, replicates: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    part_effect = rng.normal(0, PART_SD, parts)
    operator_effect = rng.normal(0, OPERATOR_SD, operators)
    rows = [(f"P{p}", f"O{o}", 10 + part_effect[p] + operator_effect[o] + rng.normal(0, REPEAT_SD))
            for p in range(parts) for o in range(operators) for _r in range(replicates)]
    return pd.DataFrame(rows, columns=["part", "operator", "y"])


def test_the_components_built_into_the_data_come_back() -> None:
    study = gage_rr(_study(parts=40, operators=6, replicates=4), "y", "part", "operator")
    comps = study.components.set_index("Source")
    assert comps.loc["Repeatability", "Std Dev"] == pytest.approx(REPEAT_SD, rel=0.1)
    assert comps.loc["Part-to-Part", "Std Dev"] == pytest.approx(PART_SD, rel=0.3)
    assert comps.loc["Operator", "Std Dev"] == pytest.approx(OPERATOR_SD, rel=0.6)
    assert study.pooled  # no interaction was built in
    total = comps.loc["Total Variation", "Variance"]
    assert comps.loc["Gage R&R", "Variance"] + comps.loc["Part-to-Part", "Variance"] == pytest.approx(total)
    assert comps.loc["Total Variation", "% Contribution"] == pytest.approx(100.0)


def test_tolerance_ndc_and_the_verdict() -> None:
    study = gage_rr(_study(), "y", "part", "operator", tolerance=12.0)
    comps = study.components.set_index("Source")
    assert comps.loc["Gage R&R", "% Tolerance"] == pytest.approx(100 * comps.loc["Gage R&R", "Study Var"] / 12.0)
    expected_ndc = int(np.floor(1.41 * comps.loc["Part-to-Part", "Std Dev"] / comps.loc["Gage R&R", "Std Dev"]))
    assert study.ndc == max(1, expected_ndc)
    assert study.sentences and "measuring system" in study.sentences[0][0]


def test_without_operators_it_is_repeatability_alone() -> None:
    study = gage_rr(_study(operators=1), "y", "part")
    comps = study.components.set_index("Source")
    assert comps.loc["Reproducibility", "Variance"] == 0.0
    with pytest.raises(ValueError):
        gage_rr(_study(operators=1, replicates=1), "y", "part")


def test_the_dialog_reports_and_draws_a_multi_vari_chart(qapp, repo) -> None:
    import time

    from app.table_operations.gage_dialog import GageDialog

    repo.import_dataframe(_study(parts=8), table_name="msa", normalize_columns=False)
    dialog = GageDialog(repo=repo, table="msa")
    try:
        dialog.roles_widget.set_casting({"response": ["y"], "part": ["part"], "operator": ["operator"]})
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 30
        while dialog._task is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Multi-Vari Chart"]
        from matplotlib.figure import Figure

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert len(drawn.axes[0].lines) >= 8 * 3
    finally:
        dialog.close()
