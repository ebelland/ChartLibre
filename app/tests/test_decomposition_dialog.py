"""Decomposition and manifold learning across several series (todo.txt P2-15).

Unlike every other series operation, this one combines several *selected*
series into one shared-grid feature matrix rather than reading each one
independently - see the dialog module's own docstring for why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.decomposition_dialog import DECOMP_PCA, SeriesDecompositionDialog


def _dialog(repo: SqliteRepo, figure_id: int, *, select_all: bool = True):
    dialog = SeriesDecompositionDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=select_all)
    return dialog


@pytest.fixture
def correlated_figure(repo: SqliteRepo):
    """Three series built from the same underlying sine wave.

    Strongly correlated by construction, so PCA's first component should
    capture most of the variance - the thing the PCA test below checks.
    """
    x = np.linspace(0.0, 10.0, 60)
    base = np.sin(x)
    rng = np.random.RandomState(0)
    tables = {
        "a": base + rng.normal(scale=0.02, size=x.size),
        "b": 2.0 * base + 1.0 + rng.normal(scale=0.02, size=x.size),
        "c": -base + 2.0 + rng.normal(scale=0.02, size=x.size),
    }
    for name, y in tables.items():
        repo.import_dataframe(
            pd.DataFrame({"x": x, "y": y}), table_name=name, normalize_columns=False
        )

    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    for index, name in enumerate(tables):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=name,
            sql_query=f"SELECT x, y FROM {name}", roles={"x": "x", "y": "y"}, style={},
        )
    return figure_id, axis_id


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Decomposition
# ----------------------------------------------------------------------
def test_pca_recovers_the_shared_signal_as_the_first_component(
    qapp, repo: SqliteRepo, correlated_figure
) -> None:
    figure_id, _axis_id = correlated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(DECOMP_PCA)

    result = dialog.compute_results()[0]
    ratios = result.metadata["explained_variance_ratio"]

    assert ratios[0] == max(ratios)
    assert ratios[0] > 0.5


# ----------------------------------------------------------------------
# Manifold learning
# ----------------------------------------------------------------------


