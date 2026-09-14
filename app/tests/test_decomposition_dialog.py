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
from app.series_operations.decomposition_dialog import (
    DECOMP_NMF,
    DECOMP_PCA,
    MANIFOLD_ISOMAP,
    MANIFOLD_LLE,
    MANIFOLD_TSNE,
    SeriesDecompositionDialog,
)


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


@pytest.fixture
def non_negative_figure(repo: SqliteRepo):
    """Two strictly non-negative series, for NMF's own happy path."""
    x = np.linspace(0.0, 10.0, 40)
    repo.import_dataframe(
        pd.DataFrame({"x": x, "y": np.abs(np.sin(x)) + 0.1}),
        table_name="p", normalize_columns=False,
    )
    repo.import_dataframe(
        pd.DataFrame({"x": x, "y": np.abs(np.cos(x)) + 0.1}),
        table_name="q", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    for index, name in enumerate(("p", "q")):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=name,
            sql_query=f"SELECT x, y FROM {name}", roles={"x": "x", "y": "y"}, style={},
        )
    return figure_id, axis_id


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------
def test_selecting_only_one_series_is_refused(qapp, repo: SqliteRepo, correlated_figure) -> None:
    figure_id, _axis_id = correlated_figure
    dialog = _dialog(repo, figure_id, select_all=False)

    with pytest.raises(ValueError, match="two or more"):
        dialog.compute_results()


def test_nmf_refuses_data_with_negative_values(
    qapp, repo: SqliteRepo, correlated_figure
) -> None:
    figure_id, _axis_id = correlated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(DECOMP_NMF)

    with pytest.raises(ValueError, match="non-negative"):
        dialog.compute_results()


def test_nmf_runs_on_non_negative_data(qapp, repo: SqliteRepo, non_negative_figure) -> None:
    figure_id, _axis_id = non_negative_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(DECOMP_NMF)

    result = dialog.compute_results()[0]

    assert result.kind == "decomposition"
    assert result.values.shape[0] == len(result.x)


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


def test_pca_apply_creates_one_series_per_component(
    qapp, repo: SqliteRepo, correlated_figure
) -> None:
    figure_id, axis_id = correlated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(DECOMP_PCA)
    dialog.parameter_values()  # sanity: does not raise before apply

    assert dialog.apply()
    new_axis_id = dialog._result_axis_id
    assert new_axis_id != axis_id
    names = [str(row["name"]) for row in repo.get_series(new_axis_id)]
    assert names == ["PC1", "PC2", "PC3"]


def test_fast_ica_runs_end_to_end(qapp, repo: SqliteRepo, correlated_figure) -> None:
    figure_id, _axis_id = correlated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText("Fast ICA")

    result = dialog.compute_results()[0]

    assert result.kind == "decomposition"
    assert len(result.component_names) >= 1


# ----------------------------------------------------------------------
# Manifold learning
# ----------------------------------------------------------------------
@pytest.mark.parametrize("model", [MANIFOLD_TSNE, MANIFOLD_ISOMAP, MANIFOLD_LLE])
def test_each_manifold_model_produces_a_2d_embedding(
    qapp, repo: SqliteRepo, correlated_figure, model: str
) -> None:
    figure_id, _axis_id = correlated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(model)

    result = dialog.compute_results()[0]

    assert result.kind == "manifold"
    assert len(result.x) == len(result.values)


def test_a_small_sample_clamps_perplexity_and_neighbors_instead_of_raising(
    qapp, repo: SqliteRepo
) -> None:
    x = np.linspace(0.0, 1.0, 6)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": x}), table_name="s1", normalize_columns=False)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": x * 2}), table_name="s2", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    for index, name in enumerate(("s1", "s2")):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=name,
            sql_query=f"SELECT x, y FROM {name}", roles={"x": "x", "y": "y"}, style={},
        )
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(MANIFOLD_TSNE)
    dialog.set_parameter_values({"n_grid": 6, "perplexity": 30.0})

    result = dialog.compute_results()[0]  # must not raise scikit-learn's own error

    assert result.kind == "manifold"


def test_manifold_output_always_lands_on_a_new_axis(
    qapp, repo: SqliteRepo, correlated_figure
) -> None:
    figure_id, axis_id = correlated_figure
    dialog = _dialog(repo, figure_id)
    dialog.model_combo.setCurrentText(MANIFOLD_TSNE)

    assert dialog.apply()

    assert dialog._result_axis_id != axis_id
    names = [str(row["name"]) for row in repo.get_series(dialog._result_axis_id)]
    assert names == ["t-SNE embedding"]
