"""The series clustering's calculation (app.analysis.cluster_series): the k it picks, the report's tables, and the
split of one result into a series per cluster."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analysis import cluster_series as cs
from app.analysis.clustering import TOOL_KMEANS


def _choice(points: np.ndarray) -> cs.ClusterSeriesChoice:
    frame = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})
    return cs.ClusterSeriesChoice(name="Blobs", frame=frame, x_col="x", y_col="y", z_col=None, roles={},
                                  source_table="blobs", source_x_column="x", source_sql_query="", source=None)


def _params(**overrides) -> dict:
    params = {"method": cs.CLUSTER_KMEANS, "scipy_tool": TOOL_KMEANS, "clusters": 2, "whiten": False,
              "kmeans_iter": 50, "kmeans_thresh": 1e-5, "max_runtime_seconds": 0.0}
    params.update(overrides)
    return params


def _three_blobs() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.vstack([rng.normal(center, 0.2, size=(40, 2)) for center in ((0, 0), (5, 5), (0, 6))])


def test_cluster_one_makes_the_clusters_the_series_has() -> None:
    result = cs.cluster_one(_choice(_three_blobs()), ["x", "y"], "Axis", _params(clusters=3), 1, None)
    assert result.summary["Clusters"] == 3
    assert result.means.shape[0] == 3 and set(result.means.columns) >= {"Cluster", "x", "y"}
    assert int(result.frame[cs.CLUSTER_COLUMN].nunique()) == 3


def test_automatic_k_picks_the_silhouette_best_k() -> None:
    result = cs.cluster_one(_choice(_three_blobs()), ["x", "y"], "Axis",
                            _params(clusters=2, automatic_k=True, k_max=6), 1, None)
    assert result.summary["Selected k"] == 3
    assert list(result.k_table["k"]) == [2, 3, 4, 5, 6]
    assert bool(result.k_table.loc[result.k_table["Chosen"], "k"].iloc[0] == 3)


def test_split_gives_one_result_per_cluster_named_by_it() -> None:
    result = cs.cluster_one(_choice(_three_blobs()), ["x", "y"], "Axis", _params(clusters=3), 1, None)
    parts = cs.split_result_by_cluster(result)
    assert len(parts) == 3
    assert [part.result_name for part in parts] == [f"Blobs - Cluster {n}" for n in (1, 2, 3)]
    assert all(part.metadata["is_split_series"] for part in parts)
