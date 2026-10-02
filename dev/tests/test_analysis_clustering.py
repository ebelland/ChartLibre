"""Clustering (app.analysis.clustering): three well-separated blobs, found."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import clustering as cl

RNG = np.random.default_rng(2)
CENTRES = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
F = np.vstack([RNG.normal(centre, 0.3, (40, 2)) for centre in CENTRES])
TRUTH = np.repeat([0, 1, 2], 40)


def _agrees_with_truth(labels: np.ndarray) -> bool:
    """Same partition as TRUTH, whatever the cluster numbers are."""
    pairs = set(zip(TRUTH.tolist(), np.asarray(labels).tolist()))
    return len(pairs) == 3 and len({b for _a, b in pairs}) == 3


@pytest.mark.parametrize("tool", [cl.TOOL_WHITEN, cl.TOOL_KMEANS2])
def test_scipy_kmeans_finds_the_three_blobs(tool: str) -> None:
    labels, _meta = cl.cluster_kmeans(F, scipy_tool=tool, clusters=3, use_whiten=False, iterations=20, threshold=1e-5)
    assert _agrees_with_truth(labels)


@pytest.mark.parametrize("tool", [cl.TOOL_KMEANS, cl.TOOL_KMEANS2, cl.TOOL_WHITEN])
def test_scipy_kmeans_gives_the_same_clusters_every_run(tool: str) -> None:
    """Seeded: one diffuse cloud, where the starting points decide the split."""
    cloud = np.random.default_rng(5).normal(0.0, 1.0, (300, 2))
    runs = [
        cl.cluster_kmeans(cloud, scipy_tool=tool, clusters=4, use_whiten=False, iterations=20, threshold=1e-5)
        for _ in range(3)
    ]
    for labels, meta in runs[1:]:
        np.testing.assert_array_equal(labels, runs[0][0])
        assert meta["centroids"] == runs[0][1]["centroids"]


def test_hierarchical_clustering_cut_at_three_finds_the_blobs() -> None:
    labels, _meta = cl.cluster_hierarchical(
        F, scipy_tool=cl.TOOL_FCLUSTER, clusters=3, linkage_method="ward", metric="euclidean",
        criterion="maxclust", distance_threshold=1.0,
    )
    assert _agrees_with_truth(labels)


@pytest.mark.parametrize(
    "tool",
    [cl.SKLEARN_KMEANS, cl.SKLEARN_AGGLOMERATIVE, cl.SKLEARN_BIRCH, cl.SKLEARN_GAUSSIAN_MIXTURE, cl.SKLEARN_SPECTRAL],
)
def test_scikit_learn_finds_the_three_blobs(tool: str) -> None:
    labels, _meta = cl.cluster_sklearn(
        F, sklearn_tool=tool, clusters=3, linkage_method="ward", metric="euclidean", eps=0.8, min_samples=5
    )
    assert _agrees_with_truth(labels)


def test_dbscan_finds_the_blobs_without_being_told_how_many() -> None:
    labels, _meta = cl.cluster_sklearn(
        F, sklearn_tool=cl.SKLEARN_DBSCAN, clusters=0, linkage_method="ward", metric="euclidean", eps=0.8, min_samples=5
    )
    assert _agrees_with_truth(labels)


def test_cluster_ids_are_stable_and_start_at_one() -> None:
    assert cl.stable_cluster_ids(np.array([7, -1, 7, 3])).tolist() == [3, 1, 3, 2]


def test_the_feature_matrix_drops_rows_that_are_not_numbers_and_reads_dates() -> None:
    frame = pd.DataFrame({"x": ["2024-01-01", "2024-01-02", "2024-01-03"], "y": [1.0, np.nan, 3.0]})
    matrix, kept = cl.numeric_matrix(frame, ["x", "y"])
    assert kept.tolist() == [True, False, True] and matrix.shape == (2, 2)
    assert matrix[1, 0] - matrix[0, 0] == pytest.approx(2 * 86400.0)


@pytest.mark.parametrize(
    ("frame", "columns", "match"),
    [(pd.DataFrame(), ["x"], "no rows"), (pd.DataFrame({"x": [1.0, 2.0]}), ["y"], "Missing"),
     (pd.DataFrame({"x": [1.0, np.nan]}), ["x"], "two finite rows")],
)
def test_a_matrix_that_cannot_be_clustered_is_a_value_error(frame, columns, match) -> None:
    with pytest.raises(ValueError, match=match):
        cl.numeric_matrix(frame, columns)


def test_a_run_past_its_time_limit_is_a_timeout_error() -> None:
    import time

    def slow() -> tuple[np.ndarray, dict]:
        time.sleep(1.0)
        return np.zeros(1), {}

    with pytest.raises(TimeoutError, match="timeout"):
        cl.run_with_timeout(0.05, slow)
    assert cl.run_with_timeout(0.0, lambda: (np.ones(1), {}))[0].tolist() == [1.0]
