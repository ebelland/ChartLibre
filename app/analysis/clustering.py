"""Clustering: the arithmetic behind the Clustering series operation.

SciPy's k-means and vector quantisation, its hierarchical linkages, and
scikit-learn's algorithms (k-means, DBSCAN, OPTICS, mean shift, spectral,
agglomerative, BIRCH, Gaussian mixtures...) over a feature matrix built from a
series' columns, with stable 1-based cluster ids and an optional time limit.
No Qt, no repository, nothing logged (todo R-01): a request the data cannot
meet is a ``ValueError``, a run past its time limit a ``TimeoutError``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster import vq

from app.utils.coercion import to_numeric_axis

TOOL_WHITEN: str = "vq.whiten + kmeans2"
TOOL_VQ: str = "vq.vq"
TOOL_KMEANS: str = "vq.kmeans"
TOOL_KMEANS2: str = "vq.kmeans2"
TOOL_FCLUSTER: str = "hierarchy.fcluster"
TOOL_FCLUSTERDATA: str = "hierarchy.fclusterdata"
TOOL_LEADERS: str = "hierarchy.leaders"

SKLEARN_KMEANS: str = "sklearn.KMeans"
SKLEARN_MINIBATCH_KMEANS: str = "sklearn.MiniBatchKMeans"
SKLEARN_BISECTING_KMEANS: str = "sklearn.BisectingKMeans"
SKLEARN_AGGLOMERATIVE: str = "sklearn.AgglomerativeClustering"
SKLEARN_DBSCAN: str = "sklearn.DBSCAN"
SKLEARN_OPTICS: str = "sklearn.OPTICS"
SKLEARN_BIRCH: str = "sklearn.Birch"
SKLEARN_MEANSHIFT: str = "sklearn.MeanShift"
SKLEARN_SPECTRAL: str = "sklearn.SpectralClustering"
SKLEARN_GAUSSIAN_MIXTURE: str = "sklearn.GaussianMixture"


def numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return finite feature matrix and retained-row mask."""
    if frame.empty:
        raise ValueError("Selected series query returned no rows.")

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {', '.join(missing)}")

    # to_numeric_axis per column, not a bare pd.to_numeric over all of them:
    # a timestamp feature - and x is the commonest feature there is - would
    # come back all-NaN, and clustering would then refuse a table of a
    # thousand rows with "at least two finite rows are required".
    matrix = np.column_stack(
        [to_numeric_axis(frame[column]) for column in columns]
    ) if list(columns) else np.empty((len(frame), 0), dtype=float)
    mask = np.asarray(np.isfinite(matrix).all(axis=1), dtype=bool).reshape(-1)

    if int(mask.sum()) < 2:
        raise ValueError("At least two finite rows are required for clustering.")

    return matrix[mask], mask


def stable_cluster_ids(labels: np.ndarray) -> np.ndarray:
    """Convert arbitrary clustering labels to stable 1-based cluster ids."""
    raw = np.asarray(labels, dtype=int).reshape(-1)
    unique = sorted(int(value) for value in np.unique(raw))
    mapping = {value: index + 1 for index, value in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in raw], dtype=int)



def run_with_timeout(
    timeout_seconds: float,
    func: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    *args: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a clustering callable with an optional wall-time timeout.

    A timeout stops waiting for the preview/apply operation and returns control
    to the dialog. Python cannot safely kill a running SciPy call inside a
    thread, so the abandoned worker is detached with ``shutdown(wait=False)``.
    """
    timeout = float(timeout_seconds)
    if timeout <= 0.0:
        return func(*args, **kwargs)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datahub-cluster")
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        future.cancel()
        message = (
            f"Clustering exceeded the configured timeout of {timeout:.1f} seconds. "
            "The preview/apply operation was stopped."
        )
        raise TimeoutError(message) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def initial_code_book(obs: np.ndarray, clusters: int) -> np.ndarray:
    """Create a deterministic code book for the raw vq operation."""
    matrix = np.asarray(obs, dtype=float)
    n_rows = matrix.shape[0]
    k = int(np.clip(int(clusters), 1, n_rows))
    indices = np.linspace(0, n_rows - 1, num=k, dtype=int)
    return np.asarray(matrix[indices, :], dtype=float)


def cluster_kmeans(
    features: np.ndarray,
    *,
    scipy_tool: str,
    clusters: int,
    use_whiten: bool,
    iterations: int,
    threshold: float,
    timeout_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster observations with the selected ``scipy.cluster.vq`` function."""

    def _compute() -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("K-means/vector quantization expects a 2D observation matrix.")

        n_rows = matrix.shape[0]
        if n_rows < 2:
            raise ValueError("K-means/vector quantization requires at least two observations.")

        k = int(np.clip(int(clusters), 1, n_rows))
        iterations_safe = max(1, int(iterations))
        threshold_safe = max(float(threshold), 0.0)
        tool = str(scipy_tool or TOOL_KMEANS2)
        whiten_applied = bool(use_whiten or tool == TOOL_WHITEN)
        obs = cast(np.ndarray, vq.whiten(matrix)) if whiten_applied else matrix

        centroids: np.ndarray
        distances: np.ndarray
        distortion: float | None = None

        if tool == TOOL_VQ:
            centroids = initial_code_book(obs, k)
            codes, distances = vq.vq(obs, centroids, check_finite=True)
        elif tool == TOOL_KMEANS:
            raw_centroids, raw_distortion = vq.kmeans(
                obs,
                k,
                iter=iterations_safe,
                thresh=threshold_safe,
                check_finite=True,
            )
            centroids = np.asarray(raw_centroids, dtype=float)
            distortion_values = np.asarray(raw_distortion, dtype=float).reshape(-1)
            distortion = float(distortion_values[0]) if distortion_values.size else 0.0
            codes, distances = vq.vq(obs, centroids, check_finite=True)
        else:
            # TOOL_KMEANS2 and TOOL_WHITEN both classify with kmeans2. TOOL_WHITEN
            # explicitly makes whitening part of the selected SciPy operation.
            centroids, labels = vq.kmeans2(
                obs,
                k,
                iter=iterations_safe,
                thresh=threshold_safe,
                minit="++",
                missing="warn",
                check_finite=True,
            )
            codes = np.asarray(labels, dtype=int)
            _codes_for_distance, distances = vq.vq(obs, centroids, check_finite=True)

        cluster_ids = stable_cluster_ids(np.asarray(codes, dtype=int))
        metadata: dict[str, Any] = {
            "scipy_tool": tool,
            "centroids": np.asarray(centroids, dtype=float).tolist(),
            "distortion": None if distortion is None else float(distortion),
            "mean_distance": float(np.mean(distances)) if distances.size else 0.0,
            "max_distance": float(np.max(distances)) if distances.size else 0.0,
            "clusters_requested": int(clusters),
            "clusters_found": int(np.unique(cluster_ids).size),
            "whiten": whiten_applied,
            "iterations": iterations_safe,
            "threshold": threshold_safe,
        }
        return cluster_ids, metadata

    return run_with_timeout(timeout_seconds, _compute)


def cluster_hierarchical(
    features: np.ndarray,
    *,
    scipy_tool: str,
    clusters: int,
    linkage_method: str,
    metric: str,
    criterion: str,
    distance_threshold: float,
    timeout_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster observations with the selected ``scipy.cluster.hierarchy`` function."""

    def _compute() -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Hierarchical clustering expects a 2D observation matrix.")

        n_rows = matrix.shape[0]
        if n_rows < 2:
            raise ValueError("Hierarchical clustering requires at least two observations.")

        tool = str(scipy_tool or TOOL_FCLUSTER)
        method = str(linkage_method or "ward")
        metric_name = str(metric or "euclidean")
        criterion_name = str(criterion or "maxclust")

        if method == "ward":
            metric_name = "euclidean"

        if criterion_name == "distance":
            threshold = float(distance_threshold)
            if threshold <= 0.0:
                threshold = 1.0
            t_value: float | int = threshold
        elif criterion_name == "inconsistent":
            t_value = max(float(distance_threshold), 1.0)
        else:
            t_value = int(np.clip(int(clusters), 1, n_rows))

        linkage_matrix: np.ndarray | None = None
        leader_nodes: list[int] = []
        leader_cluster_ids: list[int] = []

        if tool == TOOL_FCLUSTERDATA:
            labels = hierarchy.fclusterdata(
                matrix,
                t=t_value,
                criterion=criterion_name,
                metric=metric_name,
                depth=2,
                method=method,
            )
        else:
            linkage = np.asarray(
                hierarchy.linkage(
                    matrix,
                    method=method,
                    metric=metric_name,
                    optimal_ordering=True,
                ),
                dtype=float,
            )
            linkage_matrix = linkage
            if criterion_name == "distance" and float(distance_threshold) <= 0.0:
                t_value = float(np.median(linkage[:, 2])) if linkage.size else 1.0
            labels = hierarchy.fcluster(linkage, t=t_value, criterion=criterion_name)
            if tool == TOOL_LEADERS:
                leaders, leader_ids = hierarchy.leaders(linkage, labels)
                leader_nodes = [int(value) for value in np.asarray(leaders).reshape(-1)]
                leader_cluster_ids = [int(value) for value in np.asarray(leader_ids).reshape(-1)]

        cluster_ids = stable_cluster_ids(np.asarray(labels, dtype=int))
        metadata: dict[str, Any] = {
            "scipy_tool": tool,
            "clusters_requested": int(clusters),
            "clusters_found": int(np.unique(cluster_ids).size),
            "linkage_method": method,
            "metric": metric_name,
            "criterion": criterion_name,
            "distance_threshold": float(distance_threshold),
            "cut_value": float(t_value),
            "linkage_rows": 0 if linkage_matrix is None else int(linkage_matrix.shape[0]),
            "leader_nodes": leader_nodes,
            "leader_cluster_ids": leader_cluster_ids,
        }
        return cluster_ids, metadata

    return run_with_timeout(timeout_seconds, _compute)


def cluster_sklearn(
    features: np.ndarray,
    *,
    sklearn_tool: str,
    clusters: int,
    linkage_method: str,
    metric: str,
    eps: float,
    min_samples: int,
    timeout_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster observations with scikit-learn algorithms when available."""

    def _compute() -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("scikit-learn clustering expects a 2D observation matrix.")
        n_rows = matrix.shape[0]
        if n_rows < 2:
            raise ValueError("scikit-learn clustering requires at least two observations.")

        try:
            from sklearn.cluster import (
                AgglomerativeClustering,
                Birch,
                BisectingKMeans,
                DBSCAN,
                KMeans,
                MeanShift,
                MiniBatchKMeans,
                OPTICS,
                SpectralClustering,
            )
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            raise ValueError(
                "scikit-learn clustering requires the optional dependency 'scikit-learn'. "
                "Install it in the current Python environment."
            ) from exc

        tool = str(sklearn_tool or SKLEARN_KMEANS)
        k = int(np.clip(int(clusters), 1, n_rows))
        metric_name = str(metric or "euclidean")
        linkage = str(linkage_method or "ward")
        eps_value = float(eps) if float(eps) > 0.0 else 0.5
        min_samples_value = max(1, int(min_samples))
        random_state = 0

        if tool == SKLEARN_KMEANS:
            model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"inertia": float(model.inertia_)}
        elif tool == SKLEARN_MINIBATCH_KMEANS:
            model = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"inertia": float(model.inertia_)}
        elif tool == SKLEARN_BISECTING_KMEANS:
            model = BisectingKMeans(n_clusters=k, random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"inertia": float(model.inertia_)}
        elif tool == SKLEARN_AGGLOMERATIVE:
            if linkage == "ward":
                metric_name = "euclidean"
            model = AgglomerativeClustering(n_clusters=k, linkage=cast(Any, linkage), metric=metric_name)
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_DBSCAN:
            model = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric=metric_name)
            labels = model.fit_predict(matrix)
            extra = {"eps": eps_value, "min_samples": min_samples_value}
        elif tool == SKLEARN_OPTICS:
            model = OPTICS(min_samples=min_samples_value, metric=metric_name)
            labels = model.fit_predict(matrix)
            extra = {"min_samples": min_samples_value}
        elif tool == SKLEARN_BIRCH:
            model = Birch(n_clusters=k)
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_MEANSHIFT:
            model = MeanShift()
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_SPECTRAL:
            model = SpectralClustering(n_clusters=k, assign_labels="kmeans", random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_GAUSSIAN_MIXTURE:
            model = GaussianMixture(n_components=k, random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"converged": bool(model.converged_), "lower_bound": float(model.lower_bound_)}
        else:
            raise ValueError(f"Unsupported scikit-learn clustering algorithm: {tool}")

        raw = np.asarray(labels, dtype=int)
        # DBSCAN/OPTICS noise is -1. Keep it as a stable positive cluster id
        # rather than NULL so renderers can color it consistently.
        cluster_ids = stable_cluster_ids(raw)
        metadata: dict[str, Any] = {
            "scipy_tool": tool,
            "sklearn_tool": tool,
            "clusters_requested": int(clusters),
            "clusters_found": int(np.unique(cluster_ids).size),
            "metric": metric_name,
            "linkage_method": linkage,
            **extra,
        }
        return cluster_ids, metadata

    return run_with_timeout(timeout_seconds, _compute)
