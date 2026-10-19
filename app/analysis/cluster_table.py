"""Clustering of a table's rows on its columns, k chosen by the silhouette (AN-10).

As JMP's Analyze > Clustering (K Means Cluster, Hierarchical Cluster, Normal
Mixtures): the rows grouped on several continuous columns, standardized so
that each counts the same whatever its unit. With k left at 0 every k from 2
to the maximum is tried and the one with the highest mean silhouette - how
much closer each row is to its own cluster than to the next - is kept; the
table of k says how clear-cut each choice was. Clusters are numbered from the
largest. No Qt, no repository: a request the data cannot meet is a ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.multivariate import _numbers

KMEANS, WARD, MIXTURES = "kmeans", "ward", "mixtures"
METHODS: tuple[tuple[str, str], ...] = (
    (KMEANS, "K Means"),
    (WARD, "Hierarchical (Ward)"),
    (MIXTURES, "Normal Mixtures"),
)
#: Above this many rows the silhouettes are estimated on a sample of them
#: (all-pairs distances: the memory grows with the square of the rows).
SILHOUETTE_SAMPLE: int = 3000


@dataclass
class ClusterResult:
    columns: list[str]
    method: str
    k: int
    summary: dict[str, Any]
    k_table: pd.DataFrame        # k, silhouette, within-cluster SS (and BIC for mixtures), chosen
    means: pd.DataFrame          # Cluster, Count, each column's mean
    clusters: pd.Series          # Cluster of each row used, indexed like the frame
    silhouettes: pd.Series       # each used row's silhouette (empty above SILHOUETTE_SAMPLE rows)
    projection: pd.DataFrame     # Prin1, Prin2 of the standardized columns, for the plot
    notes: list[str] = field(default_factory=list)


def _labels(data: np.ndarray, method: str, k: int) -> tuple[np.ndarray, float | None]:
    """Each row's cluster (0-based) for *k* clusters, and the mixture's BIC."""
    if method == KMEANS:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(data), None
    if method == WARD:
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data), None
    if method == MIXTURES:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=k, n_init=3, random_state=0).fit(data)
        return model.predict(data), float(model.bic(data))
    raise ValueError(f"'{method}' is not one of {', '.join(key for key, _label in METHODS)}")


def _within(data: np.ndarray, labels: np.ndarray) -> float:
    return float(sum(((data[labels == g] - data[labels == g].mean(axis=0)) ** 2).sum() for g in np.unique(labels)))


def _silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    sample = SILHOUETTE_SAMPLE if len(data) > SILHOUETTE_SAMPLE else None
    return float(silhouette_score(data, labels, sample_size=sample, random_state=0))


def cluster_columns(frame: pd.DataFrame, columns: list[str], *, method: str = KMEANS, k: int = 0,
                    k_max: int = 8, standardize: bool = True) -> ClusterResult:
    """The rows of *frame* in *k* clusters on *columns* (k = 0: the best mean silhouette from 2 to *k_max*)."""
    if not columns:
        raise ValueError("choose one column or more")
    if method not in dict(METHODS):
        raise ValueError(f"'{method}' is not one of {', '.join(key for key, _label in METHODS)}")
    data = _numbers(frame, columns).dropna()
    if len(data) < 4:
        raise ValueError("fewer than four rows have a value in every column")
    spread = data.std(ddof=1)
    constant = [c for c in columns if not spread[c] > 0]
    if constant:
        raise ValueError(f"{', '.join(constant)} does not vary: it cannot separate the rows")
    centered = data - data.mean()
    matrix = (centered / spread if standardize else centered).to_numpy(float)
    largest = min(int(k_max), len(data) - 1)
    if k and not 2 <= k <= len(data) - 1:
        raise ValueError(f"k must be between 2 and {len(data) - 1} for {len(data)} rows")
    tried = [k] if k else list(range(2, max(largest, 2) + 1))
    rows = []
    fits: dict[int, np.ndarray] = {}
    for count in tried:
        labels, bic = _labels(matrix, method, count)
        fits[count] = labels
        row = {"k": count, "Silhouette": _silhouette(matrix, labels), "Within SS": _within(matrix, labels)}
        if bic is not None:
            row["BIC"] = bic
        rows.append(row)
    k_table = pd.DataFrame(rows)
    chosen = int(k) if k else int(k_table.loc[k_table["Silhouette"].fillna(-2).idxmax(), "k"])
    k_table["Chosen"] = np.where(k_table["k"] == chosen, "*", "")

    labels = fits[chosen]
    # Numbered from the largest cluster: Cluster 1 is the commonest kind of row.
    sizes = pd.Series(labels).value_counts()
    renumber = {old: new + 1 for new, old in enumerate(sizes.index)}
    clusters = pd.Series([renumber[g] for g in labels], index=data.index, name="Cluster")

    means = data.groupby(clusters.to_numpy()).mean()
    means.insert(0, "Count", clusters.value_counts().sort_index().to_numpy())
    means.insert(0, "Cluster", means.index.astype(int))
    means = means.reset_index(drop=True)

    notes: list[str] = []
    silhouettes = pd.Series(dtype=float, name="Silhouette")
    if len(data) <= SILHOUETTE_SAMPLE:
        from sklearn.metrics import silhouette_samples

        silhouettes = pd.Series(silhouette_samples(matrix, labels), index=data.index, name="Silhouette")
    else:
        notes.append(f"Silhouettes estimated on {SILHOUETTE_SAMPLE} of the {len(data)} rows; none per row.")
    if len(data) < len(frame):
        notes.append(f"{len(frame) - len(data)} row(s) miss a value in some column and are in no cluster.")
    if not k and len(tried) > 1:
        notes.append(f"k = {chosen}: the highest mean silhouette from k = 2 to {tried[-1]}.")

    if matrix.shape[1] >= 2:
        _values, vectors = np.linalg.eigh(np.cov(matrix, rowvar=False))
        top = vectors[:, ::-1][:, :2]
        projection = pd.DataFrame(matrix @ top, index=data.index, columns=["Prin1", "Prin2"])
    else:
        projection = pd.DataFrame({"Prin1": matrix[:, 0], "Prin2": np.zeros(len(matrix))}, index=data.index)

    best = float(k_table.loc[k_table["k"] == chosen, "Silhouette"].iloc[0])
    summary: dict[str, Any] = {"Method": dict(METHODS)[method], "Columns": len(columns), "Rows": len(data),
                               "Clusters": chosen, "Mean silhouette": best}
    return ClusterResult(columns=list(columns), method=method, k=chosen, summary=summary, k_table=k_table,
                         means=means, clusters=clusters, silhouettes=silhouettes, projection=projection,
                         notes=notes)
