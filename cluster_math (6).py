"""Pure clustering data preparation and computation.

This module contains no Qt widgets and does not access the repository.  The
cluster dialog materializes its inputs on the GUI thread, then passes the data
objects here so these functions can safely execute on a worker thread.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
import numpy as np
from matplotlib import rcParams

from app.analysis.clustering import (
    cluster_hierarchical,
    cluster_kmeans,
    cluster_sklearn,
    numeric_matrix,
)
from app.data.select_sql import sql_insert_select_expression
from app.logs.logger import applogger
from app.plugin_api.data_operations.results import FrameResult
from app.utils.coercion import to_numbers

CLUSTER_KMEANS: Final[str] = "K-means / vector quantization"
CLUSTER_HIERARCHICAL: Final[str] = "Hierarchical / agglomerative"
CLUSTER_SKLEARN: Final[str] = "scikit-learn"

CLUSTER_COLUMN: Final[str] = "ClusterId"
CLUSTER_BACKUP_COLUMN: Final[str] = "__ClusterId_preview_backup__"
RENDER_SEPARATE_SERIES: Final[str] = "Separate series per cluster"


@dataclass(slots=True)
class ClusterSeriesChoice:
    """Materialized source series used as one clustering input."""

    name: str
    frame: pd.DataFrame
    x_col: str
    y_col: str
    z_col: str | None
    roles: dict[str, Any]
    source_table: str
    source_x_column: str
    source_sql_query: str
    source: Any | None = None


@dataclass(slots=True)
class ClusterResult(FrameResult):
    """Clustering output for one source series or one split cluster series."""

    source_name: str
    result_name: str
    method: str
    frame: pd.DataFrame
    x_col: str
    y_col: str
    z_col: str | None
    feature_columns: list[str]
    metadata: dict[str, Any]
    summary: dict[str, Any]
    means: pd.DataFrame
    k_table: pd.DataFrame
    silhouettes: pd.Series
    projection: pd.DataFrame
    notes: list[str]

    def preview(self, dialog: Any, axis_id: int) -> None:
        dialog.apply_results_to_axis(axis_id, [self])

    def apply(self, dialog: Any, axis_id: int) -> None:
        dialog.apply_results_to_axis(axis_id, [self])


def source_table_from_sql(sql_query: str) -> str:
    """Resolve the source table referenced by a simple series SELECT."""
    match = re.search(
        r'\bFROM\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))',
        str(sql_query),
        flags=re.IGNORECASE,
    )
    if match is not None:
        return next((str(group) for group in match.groups() if group), "")
    applogger.error(
        "Selected series SQL must contain the source table in a FROM clause.",
        raise_error=True,
    )
    return ""


def source_column_for_alias(sql_query: str, alias: str) -> str:
    """Resolve the source field projected under *alias* by a simple SELECT."""
    quoted_alias = re.escape(str(alias))
    alias_pattern = (
        r'"' + quoted_alias + r'"'
        r'|\[' + quoted_alias + r'\]'
        r'|`' + quoted_alias + r'`'
        r'|' + quoted_alias + r'(?=\s|,|$)'
    )
    pattern = (
        r'(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))'
        r'\s+AS\s+(?:' + alias_pattern + r')'
    )
    match = re.search(pattern, str(sql_query), flags=re.IGNORECASE)
    if match is not None:
        return next((str(group) for group in match.groups() if group), "")
    # Generated table-series queries can expose their real role column without
    # an ``AS x`` alias.  The dialog passes that resolved role as *alias*.
    return str(alias)


def sql_with_clusterid_color(sql_query: str) -> str:
    """Expose ClusterId through the scatter color role."""
    return sql_insert_select_expression(str(sql_query), '"ClusterId" AS "color"')


def sql_with_cluster_filter(sql_query: str, cluster_id: int) -> str:
    """Add a ClusterId predicate while preserving trailing SQL clauses."""
    sql = str(sql_query).strip().rstrip(";")
    cluster_clause = f'"ClusterId"={int(cluster_id)}'
    boundary = re.search(
        r"\b(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b",
        sql,
        flags=re.IGNORECASE,
    )
    if boundary is None:
        head, tail = sql, ""
    else:
        head = sql[: boundary.start()].rstrip()
        tail = " " + sql[boundary.start():].lstrip()
    conjunction = " AND " if re.search(r"\bWHERE\b", head, flags=re.IGNORECASE) else " WHERE "
    return f"{head}{conjunction}{cluster_clause}{tail}"


def rc_cycle_color(cluster_id: int) -> str:
    """Return a stable plotting-cycle color for a one-based cluster id."""
    colors = rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return str(colors[(int(cluster_id) - 1) % len(colors)])


def _quality(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Internal validation metrics shared by SciPy and scikit-learn models."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    valid = len(unique) >= 2 and len(labels) > len(unique)
    within_ss = 0.0
    for cluster_id in unique:
        points = features[labels == cluster_id]
        if len(points):
            within_ss += float(np.sum((points - np.mean(points, axis=0)) ** 2))
    return {
        "Silhouette": float(silhouette_score(features, labels)) if valid else np.nan,
        "Within SS": within_ss,
        "Calinski-Harabasz": float(calinski_harabasz_score(features, labels)) if valid else np.nan,
        "Davies-Bouldin": float(davies_bouldin_score(features, labels)) if valid else np.nan,
    }


def _run_model(features: np.ndarray, params: Mapping[str, Any], clusters: int) -> tuple[np.ndarray, dict[str, Any]]:
    method = str(params["method"])
    if method == CLUSTER_KMEANS:
        return cluster_kmeans(features, scipy_tool=str(params["scipy_tool"]), clusters=clusters,
            use_whiten=bool(params["whiten"]), iterations=int(params["kmeans_iter"]),
            threshold=float(params["kmeans_thresh"]),
            timeout_seconds=float(params.get("max_runtime_seconds", 0.0)))
    if method == CLUSTER_HIERARCHICAL:
        return cluster_hierarchical(features, scipy_tool=str(params["scipy_tool"]), clusters=clusters,
            linkage_method=str(params["linkage_method"]), metric=str(params["metric"]),
            criterion=str(params["criterion"]), distance_threshold=float(params["distance_threshold"]),
            timeout_seconds=float(params.get("max_runtime_seconds", 0.0)))
    return cluster_sklearn(features, sklearn_tool=str(params["scipy_tool"]), clusters=clusters,
        linkage_method=str(params["linkage_method"]), metric=str(params["metric"]),
        eps=float(params.get("sklearn_eps", 0.5)), min_samples=int(params.get("sklearn_min_samples", 5)),
        timeout_seconds=float(params.get("max_runtime_seconds", 0.0)))


def _supports_k_sweep(params: Mapping[str, Any]) -> bool:
    tool = str(params.get("scipy_tool", "")).lower()
    return not any(name in tool for name in ("dbscan", "optics", "meanshift"))


def cluster_one(series: ClusterSeriesChoice, feature_columns: list[str], axis_name: str,
                params: Mapping[str, Any], figure_id: int, source_series_id: int | None) -> ClusterResult:
    """Cluster one series, including optional silhouette-based automatic k selection."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples
    features_raw, finite_mask_raw = numeric_matrix(series.frame, feature_columns)
    features = np.asarray(features_raw, dtype=np.float64)
    finite_mask = np.asarray(finite_mask_raw, dtype=np.bool_)
    requested = int(params.get("clusters", 4))
    auto = bool(params.get("automatic_k", False))
    k_max = min(int(params.get("k_max", 8)), max(2, len(features) - 1))
    candidates = range(2, k_max + 1) if auto and _supports_k_sweep(params) else (requested,)
    trials: list[tuple[int, np.ndarray, dict[str, Any], dict[str, float]]] = []
    errors: list[str] = []
    for k in candidates:
        try:
            labels, metadata = _run_model(features, params, int(k))
            labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
            trials.append((
                int(k),
                labels_array,
                metadata,
                _quality(features, labels_array),
            ))
        except Exception as exc:
            errors.append(f"k={k}: {exc}")
    if not trials:
        raise ValueError("; ".join(errors) or "No clustering candidate succeeded")
    chosen = max(trials, key=lambda item: (-np.inf if np.isnan(item[3]["Silhouette"]) else item[3]["Silhouette"]))
    chosen_k, cluster_ids, metadata, quality = chosen
    # Preserve scikit-learn estimator attributes under their exact public names
    # (trailing underscores included) and do not conflate them with the common
    # validation metrics above.
    k_rows: list[dict[str, Any]] = []
    for k, _labels, trial_meta, metrics in trials:
        row: dict[str, Any] = {"k": k, **metrics, "Chosen": k == chosen_k}
        exact = trial_meta.get("estimator_diagnostics", {})
        if isinstance(exact, Mapping):
            for key, value in exact.items():
                if isinstance(value, (int, float, np.integer, np.floating, bool, str)):
                    row[str(key)] = value
        k_rows.append(row)
    k_table = pd.DataFrame(k_rows)
    output = series.frame.copy()
    output[CLUSTER_COLUMN] = pd.Series([pd.NA] * len(output), dtype="Int64")
    output.loc[finite_mask, CLUSTER_COLUMN] = cluster_ids
    feature_frame = pd.DataFrame(features, columns=list(feature_columns))
    feature_frame[CLUSTER_COLUMN] = cluster_ids
    grouped = feature_frame.groupby(by=CLUSTER_COLUMN, as_index=False)
    means: pd.DataFrame = grouped[list(feature_columns)].mean(numeric_only=True)
    means = means.rename(columns={CLUSTER_COLUMN: "Cluster"})
    sizes: pd.DataFrame = feature_frame.groupby(
        by=CLUSTER_COLUMN,
        as_index=False,
    ).size()
    sizes = sizes.rename(columns={CLUSTER_COLUMN: "Cluster", "size": "Rows"})

    cluster_ids_array = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    unique_cluster_ids = np.unique(cluster_ids_array)
    finite_positions = np.flatnonzero(finite_mask)
    silhouettes = pd.Series(np.nan, index=output.index, dtype=float)
    if len(unique_cluster_ids) >= 2 and len(cluster_ids_array) > len(unique_cluster_ids):
        silhouette_values = np.asarray(
            silhouette_samples(features, cluster_ids_array),
            dtype=np.float64,
        )
        silhouettes.iloc[finite_positions] = silhouette_values
    n_components = 2 if features.shape[1] >= 2 else 1
    projected = PCA(n_components=n_components).fit_transform(features)
    if n_components == 1:
        projected = np.column_stack((projected[:, 0], np.zeros(len(projected))))
    projection = pd.DataFrame(
        np.nan,
        index=output.index,
        columns=("Prin1", "Prin2"),
    )
    projection.iloc[
        finite_positions,
        projection.columns.get_indexer(("Prin1", "Prin2")),
    ] = np.asarray(projected[:, :2], dtype=np.float64)
    notes = errors
    if auto and not _supports_k_sweep(params):
        notes.append("Automatic k is not applicable to this algorithm; its native cluster selection was used.")
    summary = {"Method": str(params["method"]), "Tool": str(params["scipy_tool"]),
               "Clusters": int(len(unique_cluster_ids)), "Selected k": chosen_k,
               "Rows clustered": int(len(cluster_ids)), **quality}
    metadata.update({**dict(params), "figure_id": figure_id, "axis_name": axis_name,
        "source_series_id": source_series_id, "source_series_name": series.name,
        "source_table": series.source_table, "source_x_column": series.source_x_column,
        "source_sql_query": series.source_sql_query, "feature_columns": feature_columns,
        "finite_rows": int(np.count_nonzero(finite_mask)), "total_rows": int(len(output)), "roles": series.roles,
        "base_result_name": f"{series.name} - clusters", "is_split_series": False,
        "cluster_sizes": sizes})
    return ClusterResult(source_name=series.name, result_name=f"{series.name} - clusters",
        method=str(params["method"]), frame=output, x_col=series.x_col, y_col=series.y_col,
        z_col=series.z_col, feature_columns=feature_columns, metadata=metadata, summary=summary,
        means=means, k_table=k_table, silhouettes=silhouettes, projection=projection, notes=notes)


def split_result_by_cluster(result: ClusterResult) -> list[ClusterResult]:
    """Represent one clustered frame as a result descriptor per cluster."""
    numeric_ids: pd.Series = pd.Series(
        pd.to_numeric(result.frame[CLUSTER_COLUMN], errors="coerce"),
        dtype="Float64",
    ).dropna()
    cluster_ids = sorted({
        int(value) for value in numeric_ids.to_numpy(dtype=float)
    })
    results: list[ClusterResult] = []
    for cluster_id in cluster_ids:
        metadata = dict(result.metadata)
        metadata.update(
            {
                "cluster_id": cluster_id,
                "is_split_series": True,
                "render_mode": RENDER_SEPARATE_SERIES,
            }
        )
        results.append(
            ClusterResult(
                source_name=result.source_name,
                result_name=f"{result.source_name} - Cluster {cluster_id}",
                method=result.method,
                frame=result.frame,
                x_col=result.x_col,
                y_col=result.y_col,
                z_col=result.z_col,
                feature_columns=list(result.feature_columns),
                metadata=metadata,
                summary=dict(result.summary), means=result.means, k_table=result.k_table,
                silhouettes=result.silhouettes, projection=result.projection, notes=list(result.notes),
            )
        )
    return results
