"""AN-10 Cluster Analysis: three blobs put in the data come back as three clusters, k found by the silhouette."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app.analysis.cluster_table import KMEANS, METHODS, MIXTURES, WARD, cluster_columns
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.clustering_dialog import ClusteringDialog


def _blobs(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    centres = ((0.0, 0.0), (4.0, 4.0), (8.0, 0.0))
    points = np.vstack([rng.normal(centre, 0.5, (n * (k + 1), 2)) for k, centre in enumerate(centres)])
    frame = pd.DataFrame(points, columns=["width", "height"])
    frame["height"] *= 1000.0  # another unit: standardizing keeps it from leading
    return frame


@pytest.mark.parametrize("method", [KMEANS, WARD, MIXTURES])
def test_three_blobs_are_found_and_numbered_from_the_largest(method: str) -> None:
    result = cluster_columns(_blobs(), ["width", "height"], method=method)
    assert result.k == 3
    assert result.summary["Mean silhouette"] > 0.7
    assert result.means["Count"].tolist() == [120, 80, 40]  # the largest blob is Cluster 1
    assert result.k_table.loc[result.k_table["Chosen"] == "*", "k"].tolist() == [3]
    assert result.clusters.nunique() == 3 and len(result.silhouettes) == 240


def test_a_given_k_and_rows_with_gaps() -> None:
    frame = _blobs()
    frame.loc[0, "width"] = np.nan
    result = cluster_columns(frame, ["width", "height"], k=2)
    assert result.k == 2 and len(result.k_table) == 1
    assert len(result.clusters) == len(frame) - 1 and any("miss a value" in note for note in result.notes)


def test_what_cannot_be_clustered_says_why() -> None:
    frame = _blobs().assign(flat=1.0)
    with pytest.raises(ValueError, match="does not vary"):
        cluster_columns(frame, ["width", "flat"])
    with pytest.raises(ValueError, match="between 2"):
        cluster_columns(frame.head(5), ["width"], k=9)
    assert {key for key, _label in METHODS} == {KMEANS, WARD, MIXTURES}


def test_ok_writes_the_clusters_and_draws_them(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(_blobs(), table_name="parts", normalize_columns=False)
    dialog = ClusteringDialog(repo=repo, table="parts")
    try:
        dialog.roles_widget.set_casting({"response": ["width", "height"]})
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 60
        while dialog._task is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        rows = repo.table_frame("_parts_clusters")
        assert sorted(rows["Cluster"].unique().tolist()) == [1, 2, 3]
        assert {"Silhouette", "Prin1", "Prin2"} <= set(rows.columns)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Scatter Plot", "Scatter Plot", "Parallel Plot"]
    finally:
        dialog.close()
