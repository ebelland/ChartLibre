"""Cluster Analysis (Analysis): a table's rows grouped on its columns, k chosen by the silhouette (AN-10).

As JMP's Analyze > Clustering - K Means Cluster, Hierarchical Cluster,
Normal Mixtures - on a table's continuous columns rather than on a chart's
series (that is the Clustering data operation). The report gives the table
of k (each k's mean silhouette and within-cluster sum of squares; BIC for
the mixtures) and each cluster's size and means. OK writes each row's
cluster into the table's own ClusterId column (as JMP's Save Clusters, and
as the Clustering operation does), the rows with their Cluster and Silhouette, the table of k and the means, and draws the
silhouette by k, the clusters on the first two principal components, and the
clusters' profiles as a parallel plot. The engine is app.analysis.cluster_table.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import cluster_table as ct
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, IntParam
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


#: The table's own cluster column - the name the Clustering operation writes too, so either fills it.
CLUSTER_COLUMN = "ClusterId"


class ClusteringDialog(TableOperationDialogBase):
    """K Means, hierarchical (Ward) or normal-mixture clusters of the rows."""

    Name = "Cluster Analysis"
    Description = (
        "Group the rows on several columns - K Means, Hierarchical (Ward) or Normal Mixtures - with k "
        "chosen by the silhouette; each cluster's size and means, its rows on the principal components."
    )
    Category = "Multivariate Methods"
    Order = 46
    Icon = """
    <circle cx="6" cy="7" r="1.3"/><circle cx="8.5" cy="5.5" r="1.3"/><circle cx="7.5" cy="9" r="1.3"/>
    <circle cx="16" cy="8" r="1.3"/><circle cx="18" cy="10" r="1.3"/><circle cx="15.5" cy="11" r="1.3"/>
    <circle cx="10" cy="17" r="1.3"/><circle cx="12.5" cy="18.5" r="1.3"/><circle cx="12" cy="15.5" r="1.3"/>
    <circle cx="7.3" cy="7.3" r="4" stroke-dasharray="2 2"/><circle cx="16.5" cy="9.7" r="4" stroke-dasharray="2 2"/>
    """
    Keywords: str = (
        "clustering cluster raggruppamento k means k-means kmeans hierarchical gerarchico ward normal "
        "mixtures miscele normali gaussian mixture silhouette segmentation segmentazione gruppi groups"
    )
    Goals: list[str] = ["groups"]

    ROLES = (
        ColumnRole("response", "Y, columns", "The continuous columns the rows are grouped on.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1),
    )
    PARAMETERS_TAB = "Clusters"
    PARAMS = (
        ChoiceParam("method", "Method:", choices=tuple((label, key) for key, label in ct.METHODS),
                    tooltip="K Means: round clusters of similar size, the usual choice. Hierarchical (Ward): "
                            "merges the closest rows step by step. Normal Mixtures: clusters of any shape "
                            "and spread, each row with its most likely cluster."),
        IntParam("k", "Clusters (k):", tooltip="0 tries every k up to the maximum and keeps the one with the "
                                               "highest mean silhouette.",
                 default_value=0, minimum=0, maximum=50),
        IntParam("k_max", "Largest k tried:", tooltip="With k at 0: k from 2 to this.",
                 default_value=8, minimum=2, maximum=50),
        BoolParam("standardize", "Columns standardized",
                  tooltip="Each column divided by its standard deviation, so that each counts the same "
                          "whatever its unit - leave on unless the columns share one unit and scale.",
                  default_value=True),
        BoolParam("save", "Save ClusterId in the table",
                  tooltip="As JMP's Save Clusters: each row's cluster in the table's own ClusterId column - "
                          "the one the Clustering operation writes - to colour or filter any chart by it; "
                          "0 for a row in no cluster.",
                  default_value=True),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        return ct.cluster_columns(frame, casting.columns("response"), method=str(params.get("method") or ct.KMEANS),
                                  k=int(params.get("k", 0) or 0), k_max=int(params.get("k_max", 8) or 8),
                                  standardize=bool(params.get("standardize", True)))

    def report_subtitle(self, result: ct.ClusterResult) -> str:
        return f"{self.table} · {_(dict(ct.METHODS)[result.method])} · k = {result.k}"

    def report_body(self, result: ct.ClusterResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), _(value) if isinstance(value, str) else report_html.format_number(value))
            for key, value in result.summary.items())))
        blocks.append(report_html.section(_("Cluster means"), self.report_table(result.means, digits=5)))
        blocks.append(report_html.section(_("Choosing k"), self.report_table(result.k_table, digits=4)))
        return "".join(blocks)

    def apply_results(self, result: ct.ClusterResult) -> list[dict[str, Any]]:
        rows = self._frame.copy()
        rows["Cluster"] = result.clusters.reindex(rows.index).astype("Int64")  # whole numbers, not 1.0
        if not result.silhouettes.empty:
            rows["Silhouette"] = result.silhouettes.reindex(rows.index)
        for column in result.projection.columns:
            rows[column] = result.projection[column].reindex(rows.index)
        save = bool(self.parameter_values().get("save", True)) and len(self.row_ids) == len(self._frame)
        tables = self.write_result_tables({"clusters": rows, "clusters_k": result.k_table,
                                           "clusters_means": result.means}, also=[self.table] if save else [])
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]
        if save:
            ids = self.row_ids
            self._repo.set_integer_by_row_id(self.table, CLUSTER_COLUMN, {
                int(ids.iloc[position]): int(cluster) for position, cluster in result.clusters.items()})
            written.append({"table": self.table, "column": CLUSTER_COLUMN})

        sql, roles = self.series_select(tables["clusters_k"], {"x": "k", "y": "Silhouette"})
        panels: list[tuple] = [("Scatter Plot", _("Mean silhouette by k"), "k", _("Silhouette"),
                                [(_("Silhouette"), sql, roles, {"linestyle": "-", "marker": "o"})])]
        scatter = []
        for cluster in range(1, result.k + 1):
            sql, roles = self.series_select(tables["clusters"], {"x": "Prin1", "y": "Prin2"},
                                            where=f'"Cluster" = {cluster}')
            scatter.append((_("Cluster {n}").format(n=cluster), sql, roles, {"marker": "o"}))
        panels.append(("Scatter Plot", _("Clusters on the principal components"), "Prin1", "Prin2", scatter))
        if len(result.columns) >= 2:
            # Each row's profile across the columns, coloured by its cluster.
            shown = result.columns[:6]
            columns = ", ".join(self.quoted(column) for column in shown)
            sql = (f"SELECT {columns}, 'Cluster ' || \"Cluster\" AS \"group\" FROM {self.quoted(tables['clusters'])} "
                   'WHERE "Cluster" IS NOT NULL ORDER BY "Cluster"')
            panels.append(("Parallel Plot", _("Cluster profiles"), "", "", [(_("Rows"), sql, {"group": "group"}, {})]))
        figure_id = self.create_report_figure(f"Cluster Analysis · {self.table}", self.report_document(result), panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": self.table})
        return written
