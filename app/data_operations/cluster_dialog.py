"""Chart-series clustering dialog.

Drop-in companion for the shared ``DataOperationDialogBase`` workflow.

Features
--------
- Select one chart axis and one or more source series.
- Cluster selected source data with ``scipy.cluster``:
  - K-means / vector quantization via ``scipy.cluster.vq``
  - Hierarchical / agglomerative clustering via ``scipy.cluster.hierarchy``
- Add a ``ClusterId`` column to the generated dataset.
- Render either:
  - one series colored by ``ClusterId`` through the Color role; or
  - separate chart series per cluster using ``WHERE ClusterId = x`` queries.
- Fill the results pane with a plain-text preview summary.

Notes
-----
This module reuses the shared DataOperationDialogBase preview/apply flow.
Clustering overrides only the axis-application step because it updates source
table ClusterId values and either recolors the selected series or creates split
series per cluster.

The report (app.analysis.cluster_series) gives each series its summary - the
silhouette, Calinski-Harabasz and Davies-Bouldin scores -, a row per cluster
with its size, silhouette and means, and the k tried when k is chosen
automatically. Beside the clusters on their own chart, a figure of its own
shows them on the first two principal components, the silhouette and the
means by cluster and, with automatic k, the silhouette of each k tried.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QWidget,
)

from app.analysis.cluster_series import (
    CLUSTER_BACKUP_COLUMN,
    CLUSTER_COLUMN,
    CLUSTER_HIERARCHICAL,
    CLUSTER_KMEANS,
    CLUSTER_SKLEARN,
    RENDER_SEPARATE_SERIES,
    ClusterResult,
    ClusterSeriesChoice,
    cluster_one,
    cluster_table,
    rc_cycle_color,
    source_column_for_alias,
    source_table_from_sql,
    split_result_by_cluster,
    sql_with_cluster_filter,
    sql_with_clusterid_color,
)
from app.analysis.clustering import (
    SKLEARN_AGGLOMERATIVE,
    SKLEARN_BIRCH,
    SKLEARN_BISECTING_KMEANS,
    SKLEARN_DBSCAN,
    SKLEARN_GAUSSIAN_MIXTURE,
    SKLEARN_KMEANS,
    SKLEARN_MEANSHIFT,
    SKLEARN_MINIBATCH_KMEANS,
    SKLEARN_OPTICS,
    SKLEARN_SPECTRAL,
    TOOL_FCLUSTER,
    TOOL_FCLUSTERDATA,
    TOOL_KMEANS,
    TOOL_KMEANS2,
    TOOL_LEADERS,
    TOOL_VQ,
    TOOL_WHITEN,
)
from app.data.data_source import parse_roles, resolve_role_column
from app.logs.logger import applogger
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import (
    DataOperationDialogBase,
    OperationModel,
    ResultSeriesSpec,
    SeriesJob,
    SeriesOutcome,
)
from app.plugin_api.data_operations.next_steps import NextStep
from app.plugin_api.data_operations.parameter_spec import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
)
from app.utils import report_html
from app.utils.coercion import to_numbers
from app.utils.i18n import _
from app.utils.messages import show_message

FEATURE_XY: Final[str] = "X and Y"
FEATURE_Y_ONLY: Final[str] = "Y only"
FEATURE_XYZ: Final[str] = "X, Y and Z"
FEATURE_ALL_NUMERIC: Final[str] = "All numeric columns"

RENDER_COLORED_SERIES: Final[str] = "Single series colored by ClusterId"

# The tool names are the engine's (app.analysis.clustering).

HIERARCHY_METHODS: Final[tuple[str, ...]] = (
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
)

HIERARCHY_METRICS: Final[tuple[str, ...]] = (
    "euclidean",
    "cityblock",
    "cosine",
    "correlation",
    "chebyshev",
    "minkowski",
)

HIERARCHY_CRITERIA: Final[tuple[str, ...]] = (
    "maxclust",
    "distance",
    "inconsistent",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterMethod(OperationModel):
    #: The functions of this family, in the Function combo's order, each
    #: with its own documentation page.
    tools: Mapping[str, str]


#: The clustering families, in the Method combo's order.
CLUSTER_METHODS: dict[str, ClusterMethod] = {
    CLUSTER_KMEANS: ClusterMethod(
        doc_title="SciPy k-means / vector quantization",
        doc_url="https://docs.scipy.org/doc/scipy/reference/cluster.vq.html",
        tools={
            TOOL_WHITEN: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html",
            TOOL_VQ: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html",
            TOOL_KMEANS: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html",
            TOOL_KMEANS2: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html",
        },
    ),
    CLUSTER_HIERARCHICAL: ClusterMethod(
        doc_title="SciPy hierarchical clustering",
        doc_url="https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html",
        tools={
            TOOL_FCLUSTER: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html",
            TOOL_FCLUSTERDATA: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html",
            TOOL_LEADERS: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaders.html",
        },
    ),
    CLUSTER_SKLEARN: ClusterMethod(
        doc_title="scikit-learn clustering",
        doc_url="https://scikit-learn.org/stable/modules/clustering.html",
        tools={
            SKLEARN_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
            SKLEARN_MINIBATCH_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html",
            SKLEARN_BISECTING_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html",
            SKLEARN_AGGLOMERATIVE: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
            SKLEARN_DBSCAN: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
            SKLEARN_OPTICS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
            SKLEARN_BIRCH: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html",
            SKLEARN_MEANSHIFT: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html",
            SKLEARN_SPECTRAL: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
            SKLEARN_GAUSSIAN_MIXTURE: "https://scikit-learn.org/stable/modules/mixture.html",
        },
    ),
}
#: Every function's documentation, whatever its family.
TOOL_DOCS: dict[str, str] = {
    tool: url for method in CLUSTER_METHODS.values() for tool, url in method.tools.items()
}


#: The charts of the report figure, each a box in the Charts tab.
CHART_PROJECTION: Final[str] = "chart_projection"
CHART_SILHOUETTE: Final[str] = "chart_silhouette"
CHART_MEANS: Final[str] = "chart_means"
CHART_K: Final[str] = "chart_k"
CHART_PROFILES: Final[str] = "chart_profiles"
#: The columns a profile draws at most: past these, the parallel plot is a blur.
MAX_PROFILE_COLUMNS: Final[int] = 6
#: The Bar Chart reads its columns by their role names: the categories as X, the heights as Y.
BAR_ROLES: Final[dict[str, str]] = {"X": "X", "Y": "Y"}
#: More clusters than this are not drawn as bars (DBSCAN can find hundreds): the table still lists them.
MAX_BARS: Final[int] = 60


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _number(value: Any) -> str:
    """A value as SQL: a finite number, else NULL."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NULL"
    return repr(number) if np.isfinite(number) else "NULL"


def _values_query(rows: Sequence[tuple[Any, Any]], *, category: bool) -> str:
    """The points as a query - bars of categories (X, Y) when *category*, else points (x, y)."""
    def x_sql(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'" if category else _number(value)

    x_name, y_name = ("X", "Y") if category else ("x", "y")
    return " UNION ALL ".join(f"SELECT {x_sql(x)} AS {x_name}, {_number(y)} AS {y_name}" for x, y in rows)


def _extra_axes(values: Mapping[str, Any]) -> bool:
    """The report's charts are offered for Same figure and New figure: Same axis draws the clusters only."""
    return values.get("_destination") != DataOperationDialogBase.DEST_SAME_AXIS


def _kmeans(values: Mapping[str, Any]) -> bool:
    return values.get("method") == CLUSTER_KMEANS


def _hierarchical(values: Mapping[str, Any]) -> bool:
    return values.get("method") == CLUSTER_HIERARCHICAL


#: The settings, each shown for the family and the algorithm that use it.
CLUSTER_PARAMS = (
    IntParam("clusters", "Clusters:", default_value=4, minimum=1, maximum=10_000,
             tooltip="Number of clusters to request."),
    BoolParam("automatic_k", "Choose k automatically", default_value=False,
              tooltip="Try every k from 2 to the largest and keep the one with the best mean silhouette. "
                      "DBSCAN, OPTICS and MeanShift find their own number of clusters."),
    IntParam("k_max", "Largest k:", default_value=8, minimum=2, maximum=50,
             visible_if=lambda v: bool(v.get("automatic_k")),
             tooltip="The largest number of clusters tried."),
    BoolParam("standardize", "Columns standardized", default_value=True,
              tooltip="Each feature divided by its standard deviation, so that each counts the same whatever "
                      "its unit - leave on unless the features share one unit and scale."),
    BoolParam("whiten", "Whiten features", default_value=True,
              visible_if=lambda v: _kmeans(v) and v.get("tool") in (TOOL_WHITEN, TOOL_KMEANS, TOOL_KMEANS2),
              tooltip="Normalize features by standard deviation before k-means."),
    IntParam("kmeans_iter", "K-means iterations:", default_value=50, minimum=1, maximum=10_000, visible_if=_kmeans,
             tooltip="Maximum k-means iterations."),
    FloatParam("kmeans_thresh", "K-means threshold:", default_value=1e-5, minimum=0.0, maximum=1.0, decimals=8,
               step=0.001, visible_if=_kmeans, tooltip="K-means convergence threshold."),
    FloatParam("max_runtime_seconds", "Max runtime:", default_value=15.0, minimum=0.0, maximum=3600.0, decimals=1,
               step=1.0, special_text="no timeout",
               tooltip="Maximum seconds to wait for clustering. Use 0 for no timeout."),
    ChoiceParam("linkage_method", "Linkage method:", choices=tuple(HIERARCHY_METHODS), default_value="ward",
                visible_if=lambda v: _hierarchical(v) or v.get("tool") == SKLEARN_AGGLOMERATIVE),
    ChoiceParam("metric", "Distance metric:", choices=tuple(HIERARCHY_METRICS), default_value="euclidean",
                visible_if=lambda v: _hierarchical(v) or v.get("tool") in (
                    SKLEARN_AGGLOMERATIVE, SKLEARN_DBSCAN, SKLEARN_OPTICS)),
    ChoiceParam("criterion", "Cut criterion:", choices=tuple(HIERARCHY_CRITERIA), default_value="maxclust",
                visible_if=_hierarchical),
    FloatParam("distance_threshold", "Cut threshold:", default_value=0.0, minimum=0.0, maximum=1e18, decimals=6,
               step=0.001, special_text="auto", visible_if=_hierarchical,
               tooltip="Cut threshold for distance/inconsistent criteria. Leave at auto for median linkage "
                       "distance."),
    FloatParam("sklearn_eps", "Neighborhood radius:", default_value=0.5, minimum=0.0, maximum=1e18, decimals=6,
               step=0.001, special_text="auto", visible_if=lambda v: v.get("tool") == SKLEARN_DBSCAN,
               tooltip="Neighborhood radius for DBSCAN. 0 uses the default 0.5."),
    IntParam("sklearn_min_samples", "Min samples:", default_value=5, minimum=1, maximum=1_000_000,
             visible_if=lambda v: v.get("tool") in (SKLEARN_DBSCAN, SKLEARN_OPTICS),
             tooltip="Minimum samples for DBSCAN/OPTICS core points."),
    BoolParam(CHART_PROJECTION, "Clusters on two principal components", default_value=True, tab="charts",
              visible_if=_extra_axes,
              tooltip="The rows on the first two principal components of the features, coloured by cluster."),
    BoolParam(CHART_SILHOUETTE, "Silhouette by cluster", default_value=True, tab="charts",
              visible_if=_extra_axes,
              tooltip="Each cluster's mean silhouette: near 1 well apart, near 0 on a border, below 0 misplaced."),
    BoolParam(CHART_MEANS, "Cluster means", default_value=True, tab="charts",
              visible_if=_extra_axes,
              tooltip="Each feature's mean in each cluster."),
    BoolParam(CHART_K, "Choosing k", default_value=True, tab="charts",
              visible_if=_extra_axes,
              tooltip="With automatic k: the mean silhouette of each k tried."),
    BoolParam(CHART_PROFILES, "Cluster profiles", default_value=True, tab="charts",
              visible_if=_extra_axes,
              tooltip="Each row across the features as a parallel plot, coloured by its cluster."),
)


class SeriesClusterDialog(DataOperationDialogBase):
    """Dialog that clusters selected chart-series data and colors/splits clusters."""
    Name: str  = "Clustering"
    Description = (
        "Group similar rows - k-means, hierarchical (Ward), normal mixtures, DBSCAN and more - with k chosen by "
        "the silhouette; each cluster's size and means, its rows on the principal components."
    )

    # Clustering reads an observation matrix, not a function of x, so order and
    # repeated x carry no meaning here and must not be rejected: two samples
    # with the same x are an ordinary pair of observations. Only the universal
    # empty/length/non-finite checks apply.
    INPUT_MINIMUM_POINTS = 2

    #: Computed on a worker thread: see DataOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    Icon = """
    <circle cx="7" cy="8" r="2"/>
    <circle cx="16.5" cy="7" r="2"/>
    <circle cx="10" cy="16" r="2"/>
    <circle cx="18" cy="15.5" r="2"/>
    <path d="M8.8 9.5l5.9 4.8"/>
    <path d="M15 8.3l-3.6 6.1"/>
    <path d="M12 16h4"/>
    """
    Keywords: str = (
        "cluster clustering raggruppamento gruppi groups k-means kmeans k means dbscan hierarchical gerarchico "
        "ward normal mixtures miscele normali gaussian mixture silhouette segmentation segmentazione"
    )
    Goals: list[str] = ["groups"]
    Category: str = "Multivariate Methods"
    Order: int = 45

    PARAMS = CLUSTER_PARAMS
    #: Same axis: the clusters on the series' own axis, nothing else. Same figure and New figure add the
    #: report's charts ticked in the Charts tab, as new axes of the current figure or in a figure of their own.
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS,
                    DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS

    WINDOW_SIZE = (760, 680)

    def __init__(self, **kwargs: Any) -> None:
        # source table -> did a ClusterId column exist before the preview.
        self._cluster_snapshots: dict[str, bool] = {}
        # series id -> its SQL before the preview rewrote it.
        self._series_sql_snapshots: dict[int, str] = {}
        super().__init__(**kwargs)
        # Same figure adds the report's charts to the figure the series are on.
        self.current_figure_id = self._figure_id

    def build_model_selector(self) -> QWidget:
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(list(CLUSTER_METHODS))
        self.method_combo.setToolTip(_("Choose the clustering algorithm family."))

        self.scipy_tool_combo = QComboBox(self)
        self.scipy_tool_combo.setToolTip(_("Choose the exact scipy.cluster function/workflow to use."))

        self.feature_combo = QComboBox(self)
        self.feature_combo.addItems([FEATURE_XY, FEATURE_Y_ONLY, FEATURE_XYZ, FEATURE_ALL_NUMERIC])
        self.feature_combo.setToolTip(_("Choose which numeric columns are used as clustering features."))

        self.render_mode_combo = QComboBox(self)
        self.render_mode_combo.addItems([RENDER_COLORED_SERIES, RENDER_SEPARATE_SERIES])
        self.render_mode_combo.setToolTip(
            _("Choose whether clusters are rendered as one point-colored series or as separate series.")
        )

        return self.model_form_card(
            "clusterModelCard",
            [
                (_("Family:"), self.method_combo),
                (_("Algorithm:"), self.scipy_tool_combo),
                (_("Features:"), self.feature_combo),
                (_("Render as:"), self.render_mode_combo),
            ],
        )

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(lambda *_args: self.mark_results_stale())
        self.series_selector.axis_changed.connect(lambda *_args: self.mark_results_stale())

        self.method_combo.currentIndexChanged.connect(self._refresh_tools)
        self.method_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.method_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.scipy_tool_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.scipy_tool_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.feature_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.render_mode_combo.currentIndexChanged.connect(self.mark_results_stale)


    def _refresh_tools(self) -> None:
        """Refresh the SciPy function list for the selected clustering family."""
        current = self.scipy_tool_combo.currentText()
        method = self.method_combo.currentText()
        tools = CLUSTER_METHODS.get(method, CLUSTER_METHODS[CLUSTER_SKLEARN]).tools
        self.scipy_tool_combo.blockSignals(True)
        self.scipy_tool_combo.clear()
        self.scipy_tool_combo.addItems(list(tools))
        old_index = self.scipy_tool_combo.findText(current)
        self.scipy_tool_combo.setCurrentIndex(old_index if old_index >= 0 else 0)
        self.scipy_tool_combo.blockSignals(False)
        self._update_description_link()

    def parameter_context(self) -> Mapping[str, Any]:
        """The family and the algorithm chosen: each setting shows for those that use it (CLUSTER_PARAMS)."""
        return {"method": self.method_combo.currentText(), "tool": self.scipy_tool_combo.currentText()}

    def _refresh_visibility(self, *_ignored: Any) -> None:
        if self.scipy_tool_combo.count() == 0:
            self._refresh_tools()
        super()._refresh_visibility()
        self._update_description_link()

    def _update_description_link(self) -> None:
        tool = self.scipy_tool_combo.currentText() if hasattr(self, "scipy_tool_combo") else ""
        title = tool or "SciPy clustering"
        url = TOOL_DOCS.get(tool, "https://scikit-learn.org/stable/modules/clustering.html" if str(tool).startswith("sklearn.") else "https://docs.scipy.org/doc/scipy/reference/cluster.html")
        self.set_doc_link(title, url)

    def _params(self) -> dict[str, Any]:
        params = self.parameter_values()
        params.pop("destination", None)  # the Charts tab's: where, not how
        return {
            "method": self.method_combo.currentText(),
            "scipy_tool": self.scipy_tool_combo.currentText(),
            "feature_mode": self.feature_combo.currentText(),
            "render_mode": self.render_mode_combo.currentText(),
            **params,
        }

    def _series_display_name(self, row: Any) -> str:
        return str(row["name"])

    def _series_choice_from_row(self, row: Any) -> ClusterSeriesChoice:
        name = str(row["name"])
        sql_query = str(row["sql_query"])
        if not sql_query:
            applogger.error("Selected series has no SQL query.", raise_error=True)

        source_table = source_table_from_sql(sql_query)
        source_x_column = source_column_for_alias(sql_query, "x")
        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.", raise_error=True)

        roles = parse_roles(row["roles"])
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = resolve_role_column(columns, roles, "x") or ""
        y_col = resolve_role_column(columns, roles, "y") or ""
        z_col = resolve_role_column(columns, roles, "z") or ""

        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col
        if z_col and z_col not in columns:
            z_col = ""

        # Report only. Cluster labels are assigned back to the frame's rows, so
        # the row order has to survive; and because clustering reads an
        # observation matrix rather than f(x), the only checks that fire here
        # are the universal ones - repeated x is two ordinary observations.
        self.validate_input_xy(
            self.numeric_x(frame[x_col], name),
            self.numeric_y(frame[y_col]),
            label=name,
            raise_on_error=False,
        )

        return ClusterSeriesChoice(
            name=name,
            frame=frame.copy(),
            x_col=x_col,
            y_col=y_col,
            z_col=z_col or None,
            roles=roles,
            source_table=source_table,
            source_x_column=source_x_column,
            source_sql_query=sql_query,
            source=row,
        )

    def _feature_columns(self, series: ClusterSeriesChoice) -> list[str]:
        mode = self.feature_combo.currentText()
        frame = series.frame

        if mode == FEATURE_Y_ONLY:
            return [series.y_col]

        if mode == FEATURE_XYZ:
            z_col = series.z_col or ""
            if not z_col:
                applogger.error(
                    "Selected series has no Z role/column for X, Y and Z clustering.",
                    raise_error=True,
                )
            return [series.x_col, series.y_col, z_col]

        if mode == FEATURE_ALL_NUMERIC:
            excluded = {"ClusterId"}
            numeric = [
                str(column)
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column]) and str(column) not in excluded
            ]
            if not numeric:
                applogger.error("No numeric columns are available for clustering.", raise_error=True)
            return numeric

        return [series.x_col, series.y_col]

    def prepare_job(self, **options: Any) -> SeriesJob | None:
        """Read each series and its feature columns here; the clustering runs in the job."""
        del options
        settings = (self.current_axis_name(), self._params(), self._figure_id)
        selected_rows = self.selected_series()
        if not selected_rows:
            return None

        inputs: list[tuple[str, Any]] = []
        errors: list[str] = []
        for row in selected_rows:
            try:
                series = self._series_choice_from_row(row)
                inputs.append((self._series_display_name(row), (series, self._feature_columns(series))))
            except Exception as exc:  # noqa: BLE001 - collected per series, reported with the others
                errors.append(f"{self._series_display_name(row)}: {exc}")
        return SeriesJob(inputs, settings, self.compute_series, errors)

    def compute_series(
        self,
        name: str,
        data: tuple[ClusterSeriesChoice, list[str]],
        settings: tuple[str, dict[str, Any], int],
    ) -> ClusterResult:
        del name
        series, feature_columns = data
        axis_name, params, figure_id = settings
        return self._cluster_one(series, feature_columns, axis_name, params, figure_id)

    def finish_job(self, job: Any, outcome: Any) -> list[ClusterResult]:
        """The clustered series - split one per cluster when asked; failures reported together."""
        if not isinstance(outcome, SeriesOutcome):
            return list(super().finish_job(job, outcome))
        _axis_name, params, _figure_id = job.settings
        results: list[ClusterResult] = []
        for _name, full_result in outcome.outcomes:
            if params["render_mode"] == RENDER_SEPARATE_SERIES:
                results.extend(split_result_by_cluster(full_result))
            else:
                results.append(full_result)
        errors = outcome.errors

        if errors and not results:
            applogger.error("\n".join(errors), raise_error=True)

        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )

        return results

    def _cluster_one(
        self,
        series: ClusterSeriesChoice,
        feature_columns: list[str],
        axis_name: str,
        params: Mapping[str, Any],
        figure_id: int,
    ) -> ClusterResult:
        """Cluster one series' rows; no widget, no repository - it runs in the background."""
        return cluster_one(series, feature_columns, axis_name, params, figure_id, self._source_series_id(series))

    def _snapshot_cluster_state(self, source_table: str) -> None:
        """Move the existing ClusterId aside before a preview overwrites it.

        Why an explicit column snapshot rather than relying on the preview
        SAVEPOINT: clustering does not create removable preview artifacts, it
        overwrites a column in the user's own table.  A savepoint should cover
        that too, but any repository helper that commits - directly or through
        pandas ``to_sql`` - silently ends the transaction and takes the
        savepoint with it, and the user then finds their data changed after
        pressing Close.  Renaming the column is O(1) metadata, cannot fail on
        size, and restores the original bytes rather than a re-computed
        approximation.
        """
        if source_table in self._cluster_snapshots:
            return

        try:
            had_column = self._repo.snapshot_column(
                source_table, CLUSTER_COLUMN, CLUSTER_BACKUP_COLUMN
            )
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception(
                "Failed to snapshot %s.%s before preview", source_table, CLUSTER_COLUMN
            )
            return

        self._cluster_snapshots[source_table] = had_column

    def _snapshot_series_sql(self, series_id: int) -> None:
        """Remember a series' SQL before the preview rewrites it."""
        if series_id in self._series_sql_snapshots:
            return
        try:
            self._series_sql_snapshots[series_id] = str(
                self._repo.get_series_sql_query(series_id) or ""
            )
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to snapshot SQL for series id=%s", series_id)

    def restore_operation_snapshots(self) -> bool:
        """Undo everything a cluster preview wrote.  Returns True if it did."""
        restored = False

        for source_table in list(self._cluster_snapshots):
            try:
                # Without a ClusterId before the preview there is no backup,
                # and the restore just drops the column the preview added.
                # Not delete_table_column: that records an undo entry, and a
                # preview must never reach the undo history.
                self._repo.restore_column_snapshot(
                    source_table, CLUSTER_COLUMN, CLUSTER_BACKUP_COLUMN
                )
                restored = True
            except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
                applogger.exception(
                    "Failed to restore %s.%s after preview", source_table, CLUSTER_COLUMN
                )
        self._cluster_snapshots.clear()

        for series_id, sql_query in list(self._series_sql_snapshots.items()):
            try:
                self._repo.update_series_sql_query(int(series_id), sql_query)
                restored = True
            except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
                applogger.exception("Failed to restore SQL for series id=%s", series_id)
        self._series_sql_snapshots.clear()

        return restored

    def discard_operation_snapshots(self) -> None:
        """Drop the snapshots after Apply has made the changes permanent."""
        for source_table in list(self._cluster_snapshots):
            try:
                self._repo.discard_column_snapshot(source_table, CLUSTER_BACKUP_COLUMN)
            except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
                applogger.exception(
                    "Failed to discard the %s snapshot on %s",
                    CLUSTER_BACKUP_COLUMN,
                    source_table,
                )
        self._cluster_snapshots.clear()
        self._series_sql_snapshots.clear()

    def _write_cluster_ids_to_source_table(self, result: ClusterResult) -> None:
        source_table = str(result.metadata["source_table"])
        source_x_column = str(result.metadata["source_x_column"])
        self._snapshot_cluster_state(source_table)
        self._repo.ensure_column(source_table, "ClusterId")

        cluster_values = to_numbers(result.frame["ClusterId"])
        x_values = result.frame[result.x_col]
        self._repo.clear_integer_column(source_table, "ClusterId")
        self._repo.set_ClusterId(source_table,source_x_column,x_values,cluster_values)


    def _colored_series_sql_query(self, table_name: str, result: ClusterResult) -> str:
        del table_name
        return sql_with_clusterid_color(str(result.metadata["source_sql_query"]))

    def result_series_spec(self, axis_id: int, table_name: str, result: ClusterResult) -> ResultSeriesSpec:
        del axis_id

        roles = self._base_roles(result)
        roles.update({"ClusterId": "ClusterId", "cluster": "ClusterId"})

        cluster_id = result.metadata.get("cluster_id")
        is_split = bool(result.metadata.get("is_split_series", False))

        if is_split and cluster_id is not None:
            sql_query = sql_with_cluster_filter(
                str(result.metadata["source_sql_query"]),
                int(cluster_id),
            )
            color = rc_cycle_color(int(cluster_id))
            style = {
                "generated_clustering": True,
                "clustering_dialog": "series_clustering",
                "source_name": result.source_name,
                "source_series_id": result.metadata.get("source_series_id"),
                "method": result.method,
                "features": result.feature_columns,
                "cluster_id": int(cluster_id),
                "color": color,
                "marker": "o",
                "linestyle": "",
                "use_point_colors": False,
                "render_mode": RENDER_SEPARATE_SERIES,
            }
            return ResultSeriesSpec(
                name=result.result_name,
                sql_query=sql_query,
                roles=roles,
                style=style,
            )

        sql_query = self._colored_series_sql_query(table_name, result)
        roles.update({"color": "color"})
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=sql_query,
            roles=roles,
            style={
                "generated_clustering": True,
                "clustering_dialog": "series_clustering",
                "source_name": result.source_name,
                "source_series_id": result.metadata.get("source_series_id"),
                "method": result.method,
                "features": result.feature_columns,
                "marker": "o",
                "linestyle": "",
                "use_point_colors": True,
                "color_role": "color",
                "render_mode": RENDER_COLORED_SERIES,
            },
        )


    def _base_roles(self, result: ClusterResult) -> dict[str, Any]:
        roles: dict[str, Any] = dict(result.metadata.get("roles", {}))
        roles.update({"x": result.x_col, "y": result.y_col})
        if result.z_col:
            roles["z"] = result.z_col
        return roles

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_clustering": True, "clustering_dialog": "series_clustering"}

    def format_results(self, results: Sequence[ClusterResult]) -> str:
        """Each clustered series' report, once even when it is split one series per cluster."""
        if not results:
            return ""
        unique = self._unique_report_results(results)
        subtitle = ", ".join(result.source_name for result in unique)
        return self.results_document(subtitle, *(self.report_for(result) for result in unique))

    def report_for(self, result: ClusterResult) -> str:
        """One series' report: the scores, a row per cluster, the k tried, and what to do next."""
        summary = result.summary
        parts = [report_html.note(_(note)) for note in result.notes]
        parts.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), report_html.format_number(value) if isinstance(value, (int, float, np.number))
             else _(str(value))) for key, value in summary.items())))
        parts.append(report_html.section(_("Clusters"), report_html.frame_table(
            cluster_table(result).rename(columns=self.feature_names(result)), _, digits=4)))
        if len(result.k_table) > 1:
            parts.append(report_html.section(_("Choosing k"), report_html.frame_table(result.k_table, _, digits=4)))
        box = ""
        if reading_guide.enabled():
            box = report_html.reading_guide(_("How to read this"), self.reading_for(result))
        rows = summary.get("Rows clustered", len(result.frame))
        body = report_html.section(f"{result.source_name} (N = {rows})",
                                   report_html.with_reading_guide("".join(parts), box))
        return body + self.what_next(result)

    def reading_for(self, result: ClusterResult) -> list[str]:
        """One series' "How to read this" lines, in its own section (the base's box stays empty)."""
        return [reading_guide.silhouette(result.summary.get("Silhouette"), clusters=result.summary.get("Clusters"))]

    @staticmethod
    def feature_names(result: ClusterResult) -> dict[str, str]:
        """Each feature's name in its table: the column a series' "AS x" stands for."""
        query = str(result.metadata.get("source_sql_query") or "")
        return {feature: source_column_for_alias(query, feature) for feature in result.feature_columns}

    def next_steps(self, result: ClusterResult) -> list[NextStep]:
        """Once the clusters are in the table: do they differ in the features they were made from?"""
        table = str(result.metadata.get("source_table") or "")
        query = str(result.metadata.get("source_sql_query") or "")
        if not table:
            return []
        columns = {str(column) for column in self._repo.query_df(f"SELECT * FROM {_quoted(table)} LIMIT 0").columns}
        steps = []
        for feature in result.feature_columns[:2]:
            column = source_column_for_alias(query, feature)
            if column in columns:
                steps.append(NextStep(label=_("Do the clusters differ in {column}?").format(column=column),
                                      operation="Are there differences?", table=table,
                                      roles={"response": [column], "group": [CLUSTER_COLUMN]},
                                      kinds={CLUSTER_COLUMN: "nominal"}))
        return steps

    # ------------------------------------------------------------------
    # The report figure
    # ------------------------------------------------------------------

    def report_panels(self, result: ClusterResult) -> list[tuple]:
        """The report figure's charts that are ticked and have something to show."""
        panels: list[tuple] = []
        clusters = cluster_table(result)
        components = np.asarray(result.pca.get("components", np.empty((0, 0))), dtype=float)
        if self.chart_enabled(CHART_PROJECTION) and len(result.feature_columns) >= 2 and components.shape[0] >= 2:
            mean = np.asarray(result.pca["mean"], dtype=float)
            count = len(result.feature_columns)
            center = np.asarray(result.pca.get("center", np.zeros(count)), dtype=float)
            scale = np.asarray(result.pca.get("scale", np.ones(count)), dtype=float)

            def axis(row: np.ndarray) -> str:
                # The PCA saw each feature standardized: (value - center) / scale, then less its mean.
                return " + ".join(
                    f"(({_quoted(feature)} - {_number(c)}) / {_number(s)} - {_number(m)}) * {_number(weight)}"
                    for feature, c, s, m, weight in zip(result.feature_columns, center, scale, mean, row, strict=True))

            source = sql_with_clusterid_color(str(result.metadata["source_sql_query"]))
            sql = f"SELECT {axis(components[0])} AS x, {axis(components[1])} AS y, color FROM ({source})"
            panels.append(("Scatter Plot", _("Clusters on two principal components"), "Prin1", "Prin2",
                           [(result.source_name, sql, {"x": "x", "y": "y", "color": "color"},
                             {"marker": "o", "linestyle": "", "use_point_colors": True, "color_role": "color"})]))
        if not clusters.empty and len(clusters) <= MAX_BARS:
            if self.chart_enabled(CHART_SILHOUETTE) and clusters["Mean silhouette"].notna().any():
                rows = list(zip(clusters["Cluster"], clusters["Mean silhouette"], strict=True))
                panels.append(("Bar Chart", _("Silhouette by cluster"), _("Cluster"), _("Mean silhouette"),
                               [(_("Silhouette"), _values_query(rows, category=True), BAR_ROLES, {})]))
            features = [feature for feature in result.feature_columns if feature in clusters.columns]
            names = self.feature_names(result)
            if self.chart_enabled(CHART_MEANS) and features:
                series = [(names[feature], _values_query(list(zip(clusters["Cluster"], clusters[feature], strict=True)),
                                                  category=True), BAR_ROLES, {}) for feature in features]
                panels.append(("Bar Chart", _("Cluster means"), _("Cluster"), _("Mean"), series))
        if self.chart_enabled(CHART_PROFILES) and len(result.feature_columns) >= 2:
            # Each row's profile across the features, coloured by its cluster.
            shown = result.feature_columns[:MAX_PROFILE_COLUMNS]
            names = self.feature_names(result)
            source = sql_with_clusterid_color(str(result.metadata["source_sql_query"]))
            columns = ", ".join(f"{_quoted(feature)} AS {_quoted(names.get(feature, feature))}" for feature in shown)
            sql = (f"SELECT {columns}, 'Cluster ' || color AS \"group\" FROM ({source}) "
                   "WHERE color IS NOT NULL ORDER BY color")
            panels.append(("Parallel Plot", _("Cluster profiles"), "", "",
                           [(_("Rows"), sql, {"group": "group"}, {})]))
        if self.chart_enabled(CHART_K) and len(result.k_table) > 1 and "Silhouette" in result.k_table.columns:
            rows = list(zip(result.k_table["k"], result.k_table["Silhouette"], strict=True))
            panels.append(("Scatter Plot", _("Choosing k"), "k", _("Mean silhouette"),
                           [(_("Silhouette"), _values_query(rows, category=False), {"x": "x", "y": "y"},
                             {"marker": "o", "linestyle": "-"})]))
        return panels

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """The clusters stay on their series' axis, whatever "Draw on" says: it places the report's charts."""
        del results
        return selected_axis_id

    def _draw_report_figures(self, results: Sequence[ClusterResult]) -> None:
        """Each series' report charts: new axes of the current figure, or a figure of their own; none on Same axis."""
        if self.draw_on() == self.DEST_SAME_AXIS:
            return
        for result in self._unique_report_results(results):
            panels = self.report_panels(result)
            if panels:
                self.create_report_figure(f"{_(self.Name)} · {result.source_name}",
                                          self.results_document(result.source_name, self.report_for(result)), panels)

    @staticmethod
    def _unique_report_results(results: Sequence[ClusterResult]) -> list[ClusterResult]:
        unique: dict[tuple[str, str], ClusterResult] = {}
        for result in results:
            key = (
                str(result.metadata.get("source_series_id") or result.source_name),
                str(result.metadata.get("base_result_name") or result.result_name),
            )
            unique.setdefault(key, result)
        return list(unique.values())


    def apply_results_to_axis(self, axis_id: int, results: Sequence[ClusterResult]) -> None:
        if self.render_mode_combo.currentText() == RENDER_SEPARATE_SERIES:
            self.remove_previous_generated_series(axis_id)

        updated_tables: set[str] = set()
        for result in results:
            source_table = str(result.metadata["source_table"])
            if source_table not in updated_tables:
                self._write_cluster_ids_to_source_table(result)
                updated_tables.add(source_table)
                applogger.info(f"Updated ClusterId in selected table: {source_table}")

            if self.render_mode_combo.currentText() == RENDER_SEPARATE_SERIES:
                self.create_result_series(axis_id, source_table, result)
                continue

            sql_query = self._colored_series_sql_query(source_table, result)
            source_series_id = result.metadata.get("source_series_id")
            if source_series_id is None:
                applogger.error(
                    "Selected series id is missing; cannot update source series SQL.",
                    raise_error=True,
                )
                continue
            source_series_id_int = int(source_series_id)
            self._snapshot_series_sql(source_series_id_int)
            self._repo.update_series_sql_query(source_series_id_int, sql_query)
            applogger.info(f"Updated selected series SQL for ClusterId color: {result.source_name}")

        # After the ids are in the table: the report's charts read them.
        self._draw_report_figures(results)
        self.applied.emit()

    def preview_results_to_axis(self, axis_id: int, results: Sequence[ClusterResult]) -> None:
        """Preview clustering by applying the cluster result to the active chart.

        Clustering is different from generated-table operations: the chart only
        changes after ClusterId is written to the source table and the selected
        source series SQL is updated or generated split series are created.
        Therefore preview must use the same chart-update path as apply.
        """
        self.apply_results_to_axis(int(axis_id), results)

    def operation_succeeded(self, *, commit: bool) -> None:
        """Tell the owner window, and drop the snapshots once committed.

        Preview goes through the shared pipeline like every other operation:
        reimplementing it bypassed ``cancel_operation_changes`` and the
        preview SAVEPOINT, which is why a cluster preview used to survive
        Close - it writes ClusterId into the user's own source table and
        rewrites the selected series' SQL, and neither of those is a
        removable "preview artifact".
        """
        if commit:
            # Committed: the snapshots are no longer a safety net, just clutter.
            self.discard_operation_snapshots()

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Undo a cluster preview.

        Overridden because clustering's preview is not made of removable
        artifacts: it overwrites ClusterId in the source table and rewrites the
        selected series' SQL.  Both are restored from the snapshots taken when
        the preview ran.
        """
        restored = self.restore_operation_snapshots()
        super().cancel_operation_changes(refresh=refresh and not restored)

        if restored and refresh:
            self._refresh_after_preview_state_change()

    @staticmethod
    def _source_series_id(series: ClusterSeriesChoice) -> int | None:
        source = series.source
        if source is None:
            return None
        try:
            value = source["id"]
        except (KeyError, TypeError, IndexError):
            return None
        return int(value) if value is not None else None
