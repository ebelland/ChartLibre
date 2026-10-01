"""Chart-series clustering dialog.

Drop-in companion for the shared ``SeriesOperationDialogBase`` workflow.

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
This module reuses the shared SeriesOperationDialogBase preview/apply flow.
Clustering overrides only the axis-application step because it updates source
table ClusterId values and either recolors the selected series or creates split
series per cluster.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
from matplotlib import rcParams
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QScrollArea,
    QSpinBox,
    QWidget,
)

from app.analysis.clustering import (
    TOOL_WHITEN,
    TOOL_VQ,
    TOOL_KMEANS,
    TOOL_KMEANS2,
    TOOL_FCLUSTER,
    TOOL_FCLUSTERDATA,
    TOOL_LEADERS,
    SKLEARN_KMEANS,
    SKLEARN_MINIBATCH_KMEANS,
    SKLEARN_BISECTING_KMEANS,
    SKLEARN_AGGLOMERATIVE,
    SKLEARN_DBSCAN,
    SKLEARN_OPTICS,
    SKLEARN_BIRCH,
    SKLEARN_MEANSHIFT,
    SKLEARN_SPECTRAL,
    SKLEARN_GAUSSIAN_MIXTURE,
    cluster_hierarchical,
    cluster_kmeans,
    cluster_sklearn,
    numeric_matrix,
)
from app.data.data_source import parse_roles
from app.data.select_sql import sql_insert_select_expression
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.coercion import to_numbers
from app.utils.messages import show_message
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesJob,
    SeriesOperationDialogBase,
    SeriesOutcome,
)
from app.styles.style import (
    CardFrame,
    stdSizeAndlayout,
)
from app.widgets.axis_series_selector import AxisSeriesSelector
from app.utils.i18n import _


CLUSTER_KMEANS: Final[str] = "K-means / vector quantization"
CLUSTER_HIERARCHICAL: Final[str] = "Hierarchical / agglomerative"
CLUSTER_SKLEARN: Final[str] = "scikit-learn"

FEATURE_XY: Final[str] = "X and Y"
FEATURE_Y_ONLY: Final[str] = "Y only"
FEATURE_XYZ: Final[str] = "X, Y and Z"
FEATURE_ALL_NUMERIC: Final[str] = "All numeric columns"

# Clustering writes into the user's own table, so a preview has to be able to
# put the previous state back byte for byte.  The backup column is renamed
# aside rather than copied: O(1) metadata, and it restores the original values
# rather than a re-computed approximation.
CLUSTER_COLUMN: Final[str] = "ClusterId"
CLUSTER_BACKUP_COLUMN: Final[str] = "__ClusterId_preview_backup__"

RENDER_COLORED_SERIES: Final[str] = "Single series colored by ClusterId"
RENDER_SEPARATE_SERIES: Final[str] = "Separate series per cluster"

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




@dataclass(slots=True)
class ClusterSeriesChoice:
    """Selectable chart series descriptor materialized from a source query."""

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
class ClusterResult(TableResult):
    """Clustering output for one source series or one generated cluster series."""

    source_name: str
    result_name: str
    method: str
    frame: pd.DataFrame
    x_col: str
    y_col: str
    z_col: str | None
    feature_columns: list[str]
    metadata: dict[str, Any]

    def to_df(self) -> pd.DataFrame:
        return self.frame

    # Not a table of its own: clustering writes ClusterId into the source
    # table and recolours or splits the source series, as a batch - see
    # SeriesClusterDialog.apply_results_to_axis, which both of these use.
    def preview(self, dialog: Any, axis_id: int) -> None:
        dialog.apply_results_to_axis(axis_id, [self])

    def apply(self, dialog: Any, axis_id: int) -> None:
        dialog.apply_results_to_axis(axis_id, [self])


def _source_table_from_sql(sql_query: str) -> str:
    match = re.search(
        r'\bFROM\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))',
        str(sql_query),
        flags=re.IGNORECASE,
    )
    if match is None:
        applogger.error(
            "Selected series SQL must contain the source table in a FROM clause.",
            show_dialog=True,
            raise_error=True,
        )
        return ""
    for group in match.groups():
        if group:
            return str(group)
    applogger.error(
        "Selected series source table could not be resolved.",
        show_dialog=True,
        raise_error=True,
    )
    return ""


def _source_column_for_alias(sql_query: str, alias: str) -> str:
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
    if match is None:
        applogger.error(
            f'Selected series SQL must project a source field as "{alias}".',
            show_dialog=True,
            raise_error=True,
        )
        return ""
    for group in match.groups():
        if group:
            return str(group)
    applogger.error(
        f'Source field for alias "{alias}" could not be resolved.',
        show_dialog=True,
        raise_error=True,
    )
    return ""




def _sql_with_clusterid_color(sql_query: str) -> str:
    """Return SQL that exposes ClusterId through the scatter color role."""
    return sql_insert_select_expression(
        str(sql_query),
        '"ClusterId" AS "color"',
    )



def _sql_with_cluster_filter(sql_query: str, cluster_id: int) -> str:
    """Return source SQL filtered to one ClusterId.

    Keeps ORDER BY/GROUP BY/HAVING/LIMIT/OFFSET clauses at the end by
    inserting the ClusterId predicate into the WHERE section.
    """
    sql = str(sql_query).strip().rstrip(";")
    cluster_clause = f'"ClusterId"={int(cluster_id)}'

    boundary = re.search(
        r"\b(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b",
        sql,
        flags=re.IGNORECASE,
    )

    if boundary is None:
        head = sql
        tail = ""
    else:
        head = sql[: boundary.start()].rstrip()
        tail = " " + sql[boundary.start():].lstrip()

    if re.search(r"\bWHERE\b", head, flags=re.IGNORECASE):
        return f"{head} AND {cluster_clause}{tail}"

    return f"{head} WHERE {cluster_clause}{tail}"


def _rc_cycle_color(cluster_id: int) -> str:
    colors = rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return str(colors[(int(cluster_id) - 1) % len(colors)])




class SeriesClusterDialog(SeriesOperationDialogBase):
    """Dialog that clusters selected chart-series data and colors/splits clusters."""
    Name: str  = "Clustering"
    Description = "Group similar data"

    # Clustering reads an observation matrix, not a function of x, so order and
    # repeated x carry no meaning here and must not be rejected: two samples
    # with the same x are an ordinary pair of observations. Only the universal
    # empty/length/non-finite checks apply.
    INPUT_MINIMUM_POINTS = 2

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
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
    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        applied_callback: Callable[[], None] | None = None,
        table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        if repo is None:
            applogger.error("SeriesClusterDialog requires a repository instance.", show_dialog=True, raise_error=True)

        self._repo: Any = repo
        self._figure_id = int(figure_id)
        self._applied_callback = applied_callback
        self._initial_table = table
        self._last_results: list[ClusterResult] = []
        self._field_rows: dict[str, tuple[QWidget, QWidget]] = {}

        # source table -> did a ClusterId column exist before the preview.
        self._cluster_snapshots: dict[str, bool] = {}
        # series id -> its SQL before the preview rewrote it.
        self._series_sql_snapshots: dict[int, str] = {}

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Clustering",
            parent=parent,
            width=760,
            height=680,
        )

        self.model_combo.setVisible(False)
        self._refresh_visibility()
        self.mark_results_stale()

    def create_axis_series_selector(self) -> AxisSeriesSelector:
        return AxisSeriesSelector(self._repo, self._figure_id, self)

    def init_operation_widgets(self) -> None:
        self._create_controls()

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

    def build_parameter_selector(self) -> QWidget:
        settings_widget = CardFrame(self, "clusterParamsCard")
        form_widget = QWidget(settings_widget)
        self.form = QFormLayout(form_widget)
        stdSizeAndlayout(self.form)
        self._add_parameter_rows()
        settings_widget.layout().addWidget(form_widget)

        scroll = QScrollArea(self)
        stdSizeAndlayout(scroll)
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings_widget)
        return scroll

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

        self.cluster_count_spin.valueChanged.connect(self.mark_results_stale)
        self.kmeans_iter_spin.valueChanged.connect(self.mark_results_stale)
        self.kmeans_thresh_spin.valueChanged.connect(self.mark_results_stale)
        self.whiten_check.stateChanged.connect(self.mark_results_stale)
        self.max_runtime_spin.valueChanged.connect(self.mark_results_stale)
        self.linkage_method_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.metric_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.hierarchy_criterion_combo.currentIndexChanged.connect(self.mark_results_stale)
        self.distance_threshold_spin.valueChanged.connect(self.mark_results_stale)
        self.sklearn_eps_spin.valueChanged.connect(self.mark_results_stale)
        self.sklearn_min_samples_spin.valueChanged.connect(self.mark_results_stale)


    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setDecimals(decimals)
        widget.setSingleStep(10 ** -min(decimals, 3))
        return widget

    def _create_controls(self) -> None:
        self.cluster_count_spin = self._spin(1, 10_000, 4)
        self.cluster_count_spin.setToolTip(_("Number of clusters to request."))

        self.whiten_check = QCheckBox()
        self.whiten_check.setChecked(True)
        self.whiten_check.setToolTip(_("Normalize features by standard deviation before k-means."))

        self.kmeans_iter_spin = self._spin(1, 10_000, 50)
        self.kmeans_iter_spin.setToolTip(_("Maximum k-means iterations."))

        self.kmeans_thresh_spin = self._double_spin(0.0, 1.0, 1e-5, 8)
        self.kmeans_thresh_spin.setToolTip(_("K-means convergence threshold."))

        self.max_runtime_spin = self._double_spin(0.0, 3600.0, 15.0, 1)
        self.max_runtime_spin.setSpecialValueText(_("no timeout"))
        self.max_runtime_spin.setToolTip(
            _("Maximum seconds to wait for clustering. Use 0 for no timeout.")
        )


        self.linkage_method_combo = QComboBox()
        self.linkage_method_combo.addItems(list(HIERARCHY_METHODS))
        self.linkage_method_combo.setCurrentText("ward")

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(HIERARCHY_METRICS))
        self.metric_combo.setCurrentText("euclidean")

        self.hierarchy_criterion_combo = QComboBox()
        self.hierarchy_criterion_combo.addItems(list(HIERARCHY_CRITERIA))
        self.hierarchy_criterion_combo.setCurrentText("maxclust")

        self.distance_threshold_spin = self._double_spin(0.0, 1e18, 0.0, 6)
        self.distance_threshold_spin.setSpecialValueText(_("auto"))
        self.distance_threshold_spin.setToolTip(
            _("Cut threshold for distance/inconsistent criteria. Leave at auto for median linkage distance.")
        )

        self.sklearn_eps_spin = self._double_spin(0.0, 1e18, 0.5, 6)
        self.sklearn_eps_spin.setSpecialValueText(_("auto"))
        self.sklearn_eps_spin.setToolTip(_("Neighborhood radius for DBSCAN. 0 uses the default 0.5."))

        self.sklearn_min_samples_spin = self._spin(1, 1_000_000, 5)
        self.sklearn_min_samples_spin.setToolTip(_("Minimum samples for DBSCAN/OPTICS core points."))


    def _add_parameter_rows(self) -> None:
        rows: tuple[tuple[str, str, QWidget], ...] = (
            ("clusters", "Clusters:", self.cluster_count_spin),
            ("whiten", "Whiten features:", self.whiten_check),
            ("kmeans_iter", "K-means iterations:", self.kmeans_iter_spin),
            ("kmeans_thresh", "K-means threshold:", self.kmeans_thresh_spin),
            ("max_runtime", "Max runtime:", self.max_runtime_spin),
            ("linkage_method", "Linkage method:", self.linkage_method_combo),
            ("metric", "Distance metric:", self.metric_combo),
            ("criterion", "Cut criterion:", self.hierarchy_criterion_combo),
            ("distance_threshold", "Cut threshold:", self.distance_threshold_spin),
            ("sklearn_eps", "Neighborhood radius:", self.sklearn_eps_spin),
            ("sklearn_min_samples", "Min samples:", self.sklearn_min_samples_spin),
        )
        for key, label, widget in rows:
            self.add_field_row(self.form, key, label, widget)

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

    def _refresh_visibility(self) -> None:
        if self.scipy_tool_combo.count() == 0:
            self._refresh_tools()

        method = self.method_combo.currentText()
        tool = self.scipy_tool_combo.currentText()
        visible = {"clusters", "max_runtime"}

        if method == CLUSTER_KMEANS:
            visible.update({"kmeans_iter", "kmeans_thresh"})
            if tool in {TOOL_WHITEN, TOOL_KMEANS, TOOL_KMEANS2}:
                visible.add("whiten")
        elif method == CLUSTER_HIERARCHICAL:
            visible.update({"linkage_method", "metric", "criterion", "distance_threshold"})
        else:
            if tool in {SKLEARN_AGGLOMERATIVE}:
                visible.update({"linkage_method", "metric"})
            if tool in {SKLEARN_DBSCAN}:
                visible.update({"metric", "sklearn_eps", "sklearn_min_samples"})
            if tool in {SKLEARN_OPTICS}:
                visible.update({"metric", "sklearn_min_samples"})

        self.show_field_rows(visible)
        self._update_description_link()

    def _update_description_link(self) -> None:
        tool = self.scipy_tool_combo.currentText() if hasattr(self, "scipy_tool_combo") else ""
        title = tool or "SciPy clustering"
        url = TOOL_DOCS.get(tool, "https://scikit-learn.org/stable/modules/clustering.html" if str(tool).startswith("sklearn.") else "https://docs.scipy.org/doc/scipy/reference/cluster.html")
        self.set_doc_link(title, url)

    def _params(self) -> dict[str, Any]:
        return {
            "method": self.method_combo.currentText(),
            "scipy_tool": self.scipy_tool_combo.currentText(),
            "feature_mode": self.feature_combo.currentText(),
            "render_mode": self.render_mode_combo.currentText(),
            "clusters": self.cluster_count_spin.value(),
            "whiten": self.whiten_check.isChecked(),
            "kmeans_iter": self.kmeans_iter_spin.value(),
            "kmeans_thresh": self.kmeans_thresh_spin.value(),
            "linkage_method": self.linkage_method_combo.currentText(),
            "metric": self.metric_combo.currentText(),
            "criterion": self.hierarchy_criterion_combo.currentText(),
            "distance_threshold": self.distance_threshold_spin.value(),
            "max_runtime_seconds": self.max_runtime_spin.value(),
            "sklearn_eps": self.sklearn_eps_spin.value(),
            "sklearn_min_samples": self.sklearn_min_samples_spin.value(),
        }

    def _current_axis_name(self) -> str:
        return self.series_selector.selected_axis_name()

    def _series_display_name(self, row: Any) -> str:
        return str(row["name"])

    def _series_choice_from_row(self, row: Any) -> ClusterSeriesChoice:
        name = str(row["name"])
        sql_query = str(row["sql_query"])
        if not sql_query:
            applogger.error("Selected series has no SQL query.", show_dialog=True, raise_error=True)

        source_table = _source_table_from_sql(sql_query)
        source_x_column = _source_column_for_alias(sql_query, "x")
        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.", show_dialog=True, raise_error=True)

        roles = parse_roles(row["roles"])
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = str(roles.get("x") or "")
        y_col = str(roles.get("y") or "")
        z_col = str(roles.get("z") or "")

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
                    show_dialog=True,
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
                applogger.error("No numeric columns are available for clustering.", show_dialog=True, raise_error=True)
            return numeric

        return [series.x_col, series.y_col]

    def prepare_job(self, **options: Any) -> SeriesJob | None:
        """Read each series and its feature columns here; the clustering runs in the job."""
        del options
        settings = (self._current_axis_name(), self._params(), self._figure_id)
        selected_rows = self.selected_series()
        if not selected_rows:
            return None

        inputs: list[tuple[str, Any]] = []
        errors: list[str] = []
        for row in selected_rows:
            try:
                series = self._series_choice_from_row(row)
                inputs.append((self._series_display_name(row), (series, self._feature_columns(series))))
            except Exception as exc:
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
                results.extend(self._split_result_by_cluster(full_result))
            else:
                results.append(full_result)
        errors = outcome.errors

        if errors and not results:
            applogger.error("\n".join(errors), show_dialog=True, raise_error=True)

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
        features, finite_mask = numeric_matrix(series.frame, feature_columns)

        method = str(params["method"])
        if method == CLUSTER_KMEANS:
            cluster_ids, metadata = cluster_kmeans(
                features,
                scipy_tool=str(params["scipy_tool"]),
                clusters=int(params["clusters"]),
                use_whiten=bool(params["whiten"]),
                iterations=int(params["kmeans_iter"]),
                threshold=float(params["kmeans_thresh"]),
                timeout_seconds=float(params.get("max_runtime_seconds", 0.0)),
            )
        elif method == CLUSTER_HIERARCHICAL:
            cluster_ids, metadata = cluster_hierarchical(
                features,
                scipy_tool=str(params["scipy_tool"]),
                clusters=int(params["clusters"]),
                linkage_method=str(params["linkage_method"]),
                metric=str(params["metric"]),
                criterion=str(params["criterion"]),
                distance_threshold=float(params["distance_threshold"]),
                timeout_seconds=float(params.get("max_runtime_seconds", 0.0)),
            )
        else:
            cluster_ids, metadata = cluster_sklearn(
                features,
                sklearn_tool=str(params["scipy_tool"]),
                clusters=int(params["clusters"]),
                linkage_method=str(params["linkage_method"]),
                metric=str(params["metric"]),
                eps=float(params.get("sklearn_eps", 0.5)),
                min_samples=int(params.get("sklearn_min_samples", 5)),
                timeout_seconds=float(params.get("max_runtime_seconds", 0.0)),
            )

        output = series.frame.copy()
        output["ClusterId"] = pd.Series([pd.NA] * len(output), dtype="Int64")
        output.loc[finite_mask, "ClusterId"] = cluster_ids

        metadata.update(
            {
                **dict(params),
                "figure_id": figure_id,
                "axis_name": axis_name,
                "source_series_id": self._source_series_id(series),
                "source_series_name": series.name,
                "source_table": series.source_table,
                "source_x_column": series.source_x_column,
                "source_sql_query": series.source_sql_query,
                "feature_columns": feature_columns,
                "finite_rows": int(finite_mask.sum()),
                "total_rows": int(len(output)),
                "roles": series.roles,
                "base_result_name": f"{series.name} - clusters",
                "is_split_series": False,
            }
        )

        return ClusterResult(
            source_name=series.name,
            result_name=f"{series.name} - clusters",
            method=method,
            frame=output,
            x_col=series.x_col,
            y_col=series.y_col,
            z_col=series.z_col,
            feature_columns=feature_columns,
            metadata=metadata,
        )

    def _split_result_by_cluster(self, result: ClusterResult) -> list[ClusterResult]:
        ids = to_numbers(result.frame["ClusterId"])
        cluster_ids = sorted(int(value) for value in ids.dropna().unique())
        split_results: list[ClusterResult] = []

        for cluster_id in cluster_ids:
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "cluster_id": int(cluster_id),
                    "is_split_series": True,
                    "render_mode": RENDER_SEPARATE_SERIES,
                }
            )
            split_results.append(
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
                )
            )

        return split_results

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
        except Exception:
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
        except Exception:
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
            except Exception:
                applogger.exception(
                    "Failed to restore %s.%s after preview", source_table, CLUSTER_COLUMN
                )
        self._cluster_snapshots.clear()

        for series_id, sql_query in list(self._series_sql_snapshots.items()):
            try:
                self._repo.update_series_sql_query(int(series_id), sql_query)
                restored = True
            except Exception:
                applogger.exception("Failed to restore SQL for series id=%s", series_id)
        self._series_sql_snapshots.clear()

        return restored

    def discard_operation_snapshots(self) -> None:
        """Drop the snapshots after Apply has made the changes permanent."""
        for source_table in list(self._cluster_snapshots):
            try:
                self._repo.discard_column_snapshot(source_table, CLUSTER_BACKUP_COLUMN)
            except Exception:
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
        self._repo.ensure_cluster_column(source_table)

        cluster_values = to_numbers(result.frame["ClusterId"])
        x_values = result.frame[result.x_col]
        self._repo.clear_cluster_column(source_table)
        self._repo.set_ClusterId(source_table,source_x_column,x_values,cluster_values)


    def _colored_series_sql_query(self, table_name: str, result: ClusterResult) -> str:
        del table_name
        return _sql_with_clusterid_color(str(result.metadata["source_sql_query"]))

    def result_series_spec(self, axis_id: int, table_name: str, result: ClusterResult) -> ResultSeriesSpec:
        del axis_id

        roles = self._base_roles(result)
        roles.update({"ClusterId": "ClusterId", "cluster": "ClusterId"})

        cluster_id = result.metadata.get("cluster_id")
        is_split = bool(result.metadata.get("is_split_series", False))

        if is_split and cluster_id is not None:
            sql_query = _sql_with_cluster_filter(
                str(result.metadata["source_sql_query"]),
                int(cluster_id),
            )
            color = _rc_cycle_color(int(cluster_id))
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
        """Return a plain-text preview summary only.

        The preview deliberately avoids HTML because some result widgets display
        markup as literal text depending on the Qt backend in use.
        """
        if not results:
            return ""

        unique_results = self._unique_report_results(results)
        lines: list[str] = ["Clustering preview"]
        for result in unique_results:
            ids = to_numbers(result.frame["ClusterId"])
            valid_ids = ids.dropna().astype(int)
            clusters_found = int(valid_ids.nunique()) if not valid_ids.empty else 0
            total_rows = int(len(result.frame))
            finite_rows = int(valid_ids.size)
            feature_text = ", ".join(result.feature_columns)
            lines.append(
                f"- {result.source_name}: {clusters_found} clusters, "
                f"{finite_rows}/{total_rows} clustered rows, "
                f"tool={result.metadata.get('scipy_tool', result.method)}, "
                f"features={feature_text}"
            )
        return "\n".join(lines)

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

    @staticmethod
    def _html_escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

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
                    show_dialog=True,
                    raise_error=True,
                )
                continue
            source_series_id_int = int(source_series_id)
            self._snapshot_series_sql(source_series_id_int)
            self._repo.update_series_sql_query(source_series_id_int, sql_query)
            applogger.info(f"Updated selected series SQL for ClusterId color: {result.source_name}")

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
        if self._applied_callback is not None:
            self._applied_callback()

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
