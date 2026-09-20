"""Decomposition and manifold learning across several series (todo.txt P2-15).

Every other operation in this app reads its selected series one at a time,
each producing its own independent result. This one is the exception: PCA,
ICA, NMF and the three manifold methods below all need several series
*combined* into one feature matrix - columns are series, rows are a shared
grid of x values every selected series is resampled onto - which is why
selecting only one series here is refused rather than silently doing
nothing useful.

Two families share one dialog because they share that exact "several series
in, fewer dimensions out" shape, even though what comes out differs:

- *Decomposition* (PCA/ICA/NMF) still means something as a curve over the
  shared x - each component is drawn as its own series against the grid, the
  same shape a smoothed or filtered series already has.
- *Manifold learning* (t-SNE/Isomap/LLE) is exploratory 2D structure, not a
  value over x - each grid sample becomes one unordered 2D point, drawn as a
  bare scatter on its own new axis rather than as x/y-over-grid curves.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.series_operations.parameter_spec import FloatParam, IntParam
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

DECOMP_PCA = "PCA"
DECOMP_FASTICA = "Fast ICA"
DECOMP_NMF = "NMF"
MANIFOLD_TSNE = "t-SNE"
MANIFOLD_ISOMAP = "Isomap"
MANIFOLD_LLE = "Locally Linear Embedding"

DECOMPOSITION_MODELS = (DECOMP_PCA, DECOMP_FASTICA, DECOMP_NMF)
MANIFOLD_MODELS = (MANIFOLD_TSNE, MANIFOLD_ISOMAP, MANIFOLD_LLE)
DECOMPOSITION_ALL_MODELS = DECOMPOSITION_MODELS + MANIFOLD_MODELS

DECOMPOSITION_DOCS = {
    DECOMP_PCA: (
        "Principal component analysis",
        "https://en.wikipedia.org/wiki/Principal_component_analysis",
    ),
    DECOMP_FASTICA: (
        "Independent component analysis",
        "https://en.wikipedia.org/wiki/Independent_component_analysis",
    ),
    DECOMP_NMF: (
        "Non-negative matrix factorization",
        "https://en.wikipedia.org/wiki/Non-negative_matrix_factorization",
    ),
    MANIFOLD_TSNE: (
        "t-distributed stochastic neighbor embedding",
        "https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding",
    ),
    MANIFOLD_ISOMAP: (
        "Isomap",
        "https://en.wikipedia.org/wiki/Isomap",
    ),
    MANIFOLD_LLE: (
        "Locally linear embedding",
        "https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Locally-linear_embedding",
    ),
}


@dataclass(slots=True)
class DecompositionResult:
    """One joint decomposition/embedding of several selected series.

    ``kind`` picks how ``to_frame``/the dialog's own ``result_series_specs``
    read the rest of the fields - see the module docstring for why the two
    families need different output shapes:

    - "decomposition": ``x`` is the shared grid, ``values`` is
      ``(len(x), len(component_names))`` - one curve per component.
    - "manifold": ``x`` is the embedding's own first coordinate, ``values``
      its second - one bare (x, y) point per grid sample, no ``x`` ordering
      implied.
    """

    source_names: list[str]
    model: str
    kind: str
    x: np.ndarray
    values: np.ndarray
    component_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        if self.kind == "manifold":
            return pd.DataFrame({"x": self.x, "y": self.values})
        data: dict[str, Any] = {"x": self.x}
        for index, name in enumerate(self.component_names):
            data[name] = self.values[:, index]
        return pd.DataFrame(data)


class SeriesDecompositionDialog(SeriesOperationDialogBase):
    """Decompose or embed several selected series, resampled onto one grid."""

    Name: str = "Decomposition"
    Description = "Decompose or embed several series together (PCA, ICA, NMF, t-SNE, ...)"

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        IntParam(
            "n_grid",
            "Grid points:",
            tooltip=(
                "Every selected series is linearly resampled onto this many "
                "evenly spaced points across their shared x range, before "
                "decomposing or embedding them together."
            ),
            default_value=200,
            minimum=10,
            maximum=5000,
        ),
        IntParam(
            "n_components",
            "Components:",
            tooltip="How many components to extract. Clamped to the number of selected series.",
            default_value=3,
            minimum=1,
            maximum=20,
            visible_for={"model": DECOMPOSITION_MODELS},
        ),
        FloatParam(
            "perplexity",
            "Perplexity:",
            tooltip=(
                "Roughly, how many neighbours each point is assumed to have. "
                "Lowered automatically if it does not fit the sample count."
            ),
            default_value=30.0,
            minimum=5.0,
            maximum=50.0,
            visible_for={"model": (MANIFOLD_TSNE,)},
        ),
        IntParam(
            "n_neighbors",
            "Neighbors:",
            tooltip=(
                "How many nearby points each one is connected to. Lowered "
                "automatically if it does not fit the sample count."
            ),
            default_value=5,
            minimum=2,
            maximum=50,
            visible_for={"model": (MANIFOLD_ISOMAP, MANIFOLD_LLE)},
        ),
    )

    Icon = """
    <circle cx="7" cy="7" r="2"/>
    <circle cx="16" cy="6" r="2"/>
    <circle cx="12" cy="13" r="2"/>
    <circle cx="6" cy="17" r="2"/>
    <circle cx="17" cy="17" r="2"/>
    <path d="M12 13l-5-6M12 13l4-7M12 13l-6 4M12 13l5 4" stroke-dasharray="1 2"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesDecompositionDialog requires a repository instance.")

        self._last_results: list[DecompositionResult] = []
        self._parameter_form: QFormLayout | None = None
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Decomposition",
            parent=parent,
            width=780,
            height=640,
        )
        self.series_selector.reload(select_all_series=False)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        container = QWidget(panel)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(DECOMPOSITION_ALL_MODELS)
        self.model_combo.insertSeparator(len(DECOMPOSITION_MODELS))
        self.model_combo.setToolTip(
            _("Choose a decomposition (a curve per component) or a manifold embedding (a 2D scatter).")
        )
        form.addRow(_("Model:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.mark_results_stale)

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = DECOMPOSITION_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or DECOMP_PCA


    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[DecompositionResult]:
        rows = self.selected_series()
        if len(rows) < 2:
            raise ValueError("Select two or more series to decompose or embed together.")

        model = self._model()
        params = self.parameter_values()
        n_grid = int(params.get("n_grid", 200))

        names: list[str] = []
        xy_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        errors: list[str] = []
        for row in rows:
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                names.append(name)
                xy_pairs.append((x_values, y_values))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if len(xy_pairs) < 2:
            detail = " ".join(errors)
            raise ValueError(
                f"Select two or more usable series to decompose or embed together. {detail}".strip()
            )
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        grid = self._shared_grid(xy_pairs, n_grid)
        feature_matrix = np.column_stack(
            [np.interp(grid, x_values, y_values) for x_values, y_values in xy_pairs]
        )

        if model in MANIFOLD_MODELS:
            return [self._embed(names, feature_matrix, model, params)]
        return [self._decompose(names, grid, feature_matrix, model, params)]

    @staticmethod
    def _shared_grid(
        xy_pairs: Sequence[tuple[np.ndarray, np.ndarray]], n_grid: int
    ) -> np.ndarray:
        low = max(float(x.min()) for x, _y in xy_pairs)
        high = min(float(x.max()) for x, _y in xy_pairs)
        if not (high > low):
            raise ValueError(
                "the selected series' x ranges do not overlap; nothing to resample onto a shared grid"
            )
        return np.linspace(low, high, max(2, int(n_grid)))

    def _decompose(
        self,
        names: list[str],
        grid: np.ndarray,
        feature_matrix: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> DecompositionResult:
        n_series = feature_matrix.shape[1]
        n_components = max(1, min(int(params.get("n_components", 3)), n_series))
        explained_variance_ratio: list[float] | None = None

        if model == DECOMP_PCA:
            estimator = PCA(n_components=n_components, random_state=0)
            values = estimator.fit_transform(feature_matrix)
            explained_variance_ratio = [float(v) for v in estimator.explained_variance_ratio_]
            prefix = "PC"
        elif model == DECOMP_FASTICA:
            estimator = FastICA(n_components=n_components, random_state=0, max_iter=1000)
            values = estimator.fit_transform(feature_matrix)
            prefix = "IC"
        else:
            if float(feature_matrix.min()) < 0.0:
                raise ValueError(
                    "NMF requires non-negative data; the selected series include negative values"
                )
            estimator = NMF(n_components=n_components, random_state=0, max_iter=1000)
            values = estimator.fit_transform(feature_matrix)
            prefix = "Component"

        component_names = [f"{prefix}{index + 1}" for index in range(n_components)]
        return DecompositionResult(
            source_names=names,
            model=model,
            kind="decomposition",
            x=grid,
            values=values,
            component_names=component_names,
            metadata={
                "n_series": n_series,
                "explained_variance_ratio": explained_variance_ratio,
            },
        )

    def _embed(
        self,
        names: list[str],
        feature_matrix: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> DecompositionResult:
        n_samples = feature_matrix.shape[0]

        if model == MANIFOLD_TSNE:
            perplexity = self._clamped_perplexity(params, n_samples)
            embedding = TSNE(
                n_components=2, random_state=0, perplexity=perplexity
            ).fit_transform(feature_matrix)
        elif model == MANIFOLD_ISOMAP:
            neighbors = self._clamped_neighbors(params, n_samples)
            embedding = Isomap(
                n_components=2, n_neighbors=neighbors
            ).fit_transform(feature_matrix)
        else:
            neighbors = self._clamped_neighbors(params, n_samples)
            embedding = LocallyLinearEmbedding(
                n_components=2, n_neighbors=neighbors, random_state=0
            ).fit_transform(feature_matrix)

        return DecompositionResult(
            source_names=names,
            model=model,
            kind="manifold",
            x=np.asarray(embedding[:, 0], dtype=float),
            values=np.asarray(embedding[:, 1], dtype=float),
            component_names=[],
            metadata={"n_series": feature_matrix.shape[1], "n_samples": n_samples},
        )

    @staticmethod
    def _clamped_perplexity(params: Mapping[str, Any], n_samples: int) -> float:
        requested = float(params.get("perplexity", 30.0))
        # scikit-learn requires perplexity < n_samples; leave real headroom.
        limit = max(5.0, (n_samples - 1) / 3.0)
        if requested >= n_samples or requested > limit:
            applogger.warning(
                f"Perplexity {requested:g} does not fit {n_samples} sampled "
                f"points; using {limit:g} instead.",
                show_dialog=False,
                raise_error=False,
            )
            return limit
        return requested

    @staticmethod
    def _clamped_neighbors(params: Mapping[str, Any], n_samples: int) -> int:
        requested = int(params.get("n_neighbors", 5))
        limit = max(2, n_samples - 1)
        if requested > limit:
            applogger.warning(
                f"n_neighbors {requested} does not fit {n_samples} sampled "
                f"points; using {limit} instead.",
                show_dialog=False,
                raise_error=False,
            )
            return limit
        return requested

    # ------------------------------------------------------------------
    # Where the result is drawn
    # ------------------------------------------------------------------

    def resolve_target_axis_id(
        self, selected_axis_id: int, results: Sequence[Any]
    ) -> int:
        result = results[0]
        if result.kind == "manifold":
            # An embedding shares no scale or meaning with the source axes -
            # it always gets its own new axis, never the source's or a
            # user-chosen destination.
            if self._result_axis_id is None:
                self._result_axis_id = self.create_result_axis(
                    chart_type="Scatter Plot",
                    title=result.model,
                    x_label=_("Component 1"),
                    y_label=_("Component 2"),
                    options={"grid": True, "linestyle": "", "marker": "o"},
                )
            return int(self._result_axis_id)

        axis_id = self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Scatter Plot",
            title=result.model,
            figure_name=self._result_figure_name(results),
            options={"grid": True, "linestyle": "-", "marker": ""},
        )
        return axis_id

    def _result_figure_name(self, results: Sequence[Any]) -> str:
        result = results[0]
        source = "+".join(result.source_names) if results else ""
        return f"{source} - {result.model}".strip(" -") or "Decomposition"

    def discard_operation_artifacts(self) -> None:
        self.discard_result_target()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: DecompositionResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: DecompositionResult,
    ) -> Sequence[ResultSeriesSpec]:
        del axis_id
        base_style = {
            "generated_decomposition": True,
            "decomposition_dialog": "series_decomposition",
            "model": result.model,
        }
        if result.kind == "manifold":
            return [
                ResultSeriesSpec(
                    name=f"{result.model} embedding",
                    sql_query=f'SELECT x, y FROM "{table_name}"',
                    roles={"x": "x", "y": "y"},
                    style={**base_style, "linestyle": "", "marker": "o"},
                )
            ]
        return [
            ResultSeriesSpec(
                name=name,
                sql_query=f'SELECT x, "{name}" AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style={**base_style, "linestyle": "-", "marker": ""},
            )
            for name in result.component_names
        ]

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {
            "generated_decomposition": True,
            "decomposition_dialog": "series_decomposition",
        }

    def result_table_name(self, axis_id: int, result: DecompositionResult) -> str:
        label = "_".join(result.source_names)[:60]
        return generated_table_name(
            f"Decomposition_axis{axis_id}_{label}_{result.model}",
            fallback="Decomposition_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Decomposition"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[DecompositionResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        result = results[0]
        sections = [
            report_html.section(
                _("Input"),
                report_html.summary_table(
                    (
                        (_("Series"), ", ".join(result.source_names)),
                        (_("Model"), result.model),
                    )
                ),
            )
        ]

        ratios = result.metadata.get("explained_variance_ratio")
        if ratios:
            rows = [
                (name, report_html.format_number(ratio * 100.0, digits=1) + "%")
                for name, ratio in zip(result.component_names, ratios)
            ]
            sections.append(
                report_html.section(
                    _("Explained variance"),
                    report_html.table(
                        (_("Component"), _("Share")),
                        rows,
                        align=("left", "right"),
                    ),
                )
            )
        elif result.kind == "manifold":
            sections.append(
                report_html.note(
                    _(
                        "{count} point(s) embedded from {series} series into 2 dimensions."
                    ).format(
                        count=len(result.x),
                        series=len(result.source_names),
                    )
                )
            )

        return report_html.document(_("Decomposition"), result.model, *sections)
