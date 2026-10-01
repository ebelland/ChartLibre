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
from PySide6.QtWidgets import QFormLayout, QWidget
from app.analysis import decomposition as dc
from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    CallJob,
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.series_operations.parameter_spec import FloatParam, IntParam
from app.utils import report_html
from app.utils.i18n import _

DECOMP_PCA = "PCA"
DECOMP_FASTICA = "Fast ICA"
DECOMP_NMF = "NMF"
MANIFOLD_TSNE = "t-SNE"
MANIFOLD_ISOMAP = "Isomap"
MANIFOLD_LLE = "Locally Linear Embedding"


@dataclass(frozen=True, slots=True, kw_only=True)
class DecompositionModel(OperationModel):
    #: A manifold embedding: the result is a 2D scatter of the samples, not
    #: a curve per component.
    manifold: bool = False


#: The models offered, in combo order: the decompositions, a line, then the
#: manifold embeddings.
DECOMPOSITION_ALL_MODELS: dict[str, DecompositionModel] = {
    DECOMP_PCA: DecompositionModel(
        doc_title="Principal component analysis",
        doc_url="https://en.wikipedia.org/wiki/Principal_component_analysis",
    ),
    DECOMP_FASTICA: DecompositionModel(
        doc_title="Independent component analysis",
        doc_url="https://en.wikipedia.org/wiki/Independent_component_analysis",
    ),
    DECOMP_NMF: DecompositionModel(
        doc_title="Non-negative matrix factorization",
        doc_url="https://en.wikipedia.org/wiki/Non-negative_matrix_factorization",
    ),
    MANIFOLD_TSNE: DecompositionModel(
        doc_title="t-distributed stochastic neighbor embedding",
        doc_url="https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding",
        group="manifold",
        manifold=True,
    ),
    MANIFOLD_ISOMAP: DecompositionModel(
        doc_title="Isomap",
        doc_url="https://en.wikipedia.org/wiki/Isomap",
        group="manifold",
        manifold=True,
    ),
    MANIFOLD_LLE: DecompositionModel(
        doc_title="Locally linear embedding",
        doc_url="https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Locally-linear_embedding",
        group="manifold",
        manifold=True,
    ),
}
#: The engine's name for each model.
_KIND: dict[str, str] = {
    DECOMP_PCA: dc.KIND_PCA,
    DECOMP_FASTICA: dc.KIND_FASTICA,
    DECOMP_NMF: dc.KIND_NMF,
    MANIFOLD_TSNE: dc.KIND_TSNE,
    MANIFOLD_ISOMAP: dc.KIND_ISOMAP,
    MANIFOLD_LLE: dc.KIND_LLE,
}

DECOMPOSITION_MODELS = tuple(name for name, model in DECOMPOSITION_ALL_MODELS.items() if not model.manifold)



@dataclass(slots=True)
class DecompositionResult(TableResult):
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

    @property
    def series(self) -> tuple[str, ...]:
        return tuple(self.source_names)

    def to_df(self) -> pd.DataFrame:
        if self.kind == "manifold":
            return pd.DataFrame({"x": self.x, "y": self.values})
        data: dict[str, Any] = {"x": self.x}
        for index, name in enumerate(self.component_names):
            data[name] = self.values[:, index]
        return pd.DataFrame(data)


class SeriesDecompositionDialog(SeriesOperationDialogBase):
    """Decompose or embed several selected series, resampled onto one grid."""

    MODELS = DECOMPOSITION_ALL_MODELS
    MODEL_TOOLTIP = "Choose a decomposition (a curve per component) or a manifold embedding (a 2D scatter)."

    Name: str = "Decomposition"
    Description = "Decompose or embed several series together (PCA, ICA, NMF, t-SNE, ...)"

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

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
        self._parameter_form = None


    def _model(self) -> str:
        return self.current_model(DECOMP_PCA)


    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def prepare_job(self, **options: Any) -> CallJob:
        """Read every selected series here; the grid, the matrix and the model run in the job."""
        del options
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

        def compute() -> list[DecompositionResult]:
            grid = dc.shared_grid(xy_pairs, n_grid)
            matrix = dc.feature_matrix(grid, xy_pairs)
            if DECOMPOSITION_ALL_MODELS[model].manifold:
                return [self._embed(names, matrix, model, params)]
            return [self._decompose(names, grid, matrix, model, params)]

        return CallJob(compute)

    def _decompose(
        self,
        names: list[str],
        grid: np.ndarray,
        matrix: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> DecompositionResult:
        decomposition = dc.decompose(_KIND[model], matrix, int(params.get("n_components", 3)))
        return DecompositionResult(
            source_names=names,
            model=model,
            kind="decomposition",
            x=grid,
            values=decomposition.values,
            component_names=decomposition.component_names,
            metadata={
                "n_series": matrix.shape[1],
                "explained_variance_ratio": decomposition.explained_variance_ratio,
            },
        )

    def _embed(
        self,
        names: list[str],
        matrix: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> DecompositionResult:
        embedding = dc.embed(
            _KIND[model],
            matrix,
            perplexity=float(params.get("perplexity", 30.0)),
            n_neighbors=int(params.get("n_neighbors", 5)),
        )
        for note in embedding.notes:
            applogger.warning(note, show_dialog=False, raise_error=False)
        return DecompositionResult(
            source_names=names,
            model=model,
            kind="manifold",
            x=embedding.x,
            values=embedding.y,
            component_names=[],
            metadata={"n_series": matrix.shape[1], "n_samples": matrix.shape[0]},
        )

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
            figure_name=self.result_figure_name(results, result.model, source="+".join(result.source_names)),
            options={"grid": True, "linestyle": "-", "marker": ""},
        )
        return axis_id

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

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
