"""Principal Components (Analysis): a few directions that carry most of the columns' variation (AN-04).

As JMP's Principal Components platform: the eigenvalues - the variance each
component carries, and its share - the loadings (each column's correlation
with each component) and every row's scores, on the correlation matrix
(columns standardized) or the covariance matrix. OK writes the scores beside
the rows and the eigenvalues and loadings as tables, and draws the scree
plot, the score plot and the loading plot. The engine is
app.analysis.multivariate.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import multivariate as mv
from app.plugin_api.data_operations.parameter_spec import ChoiceParam
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class PrincipalComponentsDialog(TableOperationDialogBase):
    """Principal components: eigenvalues, loadings and each row's scores."""

    Name = "Principal Components"
    Description = (
        "A few directions that carry most of the variation of many columns: eigenvalues and their "
        "share, loadings, each row's scores; scree, score and loading plots."
    )
    Category = "Multivariate Methods"
    Order = 40
    Icon = """
    <circle cx="7" cy="16" r="1.3"/><circle cx="10" cy="13" r="1.3"/><circle cx="13" cy="11" r="1.3"/>
    <circle cx="16" cy="8" r="1.3"/><circle cx="12" cy="14" r="1.3"/><circle cx="9" cy="10" r="1.3"/>
    <path d="M4 20L20 4"/><path d="M8 6l8 12" stroke-dasharray="2 2"/>
    """
    Keywords: str = (
        "principal components componenti principali pca principal component analysis analisi delle componenti "
        "principali eigenvalues autovalori loadings scores punteggi scree dimension reduction riduzione "
        "dimensionale"
    )
    Goals: list[str] = ["explain", "groups"]

    ROLES = (
        ColumnRole("response", "Y, columns", "The continuous columns whose variation is summed up.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=2),
        ColumnRole("by", "By", "Optional: a column of groups - species, batch, site; the score plot and the "
                               "biplot colour the rows by it.", maximum=1),
    )
    #: More groups than this are one colour: a legend of fifty is no help.
    MAX_GROUPS = 20
    PARAMETERS_TAB = "Principal Components"
    PARAMS = (
        ChoiceParam("on", "On:", choices=tuple((label, key) for key, label in mv.PCA_ON),
                    tooltip="Correlations weigh every column the same, whatever its unit - the usual choice. "
                            "Covariances let the columns with the widest spread lead, for columns in one unit."),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        if not found and set(casting.columns("response")) & set(casting.columns("by")):
            found.append(_("The By column cannot also be one of the columns."))
        return found

    def _row_series(self, table: str, x: str, y: str) -> list[tuple]:
        """The rows as one series, or one per group of the By column."""
        by = self.roles_widget.casting().columns("by")
        if by:
            levels = [value for value in pd.unique(self._frame[by[0]].dropna())][: self.MAX_GROUPS + 1]
            if 1 < len(levels) <= self.MAX_GROUPS:
                series = []
                for level in sorted(levels, key=str):
                    sql, roles = self.series_select(table, {"x": x, "y": y},
                                                    where=f"{self.quoted(by[0])} = {self.literal(str(level))}")
                    series.append((str(level), sql, roles, {}))
                return series
        sql, roles = self.series_select(table, {"x": x, "y": y})
        return [(_("Rows"), sql, roles, {})]

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        return mv.principal_components(frame, casting.columns("response"),
                                       on=str(params.get("on") or mv.CORRELATIONS))

    def report_subtitle(self, result: mv.PCAResult) -> str:
        return f"{self.table} · {_(dict(mv.PCA_ON)[result.on])}"

    def report_body(self, result: mv.PCAResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), report_html.format_number(value)) for key, value in result.summary.items())))
        blocks.append(report_html.section(_("Eigenvalues"), self.report_table(result.eigenvalues, digits=4)))
        blocks.append(report_html.section(_("Loading matrix"), self.report_table(result.loadings, digits=4)))
        blocks.append(report_html.section(_("Eigenvectors"), self.report_table(result.eigenvectors, digits=4)))
        return "".join(blocks)

    def apply_results(self, result: mv.PCAResult) -> list[dict[str, Any]]:
        frame = self._frame.copy()
        for column in result.scores.columns:
            frame[column] = result.scores[column].reindex(frame.index)
        tables = self.write_result_tables({"pca": frame, "pca_eigenvalues": result.eigenvalues,
                                           "pca_loadings": result.loadings})
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]

        sql, roles = self.series_select(tables["pca_eigenvalues"], {"x": "Number", "y": "Eigenvalue"})
        panels: list[tuple] = [("Scatter Plot", _("Scree plot"), _("Component"), _("Eigenvalue"),
                                [(_("Eigenvalue"), sql, roles, {"linestyle": "-", "marker": "o"})])]
        if len(result.scores.columns) >= 2:
            # The axes through zero: which side of the centre a row, or a column, falls on.
            zero = {"lines": [{"orientation": orientation, "value": 0.0,
                               "kwargs": {"color": "0.6", "linestyle": "--", "linewidth": 0.8}}
                              for orientation in ("horizontal", "vertical")]}
            panels.append(("Scatter Plot", _("Score plot"), "Prin1", "Prin2",
                           self._row_series(tables["pca"], "Prin1", "Prin2"), zero))
            loading_series = []
            for column in result.columns:
                sql, roles = self.series_select(tables["pca_loadings"], {"x": "Prin1", "y": "Prin2"},
                                                where=f'"Variable" = {self.literal(column)}')
                loading_series.append((column, sql, roles, {"marker": "o"}))
            panels.append(("Scatter Plot", _("Loading plot"), "Prin1", "Prin2", loading_series, zero))
            panels.append(self._biplot(result, tables["pca"], zero))
        figure_id = self.create_report_figure(f"Principal Components · {self.table}", self.report_document(result),
                                              panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": self.table})
        return written

    def _biplot(self, result: mv.PCAResult, table: str, zero: dict[str, Any]) -> tuple:
        """The rows' scores and, as arrows from the centre, the columns' loadings - scaled to the scores' spread."""
        loadings = result.loadings[["Prin1", "Prin2"]].to_numpy(float)
        scores = result.scores[["Prin1", "Prin2"]].to_numpy(float)
        reach = float(abs(scores).max()) if scores.size else 1.0
        scale = 0.85 * reach / max(float(abs(loadings).max()), 1e-12)
        annotations = []
        for name, (x, y) in zip(result.columns, loadings * scale):
            annotations.append({"type": "arrow", "x": float(x), "y": float(y), "text": "",
                                "kwargs": {"xytext": [0.0, 0.0], "textcoords": "data",
                                           "arrowprops": {"arrowstyle": "->", "color": "#c2410c", "lw": 1.2}}})
            annotations.append({"type": "text", "x": float(x), "y": float(y), "text": str(name),
                                "kwargs": {"color": "#c2410c", "fontsize": 8,
                                           "ha": "left" if x >= 0 else "right", "va": "bottom" if y >= 0 else "top"}})
        series = [(*item[:3], {**item[3], "alpha": 0.6}) for item in self._row_series(table, "Prin1", "Prin2")]
        return ("Scatter Plot", _("Biplot"), "Prin1", "Prin2", series, {**zero, "annotations": annotations})
