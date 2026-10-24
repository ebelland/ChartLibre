"""Factor Analysis (Analysis): a few hidden factors behind many correlated columns (AN-04).

As JMP's Factor Analysis (Multivariate Methods): the common factors of the
columns' correlations, by Maximum Likelihood or Principal Axis, rotated so
that each column loads on few of them - Varimax, Quartimax, Equamax
(orthogonal) or Promax (oblique, with the factors' correlations). The report
gives the variance each factor explains, the rotated loadings, each column's
communality and uniqueness. OK writes the rows with their factor scores, the
loadings and the eigenvalues, and draws the scree plot, the loading plot and
the score plot. The engine is app.analysis.multivariate.factor_analysis.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import multivariate as mv
from app.plugin_api.data_operations.parameter_spec import ChoiceParam, IntParam
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class FactorAnalysisDialog(TableOperationDialogBase):
    """Common factors: rotated loadings, communalities, each row's scores."""

    Name = "Factor Analysis"
    Description = (
        "A few hidden factors behind many correlated columns: Maximum Likelihood or Principal Axis, "
        "rotated (Varimax, Quartimax, Equamax, Promax); loadings, communalities, each row's scores."
    )
    Category = "Multivariate Methods"
    Order = 41
    Icon = """
    <circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/>
    <rect x="15" y="3" width="5" height="3.5" rx="0.8"/><rect x="15" y="10.2" width="5" height="3.5" rx="0.8"/>
    <rect x="15" y="17.5" width="5" height="3.5" rx="0.8"/>
    <path d="M8 6.5l7-1.5M8 7l7 4.5M8 17l7-4.5M8 17.5l7 1.5"/>
    """
    Keywords: str = (
        "factor analysis analisi fattoriale fattori latenti latent factors varimax quartimax equamax "
        "promax rotation rotazione loadings saturazioni communality comunalità maximum likelihood "
        "principal axis assi principali"
    )
    Goals: list[str] = ["explain", "groups"]

    ROLES = (
        ColumnRole("response", "Y, columns", "The continuous columns whose correlations the factors explain.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=3),
        ColumnRole("by", "By", "Optional: a column of groups - species, batch, site; the score plot colours "
                               "the rows by it.", maximum=1),
    )
    PARAMETERS_TAB = "Factor Analysis"
    PARAMS = (
        IntParam("factors", "Factors:", tooltip="0 takes as many as the correlations have eigenvalues above 1 "
                                                "(see the scree plot).",
                 default_value=0, minimum=0, maximum=50),
        ChoiceParam("method", "Method:", choices=tuple((label, key) for key, label in mv.FA_METHODS),
                    tooltip="Maximum Likelihood: JMP's default, for columns roughly normal. "
                            "Principal Axis: no assumption of normality, steadier with few rows."),
        ChoiceParam("rotation", "Rotation:", choices=tuple((label, key) for key, label in mv.ROTATIONS),
                    tooltip="Varimax: each factor with few large loadings, the usual choice. Quartimax: each "
                            "column on few factors. Equamax: between the two. Promax: lets the factors "
                            "correlate, when the traits they stand for do."),
    )


    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        if not found and set(casting.columns("response")) & set(casting.columns("by")):
            found.append(_("The By column cannot also be one of the columns."))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        return mv.factor_analysis(frame, casting.columns("response"), factors=int(params.get("factors", 0) or 0),
                                  method=str(params.get("method") or mv.ML),
                                  rotation=str(params.get("rotation") or mv.VARIMAX))

    def report_subtitle(self, result: mv.FactorResult) -> str:
        return (f"{self.table} · {_(dict(mv.FA_METHODS)[result.method])} · "
                f"{_(dict(mv.ROTATIONS)[result.rotation])}")

    def report_body(self, result: mv.FactorResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), report_html.format_number(value)) for key, value in result.summary.items())))
        blocks.append(report_html.section(_("Variance explained"), self.report_table(result.variance, digits=4)))
        blocks.append(report_html.section(_("Rotated factor loadings"), self.report_table(result.loadings, digits=3)))
        blocks.append(report_html.section(_("Communalities"), self.report_table(result.communalities, digits=3)))
        if not result.factor_correlations.empty:
            blocks.append(report_html.section(_("Factor correlations"),
                                              self.report_table(result.factor_correlations, digits=3)))
        return "".join(blocks)

    def apply_results(self, result: mv.FactorResult) -> list[dict[str, Any]]:
        frame = self._frame.copy()
        for column in result.scores.columns:
            frame[column] = result.scores[column].reindex(frame.index)
        tables = self.write_result_tables({"factors": frame, "factor_loadings": result.loadings,
                                           "factor_eigenvalues": result.eigenvalues})
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]

        one = {"lines": [{"orientation": "horizontal", "value": 1.0,
                          "kwargs": {"color": "0.6", "linestyle": "--", "linewidth": 0.8}}]}
        sql, roles = self.series_select(tables["factor_eigenvalues"], {"x": "Number", "y": "Eigenvalue"})
        panels: list[tuple] = [("Scatter Plot", _("Scree plot"), _("Number"), _("Eigenvalue"),
                                [(_("Eigenvalue"), sql, roles, {"linestyle": "-", "marker": "o"})], one)]
        if result.factors >= 2:
            zero = [{"orientation": orientation, "value": 0.0,
                     "kwargs": {"color": "0.6", "linestyle": "--", "linewidth": 0.8}}
                    for orientation in ("horizontal", "vertical")]
            # Each column a point labelled with its name: which factor it belongs to.
            sql, roles = self.series_select(tables["factor_loadings"], {"x": "Factor1", "y": "Factor2"})
            labels = [{"type": "text", "x": float(row.Factor1), "y": float(row.Factor2), "text": str(row.Variable),
                       "kwargs": {"fontsize": 8, "ha": "left", "va": "bottom"}}
                      for row in result.loadings.itertuples()]
            panels.append(("Scatter Plot", _("Factor loading plot"), "Factor1", "Factor2",
                           [(_("Columns"), sql, roles, {"marker": "o"})], {"lines": zero, "annotations": labels}))
            by = self.columns("by")
            panels.append(("Scatter Plot", _("Score plot"), "Factor1", "Factor2",
                           self.grouped_series(tables["factors"], "Factor1", "Factor2", by[0] if by else ""),
                           {"lines": zero}))
        return self.draw_report(result, self.table, panels, written=written)
