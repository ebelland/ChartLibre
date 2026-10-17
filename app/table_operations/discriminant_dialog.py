"""Discriminant (Analysis): which group a row belongs to, from continuous measures (AN-20).

As JMP's Discriminant platform: Y is the group, the X covariates the
measures. The report gives the misclassification, Wilks' lambda, the
canonical analysis and the confusion matrix; OK writes each row's predicted
group, its probabilities and canonical scores beside its data, and draws the
canonical plot. The engine is app.analysis.discriminant.
"""
from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import discriminant as da
from app.plugin_api.data_operations.parameter_spec import ChoiceParam
from app.plugin_api.table_operations.column_roles import CONTINUOUS, NOMINAL, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class DiscriminantDialog(TableOperationDialogBase):
    """Discriminant analysis: classify the rows into the groups of Y from the X measures."""

    Name = "Discriminant"
    Description = (
        "Which group a row belongs to, from continuous measures: linear or quadratic "
        "discriminant analysis, canonical plot, Wilks' lambda and the misclassified rows."
    )
    Category = "Multivariate Methods"
    Order = 42
    Icon = """
    <circle cx="6" cy="8" r="1.6"/><circle cx="8.5" cy="5" r="1.6"/><circle cx="5" cy="12.5" r="1.6"/>
    <circle cx="17" cy="15" r="1.6"/><circle cx="19" cy="19" r="1.6"/><circle cx="15" cy="19.5" r="1.6"/>
    <path d="M3 21L21 3"/>
    """
    Keywords: str = (
        "discriminant discriminante lda qda linear discriminant analysis analisi discriminante "
        "classification classificazione canonical canonica wilks groups gruppi"
    )
    Goals: list[str] = ["classify", "groups"]

    ROLES = (
        ColumnRole("response", "Y, group", "The column whose groups the rows are classified into.",
                   kinds=(NOMINAL, ORDINAL), minimum=1, maximum=1),
        ColumnRole("covariate", "X, covariates", "The continuous measures that tell the groups apart.",
                   kinds=(CONTINUOUS,), minimum=1),
    )
    PARAMETERS_TAB = "Discriminant"
    PARAMS = (
        ChoiceParam("method", "Method:", choices=tuple((label, key) for key, label in da.METHODS),
                    tooltip="Linear: the groups share one covariance, the boundaries are planes. "
                            "Quadratic: each group has its own, the boundaries curve; it needs more rows."),
        ChoiceParam("priors", "Prior probabilities:", choices=tuple((label, key) for key, label in da.PRIORS),
                    tooltip="How likely each group is before the measures are seen: its share of the rows, "
                            "or the same for every group."),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        overlap = set(casting.columns("response")) & set(casting.columns("covariate"))
        if not found and overlap:
            found.append(_("{columns} is both Y and X.").format(columns=", ".join(sorted(overlap))))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        return da.discriminant(
            frame,
            casting.columns("response")[0],
            casting.columns("covariate"),
            method=str(params.get("method") or da.LINEAR),
            priors=str(params.get("priors") or da.PROPORTIONAL),
        )

    def format_results(self, result: da.DiscriminantResult) -> str:
        return report_html.document(_(self.Name), f"{self.table} · {result.response}", self._report(result))

    def _report(self, result: da.DiscriminantResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        method = _(dict(da.METHODS)[result.method])
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            [(_("Method"), method)]
            + [(_(key), report_html.format_number(value)) for key, value in result.summary.items()]
        )))
        confusion = result.confusion.reset_index()
        blocks.append(report_html.section(
            _("Confusion matrix"),
            report_html.table([_("Actual")] + [html.escape(str(column)) for column in result.confusion.columns],
                              self.report_rows(confusion)),
        ))
        blocks.append(report_html.section(_("Canonical analysis"), self.report_table(result.canonical)))
        blocks.append(report_html.section(_("Standardized canonical coefficients"),
                                          self.report_table(result.coefficients)))
        blocks.append(report_html.section(_("Group means"), self.report_table(result.means)))
        return report_html.section(_("Response {name}").format(name=result.response), *blocks)

    def apply_results(self, result: da.DiscriminantResult) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        names = {key: repo.free_table_name(f"{table}_discriminant{suffix}") for key, suffix in (
            ("rows", ""), ("canonical", "_canonical"), ("coefficients", "_coefficients"), ("means", "_means"),
        )}
        repo.snapshot_for_undo([*names.values(), *repo.DESCRIPTOR_TABLES], label=f"Discriminant on '{table}'")

        # Each row's group, probabilities and canonical scores beside its data, as Partition does.
        frame = self._frame.copy()
        predicted = f"Predicted {result.response}"
        frame[predicted] = result.predicted.reindex(frame.index)
        for part in (result.probabilities, result.scores):
            for column in part.columns:
                frame[column] = part[column].reindex(frame.index)
        repo.import_dataframe(frame, table_name=names["rows"], normalize_columns=False)
        for key in ("canonical", "coefficients", "means"):
            repo.import_dataframe(getattr(result, key), table_name=names[key], normalize_columns=False)
        written: list[dict[str, Any]] = [{"table": name} for name in names.values()]

        notes = report_html.document(_(self.Name), f"{table} · {result.response}", self._report(result))
        figure_id = self.create_report_figure(f"Discriminant · {result.response}", notes,
                                              [self._canonical_plot(result, names["rows"])])
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": result.response})
        return written

    def _canonical_plot(self, result: da.DiscriminantResult, rows: str) -> tuple:
        """JMP's canonical plot: the rows on the first two canonical axes, a series per group.
        With two groups there is one axis: the scores by group, as box plots."""
        canonical = list(result.scores.columns)
        response = self.quoted(result.response)
        if len(canonical) < 2:
            sql, roles = self.series_select(rows, {"value": canonical[0], "group": result.response},
                                            where=f"{response} IS NOT NULL")
            return ("Box Plot", _("Canonical scores by {response}").format(response=result.response),
                    result.response, canonical[0], [(result.response, sql + f" ORDER BY {response}", roles, {})])
        series = []
        for level in result.levels:
            sql, roles = self.series_select(rows, {"x": canonical[0], "y": canonical[1]},
                                            where=f"{response} = {self.literal(str(level))}")
            series.append((str(level), sql, roles, {}))
        return ("Scatter Plot", _("Canonical plot"), canonical[0], canonical[1], series)
