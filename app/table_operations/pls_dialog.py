"""PLS (Analysis): Y from many collinear X - a spectrum's wavelengths - through a few components (AN-17).

Y and the X are cast from the table's columns. The number of components is
chosen by cross-validation unless set. The report gives the
cross-validation table, the variables' importance (VIP), the loadings and
the coefficients; OK writes each row's scores and prediction beside its
data, the VIP, loadings, coefficients and cross-validation as tables, and a
figure: RMSECV by components, VIP along the variables, the scores, and
actual by predicted. The engine is app.analysis.pls.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.analysis import pls as pl
from app.plugin_api.data_operations.parameter_spec import BoolParam, IntParam
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class PLSDialog(TableOperationDialogBase):
    """Partial least squares: predict Y from many X that move together."""

    Name = "Partial Least Squares"
    Description = (
        "Y from many X that move together - a spectrum's wavelengths: a few latent components, "
        "their number chosen by cross-validation; variable importance (VIP), loadings, scores, predictions."
    )
    Category = "Multivariate Methods"
    Order = 43
    Icon = """
    <path d="M3 18c2-9 4-12 6-6s3 6 5-2 3-6 7 1"/><path d="M3 21h18"/>
    """
    Keywords: str = (
        "pls partial least squares minimi quadrati parziali spectra spettri spectrum spettro nir "
        "chemometrics chemiometria calibration calibrazione vip latent components componenti latenti "
        "multivariate calibration"
    )
    Goals: list[str] = ["predict", "explain"]

    ROLES = (
        ColumnRole("response", "Y", "The values to predict: a concentration, a property.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1),
        ColumnRole("predictor", "X", "The many measures to predict them from: a spectrum's columns.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=2),
    )
    PARAMETERS_TAB = "PLS"
    PARAMS = (
        IntParam("components", "Components:", tooltip="0 chooses them by cross-validation: another component only while "
                                                      "it cuts the prediction error by 5% or more.",
                 default_value=0, minimum=0, maximum=pl.MAX_COMPONENTS),
        IntParam("folds", "Cross-validation folds:", tooltip="Into how many parts the rows are split; each part "
                                                             "is predicted by a model fitted on the others.",
                 default_value=7, minimum=2, maximum=20, advanced=True),
        BoolParam("scale", "Scale X to unit variance", tooltip="For X in different units. Off for a spectrum: "
                                                                "scaling gives the wavelengths that carry only "
                                                                "noise the weight of the bands.",
                  default_value=False, advanced=True),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        casting = self.roles_widget.casting()
        overlap = set(casting.columns("response")) & set(casting.columns("predictor"))
        if not found and overlap:
            found.append(_("{columns} is both Y and X.").format(columns=", ".join(sorted(overlap))))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        components = int(params.get("components", 0) or 0)
        return pl.pls(frame, casting.columns("response"), casting.columns("predictor"),
                      components=components or None, folds=int(params.get("folds", 7)),
                      scale=bool(params.get("scale", False)))

    def report_subtitle(self, result: pl.PLSResult) -> str:
        return f"{self.table} · {', '.join(result.responses)}"

    def report_body(self, result: pl.PLSResult) -> str:
        blocks = [report_html.note(_(note)) for note in result.notes]
        blocks.append(report_html.section(_("Summary"), report_html.summary_table(
            (_(key), report_html.format_number(value)) for key, value in result.summary.items())))
        blocks.append(report_html.section(_("Cross-validation"), self.report_table(result.cross_validation)))
        top = result.vip.sort_values("VIP", ascending=False).head(15)
        blocks.append(report_html.section(_("Variable importance (VIP), highest 15"), self.report_table(top)))
        blocks.append(report_html.section(_("Coefficients"), self.report_table(result.coefficients.head(50))))
        return "".join(blocks)

    def apply_results(self, result: pl.PLSResult) -> list[dict[str, Any]]:
        frame = self._frame.copy()
        for part in (result.scores, result.predicted):
            for column in part.columns:
                frame[column] = part[column].reindex(frame.index)
        tables = self.write_result_tables({"pls": frame, "pls_vip": result.vip, "pls_loadings": result.loadings,
                                           "pls_coefficients": result.coefficients, "pls_cv": result.cross_validation})
        names = {"rows": tables["pls"], "vip": tables["pls_vip"], "cv": tables["pls_cv"]}
        written: list[dict[str, Any]] = [{"table": name} for name in tables.values()]

        line = {"linestyle": "-", "marker": "o"}
        sql, roles = self.series_select(names["cv"], {"x": "Components", "y": "RMSECV"})
        panels = [("Scatter Plot", _("RMSECV by number of components"), _("Components"), "RMSECV",
                   [("RMSECV", sql, roles, line)])]
        sql, roles = self.series_select(names["vip"], {"x": "Order", "y": "VIP"})
        panels.append(("Scatter Plot", _("Variable importance (VIP)"), _("Variable (in column order)"), "VIP",
                       [("VIP", sql, roles, {"linestyle": "-", "marker": ""})],
                       {"lines": [{"orientation": "horizontal", "value": pl.VIP_THRESHOLD,
                                   "kwargs": {"color": "0.5", "linestyle": "--", "linewidth": 1}}]}))
        if "T2" in result.scores.columns:
            sql, roles = self.series_select(names["rows"], {"x": "T1", "y": "T2"})
            panels.append(("Scatter Plot", _("Scores"), "T1", "T2", [(_("Rows"), sql, roles, {})]))
        response = result.responses[0]
        sql, roles = self.series_select(names["rows"], {"x": f"Predicted {response}", "y": response})
        panels.append(("Scatter Plot", _("Actual by predicted"), f"Predicted {response}", response,
                       [(response, sql, roles, {})]))
        notes = self.report_document(result)
        figure_id = self.create_report_figure(f"PLS · {response}", notes, panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": response})
        return written
