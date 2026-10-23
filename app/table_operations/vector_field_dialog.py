"""Vector Field (I-04): the divergence and curl of a field sampled on a grid, in 2D or 3D.

Choose the coordinates (x, y, and z for 3D) and the components (u, v, and w).
app.analysis.vector_calculus differentiates the field; OK writes the nodes
with their divergence and curl, and draws them: in 2D a heatmap of each, in
3D the field coloured by its divergence and the curl coloured by its size.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from app.analysis import vector_calculus as vc
from app.plugin_api.table_operations.column_roles import CONTINUOUS, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _


class VectorFieldDialog(TableOperationDialogBase):
    Name = "Vector Field"
    Description = ("A vector field sampled on a grid: its divergence (where it spreads out or gathers) and its "
                   "curl (where and how fast it turns), in 2D or 3D.")
    Category = "Curves"
    CHART_PANELS = ('Heatmap', 'Quiver 3D')
    Order = 70
    Icon = """
    <path d="M4 18l5-5"/><path d="M9 13l-1 3"/><path d="M9 13l-3 1"/><path d="M12 8l6-3"/><path d="M18 5l-3 0"/>
    <path d="M18 5l-1 3"/><path d="M14 18l5-1"/>
    """
    Keywords: str = ("vector field campo vettoriale divergence divergenza curl rotore vorticity vorticità flow "
                     "flusso gradient gradiente 3d quiver")
    Goals: list[str] = ["signal", "relate"]

    ROLES = (
        ColumnRole("x", "X", "The nodes' x.", kinds=(CONTINUOUS,), minimum=1, maximum=1),
        ColumnRole("y", "Y", "The nodes' y.", kinds=(CONTINUOUS,), minimum=1, maximum=1),
        ColumnRole("z", "Z", "Optional: the nodes' z, for a field in space.", kinds=(CONTINUOUS,), maximum=1),
        ColumnRole("u", "U", "The field's x component.", kinds=(CONTINUOUS,), minimum=1, maximum=1),
        ColumnRole("v", "V", "The field's y component.", kinds=(CONTINUOUS,), minimum=1, maximum=1),
        ColumnRole("w", "W", "Optional: the field's z component, with Z.", kinds=(CONTINUOUS,), maximum=1),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def _columns(self) -> dict[str, str]:
        casting = self.roles_widget.casting()
        return {role: (casting.columns(role) or [""])[0] for role in ("x", "y", "z", "u", "v", "w")}

    def problems(self) -> list[str]:
        found = super().problems()
        columns = self._columns()
        if not found and bool(columns["z"]) != bool(columns["w"]):
            found.append(_("A field in space needs both Z and W; a flat one neither."))
        used = [name for name in columns.values() if name]
        if not found and len(set(used)) != len(used):
            found.append(_("Each role needs a column of its own."))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del casting, params, model
        return vc.derivatives(frame, **self._columns())

    def report_body(self, result: vc.FieldDerivatives) -> str:
        columns = ["divergence", "curl"] if result.dims == 2 else ["divergence", "curl_magnitude"]
        rows = []
        for column in columns:
            values = result.frame[column].to_numpy(float)
            rows.append((_(column.replace("_", " ").capitalize()), report_html.format_number(float(np.min(values))),
                         report_html.format_number(float(np.mean(values))),
                         report_html.format_number(float(np.max(values)))))
        table = report_html.table((_("Quantity"), _("Min"), _("Mean"), _("Max")), rows,
                                  align=("left", "right", "right", "right"))
        note = report_html.note(_("Central differences on the grid (numpy.gradient); one-sided at its edges, where "
                                  "they are less accurate."))
        return report_html.section(_("{n} nodes, {dims}D").format(n=len(result.frame), dims=result.dims), table, note)

    def report_subtitle(self, result: vc.FieldDerivatives) -> str:
        return f"{self.table} · {result.dims}D"

    def apply_results(self, result: vc.FieldDerivatives) -> list[dict[str, Any]]:
        name = self.write_result_tables({"field": result.frame})["field"]
        columns = self._columns()
        if result.dims == 2:
            panels = []
            for quantity, title in (("divergence", _("Divergence")), ("curl", _("Curl"))):
                sql, roles = self.series_select(name, {"x": "x", "y": "y", "z": quantity})
                panels.append(("Heatmap", title, columns["x"], columns["y"], [(title, sql, roles, {})],
                               {"colorbar": True, "cmap": "RdBu_r"}))
        else:
            field_sql, field_roles = self.series_select(name, {"x": "x", "y": "y", "z": "z", "u": "u", "v": "v",
                                                               "w": "w", "color": "divergence"})
            curl_sql, curl_roles = self.series_select(name, {"x": "x", "y": "y", "z": "z", "u": "curl_x",
                                                             "v": "curl_y", "w": "curl_z", "color": "curl_magnitude"})
            panels = [
                ("Quiver 3D", _("The field, coloured by its divergence"), columns["x"], columns["y"],
                 [(_("Field"), field_sql, field_roles, {})]),
                ("Quiver 3D", _("Its curl, coloured by its size"), columns["x"], columns["y"],
                 [(_("Curl"), curl_sql, curl_roles, {})]),
            ]
        figure_id = self.create_report_figure(f"{_(self.Name)} · {self.table}", self.report_document(result), panels)
        self.created_figure_ids = [figure_id]
        return [{"table": name}, {"figure": figure_id, "name": str(self.table)}]
