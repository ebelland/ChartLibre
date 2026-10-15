"""Distribution (Analysis): one column's quantiles, moments, normality and capability (AN-02).

Cast one or more columns as Y; give a lower and/or an upper specification
limit (and a target) for process capability. The report gives, per column,
the quantile table, the moments, the normality tests and - with a limit -
Cp, Cpk, Pp, Ppk, Cpm and the share out of specification. OK writes those
as tables and draws, per column, the histogram with its normal curve and the
limits, beside the normal Q-Q plot. The engine is app.analysis.distribution.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.analysis import distribution as dist
from app.data_operations.parameter_spec import TextParam
from app.table_operations.column_roles import CONTINUOUS, ORDINAL, ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

_LIMIT_TOOLTIP = "Leave empty when there is none. The same limits apply to every Y column."


@dataclass(slots=True)
class DistributionReport:
    results: list[dist.DistributionResult]


def _limit(text: Any) -> float | None:
    """A typed limit, or None for an empty field."""
    text = str(text or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"'{text}' is not a number.") from exc


class DistributionDialog(TableOperationDialogBase):
    """Each column's distribution, and its capability against specification limits."""

    Name = "Distribution"
    Description = ("One column at a time: quantiles, moments, normality tests, and process capability "
                   "(Cp, Cpk, Pp, Ppk) against specification limits.")
    Category = "Distributions"
    Icon = """
    <path d="M4 20V14h3v6M8.5 20V8h3v12M13 20V5h3v15M17.5 20v-9h3v9"/><path d="M3 20h18"/>
    """

    ROLES = (
        ColumnRole("response", "Y", "The columns to describe, one report each.",
                   kinds=(CONTINUOUS, ORDINAL), minimum=1),
    )
    PARAMETERS_TAB = "Capability"
    PARAMS = (
        TextParam("lsl", "Lower spec limit (LSL):", tooltip=_LIMIT_TOOLTIP, placeholder="none"),
        TextParam("usl", "Upper spec limit (USL):", tooltip=_LIMIT_TOOLTIP, placeholder="none"),
        TextParam("target", "Target:", tooltip="For Cpm; needs both limits. " + _LIMIT_TOOLTIP,
                  placeholder="none"),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    def problems(self) -> list[str]:
        found = super().problems()
        params = self.parameter_values()
        try:
            lsl, usl, _target = (_limit(params.get(key)) for key in ("lsl", "usl", "target"))
        except ValueError as exc:
            return [*found, str(exc)]
        if lsl is not None and usl is not None and lsl >= usl:
            found.append(_("The lower specification limit must be below the upper one."))
        return found

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        limits = {key: _limit(params.get(key)) for key in ("lsl", "usl", "target")}
        return DistributionReport([dist.distribution(frame, column, **limits)
                                   for column in casting.columns("response")])

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def format_results(self, report: DistributionReport) -> str:
        return report_html.document(_(self.Name), str(self.table), self._report(report))

    def _report(self, report: DistributionReport) -> str:
        blocks = []
        for result in report.results:
            parts = [report_html.note(_(note)) for note in result.notes]
            parts.append(report_html.section(_("Quantiles"), self.report_table(result.quantiles)))
            parts.append(report_html.section(_("Moments"), self.report_table(result.moments)))
            parts.append(report_html.section(_("Normality"), self.report_table(result.normality,
                                                                                p_columns=("pvalue",))))
            capability = result.capability
            if capability is not None:
                indices = pd.DataFrame({"Index": list(capability.indices), "Value": list(capability.indices.values())})
                sigmas = report_html.summary_table((
                    (_("Sigma (within, moving range)"), report_html.format_number(capability.sigma_within)),
                    (_("Sigma (overall)"), report_html.format_number(capability.sigma_overall)),
                ))
                parts.append(report_html.section(_("Capability"), sigmas + self.report_table(indices)
                                                 + self.report_table(capability.outside)))
            blocks.append(report_html.section(f"{result.column} (N = {result.n})", "".join(parts)))
        return "".join(blocks)

    # ------------------------------------------------------------------
    # OK: tables and a figure
    # ------------------------------------------------------------------

    def apply_results(self, report: DistributionReport) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        frames: dict[str, list[pd.DataFrame]] = {"quantiles": [], "moments": [], "normality": [], "capability": []}
        for result in report.results:
            for key, frame in (("quantiles", result.quantiles), ("moments", result.moments),
                               ("normality", result.normality)):
                frames[key].append(frame.assign(Column=result.column))
            if result.capability is not None:
                indices = pd.DataFrame({"Index": list(result.capability.indices),
                                        "Value": list(result.capability.indices.values())})
                frames["capability"].append(indices.assign(Column=result.column))
        names = {key: repo.free_table_name(f"{table}_distribution_{key}") for key, parts in frames.items() if parts}
        repo.snapshot_for_undo([*names.values(), *repo.DESCRIPTOR_TABLES], label=f"Distribution on '{table}'")
        written: list[dict[str, Any]] = []
        for key, name in names.items():
            frame = pd.concat(frames[key], ignore_index=True)
            frame = frame[["Column", *[c for c in frame.columns if c != "Column"]]]
            repo.import_dataframe(frame, table_name=name, normalize_columns=False)
            written.append({"table": name})

        panels = []
        for result in report.results:
            column = result.column
            lines = []
            capability = result.capability
            if capability is not None:
                for value, label, style in ((capability.lsl, "LSL", "--"), (capability.usl, "USL", "--"),
                                            (capability.target, _("Target"), ":")):
                    if value is not None:
                        lines.append({"orientation": "vertical", "value": value,
                                      "kwargs": {"color": "#d62728", "linestyle": style, "linewidth": 1.2,
                                                 "label": label}})
            sql, roles = self.series_select(table, {"value": column})
            panels.append(("Histogram", column, column, _("Density"),
                           [(column, sql, roles, {})], {"lines": lines, "distribution_fit": "norm"}))
            panels.append(("Q-Q Plot", _("Normal quantile plot"), _("Normal quantiles"), column,
                           [(column, sql, roles, {})]))
        columns = ", ".join(result.column for result in report.results)
        notes = report_html.document(_(self.Name), f"{table} · {columns}", self._report(report))
        figure_id = self.create_report_figure(f"{_(self.Name)} · {columns}", notes, panels)
        self.created_figure_ids = [figure_id]
        written.append({"figure": figure_id, "name": columns})
        return written
