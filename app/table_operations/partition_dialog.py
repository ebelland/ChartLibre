"""Partition (Analysis): a decision tree of one response on many factors, as JMP's.

Y and the X factors are cast from the table's columns; Split and Prune grow
or cut the tree a split at a time, as JMP's buttons do, and Automatic grows
it while the validation rows say it still predicts better. OK writes each
row's leaf and prediction beside its data, the leaves, the history and the
column contributions as tables, and a figure of the charts ticked with the
report in its notes. The engine is app.analysis.partition.
"""
from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from app.analysis import partition as pt
from app.series_operations.parameter_spec import FloatParam, IntParam
from app.styles.style import TitledCard
from app.table_operations.column_roles import NOMINAL, ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

#: The charts OK can draw: (key, check box label, ticked by default).
CHARTS: tuple[tuple[str, str, bool], ...] = (
    ("contributions", "Column contributions", True),
    ("history", "R² or misclassification by splits", True),
    ("actual", "Actual by predicted (continuous Y)", True),
    ("leaves", "Leaf means or shares", False),
)

_JMP_HELP = "https://www.jmp.com/support/help/en/18.0/index.shtml#page/jmp/"
PARTITION_DOC = _JMP_HELP + "overview-of-the-partition-platform.shtml"


class PartitionDialog(TableOperationDialogBase):
    """Partition: split the rows by the factors that best separate the response."""

    Name = "Partition"
    Description = (
        "Decision tree of a continuous or nominal response on many factors: "
        "split and prune a step at a time, or automatically on validation rows; "
        "leaves, column contributions and each row's leaf and prediction."
    )
    Category = "Models"
    Icon = """
    <circle cx="12" cy="4.5" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="18" cy="12" r="2"/>
    <circle cx="3.5" cy="19.5" r="1.6"/><circle cx="9" cy="19.5" r="1.6"/>
    <path d="M10.6 6l-3.2 4.4M13.4 6l3.2 4.4M5 13.8l-1 4M7 13.8l1.4 4"/>
    """

    ROLES = (
        ColumnRole("response", "Y", "The column to predict: continuous gives a regression tree, nominal a classification tree.",
                   minimum=1, maximum=1),
        ColumnRole("factor", "X", "The columns the tree may split on.", minimum=1),
        ColumnRole("validation", "Validation",
                   "Optional: 1 (or Validation) on the rows kept out of the fit, to judge it.", maximum=1),
    )
    #: The tree's own settings, beside the Split and Prune buttons.
    PARAMETERS_TAB = "Tree"

    PARAMS = (
        IntParam("splits", "Splits:", tooltip="How many times the rows are split: one fewer than the leaves.",
                 default_value=3, minimum=1, maximum=60),
        IntParam("min_leaf", "Minimum leaf size:", tooltip="The fewest rows a leaf may hold.",
                 default_value=5, minimum=1, maximum=10_000),
        FloatParam("validation_fraction", "Validation share:",
                   tooltip="With no validation column: this share of rows, at random, judges the tree. 0 for none.",
                   default_value=0.0, minimum=0.0, maximum=0.6, decimals=2, step=0.05),
        IntParam("seed", "Random seed:", tooltip="Which rows the validation share draws.",
                 default_value=1, minimum=0, maximum=1_000_000),
    )

    def __init__(self, **kwargs: Any) -> None:
        self._auto = False
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Inputs: Split, Prune, Automatic; the charts
    # ------------------------------------------------------------------

    def build_extra_inputs(self, layout: QVBoxLayout) -> None:
        grow = TitledCard(self, _("Grow the tree"))
        row = QHBoxLayout()
        row.setSpacing(4)
        for text, tooltip, action in (
            (_("Split"), _("One more split: the one that best separates the response."), lambda: self._step(+1)),
            (_("Prune"), _("Undo the last split."), lambda: self._step(-1)),
            (_("Automatic"), _("Split while the validation rows say the tree predicts better."), self._automatic),
        ):
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(action)
            row.addWidget(button)
        row.addStretch(1)
        grow.card.layout().addLayout(row)
        link = QLabel(f"{html.escape(_('JMP documentation:'))} <a href='{PARTITION_DOC}'>{html.escape(_('Partition'))}</a>")
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        grow.card.layout().addWidget(link)
        self.roles_widget.add_tab(_(self.PARAMETERS_TAB)).addWidget(grow)

        charts = TitledCard(self, _("Charts to draw"))
        grid = QVBoxLayout()
        grid.setSpacing(10)
        self._chart_checks: dict[str, QCheckBox] = {}
        for key, label, ticked in CHARTS:
            check = QCheckBox(_(label))
            check.setChecked(ticked)
            # An attribute of its own, so the window remembers it.
            setattr(self, f"_chart_{key}", check)
            self._chart_checks[key] = check
            grid.addWidget(check)
        charts.card.layout().addLayout(grid)
        self.roles_widget.add_tab(_("Charts")).addWidget(charts)

    def _step(self, change: int) -> None:
        """Split or Prune: one split more or fewer, and the preview at once."""
        assert self._parameter_form is not None
        splits = int(self.parameter_values().get("splits", 3)) + change
        self._auto = False
        self._parameter_form.set_values({"splits": max(1, splits)})
        self.inputs_changed()
        self.preview()

    def _automatic(self) -> None:
        self._auto = True
        self.inputs_changed()
        self.preview()

    def charts(self) -> list[str]:
        return [key for key, check in self._chart_checks.items() if check.isChecked()]

    def problems(self) -> list[str]:
        found = super().problems()
        if found:
            return found
        casting = self.roles_widget.casting()
        overlap = set(casting.columns("response")) & set(casting.columns("factor"))
        if overlap:
            found.append(_("{columns} is both Y and X.").format(columns=", ".join(sorted(overlap))))
        return found

    def _inputs(self) -> tuple[Any, ...]:
        return (*super()._inputs(), self._auto)

    def parameter_values(self) -> dict[str, Any]:
        values = super().parameter_values()
        values["auto"] = self._auto
        return values

    # ------------------------------------------------------------------
    # The calculation
    # ------------------------------------------------------------------

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        validation = casting.columns("validation")
        spec = pt.PartitionSpec(
            response=casting.columns("response")[0],
            factors=casting.columns("factor"),
            kinds={name: (pt.NOMINAL if kind == NOMINAL else pt.CONTINUOUS) for name, kind in casting.kinds.items()},
            splits=int(params.get("splits", 3)),
            min_leaf=int(params.get("min_leaf", 5)),
            validation_column=validation[0] if validation else None,
            validation_fraction=float(params.get("validation_fraction", 0.0)),
            seed=int(params.get("seed", 1)),
            auto=bool(params.get("auto")),
        )
        return pt.partition(frame, spec)

    def format_results(self, result: pt.PartitionResult) -> str:
        if self._auto and self._parameter_form is not None:
            # What Automatic chose becomes the count Split and Prune go on from.
            self._parameter_form.set_values({"splits": result.splits})
        return report_html.document(_(self.Name), f"{self.table} · {result.response}", self._report(result))

    @staticmethod
    def _rows(frame: pd.DataFrame, digits: int = 6) -> list[list[str]]:
        rows = []
        for record in frame.to_dict("records"):
            cells = []
            for value in record.values():
                if isinstance(value, str):
                    cells.append(html.escape(value))
                elif isinstance(value, (int,)) and not isinstance(value, bool):
                    cells.append(str(value))
                else:
                    cells.append(report_html.format_number(value, digits=digits))
            rows.append(cells)
        return rows

    def _report(self, result: pt.PartitionResult) -> str:
        blocks = []
        if result.note:
            blocks.append(report_html.note(result.note))
        kind = _("Regression tree") if result.kind == pt.CONTINUOUS else _("Classification tree")
        blocks.append(report_html.section(
            _("Summary"),
            report_html.summary_table(
                [(_("Tree"), kind)] + [(_(key), report_html.format_number(value)) for key, value in result.summary.items()]
            ),
        ))
        blocks.append(report_html.section(
            _("Column contributions"),
            report_html.table([_(c) for c in result.contributions.columns], self._rows(result.contributions)),
        ))
        blocks.append(report_html.section(
            _("Leaf report"),
            report_html.table([_(c) for c in result.leaves.columns], self._rows(result.leaves)),
        ))
        blocks.append(report_html.section(
            _("Split history"),
            report_html.table([_(c) for c in result.history.columns], self._rows(result.history)),
        ))
        if result.confusion is not None:
            confusion = result.confusion.reset_index()
            blocks.append(report_html.section(
                _("Confusion matrix (training rows)"),
                report_html.table([_("Actual")] + [html.escape(str(c)) for c in result.confusion.columns],
                                  self._rows(confusion)),
            ))
        return report_html.section(_("Response {name}").format(name=result.response), *blocks)

    # ------------------------------------------------------------------
    # OK: the results into the project
    # ------------------------------------------------------------------

    def apply_results(self, result: pt.PartitionResult) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        names = {key: repo.free_table_name(f"{table}_partition{suffix}") for key, suffix in (
            ("rows", ""), ("leaves", "_leaves"), ("history", "_history"), ("contributions", "_contributions"),
        )}
        charts = self.charts()
        repo.snapshot_for_undo(
            [*names.values(), *(repo.DESCRIPTOR_TABLES if charts else ())],
            label=f"Partition on '{table}'",
        )

        # Each row's leaf and prediction beside its data, like ClusterId.
        frame = self._frame.copy()
        frame["Leaf"] = result.leaf.reindex(frame.index)
        frame[f"Predicted {result.response}"] = result.predicted.reindex(frame.index)
        if result.probabilities is not None:
            for column in result.probabilities.columns:
                frame[column] = result.probabilities[column].reindex(frame.index)
        frame["Partition role"] = result.role.reindex(frame.index)
        repo.import_dataframe(frame, table_name=names["rows"], normalize_columns=False)
        for key in ("leaves", "history", "contributions"):
            repo.import_dataframe(getattr(result, key), table_name=names[key], normalize_columns=False)
        written: list[dict[str, Any]] = [{"table": name} for name in names.values()]

        self.created_figure_ids = []
        panels = self._panels(result, names, charts)
        if panels:
            notes = report_html.document(_(self.Name), f"{table} · {result.response}", self._report(result))
            figure_id = self.create_report_figure(f"Partition · {result.response}", notes, panels)
            self.created_figure_ids.append(figure_id)
            written.append({"figure": figure_id, "name": result.response})
        return written

    def _panels(self, result: pt.PartitionResult, names: Mapping[str, str], charts: list[str]) -> list[tuple]:
        """(chart type, title, x label, y label, [(series name, sql, roles, style)]) per chart ticked."""
        panels = []
        nominal = result.kind == pt.NOMINAL
        measure = "Misclassification" if nominal else "R²"
        if "contributions" in charts:
            share = "G^2" if nominal else "SS"
            sql, roles = self.series_select(names["contributions"], {"Y": share, "X": "Column"})
            panels.append(("Pareto Chart", _("Column contributions"), "", _(share),
                           [(result.response, sql, roles, {})]))
        if "history" in charts:
            series = []
            for column in result.history.columns[1:]:
                sql, roles = self.series_select(names["history"], {"x": "Splits", "y": column})
                series.append((_(column), sql, roles, {"linestyle": "-", "marker": "o"}))
            panels.append(("Scatter Plot", _("{measure} by number of splits").format(measure=_(measure)),
                           _("Splits"), _(measure), series))
        if "actual" in charts and not nominal:
            predicted = f"Predicted {result.response}"
            sql, roles = self.series_select(names["rows"], {"x": predicted, "y": result.response})
            panels.append(("Scatter Plot", _("Actual by predicted"), predicted, result.response,
                           [(result.response, sql, roles, {})]))
        if "leaves" in charts:
            value = "Mean" if not nominal else f"Prob({result.levels[0]})" if result.levels else "Count"
            sql, roles = self.series_select(names["leaves"], {"X": "Leaf", "Y": value})
            panels.append(("Bar Chart", _("Leaves"), _("Leaf"), _(value), [(result.response, sql, roles, {})]))
        return panels

