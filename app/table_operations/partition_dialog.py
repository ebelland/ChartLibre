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
from dataclasses import dataclass
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from app.analysis import partition as pt
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam
from app.styles.style import TitledCard
from app.plugin_api.table_operations.column_roles import CATEGORICAL, ORDINAL, ColumnCasting, ColumnRole
from app.plugin_api.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

#: The charts OK can draw: (key, check box label, ticked by default).
#: The charts Partition can draw: key, label, ticked at first, and the Y it needs ("" for any).
CHARTS: tuple[tuple[str, str, bool, str], ...] = (
    ("partition", "Partition graph (continuous Y)", True, "continuous"),
    ("box", "Box plots by leaf (continuous Y)", False, "continuous"),
    ("contributions", "Column contributions", True, ""),
    ("history", "R² or misclassification by splits", True, ""),
    ("actual", "Actual by predicted (continuous Y)", True, "continuous"),
    ("leaves", "Leaf means or shares", False, ""),
    ("importance", "Permutation importance", False, ""),
    ("dependence", "Partial dependence (most important factors)", False, ""),
    ("roc", "ROC curve (nominal Y)", True, "categorical"),
)

_JMP_HELP = "https://www.jmp.com/support/help/en/18.0/index.shtml#page/jmp/"
PARTITION_DOC = _JMP_HELP + "overview-of-the-partition-platform.shtml"


def _tree_kinds(casting: ColumnCasting, frame: pd.DataFrame) -> dict[str, str]:
    """The engine's two kinds. As JMP's Partition: an ordinal X is split in its
    order, as a continuous one (if its levels are numbers); an ordinal Y is
    classified, as a nominal one."""
    responses = set(casting.columns("response"))
    kinds = {}
    for name, kind in casting.kinds.items():
        if kind == ORDINAL:
            numbers = name in frame and pd.to_numeric(frame[name].dropna(), errors="coerce").notna().all()
            kinds[name] = pt.CONTINUOUS if numbers and name not in responses else pt.NOMINAL
        else:
            kinds[name] = pt.NOMINAL if kind in CATEGORICAL else pt.CONTINUOUS
    return kinds


#: The rows table's column placing each row along the partition graph.
PARTITION_X = "Partition x"


def _leaf_label(rule: str) -> str:
    """A leaf's rule as a label: one condition per line."""
    return str(rule).replace(" & ", "\n")


@dataclass(slots=True)
class _GraphLayout:
    """Where the partition graph puts things, in its 0..1 horizontal space."""

    positions: pd.Series  # per row with a leaf
    segments: pd.DataFrame  # Leaf, x, y: each leaf's mean, from its left edge to its right one
    leaves: pd.DataFrame  # Leaf, start, mean, label
    mean: float  # of all the training rows


def _graph_layout(result: pt.PartitionResult) -> _GraphLayout:
    """Each leaf as wide as its share of the rows, left to right as the tree has them; its rows evenly inside it."""
    leaf = result.leaf.dropna().astype(int).sort_index(kind="stable").sort_values(kind="stable")
    counts = leaf.value_counts().reindex(sorted(leaf.unique()))
    widths = counts / counts.sum()
    starts = widths.cumsum() - widths
    rank = leaf.groupby(leaf).cumcount()
    positions = starts.reindex(leaf).to_numpy() + (rank.to_numpy() + 0.5) / counts.reindex(leaf).to_numpy() \
        * widths.reindex(leaf).to_numpy()
    means = result.leaves.set_index("Leaf")["Mean"]
    rules = result.leaves.set_index("Leaf")["Rule"]
    segments, leaves = [], []
    for number, start in starts.items():
        mean = float(means.get(number, float("nan")))
        end = float(start + widths[number])
        segments += [{"Leaf": number, "x": float(start), "y": mean}, {"Leaf": number, "x": end, "y": mean}]
        leaves.append({"Leaf": number, "start": float(start), "mean": mean,
                       "label": f"{number}. {_leaf_label(rules.get(number, ''))}"})
    # The leaves' means weighted by their training rows: the mean of all of them.
    mean = float(result.leaves["Mean"].mul(result.leaves["Count"]).sum() / max(result.leaves["Count"].sum(), 1))
    return _GraphLayout(pd.Series(positions, index=leaf.index), pd.DataFrame(segments), pd.DataFrame(leaves), mean)


class PartitionDialog(TableOperationDialogBase):
    """Partition: split the rows by the factors that best separate the response."""

    Name = "Partition"
    Description = (
        "Decision tree of a continuous or nominal response on many factors: "
        "split and prune a step at a time, or automatically on validation rows; "
        "leaves, column contributions and each row's leaf and prediction."
    )
    Category = "Predictive and Specialized Modeling"
    Order = 30
    Icon = """
    <circle cx="12" cy="4.5" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="18" cy="12" r="2"/>
    <circle cx="3.5" cy="19.5" r="1.6"/><circle cx="9" cy="19.5" r="1.6"/>
    <path d="M10.6 6l-3.2 4.4M13.4 6l3.2 4.4M5 13.8l-1 4M7 13.8l1.4 4"/>
    """
    Keywords: str = (
        "decision tree albero decisionale albero di decisione classification classificazione "
        "regression tree cart split partizioni partizione segmentation segmentazione rules regole "
        "drivers importance importanza"
    )
    Goals: list[str] = ["explain", "predict", "classify", "groups"]

    ROLES = (
        ColumnRole("response", "Y", "The column to predict: continuous gives a regression tree, nominal a classification tree.",
                   minimum=1, maximum=1),
        ColumnRole("factor", "X", "The columns the tree may split on.", minimum=1),
        ColumnRole("validation", "Validation",
                   "Optional: 1 (or Validation) on the rows kept out of the fit, to judge it.", maximum=1),
    )
    DISTINCT_ROLES = (("response", "factor", "{columns} is both Y and X."),)
    #: The tree's own settings, beside the Split and Prune buttons.
    PARAMETERS_TAB = "Tree"

    PARAMS = (
        ChoiceParam("method", "Method:", choices=tuple((label, key) for key, label in pt.METHODS),
                    tooltip="One tree you can read; a Bootstrap Forest (many trees on resampled rows, averaged); "
                            "or a Boosted Tree (many small trees, each correcting the ones before)."),
        IntParam("splits", "Splits:", tooltip="How many times the rows are split: one fewer than the leaves. "
                                              "For a Boosted Tree, the splits of each tree.",
                 default_value=3, minimum=1, maximum=60),
        IntParam("trees", "Trees:", tooltip="How many trees a Bootstrap Forest or a Boosted Tree grows.",
                 default_value=100, minimum=1, maximum=2000),
        FloatParam("learning_rate", "Learning rate:",
                   tooltip="Boosted Tree: how much each tree corrects the ones before. Smaller needs more trees.",
                   default_value=0.1, minimum=0.01, maximum=1.0, decimals=2, step=0.05),
        IntParam("min_leaf", "Minimum leaf size:", tooltip="The fewest rows a leaf may hold.",
                 default_value=5, minimum=1, maximum=10_000, advanced=True),
        FloatParam("validation_fraction", "Validation share:",
                   tooltip="With no validation column: this share of rows, at random, judges the tree. 0 for none.",
                   default_value=0.0, minimum=0.0, maximum=0.6, decimals=2, step=0.05, advanced=True),
        IntParam("seed", "Random seed:", tooltip="Which rows the validation share draws.",
                 default_value=1, minimum=0, maximum=1_000_000, advanced=True),
        # The Charts tab: only the charts the Y allows are offered (chart_context).
        *(BoolParam(f"chart_{key}", label, default_value=ticked, tab="charts",
                    visible_for={"y": (y, "any")} if y else {})
          for key, label, ticked, y in CHARTS),
    )

    def __init__(self, **kwargs: Any) -> None:
        self._auto = False
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
        return [key for key, *_rest in CHARTS if self.chart_enabled(f"chart_{key}")]

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
            kinds=_tree_kinds(casting, frame),
            splits=int(params.get("splits", 3)),
            min_leaf=int(params.get("min_leaf", 5)),
            validation_column=validation[0] if validation else None,
            validation_fraction=float(params.get("validation_fraction", 0.0)),
            seed=int(params.get("seed", 1)),
            auto=bool(params.get("auto")),
            method=str(params.get("method") or pt.TREE),
            trees=int(params.get("trees", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
        )
        return pt.partition(frame, spec)

    def format_results(self, result: pt.PartitionResult) -> str:
        if self._auto and self._parameter_form is not None and result.method == pt.TREE:
            # What Automatic chose becomes the count Split and Prune go on from.
            self._parameter_form.set_values({"splits": result.splits})
        return self.report_document(result)

    def report_subtitle(self, result: pt.PartitionResult) -> str:
        return f"{self.table} · {result.response}"

    def report_body(self, result: pt.PartitionResult) -> str:
        blocks = []
        if result.note:
            blocks.append(report_html.note(result.note))
        kind = (dict(pt.METHODS)[result.method] if result.method != pt.TREE
                else "Regression tree" if result.kind == pt.CONTINUOUS else "Classification tree")
        kind = _(kind)
        blocks.append(report_html.section(
            _("Summary"),
            report_html.summary_table(
                [(_("Tree"), kind)] + [(_(key), report_html.format_number(value)) for key, value in result.summary.items()]
            ),
        ))
        blocks.append(report_html.section(
            _("Column contributions"),
            self.report_table(result.contributions),
        ))
        if not result.leaves.empty:
            blocks.append(report_html.section(
                _("Leaf report"),
                self.report_table(result.leaves),
            ))
        blocks.append(report_html.section(
            _("Split history") if result.method == pt.TREE else _("Fit by number of trees"),
            self.report_table(result.history),
        ))
        if not result.importance.empty:
            measure = "accuracy" if result.kind == pt.NOMINAL else "R²"
            blocks.append(report_html.section(
                _("Permutation importance"),
                report_html.note(_("How much the {measure} drops when a column is shuffled, on the {rows} rows - "
                                   "for any model, the forest and the boosted tree included.").format(
                    measure=_(measure), rows=_(str(result.importance["Rows"].iloc[0]).lower()))),
                self.report_table(result.importance.drop(columns=["Rows"])),
            ))
        if result.confusion is not None:
            confusion = result.confusion.reset_index()
            blocks.append(report_html.section(
                _("Confusion matrix (training rows)"),
                report_html.table([_("Actual")] + [html.escape(str(c)) for c in result.confusion.columns],
                                  self.report_rows(confusion)),
            ))
        return report_html.section(_("Response {name}").format(name=result.response), *blocks)

    # ------------------------------------------------------------------
    # OK: the results into the project
    # ------------------------------------------------------------------

    def apply_results(self, result: pt.PartitionResult) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        charts = self.charts()
        # A forest or a boosted model has no single tree: no leaves to write or draw.
        tree = not result.leaves.empty
        graph = "partition" in charts and result.kind != pt.NOMINAL and tree
        names = {key: self.result_table_name(f"partition{suffix}") for key, suffix in (
            ("rows", ""), *((("leaves", "_leaves"),) if tree else ()), ("history", "_history"),
            ("contributions", "_contributions"), *((("graph", "_graph"),) if graph else ()),
            *((("importance", "_importance"),) if not result.importance.empty else ()),
            *((("dependence", "_dependence"),) if not result.dependence.empty else ()),
            *((("roc", "_roc"),) if not result.roc.empty else ()),
        )}
        repo.snapshot_for_undo(
            [*names.values(), *(repo.DESCRIPTOR_TABLES if charts else ())],
            label=f"Partition on '{table}'",
        )

        # Each row's leaf and prediction beside its data, like ClusterId.
        frame = self._frame.copy()
        if tree:
            frame["Leaf"] = result.leaf.reindex(frame.index)
        frame[f"Predicted {result.response}"] = result.predicted.reindex(frame.index)
        if result.probabilities is not None:
            for column in result.probabilities.columns:
                frame[column] = result.probabilities[column].reindex(frame.index)
        frame["Partition role"] = result.role.reindex(frame.index)
        layout = None
        if graph:
            layout = _graph_layout(result)
            frame[PARTITION_X] = layout.positions.reindex(frame.index)
            repo.import_dataframe(layout.segments, table_name=names["graph"], normalize_columns=False)
        repo.import_dataframe(frame, table_name=names["rows"], normalize_columns=False)
        for key in ("leaves", "history", "contributions", "importance", "dependence"):
            if key in names:
                repo.import_dataframe(getattr(result, key), table_name=names[key], normalize_columns=False)
        if "roc" in names:
            chance = pd.DataFrame({"Level": "Chance", "FPR": [0.0, 1.0], "TPR": [0.0, 1.0]})
            repo.import_dataframe(pd.concat([result.roc, chance], ignore_index=True), table_name=names["roc"],
                                  normalize_columns=False)
        written: list[dict[str, Any]] = [{"table": name} for name in names.values()]

        panels = self._panels(result, names, charts, layout)
        return self.draw_report(result, result.response, panels, written=written) if panels else written

    def _panels(self, result: pt.PartitionResult, names: Mapping[str, str], charts: list[str],
                layout: _GraphLayout | None = None) -> list[tuple]:
        """(chart type, title, x label, y label, [(series name, sql, roles, style)], options) per chart ticked."""
        panels = []
        nominal = result.kind == pt.NOMINAL
        measure = "Misclassification" if nominal else "R²"
        if layout is not None:
            panels.append(self._partition_graph(result, names, layout))
        if "box" in charts and not nominal and "leaves" in names:
            # Numbered as in the leaves chart and the partition graph: the rules
            # are too long to sit side by side under the boxes.
            leaf = self.quoted("Leaf")
            sql = (f"SELECT {self.quoted(result.response)} AS value, CAST(CAST({leaf} AS INTEGER) AS TEXT) AS "
                   f"\"group\" FROM {self.quoted(names['rows'])} WHERE {leaf} IS NOT NULL ORDER BY {leaf}")
            panels.append(("Box Plot", _("{response} by leaf").format(response=result.response), _("Leaf"),
                           result.response, [(result.response, sql, {"value": "value", "group": "group"}, {})]))
        if "contributions" in charts:
            share = "G^2" if nominal else "SS"
            sql, roles = self.series_select(names["contributions"], {"Y": share, "X": "Column"})
            panels.append(("Pareto Chart", _("Column contributions"), "", _(share),
                           [(result.response, sql, roles, {})]))
        if "history" in charts:
            series = []
            for column in result.history.columns[1:]:
                sql, roles = self.series_select(names["history"], {"x": result.history.columns[0], "y": column})
                series.append((_(column), sql, roles, {"linestyle": "-", "marker": "o"}))
            panels.append(("Scatter Plot", (_("{measure} by number of splits") if result.method == pt.TREE
                                            else _("{measure} by number of trees")).format(measure=_(measure)),
                           _(result.history.columns[0]), _(measure), series))
        if "actual" in charts and not nominal:
            predicted = f"Predicted {result.response}"
            sql, roles = self.series_select(names["rows"], {"x": predicted, "y": result.response})
            panels.append(("Scatter Plot", _("Actual by predicted"), predicted, result.response,
                           [(result.response, sql, roles, {})]))
        if "importance" in charts and "importance" in names:
            sql, roles = self.series_select(names["importance"], {"Y": "Importance", "X": "Column"})
            panels.append(("Pareto Chart", _("Permutation importance"), "", _("Importance"),
                           [(result.response, sql, roles, {})]))
        if "dependence" in charts and "dependence" in names:
            y_label = f"Prob({result.levels[0]})" if nominal and result.levels else result.response
            for factor in dict.fromkeys(result.dependence["Factor"]):
                where = f"{self.quoted('Factor')} = {self.literal(str(factor))}"
                if factor in result.factors and self._factor_is_nominal(factor):
                    sql, roles = self.series_select(names["dependence"], {"X": "Label", "Y": "Predicted"}, where=where)
                    panels.append(("Bar Chart", _("Partial dependence on {factor}").format(factor=factor), factor,
                                   y_label, [(str(factor), sql, roles, {})]))
                else:
                    sql, roles = self.series_select(names["dependence"], {"x": "Value", "y": "Predicted"}, where=where)
                    panels.append(("Scatter Plot", _("Partial dependence on {factor}").format(factor=factor), factor,
                                   y_label, [(str(factor), sql, roles, {"linestyle": "-", "marker": "o"})]))
        if "roc" in charts and "roc" in names:
            series = []
            for level in [*dict.fromkeys(result.roc["Level"]), "Chance"]:
                sql, roles = self.series_select(names["roc"], {"x": "FPR", "y": "TPR"},
                                                where=f"{self.quoted('Level')} = {self.literal(str(level))}")
                if level == "Chance":
                    series.append((_("Chance"), sql, roles, {"linestyle": "--", "marker": "", "color": "0.6"}))
                else:
                    series.append((f"{level} (AUC {result.auc.get(str(level), float('nan')):.3f})", sql, roles,
                                   {"linestyle": "-", "marker": ""}))
            panels.append(("Scatter Plot", _("ROC curve"), _("False positive rate"), _("True positive rate"), series))
        if "leaves" in charts and "leaves" in names:
            value = "Mean" if not nominal else f"Prob({result.levels[0]})" if result.levels else "Count"
            sql, roles = self.series_select(names["leaves"], {"X": "Leaf", "Y": value})
            # The leaf's number as a category, not a float axis value ("1", not "1.0").
            sql = sql.replace(f"{self.quoted('Leaf')} AS", f"CAST(CAST({self.quoted('Leaf')} AS INTEGER) AS TEXT) AS", 1)
            panels.append(("Bar Chart", _("Leaves"), _("Leaf"), _(value), [(result.response, sql, roles, {})]))
        return panels

    def _factor_is_nominal(self, factor: str) -> bool:
        return self.roles_widget.casting().kinds.get(factor) in CATEGORICAL

    def _partition_graph(self, result: pt.PartitionResult, names: Mapping[str, str], layout: _GraphLayout) -> tuple:
        """JMP's partition graph: Y by row, the rows grouped by leaf, each leaf as wide as its rows,
        a line at each leaf's mean labelled with its rule and a grey one at the mean of all rows."""
        role = self.quoted("Partition role")
        series = []
        for name, where, style in (
            (_("Training"), f"{role} = 'Training'", {"color": "black", "s": 14}),
            (_("Validation"), f"{role} = 'Validation'", {"color": "0.55", "s": 14, "marker": "x"}),
        ):
            if not (result.role == ("Training" if name == _("Training") else "Validation")).any():
                continue
            sql, roles = self.series_select(names["rows"], {"x": PARTITION_X, "y": result.response},
                                            where=f"{where} AND {self.quoted(PARTITION_X)} IS NOT NULL")
            series.append((name, sql, roles, style))
        for leaf in layout.segments["Leaf"].unique():
            sql, roles = self.series_select(names["graph"], {"x": "x", "y": "y"},
                                            where=f"{self.quoted('Leaf')} = {int(leaf)}")
            series.append((_("Leaf {leaf}").format(leaf=int(leaf)), sql, roles,
                           {"linestyle": "-", "marker": "", "color": "black", "linewidth": 1.2,
                            "show_in_legend": False}))
        mean_all = layout.mean
        annotations = [{"type": "text", "x": 0.0, "y": mean_all, "text": _("All rows"),
                        "kwargs": {"color": "0.55", "fontsize": 8, "va": "bottom"}}]
        for row in layout.leaves.itertuples():
            annotations.append({"type": "text", "x": float(row.start), "y": float(row.mean), "text": row.label,
                                "kwargs": {"fontsize": 8, "va": "bottom"}})
        lines = [{"orientation": "horizontal", "value": mean_all, "kwargs": {"color": "0.6", "linewidth": 1}}]
        lines += [{"orientation": "vertical", "value": float(x),
                   "kwargs": {"color": "0.8", "linewidth": 0.8, "linestyle": ":"}} for x in layout.leaves["start"][1:]]
        return ("Scatter Plot", _("Partition for {response}").format(response=result.response),
                _("Rows, by leaf"), result.response, series, {"annotations": annotations, "lines": lines})

