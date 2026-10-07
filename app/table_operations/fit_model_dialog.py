"""Fit Model: least squares of responses on numeric and categorical factors, after JMP.

Cast the columns: Y (one or more responses) and the factors, each continuous
or nominal. The Effects tab holds the model's terms: a macro to start from -
main effects, a factorial to degree 2, a response surface - and Add, Cross
and Remove to change it. A table the DOE dialog made comes with its roles
cast and the macro its design was built for.

Preview reports each response: the effect summary (LogWorth, most
significant first), the summary of fit, the analysis of variance, the lack of
fit when there are replicates, and the parameter estimates in coded units.
OK writes the predicted values and residuals beside the data, the effects in
a table of their own, and for each response a figure: actual by predicted,
the effects' Pareto, residual by predicted and a normal quantile plot of the
residuals. See app/analysis/linear_model.py for the arithmetic.
"""
from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.analysis import linear_model as lm
from app.series_operations.parameter_spec import BoolParam, FloatParam
from app.table_operations.column_roles import CONTINUOUS, NOMINAL, ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _

#: The macro a DOE model is analysed with by default.
_DOE_MACROS: dict[str, str] = {
    "two_level": "factorial",
    "full_factorial": "factorial",
    "box_behnken": "surface",
    "ccd_face": "surface",
    "ccd_circumscribed": "surface",
    "latin_hypercube": "surface",
}

_MACROS: tuple[tuple[str, str], ...] = (
    ("main", "Main effects"),
    ("factorial", "Factorial to degree 2"),
    ("surface", "Response surface"),
)


class FitModelDialog(TableOperationDialogBase):
    """Fit Model: responses on factors, by least squares."""

    Name = "Fit Model"
    Description = (
        "Least-squares model of one or more responses on continuous and nominal "
        "factors: effect tests, estimates, analysis of variance and lack of fit - "
        "for a designed experiment, or any table."
    )
    Category = "Models"
    Icon = """
    <path d="M4 20V4"/><path d="M4 20h16"/>
    <circle cx="8" cy="15" r="1"/><circle cx="11" cy="11" r="1"/><circle cx="15" cy="10" r="1"/><circle cx="18" cy="6" r="1"/>
    <path d="M6 17L19 5"/>
    """

    ROLES = (
        ColumnRole("response", "Y, response", "The columns to model: one model each.", kinds=(CONTINUOUS,), minimum=1),
        ColumnRole("factors", "Factors", "The columns the responses are modelled on.", minimum=1),
    )
    PARAMS = (
        BoolParam(
            "reduce",
            "Remove terms that are not significant",
            tooltip=(
                "Backwards: the least significant term goes first, until every "
                "term left is significant - never a term a higher-order one "
                "still contains."
            ),
            default_value=False,
        ),
        FloatParam(
            "alpha",
            "Significance level (α):",
            tooltip="A term is significant when its p-value is at most this.",
            default_value=0.05, minimum=0.001, maximum=0.5, decimals=3, step=0.01,
        ),
        BoolParam(
            "charts",
            "Make the charts",
            tooltip=(
                "For each response, a figure: actual by predicted, the effects' "
                "Pareto, residual by predicted, and a normal quantile plot of "
                "the residuals."
            ),
            default_value=True,
        ),
    )

    def __init__(self, **kwargs: Any) -> None:
        self._terms: list[lm.Term] = []
        self._terms_edited = False
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Roles: from the DOE design when the table has one
    # ------------------------------------------------------------------

    def default_casting(self, table: str, frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
        design = self.doe_design(table)
        if design is None:
            return {}, {}
        factors = [f["name"] for f in design.get("factors", []) if f.get("name") in frame.columns]
        kinds = {
            f["name"]: NOMINAL if f.get("kind") in ("categorical", NOMINAL) else CONTINUOUS
            for f in design.get("factors", [])
            if f.get("name") in frame.columns
        }
        responses = [name for name in design.get("responses", []) if name in frame.columns]
        kinds.update({name: CONTINUOUS for name in responses})
        return {"response": responses, "factors": factors}, kinds

    # ------------------------------------------------------------------
    # The Effects tab: the model's terms
    # ------------------------------------------------------------------

    def build_extra_inputs(self, tabs: QTabWidget) -> None:
        page = QWidget(tabs)
        layout = QGridLayout(page)
        layout.setHorizontalSpacing(8)

        layout.addWidget(self._muted(_("Factors")), 0, 0)
        self._factor_list = QListWidget(page)
        self._factor_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self._factor_list, 1, 0)

        buttons = QVBoxLayout()
        for text, tooltip, action in (
            (_("Add"), _("Add the selected factors as main effects."), self._add_selected),
            (_("Cross"), _("Add the interaction of the selected factors; one factor alone gives its square."), self._cross_selected),
            (_("Remove"), _("Remove the selected terms."), self._remove_selected),
        ):
            button = QPushButton(text, page)
            button.setToolTip(tooltip)
            button.clicked.connect(action)
            buttons.addWidget(button)
        # The macros as a menu, as in JMP: their names do not fit beside each other.
        macros = QPushButton(_("Macro"), page)
        macros.setToolTip(_("Replace the effects with a standard model."))
        menu = QMenu(macros)
        for key, label in _MACROS:
            menu.addAction(_(label), lambda k=key: self._apply_macro(k, edited=True))
        macros.setMenu(menu)
        buttons.addSpacing(12)
        buttons.addWidget(macros)
        buttons.addStretch(1)
        layout.addLayout(buttons, 1, 1)

        layout.addWidget(self._muted(_("Model effects")), 0, 2)
        self._term_list = QListWidget(page)
        self._term_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self._term_list, 1, 2)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(2, 2)
        tabs.addTab(page, _("Effects"))

    def table_or_none(self) -> str | None:
        combo = getattr(self, "_table_combo", None)
        return combo.currentText() if combo is not None else None

    @staticmethod
    def _muted(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("muted", True)
        return label

    def _factors(self) -> list[lm.Factor]:
        """The cast factors, with the design's low/high where the table has a design."""
        casting = self.roles_widget.casting()
        design = self.doe_design(self.table) or {}
        ranges = {f.get("name"): (f.get("low"), f.get("high")) for f in design.get("factors", [])}
        factors = []
        for name in casting.columns("factors"):
            kind = NOMINAL if casting.kinds.get(name) == NOMINAL else CONTINUOUS
            low, high = ranges.get(name, (None, None)) if kind == CONTINUOUS else (None, None)
            factors.append(lm.Factor(name, kind, low, high))
        return factors

    def inputs_changed(self) -> None:
        if getattr(self, "_terms_table", None) != self.table_or_none():
            # Another table: start again from its design's macro.
            self._terms_table = self.table_or_none()
            self._terms_edited = False
        if hasattr(self, "_term_list"):
            names = [f.name for f in self._factors()]
            self._factor_list.clear()
            self._factor_list.addItems(names)
            if not self._terms_edited:
                design = self.doe_design(self.table) or {}
                self._apply_macro(_DOE_MACROS.get(str(design.get("model")), "main"), edited=False)
            else:
                # A factor taken out of its role takes its terms with it.
                self._terms = [t for t in self._terms if all(n in names for n in t)]
                self._show_terms()
        super().inputs_changed()

    def _apply_macro(self, key: str, *, edited: bool) -> None:
        factors = self._factors()
        self._terms = {
            "main": lm.main_effects,
            "factorial": lambda f: lm.factorial(f, 2),
            "surface": lm.response_surface,
        }[key](factors)
        self._terms_edited = edited
        self._show_terms()
        if edited:
            super().inputs_changed()

    def _selected_factors(self) -> list[str]:
        return [item.text() for item in self._factor_list.selectedItems()]

    def _add_terms(self, terms: list[lm.Term]) -> None:
        for term in terms:
            if term not in self._terms:
                self._terms.append(term)
        self._terms_edited = True
        self._show_terms()
        super().inputs_changed()

    def _add_selected(self) -> None:
        self._add_terms([(name,) for name in self._selected_factors()])

    def _cross_selected(self) -> None:
        names = self._selected_factors()
        if len(names) == 1:
            kinds = {f.name: f.kind for f in self._factors()}
            if kinds.get(names[0]) == CONTINUOUS:
                self._add_terms([(names[0], names[0])])
        elif len(names) > 1:
            self._add_terms([tuple(names)])

    def _remove_selected(self) -> None:
        rows = sorted((self._term_list.row(item) for item in self._term_list.selectedItems()), reverse=True)
        for row in rows:
            del self._terms[row]
        self._terms_edited = True
        self._show_terms()
        super().inputs_changed()

    def _show_terms(self) -> None:
        self._term_list.clear()
        for term in self._terms:
            QListWidgetItem(lm.term_label(term), self._term_list)

    def problems(self) -> list[str]:
        found = super().problems()
        if not found and not self._terms:
            found.append(_("The model has no effects: add some on the Effects tab."))
        return found

    def parameter_values(self) -> dict[str, Any]:
        """The parameters, and the model: its terms and the factors' types and coding."""
        values = super().parameter_values()
        values["terms"] = [list(t) for t in self._terms]
        values["factors"] = [
            {"name": f.name, "kind": f.kind, "low": f.low, "high": f.high} for f in self._factors()
        ]
        return values

    # ------------------------------------------------------------------
    # Computing and reporting
    # ------------------------------------------------------------------

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        del model
        spec = lm.ModelSpec(
            factors=[lm.Factor(f["name"], f["kind"], f["low"], f["high"]) for f in params["factors"]],
            responses=casting.columns("response"),
            terms=[tuple(t) for t in params["terms"]],
            reduce=bool(params.get("reduce")),
            alpha=float(params.get("alpha", 0.05)),
        )
        return lm.fit_models(frame, spec)

    def format_results(self, result: list[lm.ResponseFit]) -> str:
        sections = [self._report(fit) for fit in result]
        subtitle = _("{table} · {count} response(s)").format(table=self.table, count=len(result))
        return report_html.document(_(self.Name), subtitle, *sections)

    @staticmethod
    def _rows(frame: pd.DataFrame, p_columns: tuple[str, ...] = ()) -> list[list[str]]:
        rows = []
        for record in frame.itertuples(index=False):
            cells = []
            for column, value in zip(frame.columns, record):
                if column in p_columns:
                    cells.append(report_html.format_p_value(value))
                elif isinstance(value, str):
                    cells.append(html.escape(value))
                elif column == "DF":
                    cells.append(str(int(value)))
                else:
                    cells.append(report_html.format_number(value))
            rows.append(cells)
        return rows

    def _report(self, fit: lm.ResponseFit) -> str:
        blocks = []
        if fit.note:
            blocks.append(report_html.note(_(fit.note)))
        effects = fit.effect_tests.sort_values("LogWorth", ascending=False, na_position="last")
        blocks.append(report_html.section(
            _("Effect summary"),
            report_html.table(
                [_("Source"), _("LogWorth"), _("Prob > F")],
                [[html.escape(r["Term"]), report_html.format_number(r["LogWorth"], digits=3),
                  report_html.format_p_value(r["Prob > F"])]
                 for r in effects.to_dict("records")],
            ),
        ))
        if fit.removed:
            blocks.append(report_html.note(
                _("Removed as not significant: {terms}.").format(terms=", ".join(lm.term_label(t) for t in fit.removed))
            ))
        blocks.append(report_html.section(
            _("Summary of fit"),
            report_html.summary_table((_(key), report_html.format_number(value)) for key, value in fit.summary.items()),
        ))
        blocks.append(report_html.section(
            _("Analysis of variance"),
            report_html.table([_(c) for c in fit.anova.columns], self._rows(fit.anova, ("Prob > F",))),
        ))
        if fit.lack_of_fit is not None:
            blocks.append(report_html.section(
                _("Lack of fit"),
                report_html.table([_(c) for c in fit.lack_of_fit.columns], self._rows(fit.lack_of_fit, ("Prob > F",))),
            ))
        blocks.append(report_html.section(
            _("Parameter estimates (coded units)"),
            report_html.table([_(c) for c in fit.estimates.columns], self._rows(fit.estimates, ("Prob > |t|",))),
        ))
        blocks.append(report_html.section(
            _("Effect tests"),
            report_html.table(
                [_(c) for c in fit.effect_tests.columns if c != "LogWorth"],
                self._rows(fit.effect_tests.drop(columns=["LogWorth"]), ("Prob > F",)),
            ),
        ))
        return report_html.section(_("Response {name}").format(name=fit.response), *blocks)

    # ------------------------------------------------------------------
    # OK: the results into the project
    # ------------------------------------------------------------------

    @staticmethod
    def _columns_for(response: str) -> tuple[str, str, str]:
        return f"Predicted {response}", f"Residual {response}", f"Studentized Residual {response}"

    def apply_results(self, result: list[lm.ResponseFit]) -> list[dict[str, Any]]:
        repo, table = self._repo, self.table
        results_name = repo.free_table_name(f"{table}_fit")
        effects_name = repo.free_table_name(f"{table}_fit_effects")
        charts = bool(self.parameter_values().get("charts", True))
        repo.snapshot_for_undo(
            [results_name, effects_name, *(repo.DESCRIPTOR_TABLES if charts else ())],
            label=f"Fit Model on '{table}'",
        )

        frame = self._frame.copy()
        effects = []
        for fit in result:
            predicted, residual, studentized = self._columns_for(fit.response)
            frame[predicted] = fit.predicted.reindex(frame.index)
            frame[residual] = fit.residuals.reindex(frame.index)
            frame[studentized] = fit.studentized.reindex(frame.index)
            ordered = fit.effect_tests.sort_values("LogWorth", na_position="first")
            for row in ordered.to_dict("records"):
                effects.append({"Response": fit.response, "Term": row["Term"], "LogWorth": row["LogWorth"],
                                "F Ratio": row["F Ratio"], "Prob > F": row["Prob > F"]})
        repo.import_dataframe(frame, table_name=results_name, normalize_columns=False)
        repo.import_dataframe(pd.DataFrame(effects), table_name=effects_name, normalize_columns=False)
        written: list[dict[str, Any]] = [{"table": results_name}, {"table": effects_name}]

        self.created_figure_ids = []
        if charts:
            for fit in result:
                figure_id = self._make_figure(fit, results_name, effects_name)
                self.created_figure_ids.append(figure_id)
                written.append({"figure": figure_id, "name": fit.response})
        return written

    def _make_figure(self, fit: lm.ResponseFit, results: str, effects: str) -> int:
        repo = self._repo
        predicted, residual, _studentized = self._columns_for(fit.response)
        figure_id = int(repo.create_figure_descriptor(name=f"Fit Model · {fit.response}", nrows=2, ncols=2))
        quoted_results = '"' + results.replace('"', '""') + '"'
        quoted_effects = '"' + effects.replace('"', '""') + '"'
        response_literal = "'" + fit.response.replace("'", "''") + "'"
        panels = (
            ("Scatter Plot", _("Actual by predicted"), predicted, fit.response,
             f"SELECT * FROM {quoted_results}", {"x": predicted, "y": fit.response}),
            ("Horizontal Bar Chart", _("Effect summary (LogWorth)"), "LogWorth", "",
             f"SELECT * FROM {quoted_effects} WHERE \"Response\" = {response_literal}", {"Y": "LogWorth", "X": "Term"}),
            ("Scatter Plot", _("Residual by predicted"), predicted, residual,
             f"SELECT * FROM {quoted_results}", {"x": predicted, "y": residual}),
            ("Q-Q Plot", _("Residual normal quantile plot"), "", residual,
             f"SELECT * FROM {quoted_results}", {"value": residual}),
        )
        for index, (chart, title, x_label, y_label, sql, roles) in enumerate(panels):
            axis_id = int(repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type=chart,
                title=title, x_label=x_label, y_label=y_label, options={},
            ))
            repo.create_series_descriptor(
                axis_id=axis_id, series_index=0, name=fit.response,
                sql_query=sql, roles=roles, style={},
            )
        return figure_id
