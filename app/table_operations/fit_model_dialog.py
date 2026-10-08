"""Fit Model: least squares of responses on continuous and nominal factors, after JMP.

One page, as JMP's launch window but simpler: the table's columns on the
left, the responses cast as Y beside them, and under Y the model's effects -
built from the columns selected on the left with Add (main effects), Cross
(an interaction; one column alone, its square) and the macros (main effects,
a factorial to a degree, a response surface). The factors are the columns
the effects use. A table the DOE dialog made comes with Y cast and the model
its design was built for.

Preview reports each response: the effect summary (LogWorth, most
significant first), the summary of fit, the analysis of variance, the lack of
fit when there are replicates, the parameter estimates in coded units and
the effect tests. OK writes the predicted values and residuals beside the
data, the effects and the prediction profile in tables of their own, and for
each response a figure of the charts ticked - its report in the figure's
notes. See app/analysis/linear_model.py for the arithmetic.
"""
from __future__ import annotations

import html
import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from app.analysis import linear_model as lm
from app.series_operations.parameter_spec import BoolParam, FloatParam
from app.table_operations.column_roles import CONTINUOUS, NOMINAL, ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.utils import report_html
from app.utils.i18n import _
from app.widgets.chart_panel import FIGURE_VIEW_OPTIONS_KEY

#: The macro a DOE model is analysed with by default.
_DOE_MACROS: dict[str, str] = {
    "two_level": "factorial",
    "full_factorial": "factorial",
    "box_behnken": "surface",
    "ccd_face": "surface",
    "ccd_circumscribed": "surface",
    "latin_hypercube": "surface",
}

#: The charts OK can draw, as (key, check box label, ticked by default).
CHARTS: tuple[tuple[str, str, bool], ...] = (
    ("actual", "Actual by predicted", True),
    ("pareto", "Effects Pareto", True),
    ("residual", "Residual by predicted", True),
    ("qq", "Residual Q-Q", False),
    ("profile", "Prediction profile", True),
    ("interaction", "Interaction", False),
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
        ColumnRole("response", "Y", "The columns to model: one model each.", kinds=(CONTINUOUS,), minimum=1),
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
    )

    def __init__(self, **kwargs: Any) -> None:
        self._terms: list[lm.Term] = []
        self._terms_edited = False
        self.created_figure_ids: list[int] = []
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Y from the DOE design, when the table has one
    # ------------------------------------------------------------------

    def default_casting(self, table: str, frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
        design = self.doe_design(table)
        if design is None:
            return {}, {}
        kinds = {
            f["name"]: NOMINAL if f.get("kind") in ("categorical", NOMINAL) else CONTINUOUS
            for f in design.get("factors", [])
            if f.get("name") in frame.columns
        }
        responses = [name for name in design.get("responses", []) if name in frame.columns]
        kinds.update({name: CONTINUOUS for name in responses})
        return {"response": responses}, kinds

    def _design_factors(self) -> list[str]:
        design = self.doe_design(self.table) or {}
        return [f["name"] for f in design.get("factors", []) if f.get("name") in self._frame.columns]

    # ------------------------------------------------------------------
    # The effects, under Y, and the charts
    # ------------------------------------------------------------------

    def build_extra_inputs(self, layout: QVBoxLayout) -> None:
        heading = QLabel(_("Model effects"))
        heading.setProperty("muted", True)
        layout.addWidget(heading)

        # Buttons in a column beside the list, as in JMP: in a row they made
        # the window wider than a laptop's.
        effects = QHBoxLayout()
        buttons = QVBoxLayout()
        buttons.setSpacing(4)
        for text, tooltip, action in (
            (_("Add"), _("Add the selected columns as main effects."), self._add_selected),
            (_("Cross"), _("Add the interaction of the selected columns; one column alone gives its square."),
             self._cross_selected),
            (_("Remove"), _("Remove the selected effects."), self._remove_selected),
        ):
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(action)
            buttons.addWidget(button)
        # The macros as a menu, as in JMP: their names do not fit side by side.
        macros = QPushButton(_("Macro"))
        macros.setToolTip(_("Replace the effects with a standard model of the selected columns."))
        menu = QMenu(macros)
        menu.addAction(_("Main effects"), lambda: self._apply_macro("main", edited=True))
        menu.addAction(_("Factorial to degree"), lambda: self._apply_macro("factorial", edited=True))
        menu.addAction(_("Response surface"), lambda: self._apply_macro("surface", edited=True))
        macros.setMenu(menu)
        buttons.addWidget(macros)
        degree = QHBoxLayout()
        degree.addWidget(QLabel(_("Degree")))
        self._degree = QSpinBox()
        self._degree.setRange(1, 6)
        self._degree.setValue(2)
        self._degree.setToolTip(_("How many factors the factorial macro's interactions combine, at most."))
        degree.addWidget(self._degree)
        buttons.addLayout(degree)
        buttons.addStretch(1)
        effects.addLayout(buttons)

        self._term_list = QListWidget()
        self._term_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._term_list.setToolTip(_("Double-click an effect to remove it."))
        self._term_list.setMinimumHeight(90)
        self._term_list.setMinimumWidth(140)
        self._term_list.itemDoubleClicked.connect(lambda _item: self._remove_selected())
        effects.addWidget(self._term_list, 1)
        layout.addLayout(effects, 1)

        charts = QLabel(_("Charts"))
        charts.setProperty("muted", True)
        layout.addWidget(charts)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        self._chart_checks: dict[str, QCheckBox] = {}
        for index, (key, label, ticked) in enumerate(CHARTS):
            check = QCheckBox(_(label))
            check.setChecked(ticked)
            # An attribute of its own, so the window remembers it.
            setattr(self, f"_chart_{key}", check)
            self._chart_checks[key] = check
            grid.addWidget(check, index // 2, index % 2)
        layout.addLayout(grid)

    def table_or_none(self) -> str | None:
        combo = getattr(self, "_table_combo", None)
        return combo.currentText() if combo is not None else None

    def _factors(self) -> list[lm.Factor]:
        """The columns the effects use, in order, typed, with the design's low/high where there is one."""
        kinds = self.roles_widget.casting().kinds
        design = self.doe_design(self.table) or {}
        ranges = {f.get("name"): (f.get("low"), f.get("high")) for f in design.get("factors", [])}
        names: list[str] = []
        for term in self._terms:
            for name in term:
                if name not in names:
                    names.append(name)
        factors = []
        for name in names:
            kind = NOMINAL if kinds.get(name) == NOMINAL else CONTINUOUS
            low, high = ranges.get(name, (None, None)) if kind == CONTINUOUS else (None, None)
            factors.append(lm.Factor(name, kind, low, high))
        return factors

    def inputs_changed(self) -> None:
        if getattr(self, "_terms_table", None) != self.table_or_none():
            # Another table: start again from its design's model.
            self._terms_table = self.table_or_none()
            self._terms_edited = False
            if hasattr(self, "_term_list"):
                design = self.doe_design(self.table) or {}
                self._apply_macro(_DOE_MACROS.get(str(design.get("model")), "main"), edited=False,
                                  columns=self._design_factors())
        elif hasattr(self, "_term_list"):
            # Columns gone, or a column now nominal, take their effects with them.
            kinds = self.roles_widget.casting().kinds
            self._terms = [
                t for t in self._terms
                if all(n in kinds for n in t) and not (len(t) == 2 and t[0] == t[1] and kinds.get(t[0]) == NOMINAL)
            ]
            self._show_terms()
        super().inputs_changed()

    def _apply_macro(self, key: str, *, edited: bool, columns: list[str] | None = None) -> None:
        """Replace the effects with a macro over *columns*: the selected ones, else the model's factors."""
        if columns is None:
            columns = self.roles_widget.selected_columns() or [f.name for f in self._factors()]
        kinds = self.roles_widget.casting().kinds
        factors = [lm.Factor(name, NOMINAL if kinds.get(name) == NOMINAL else CONTINUOUS) for name in columns]
        degree = self._degree.value() if hasattr(self, "_degree") else 2
        self._terms = {
            "main": lm.main_effects,
            "factorial": lambda f: lm.factorial(f, degree),
            "surface": lm.response_surface,
        }[key](factors)
        self._terms_edited = edited
        self._show_terms()
        if edited:
            super().inputs_changed()

    def _add_terms(self, terms: list[lm.Term]) -> None:
        for term in terms:
            if term not in self._terms:
                self._terms.append(term)
        self._terms_edited = True
        self._show_terms()
        super().inputs_changed()

    def _add_selected(self) -> None:
        self._add_terms([(name,) for name in self.roles_widget.selected_columns()])

    def _cross_selected(self) -> None:
        names = self.roles_widget.selected_columns()
        if len(names) == 1:
            if self.roles_widget.casting().kinds.get(names[0]) == CONTINUOUS:
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
        if found:
            return found
        if not self._terms:
            return [_("The model has no effects: select columns and press Add.")]
        both = sorted(set(self.roles_widget.casting().columns("response")) & {f.name for f in self._factors()})
        if both:
            return [_("A column cannot be both a response and a factor: {columns}.").format(columns=", ".join(both))]
        return []

    def charts(self) -> list[str]:
        return [key for key, check in self._chart_checks.items() if check.isChecked()]

    def parameter_values(self) -> dict[str, Any]:
        """The parameters, and the model: its terms and the factors' types and coding."""
        values = super().parameter_values()
        values["terms"] = [list(t) for t in self._terms]
        values["factors"] = [
            {"name": f.name, "kind": f.kind, "low": f.low, "high": f.high} for f in self._factors()
        ]
        values["charts"] = self.charts()
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
        names = {
            "results": repo.free_table_name(f"{table}_fit"),
            "effects": repo.free_table_name(f"{table}_fit_effects"),
            "profile": repo.free_table_name(f"{table}_fit_profile"),
        }
        charts = self.charts()
        repo.snapshot_for_undo(
            [*names.values(), *(repo.DESCRIPTOR_TABLES if charts else ())],
            label=f"Fit Model on '{table}'",
        )

        frame = self._frame.copy()
        effects, profiles = [], []
        for fit in result:
            predicted, residual, studentized = self._columns_for(fit.response)
            frame[predicted] = fit.predicted.reindex(frame.index)
            frame[residual] = fit.residuals.reindex(frame.index)
            frame[studentized] = fit.studentized.reindex(frame.index)
            for row in fit.effect_tests.to_dict("records"):
                effects.append({"Response": fit.response, "Term": row["Term"], "LogWorth": row["LogWorth"],
                                "F Ratio": row["F Ratio"], "Prob > F": row["Prob > F"]})
            if not fit.profile.empty:
                profiles.append(fit.profile.assign(Response=fit.response))
        repo.import_dataframe(frame, table_name=names["results"], normalize_columns=False)
        repo.import_dataframe(pd.DataFrame(effects), table_name=names["effects"], normalize_columns=False)
        written: list[dict[str, Any]] = [{"table": names["results"]}, {"table": names["effects"]}]
        if profiles:
            repo.import_dataframe(pd.concat(profiles, ignore_index=True), table_name=names["profile"],
                                  normalize_columns=False)
            written.append({"table": names["profile"]})

        self.created_figure_ids = []
        if charts:
            for fit in result:
                figure_id = self._make_figure(fit, names, charts)
                self.created_figure_ids.append(figure_id)
                written.append({"figure": figure_id, "name": fit.response})
        return written

    @staticmethod
    def _quoted(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def _literal(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    def _select(self, table: str, roles: Mapping[str, str], where: str = "") -> tuple[str, dict[str, str]]:
        """A series query naming each role's column as the role, the way every series reads its data."""
        columns = ", ".join(f"{self._quoted(column)} AS {self._quoted(role)}" for role, column in roles.items())
        sql = f"SELECT {columns} FROM {self._quoted(table)}" + (f" WHERE {where}" if where else "")
        return sql, {role: role for role in roles}

    def _panels(self, fit: lm.ResponseFit, names: Mapping[str, str], charts: list[str]) -> list[tuple]:
        """(chart type, title, x label, y label, [(series name, sql, roles, style)]) per chart ticked."""
        predicted, residual, _studentized = self._columns_for(fit.response)
        response = self._literal(fit.response)
        panels = []

        def one(table: str, roles: Mapping[str, str], where: str = "", name: str = fit.response,
                style: Mapping[str, Any] | None = None) -> tuple:
            sql, mapped = self._select(names[table], roles, where)
            return (name, sql, mapped, dict(style or {}))

        if "actual" in charts:
            panels.append(("Scatter Plot", _("Actual by predicted"), predicted, fit.response,
                           [one("results", {"x": predicted, "y": fit.response})]))
        if "pareto" in charts:
            panels.append(("Pareto Chart", _("Effects Pareto (LogWorth)"), "", "LogWorth",
                           [one("effects", {"Y": "LogWorth", "X": "Term"}, f'"Response" = {response}')]))
        if "residual" in charts:
            panels.append(("Scatter Plot", _("Residual by predicted"), predicted, residual,
                           [one("results", {"x": predicted, "y": residual})]))
        if "qq" in charts:
            panels.append(("Q-Q Plot", _("Residual Q-Q"), "", residual, [one("results", {"value": residual})]))
        if "profile" in charts and not fit.profile.empty:
            series = [
                one("profile", {"x": "Coded", "y": "Predicted"},
                    f'"Response" = {response} AND "Factor" = {self._literal(factor)}',
                    name=factor, style={"linestyle": "-"})
                for factor in dict.fromkeys(fit.profile["Factor"])
            ]
            panels.append(("Scatter Plot", _("Prediction profile"), _("Factor (coded, -1 to +1)"), fit.response, series))
        if "interaction" in charts:
            pairs = [t for t in fit.terms if len(t) == 2 and t[0] != t[1]]
            if pairs:
                tests = fit.effect_tests.set_index("Term")["LogWorth"]
                a, b = max(pairs, key=lambda t: tests.get(lm.term_label(t), -math.inf))
                panels.append(("Interaction Plot", _("Interaction {a} × {b}").format(a=a, b=b), a, fit.response,
                               [one("results", {"x": a, "y": fit.response, "trace": b})]))
        return panels

    def _make_figure(self, fit: lm.ResponseFit, names: Mapping[str, str], charts: list[str]) -> int:
        """A figure of the charts ticked, two to a row, with the response's report in its notes."""
        repo = self._repo
        panels = self._panels(fit, names, charts)
        columns = 2 if len(panels) > 1 else 1
        rows = max(1, math.ceil(len(panels) / columns))
        notes = report_html.document(_("Fit Model"), f"{self.table} · {fit.response}", self._report(fit))
        figure_id = int(repo.create_figure_descriptor(
            name=f"Fit Model · {fit.response}", nrows=rows, ncols=columns,
            options={FIGURE_VIEW_OPTIONS_KEY: {"notes_html": notes}},
        ))
        for index, (chart, title, x_label, y_label, series) in enumerate(panels):
            axis_id = int(repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type=chart,
                title=title, x_label=x_label, y_label=y_label, options={},
            ))
            for position, (name, sql, roles, style) in enumerate(series):
                repo.create_series_descriptor(
                    axis_id=axis_id, series_index=position, name=name, sql_query=sql, roles=roles, style=style,
                )
        return figure_id
