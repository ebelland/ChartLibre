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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
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
from app.series_operations.parameter_spec import FloatParam
from app.table_operations.column_roles import CONTINUOUS, NOMINAL, ColumnCasting, ColumnRole
from app.table_operations.dialog_base import TableOperationDialogBase
from app.styles.style import TitledCard
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

#: The charts OK can draw, as (key, check box label, ticked by default).
CHARTS: tuple[tuple[str, str, bool], ...] = (
    ("actual", "Actual by predicted", True),
    ("pareto", "Effects Pareto", True),
    ("residual", "Residual by predicted", True),
    ("qq", "Residual Q-Q", False),
    ("profile", "Prediction profile", True),
    ("interaction", "Interaction", False),
)


#: JMP's emphases for least squares: key, name, the charts the report opens with.
EMPHASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("leverage", "Effect Leverage", ("actual", "residual", "interaction")),
    ("screening", "Effect Screening", ("pareto", "profile")),
    ("minimal", "Minimal Report", ()),
)

#: JMP 18's help, page by page.
_JMP_HELP = "https://www.jmp.com/support/help/en/18.0/index.shtml#page/jmp/"
DOCS: dict[str, tuple[tuple[str, str], ...]] = {
    "standard": (("Fit Model", _JMP_HELP + "overview-of-the-fit-model-platform.shtml"),
                 ("Standard Least Squares", _JMP_HELP + "standard-least-squares-models.shtml")),
    "stepwise": (("Stepwise", _JMP_HELP + "overview-of-stepwise-regression.shtml"),),
    "glm": (("Generalized Linear Model", _JMP_HELP + "generalized-linear-models.shtml"),),
    "nominal": (("Logistic", _JMP_HELP + "overview-of-the-nominal-and-ordinal-logistic-personalities.shtml"),),
    "ordinal": (("Logistic", _JMP_HELP + "overview-of-the-nominal-and-ordinal-logistic-personalities.shtml"),),
}
EMPHASIS_DOC = _JMP_HELP + "standard-least-squares-options-in-the-fit-model-launch-window.shtml"
PROFILER_DOC = _JMP_HELP + "overview-of-the-prediction-profiler.shtml"


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
        ColumnRole("response", "Y", "The columns to model: one model each.", minimum=1),
    )
    #: Alpha beside the personality: both say how the model is fitted.
    PARAMETERS_TAB = "Model"

    PARAMS = (
        FloatParam(
            "alpha",
            "Significance level (α):",
            tooltip=(
                "A term is significant when its p-value is at most this; the "
                "Stepwise personality adds or keeps only such terms."
            ),
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
        # Personality and Emphasis first, top right as in JMP's launch window.
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self._personality = QComboBox()
        for key, label in lm.PERSONALITIES:
            self._personality.addItem(_(label), key)
        self._personality.setToolTip(_("How the responses are modelled."))
        self._personality.currentIndexChanged.connect(lambda _i: self._on_personality_changed())
        form.addRow(_("Personality:"), self._personality)
        self._emphasis = QComboBox()
        for key, label, _charts in EMPHASES:
            self._emphasis.addItem(_(label), key)
        self._emphasis.setToolTip(_("Which charts the report opens with; each can still be ticked or not."))
        self._emphasis.currentIndexChanged.connect(lambda _i: self._apply_emphasis())
        form.addRow(_("Emphasis:"), self._emphasis)
        self._family = QComboBox()
        for key, label in lm.FAMILIES:
            self._family.addItem(_(label), key)
        self._family.setToolTip(_("The response's distribution; its canonical link (log for Gamma)."))
        self._family.currentIndexChanged.connect(lambda _i: self.inputs_changed())
        form.addRow(_("Distribution:"), self._family)
        self._direction = QComboBox()
        for key, label in lm.STEPWISE:
            self._direction.addItem(_(label), key)
        self._direction.setToolTip(_("Backward drops terms from the full model; forward adds them to the mean."))
        self._direction.currentIndexChanged.connect(lambda _i: self.inputs_changed())
        form.addRow(_("Direction:"), self._direction)
        for combo in (self._personality, self._emphasis, self._family, self._direction):
            # Long names elide instead of widening the window.
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self._form = form
        self._doc_link = QLabel()
        self._doc_link.setOpenExternalLinks(True)
        self._doc_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._doc_link.setWordWrap(True)
        # Tabs beside the columns: the roles and the effects built from them,
        # the model's settings, the charts - one page of all of it was taller
        # than a laptop's screen.
        self.roles_widget.tabs.setTabText(0, _("Roles and effects"))
        model = TitledCard(self, _("Personality"))
        model_layout = model.card.layout()
        model_layout.addLayout(form)
        model_layout.addWidget(self._doc_link)
        self.roles_widget.add_tab(_(self.PARAMETERS_TAB)).addWidget(model)

        # Buttons in two short rows above the list: a column of them beside it
        # made the frame taller than the list needs to be.
        effects = QVBoxLayout()
        effects.setSpacing(6)
        buttons = QHBoxLayout()
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
        buttons.addStretch(1)
        effects.addLayout(buttons)
        degree = QHBoxLayout()
        degree.addWidget(QLabel(_("Degree")))
        self._degree = QSpinBox()
        self._degree.setRange(1, 6)
        self._degree.setValue(2)
        self._degree.setToolTip(_("How many factors the factorial macro's interactions combine, at most."))
        degree.addWidget(self._degree)
        degree.addStretch(1)
        effects.addLayout(degree)

        self._term_list = QListWidget()
        self._term_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._term_list.setToolTip(_("Double-click an effect to remove it."))
        # As tall as the page allows, and no less than a few effects.
        term_rows = max(self._term_list.fontMetrics().height() + 4, 18)
        self._term_list.setMinimumHeight(4 * term_rows)
        self._term_list.setMinimumWidth(140)
        self._term_list.itemDoubleClicked.connect(lambda _item: self._remove_selected())
        effects.addWidget(self._term_list, 1)
        effects_card = TitledCard(self, _("Construct Model Effects"))
        effects_card.card.layout().addLayout(effects)
        effects.setStretch(effects.count() - 1, 1)
        layout.addWidget(effects_card, 1)

        charts = TitledCard(self, _("Charts to draw"))
        grid = QVBoxLayout()
        # Spaced as a list of choices, not packed as a block of text.
        grid.setSpacing(10)
        self._chart_checks: dict[str, QCheckBox] = {}
        for index, (key, label, ticked) in enumerate(CHARTS):
            check = QCheckBox(_(label))
            check.setChecked(ticked)
            # An attribute of its own, so the window remembers it.
            setattr(self, f"_chart_{key}", check)
            self._chart_checks[key] = check
            grid.addWidget(check)
        charts.card.layout().addLayout(grid)
        self.roles_widget.add_tab(_("Charts")).addWidget(charts)
        self._on_personality_changed()

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
        casting = self.roles_widget.casting()
        if self.personality() in lm.CATEGORICAL_RESPONSE:
            continuous = [name for name in casting.columns("response") if casting.kinds.get(name) != NOMINAL]
            if continuous:
                return [_("{personality} needs a nominal Y; {columns} is continuous - set its type to Nominal, "
                          "or choose a least-squares personality.").format(
                    personality=self._personality.currentText(), columns=", ".join(continuous))]
        if self.personality() not in lm.CATEGORICAL_RESPONSE:
            nominal = [name for name in casting.columns("response") if casting.kinds.get(name) == NOMINAL]
            if nominal:
                return [_("{personality} needs a continuous Y; {columns} is nominal - choose Nominal or Ordinal Logistic.").format(
                    personality=self._personality.currentText(), columns=", ".join(nominal))]
        both = sorted(set(casting.columns("response")) & {f.name for f in self._factors()})
        if both:
            return [_("A column cannot be both a response and a factor: {columns}.").format(columns=", ".join(both))]
        return []

    def personality(self) -> str:
        return str(self._personality.currentData() or "standard")

    def _on_personality_changed(self) -> None:
        """Show what the personality asks for, offer the charts it can draw, link its documentation."""
        personality = self.personality()
        categorical = personality in lm.CATEGORICAL_RESPONSE
        for widget, shown in (
            (self._emphasis, personality in ("standard", "stepwise")),
            (self._family, personality == "glm"),
            (self._direction, personality == "stepwise"),
        ):
            widget.setVisible(shown)
            label = self._form.labelForField(widget)
            if label is not None:
                label.setVisible(shown)
        for key, check in self._chart_checks.items():
            usable = not categorical or key in ("pareto", "profile")
            check.setEnabled(usable)
            if not usable:
                check.setChecked(False)
        links = [(_(label), url) for label, url in DOCS.get(personality, ())]
        if personality in ("standard", "stepwise"):
            links.append((_("Emphasis"), EMPHASIS_DOC))
        links.append((_("Prediction profiler"), PROFILER_DOC))
        self._doc_link.setText(
            _("JMP documentation:") + " " + " · ".join(f"<a href='{html.escape(url)}'>{html.escape(text)}</a>" for text, url in links)
        )
        self.inputs_changed()

    def _apply_emphasis(self) -> None:
        """Tick the charts the emphasis opens with, as JMP's emphases choose their reports."""
        key = str(self._emphasis.currentData())
        charts = next((charts for k, _label, charts in EMPHASES if k == key), ())
        for chart, check in self._chart_checks.items():
            check.setChecked(chart in charts and check.isEnabled())

    def charts(self) -> list[str]:
        return [key for key, check in self._chart_checks.items() if check.isChecked() and check.isEnabled()]

    def parameter_values(self) -> dict[str, Any]:
        """The parameters, and the model: its terms and the factors' types and coding."""
        values = super().parameter_values()
        values["terms"] = [list(t) for t in self._terms]
        values["factors"] = [
            {"name": f.name, "kind": f.kind, "low": f.low, "high": f.high} for f in self._factors()
        ]
        values["charts"] = self.charts()
        values["personality"] = self.personality()
        values["family"] = str(self._family.currentData())
        values["direction"] = str(self._direction.currentData())
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
            alpha=float(params.get("alpha", 0.05)),
            personality=str(params.get("personality", "standard")),
            family=str(params.get("family", "normal")),
            direction=str(params.get("direction", "backward")),
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
                    cells.append("" if pd.isna(value) else str(int(value)))
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
                [_("Source"), _("LogWorth"), _(fit.p_column)],
                [[html.escape(r["Term"]), report_html.format_number(r["LogWorth"], digits=3),
                  report_html.format_p_value(r[fit.p_column])]
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
            _(fit.whole_title),
            report_html.table([_(c) for c in fit.anova.columns], self._rows(fit.anova, ("Prob > F", "Prob > ChiSq"))),
        ))
        if fit.lack_of_fit is not None:
            blocks.append(report_html.section(
                _("Lack of fit"),
                report_html.table([_(c) for c in fit.lack_of_fit.columns], self._rows(fit.lack_of_fit, ("Prob > F",))),
            ))
        blocks.append(report_html.section(
            _("Parameter estimates (coded units)"),
            report_html.table([_(c) for c in fit.estimates.columns],
                              self._rows(fit.estimates, ("Prob > |t|", "Prob > |z|"))),
        ))
        blocks.append(report_html.section(
            _("Effect tests"),
            report_html.table(
                [_(c) for c in fit.effect_tests.columns if c != "LogWorth"],
                self._rows(fit.effect_tests.drop(columns=["LogWorth"]), ("Prob > F", "Prob > ChiSq")),
            ),
        ))
        if fit.levels:
            blocks.insert(0, report_html.note(
                _("Levels: {levels}. The predictions are the probability of {first}.").format(
                    levels=", ".join(fit.levels), first=fit.levels[0])))
        title = _("Response {name}").format(name=fit.response)
        return report_html.section(f"{title} · {_(dict(lm.PERSONALITIES)[fit.personality])}", *blocks)

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
                effects.append({"Response": fit.response, **row})
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

    def _panels(self, fit: lm.ResponseFit, names: Mapping[str, str], charts: list[str]) -> list[tuple]:
        """(chart type, title, x label, y label, [(series name, sql, roles, style)]) per chart ticked."""
        predicted, residual, _studentized = self._columns_for(fit.response)
        response = self.literal(fit.response)
        panels = []

        def one(table: str, roles: Mapping[str, str], where: str = "", name: str = fit.response,
                style: Mapping[str, Any] | None = None) -> tuple:
            sql, mapped = self.series_select(names[table], roles, where)
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
                    f'"Response" = {response} AND "Factor" = {self.literal(factor)}',
                    name=factor, style={"linestyle": "-"})
                for factor in dict.fromkeys(fit.profile["Factor"])
            ]
            y_label = f"Prob({fit.levels[0]})" if fit.levels else fit.response
            panels.append(("Scatter Plot", _("Prediction profile"), _("Factor (coded, -1 to +1)"), y_label, series))
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
        notes = report_html.document(_("Fit Model"), f"{self.table} · {fit.response}", self._report(fit))
        return self.create_report_figure(f"Fit Model · {fit.response}", notes, self._panels(fit, names, charts))
