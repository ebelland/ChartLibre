"""Read-only statistics dialog for selected chart series.

The dialog inherits the shared DataOperationDialogBase shell. It does not
create generated chart series and does not write result tables. It reports:

- descriptive statistics for every checked series;
- one-sample tests for every checked series;
- paired-sample and independent-sample tests for every pair when multiple
  series are checked, and a comparison of all of them as groups (ANOVA,
  Welch's ANOVA, Kruskal-Wallis, Tukey HSD);
- optional normality and correlation sections, controlled by the model combo.

The arithmetic is in app.analysis.statistics; this module reads the series,
collects the parameters and lays the results out.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import html
import itertools
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
)

from app.analysis import NUMERICAL_FAILURES
from app.analysis import statistics as st
from app.data.data_source import quote_identifier
from app.data.data_source import row_value,parse_roles
from app.data.select_sql import projection_source
from app.plugin_api.data_operations.parameter_form import FIELD_MAX_WIDTH
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam
from app.plugin_api.data_operations.results import OperationResult
from app.plugin_api.data_operations.dialog_base import (
    ResultSeriesSpec,
    DataOperationDialogBase,
)
from app.styles.style import CardFrame
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.utils import report_html
from app.utils.distribution_fit import (
    CURATED_DISTRIBUTIONS,
    DEFAULT_RANK,
    DistributionFit,
    fit_distributions,
)
from app.utils.i18n import _
from app.utils.coercion import to_numbers

_SERIES_ROLE = Qt.ItemDataRole.UserRole


@dataclass(slots=True)
class SeriesStatsSample:
    """One selected series converted to finite numeric arrays."""

    name: str
    x: np.ndarray
    y: np.ndarray
    rowids: np.ndarray


@dataclass(slots=True)
class SeriesTable:
    """One selected series as its query returns it, for the models that read
    columns by role - categories, factors, times - rather than one sample."""

    name: str
    frame: pd.DataFrame
    roles: dict[str, Any]
    sql: str = ""

    def label(self, column: str) -> str:
        """The source column behind an alias - "species" for "species AS x"."""
        return projection_source(self.sql, column) or column


#: Models that read a series' columns by role (todo R-04): each checked
#: series is analysed on its own, and none of them is part of "All".
FRAME_MODELS: frozenset[str] = frozenset({"contingency", "two_way", "survival"})

#: Where each model finds its columns: role names, tried in order, each as
#: the role's own column and then as a column of that name.
MODEL_COLUMNS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "contingency": {"rows": ("x", "group"), "columns": ("y", "trace"), "weight": ("weight",), "strata": ("strata",)},
    "two_way": {"first": ("x",), "second": ("trace", "group"), "response": ("y", "value")},
    "survival": {"time": ("time", "x"), "event": ("event",), "group": ("group",)},
}

#: The columns a model cannot do without.
REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "contingency": ("rows", "columns"),
    "two_way": ("first", "second", "response"),
    "survival": ("time", "event"),
}


def model_columns(model: str, frame: pd.DataFrame, roles: Mapping[str, Any]) -> dict[str, str]:
    """The frame column for each part *model* reads, found by role then by name."""
    found: dict[str, str] = {}
    for part, names in MODEL_COLUMNS.get(model, {}).items():
        for name in names:
            mapped = str(roles.get(name, "") or "").strip()
            column = mapped if mapped in frame.columns else (name if name in frame.columns else "")
            if column:
                found[part] = column
                break
    return found


@dataclass(slots=True)
class SeriesStatsResult(OperationResult):
    """Complete statistics result for one dialog run."""

    samples: list[SeriesStatsSample]
    popmean: float
    alternative: str
    model: str
    trim_percent: float
    distribution: str = "best"
    exhaustive: bool = False
    rank_by: str = DEFAULT_RANK
    #: The series as tables, for FRAME_MODELS.
    tables: list[SeriesTable] = field(default_factory=list)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "popmean": self.popmean,
            "alternative": self.alternative,
            "trim_percent": self.trim_percent,
            "distribution": self.distribution,
        }

    @property
    def series(self) -> tuple[str, ...]:
        return tuple(sample.name for sample in self.samples)

    def to_df(self) -> pd.DataFrame:
        """The values that were tested, one row per point."""
        return pd.DataFrame(
            {
                "series": np.concatenate([[sample.name] * sample.y.size for sample in self.samples])
                if self.samples else [],
                "x": np.concatenate([sample.x for sample in self.samples]) if self.samples else [],
                "y": np.concatenate([sample.y for sample in self.samples]) if self.samples else [],
            }
        )

    # A report, not a table: the dialog publishes format_results and, when
    # asked, draws the chart the model describes (see _run_statistics).
    # Nothing is saved per result.
    def preview(self, dialog: Any, axis_id: int) -> None:
        del dialog, axis_id

    def apply(self, dialog: Any, axis_id: int) -> None:
        del dialog, axis_id


def _models_with(*per_model: Mapping[str, Sequence[Any]]) -> tuple[str, ...]:
    """The models with something in any of *per_model* (a chart, a distribution): a row's visible_for."""
    return tuple(dict.fromkeys(model for table in per_model for model, entries in table.items() if entries))


class SeriesStatisticsDialog(DataOperationDialogBase):
    """Read-only statistical summary and hypothesis-test dialog.

    The model selector controls which sections are shown:
    - all: descriptive, one-sample, normality, paired, and correlation results;
    - descriptive: summary statistics only;
    - one_sample: one-sample tests only;
    - normality: normality/shape tests only;
    - randomness: runs, turning-point, Durbin-Watson and Ljung-Box tests of the order (opt-in);
    - paired: paired-sample tests only;
    - independent: two-sample tests for every pair (Welch, Student, Mann-Whitney);
    - groups: all checked series as groups (ANOVA, Kruskal-Wallis, Tukey HSD);
    - correlation: paired correlation/association tests only.
    """
    Name: str = "Statistics"
    Description = "Compute metrics"

    # Order-independent summaries, so only the universal checks apply. Two
    # points is the floor: a standard deviation of one value is not a number
    # anyone wants reported.
    INPUT_MINIMUM_POINTS = 2

    Icon = """
    <path d="M5 19h14"/>
    <path d="M7 16v-5"/>
    <path d="M12 16V7"/>
    <path d="M17 16v-8"/>
    <path d="M6 5h12"/>
    """
    Keywords: str = (
        "statistiche descriptive descrittive mean media median mediana standard deviation deviazione "
        "standard summary riepilogo min max quantile percentile"
    )
    Goals: list[str] = ["describe"]
    Category: str = "Basic Analysis"
    Order: int = 13

    WINDOW_SIZE = (1060, 720)
    # Most useful on several series at once: with all of them ticked, the All/Paired models show
    # paired-sample results at once when there is more than one.
    SELECTS_ALL_SERIES = True

    def __init__(self, **kwargs: Any) -> None:
        # Set before super(): the base constructor calls the builder hooks, which reach these. The chart this
        # dialog adds when asked: created on the first Preview, reused, removed on Close unless Apply kept it.
        self._result_chart_type: str = ""
        self._result_axis_options: dict[str, Any] = {}
        self._fit_cache: dict[tuple[bytes, int, bool, str], list[DistributionFit]] = {}
        super().__init__(**kwargs)
        self.set_doc_link("SciPy statistics",
                          "https://docs.scipy.org/doc/scipy/reference/stats.html#summary-statistics")
        self._sync_model_controls()

    # ------------------------------------------------------------------
    # UI hooks
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        self.model_combo.clear()
        self.model_combo.addItem(_("All statistics"), "all")
        self.model_combo.addItem(_("Descriptive statistics"), "descriptive")
        self.model_combo.addItem(_("One-sample tests"), "one_sample")
        self.model_combo.addItem(_("Normality / shape tests"), "normality")
        self.model_combo.addItem(_("Randomness / serial independence"), "randomness")
        self.model_combo.addItem(_("Paired-sample tests"), "paired")
        self.model_combo.addItem(_("Independent-sample tests"), "independent")
        self.model_combo.addItem(_("Group comparison (ANOVA)"), "groups")
        self.model_combo.addItem(_("Correlation / association"), "correlation")
        self.model_combo.addItem(_("Distribution fit"), "distribution")
        self.model_combo.addItem(_("Contingency table (chi-squared, Fisher)"), "contingency")
        self.model_combo.addItem(_("Two-way ANOVA"), "two_way")
        self.model_combo.addItem(_("Survival (Kaplan-Meier, log-rank)"), "survival")
        self.model_combo.setToolTip(_("Choose which statistics section to display."))

    def build_model_selector(self) -> QWidget:
        panel = CardFrame(self, "statisticsModelCard")
        layout = panel.layout()
        self.model_combo.setMaximumWidth(FIELD_MAX_WIDTH)
        layout.addWidget(self.model_combo)
        note = QLabel(
            _("Check one or more series. One-sample tests are calculated for "
            "each checked series. Paired and independent tests are calculated "
            "for every pair, and the group comparison for all of them, when "
            "two or more series are checked."),
            panel,
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)
        # Everything at the top: the space left goes below, not around the note.
        layout.addStretch(1)
        return panel

    def _sync_model_controls(self) -> None:
        """Show and fill the chart controls the selected model can use.

        Hidden rather than disabled when a model has no chart: "all" prints
        every section at once and no single picture illustrates that, and a
        greyed-out control invites the user to hunt for what would enable it.

        The current choice is preserved across the refill where the new model
        still offers it, so moving between two models that both draw a
        histogram does not silently reset the chart to something else.
        """
        model = str(self.model_combo.currentData() or "all")

        charts = self.CHARTS_BY_MODEL.get(model, ())
        has_chart = bool(charts)
        # Shown only for a model one chart illustrates: the PARAMS' visible_for.
        if self.chart_form is not None:
            self.chart_form.set_choices("chart_type", [(chart_type, chart_type) for chart_type, _title in charts])

        # The distributions the model's chart can draw (shown where there are some: the PARAMS' visible_for).
        distributions = self.DISTRIBUTIONS_BY_MODEL.get(model, ())
        if not distributions and has_chart:
            distributions = self.DEFAULT_DISTRIBUTIONS
        if self._parameter_form_spec is not None:
            self._parameter_form_spec.set_choices("distribution", [(label, value) for value, label in distributions])

    def connect_operation_signals(self) -> None:
        super().connect_operation_signals()  # the rows the model shows, and the results stale
        self.model_combo.currentIndexChanged.connect(lambda *_args: self._sync_model_controls())

    # ------------------------------------------------------------------
    # Base pipeline overrides: read-only operation
    # ------------------------------------------------------------------
    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"statistics_dialog": "series_statistics"}

    def result_series_spec(self, axis_id: int, table_name: str, result: Any) -> ResultSeriesSpec:
        raise NotImplementedError("Statistics results are read-only and do not create chart series.")

    def preview(self) -> bool:
        return self._run_statistics(verb="Preview")

    def apply(self) -> bool:
        self.remember_window()
        return self._run_statistics(verb="Applied")

    def ok(self) -> None:
        if self.apply():
            self.accept()

    def compute_results(self) -> Sequence[SeriesStatsResult]:
        model = str(self.model_combo.currentData() or "all")
        tables: list[SeriesTable] = []
        if model in FRAME_MODELS:
            samples = []
            for row in self._rows_for_statistics():
                frame = self._series_frame(str(row["sql_query"]))
                if not frame.empty:
                    tables.append(SeriesTable(
                        str(row["name"] or f"Series {row['id']}"), frame, parse_roles(row["roles"]), str(row["sql_query"] or "")
                    ))
            if not tables:
                raise ValueError("Select at least one source series.")
        else:
            samples = self._selected_samples()
            if not samples:
                raise ValueError("Select at least one source series.")
        p = self.parameter_values()
        return [
            SeriesStatsResult(
                tables=tables,
                samples=samples,
                popmean=float(p["popmean"]),
                alternative=str(p["alternative"] or "two-sided"),
                model=str(self.model_combo.currentData() or "all"),
                trim_percent=float(p["trim_percent"]) / 100.0,
                distribution=str(p["distribution"] or "best"),
                exhaustive=bool(p["exhaustive"]),
                rank_by=str(p["rank_by"] or DEFAULT_RANK),
            )
        ]

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        result = results[0]
        if not isinstance(result, SeriesStatsResult):
            return ""
        return self._format_statistics_html(result)

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------
    def _checked_series_rows(self) -> list[Any]:
        """Return every checked series row, not just the current one.

        Through the base's selected_series, which is what draws a table's
        columns first when the data comes from a table.
        """
        return list(self.selected_series())

    def _rows_for_statistics(self) -> list[Any]:
        """Return the checked rows. Only those, and all of them.

        This used to substitute *every visible series on the axis* whenever
        one row was checked and the model was one that needs pairs -
        reasoning that the dialog is often opened from a single active
        series while the requested test needs two. The result was that
        unchecking two of three series changed nothing: all three were
        still measured, still reported, and the checkbox that had just
        been cleared was contradicted on screen.

        A guess about intent does not get to overrule an explicit action.
        A model that needs two samples and has one says so instead - the
        report ends with that note - which is a thing the user can act on.
        """
        return self._checked_series_rows()

    def _selected_samples(self) -> list[SeriesStatsSample]:
        """Return numeric samples for all selected rows.

        Statistics should not depend on the X axis being numeric. The shared
        ``get_data`` helper is designed for chart operations that need finite X
        and Y values; for this dialog we only require a numeric measurement
        column. This loader therefore reads the series SQL directly, tries the
        configured Y/Z role first, then falls back to the first numeric column.
        """
        samples: list[SeriesStatsSample] = []
        skipped: list[str] = []

        for row in self._rows_for_statistics():
            sample = self._sample_from_series_row(row)
            if sample is None:
                skipped.append(str(row["name"] or f"Series {row['id']}"))
                continue
            samples.append(sample)

        if skipped:
            applogger.warning(
                "Statistics skipped %d series with no numeric values: %s",
                len(skipped),
                ", ".join(skipped),
            )
        return samples

    def _sample_from_series_row(self, row: Any) -> SeriesStatsSample | None:
        """Build one numeric sample from a series descriptor row."""
        name = str(row["name"] or f"Series {row['id']}")
        roles = parse_roles(row["roles"])
        frame = self._series_frame(str(row["sql_query"]))
        if frame.empty:
            return None

        y_column = self._best_numeric_column(frame, roles)
        if y_column is None:
            return None

        y = to_numbers(frame[y_column]).to_numpy(dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            return None

        x = self._x_values_for_statistics(frame, roles, finite)
        rowids = np.arange(1, int(np.sum(finite)) + 1, dtype=int)
        sample_name = name if str(y_column) in {name, str(roles.get("y", "")), str(roles.get("z", ""))} else f"{name} [{y_column}]"
        return SeriesStatsSample(
            name=sample_name,
            x=x,
            y=y[finite],
            rowids=rowids,
        )

    def _series_frame(self, sql_query: str) -> pd.DataFrame:
        """Execute a series SQL query as a read-only DataFrame.

        Avoid ``query_source_table`` here: some chart descriptors contain SQL
        fragments or generated-series queries that do not resolve to a physical
        user table, and passing those through the table-name resolver logs
        ``Invalid table name: ''``. Statistics only needs the query result rows.
        """
        sql = str(sql_query or "").strip().rstrip(";").strip()
        if not sql:
            return pd.DataFrame()

        try:
            if sql.lower().startswith(("select", "with")):
                return self._repo.query_df(sql)

            # Treat a bare descriptor value as a physical table name.
            table_name = sql.strip().strip('"')
            if table_name:
                return self._repo.table_frame(table_name)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.debug("Failed to read series data for statistics.", exc_info=True)
        return pd.DataFrame()

    def _best_numeric_column(self, frame: pd.DataFrame, roles: Mapping[str, Any]) -> str | None:
        """Return the most appropriate numeric measurement column."""
        preferred = [
            str(roles.get("y", "")).strip(),
            str(roles.get("z", "")).strip(),
            str(roles.get("value", "")).strip(),
            str(roles.get("values", "")).strip(),
        ]
        for column in preferred:
            if column and column in frame.columns:
                values = to_numbers(frame[column])
                if values.notna().any():
                    return column

        excluded = {
            "hide",
            "clusterid",
            "__rowid__",
            str(roles.get("x", "")).strip().lower(),
        }
        best_column: str | None = None
        best_count = 0
        for column in frame.columns:
            if str(column).strip().lower() in excluded:
                continue
            values = to_numbers(frame[column])
            count = int(values.notna().sum())
            if count > best_count:
                best_column = str(column)
                best_count = count
        return best_column

    def _x_values_for_statistics(
        self,
        frame: pd.DataFrame,
        roles: Mapping[str, Any],
        finite_y: np.ndarray,
    ) -> np.ndarray:
        """Return optional numeric X values, or row order when X is categorical."""
        x_column = str(roles.get("x", "")).strip()
        if x_column and x_column in frame.columns:
            x_values = self.numeric_x(frame[x_column])
            if np.isfinite(x_values[finite_y]).any():
                return x_values[finite_y]
        return np.arange(1, int(np.sum(finite_y)) + 1, dtype=float)

    # ------------------------------------------------------------------
    # Distribution fit
    # ------------------------------------------------------------------
    def _distribution_fits(
        self, values: np.ndarray, *, exhaustive: bool, rank_by: str
    ) -> list[DistributionFit]:
        """Return the ranked fits for one sample, memoised.

        Memoised because the dialog refreshes on every control it owns, and a
        sweep costs half a second curated or a dozen seconds exhaustive.
        Without this, nudging the reference value - which has nothing to do
        with the fit - would refit every candidate and freeze the window.

        Keyed on the sample's bytes: two different series with identical values
        genuinely have identical fits, and the key stays correct when a series
        is edited underneath us.
        """
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        key = (finite.tobytes(), finite.size, exhaustive, rank_by)

        cached = self._fit_cache.get(key)
        if cached is None:
            cached = fit_distributions(
                finite, exhaustive=exhaustive, rank_by=rank_by
            )
            # Bounded so a long session cannot grow without limit; the useful
            # entries are the samples currently checked, which is a handful.
            if len(self._fit_cache) > 32:
                self._fit_cache.clear()
            self._fit_cache[key] = cached
        return cached

    def _distribution_section(self, result: SeriesStatsResult) -> str:
        """Rank the candidate distributions for every checked series."""
        blocks: list[str] = []
        for sample in result.samples:
            fits = self._distribution_fits(
                sample.y, exhaustive=result.exhaustive, rank_by=result.rank_by
            )
            if not fits:
                blocks.append(
                    report_html.note(
                        f"{sample.name}: no candidate distribution could be "
                        "fitted to this sample."
                    )
                )
                continue

            rows = [
                [
                    fit.name,
                    self._format_number(fit.ks_statistic),
                    self._format_pvalue(fit.pvalue),
                    self._format_number(fit.aic),
                    self._format_number(fit.bic),
                    # scipy's own parameter order, which is what its
                    # documentation for that distribution describes.
                    ", ".join(f"{value:.6g}" for value in fit.params),
                ]
                for fit in fits[: self.DISTRIBUTION_ROWS]
            ]
            blocks.append(
                report_html.section(
                    f"{sample.name} (n = {fits[0].n})",
                    self._table(
                        ["Distribution", "KS D", "p", "AIC", "BIC", "Parameters"],
                        rows,
                    ),
                )
            )

            if len(fits) > self.DISTRIBUTION_ROWS:
                # No silent truncation: a ranking that quietly drops candidates
                # reads as "these are all of them".
                blocks.append(
                    report_html.note(_(
                        "{count} further candidates fitted worse and are not shown."
                    ).format(count=len(fits) - self.DISTRIBUTION_ROWS))
                )

        blocks.append(
            report_html.note(_(
                "Ranked by {rank}. D is the largest gap "
                "between the sample's empirical CDF and the fitted one, and is "
                "shown as a diagnostic rather than as the ordering: on samples "
                "drawn from known families, ranking by D alone puts "
                "four-parameter beta first every time, because it measures how "
                "close the curve got and not how many free parameters it took. "
                "The p-value is optimistic for the same reason it always is "
                "here - the parameters were estimated from the sample being "
                "tested, so it asks whether the data could have come from this "
                "fitted curve, not from that family."
            ).format(rank=result.rank_by.upper()))
        )
        return "".join(blocks)

    def _anderson_options(self) -> dict[str, Any]:
        """The Anderson-Darling p-value settings, as the engine takes them."""
        p = self.parameter_values()
        batch = int(p["monte_carlo_batch"])
        return {
            "anderson_method": str(p["anderson_method"] or "interpolate"),
            "resamples": int(p["monte_carlo_resamples"]),
            "batch": None if batch <= 0 else batch,
        }

    @staticmethod
    def _paired(left: SeriesStatsSample, right: SeriesStatsSample) -> tuple[np.ndarray, np.ndarray, str]:
        return st.paired_values(left.x, left.y, right.x, right.y)

    # ------------------------------------------------------------------
    # HTML formatting
    # ------------------------------------------------------------------
    def _format_number(self, value: Any) -> str:
        """Format numeric results compactly but readably for HTML tables."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return html.escape(str(value))
        if not np.isfinite(number):
            return ""
        abs_number = abs(number)
        if abs_number == 0:
            return "0"
        if abs_number < 1.0e-4 or abs_number >= 1.0e6:
            return f"{number:.3e}"
        if abs_number >= 1000:
            return f"{number:,.2f}"
        if abs_number >= 100:
            return f"{number:,.3f}"
        if abs_number >= 10:
            return f"{number:,.4f}".rstrip("0").rstrip(".")
        return f"{number:,.5f}".rstrip("0").rstrip(".")

    def _format_pvalue(self, value: Any) -> str:
        """Return a p-value in the house notation (``<.0001`` when small)."""
        return report_html.format_p_value(value)

    def _table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
        """Return a headed table in the shared report style.

        Text columns stay left-aligned; everything else is right-aligned, so a
        column of numbers can be read down the page.
        """
        text_columns = {"series", "pair", "groups", "test", "note", "alignment"}
        align = [
            "left" if str(header).strip().lower() in text_columns else "right"
            for header in headers
        ]
        # Translated here, after the alignment read the English names.
        return report_html.table(
            [_(str(header)) for header in headers],
            rows,
            align=align,
            empty_message=_("No applicable results for this selection."),
        )

    def _show_section(self, model: str, section: str) -> bool:
        if section in FRAME_MODELS:
            return model == section
        # "randomness" is opt-in too: its tests read the rows in the order they are
        # in, which says nothing for a sample that is not a sequence.
        # "distribution" is not part of "all": a sweep costs half a second per
        # series curated and a dozen seconds exhaustive, and All is the default
        # model.  Making it opt-in keeps the default fast.
        if section in ("distribution", "randomness"):
            return model == section
        return model == "all" or model == section

    def _format_statistics_html(self, result: SeriesStatsResult) -> str:
        """Return the report: what was run, then one section per model part."""
        if result.model in FRAME_MODELS:
            return self._format_table_models_html(result)
        sections: list[str] = [
            report_html.summary_table(
                [
                    (_("Series"), len(result.samples)),
                    (_("Model"), result.model),
                    (_("Reference value"), f"{result.popmean:g}"),
                    (_("Alternative"), result.alternative),
                    (
                        _("Anderson method"),
                        str(self.parameter_values()["anderson_method"] or "interpolate"),
                    ),
                ]
            )
        ]
        parts = sections

        if self._show_section(result.model, "descriptive"):
            desc_rows = []
            for sample in result.samples:
                desc = st.describe(sample.y, trim_proportion=result.trim_percent)
                desc_rows.append([
                    html.escape(sample.name), str(desc.get("n", "")),
                    self._format_number(desc.get("mean")), self._format_number(desc.get("trimmed_mean")),
                    self._format_number(desc.get("gmean")), self._format_number(desc.get("hmean")),
                    self._format_number(desc.get("median")), self._format_number(desc.get("mode")), str(desc.get("mode_count", "")),
                    self._format_number(desc.get("std")), self._format_number(desc.get("var")), self._format_number(desc.get("sem")),
                    self._format_number(desc.get("min")), self._format_number(desc.get("q1")), self._format_number(desc.get("q3")),
                    self._format_number(desc.get("max")), self._format_number(desc.get("range")), self._format_number(desc.get("iqr")),
                    self._format_number(desc.get("mad")), self._format_number(desc.get("skewness")), self._format_number(desc.get("kurtosis")),
                    self._format_number(desc.get("entropy")),
                    f"{self._format_number(desc.get('ci95_low'))} .. {self._format_number(desc.get('ci95_high'))}",
                ])
            parts.append(report_html.section(
                _("Descriptive statistics"),
                self._table([
                "Series", "n", "Mean", "Trim mean", "GMean", "HMean", "Median", "Mode", "Mode n",
                "Std", "Var", "SEM", "Min", "Q1", "Q3", "Max", "Range", "IQR", "MAD", "Skew", "Kurtosis", "Entropy", "95% CI mean",
            ], desc_rows),
            ))

        if self._show_section(result.model, "one_sample"):
            rows = []
            for sample in result.samples:
                for test in st.one_sample_tests(sample.y, popmean=result.popmean, alternative=result.alternative):
                    rows.append(_html_test_row(sample.name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("One-sample tests"),
                self._table(["Series", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if self._show_section(result.model, "normality"):
            rows = []
            for sample in result.samples:
                for test in st.normality_tests(sample.y, **self._anderson_options()):
                    rows.append(_html_test_row(sample.name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Normality / shape tests"),
                self._table(["Series", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if self._show_section(result.model, "randomness"):
            rows = []
            for sample in result.samples:
                for test in st.randomness_tests(sample.y):
                    rows.append(_html_test_row(sample.name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Randomness / serial independence"),
                self._table(["Series", "Test", "n", "Statistic", "p-value", "Note"], rows)
                + report_html.note(_(
                    "These tests read the values in the order they are in. A small p-value says the "
                    "order is not random: a trend, a cycle or a memory from one value to the next."
                )),
            ))

        if len(result.samples) > 1 and self._show_section(result.model, "paired"):
            rows = []
            for left, right in itertools.combinations(result.samples, 2):
                pair_name = _("{left} vs {right}").format(left=left.name, right=right.name)
                a, b, alignment = self._paired(left, right)
                for test in st.paired_tests(a, b, alignment=alignment, alternative=result.alternative):
                    rows.append(_html_test_row(pair_name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Paired-sample tests"),
                self._table(["Pair", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if len(result.samples) > 1 and self._show_section(result.model, "independent"):
            rows = []
            for left, right in itertools.combinations(result.samples, 2):
                pair_name = _("{left} vs {right}").format(left=left.name, right=right.name)
                for test in st.independent_tests(left.y, right.y, alternative=result.alternative):
                    rows.append(_html_test_row(pair_name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Independent-sample tests"),
                self._table(["Pair", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if len(result.samples) > 1 and self._show_section(result.model, "groups"):
            groups = {sample.name: sample.y for sample in result.samples}
            rows = [
                _html_test_row(_("All checked series"), test, self._format_number, self._format_pvalue)
                for test in st.group_tests(groups)
            ]
            section = self._table(["Groups", "Test", "n", "Statistic", "p-value", "Note"], rows)
            tukey = st.tukey_hsd(groups)
            if tukey:
                section += self._table(
                    ["Pair", "Difference", "p (Tukey)", "95% CI low", "95% CI high"],
                    [
                        [
                            html.escape(row["pair"]),
                            self._format_number(row["difference"]),
                            self._format_pvalue(row["pvalue"]),
                            self._format_number(row["ci_low"]),
                            self._format_number(row["ci_high"]),
                        ]
                        for row in tukey
                    ],
                )
                section += report_html.note(_(
                    "Tukey's p-values are already adjusted for the number of "
                    "pairs, so each can be read against 0.05 on its own. Where "
                    "Levene says the variances differ, read Welch's ANOVA rather "
                    "than the classical one."
                ))
            parts.append(report_html.section(_("Group comparison (ANOVA)"), section))

        if len(result.samples) > 1 and self._show_section(result.model, "correlation"):
            rows = []
            for left, right in itertools.combinations(result.samples, 2):
                pair_name = _("{left} vs {right}").format(left=left.name, right=right.name)
                a, b, alignment = self._paired(left, right)
                for test in st.correlation_tests(a, b, alignment=alignment):
                    rows.append(_html_test_row(pair_name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Correlation / association"),
                self._table(["Pair", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if self._show_section(result.model, "distribution"):
            parts.append(
                report_html.section(
                    _("Distribution fit"), self._distribution_section(result)
                )
            )

        if len(result.samples) < 2 and result.model in {"all", "paired", "independent", "groups", "correlation"}:
            parts.append(
                report_html.note(_(
                    "Paired, independent, group and correlation sections need "
                    "at least two selected series; check another one in the "
                    "Series list."
                ))
            )

        return report_html.document(
            _("Statistics"),
            _("{count} series").format(count=len(result.samples)),
            *parts,
        )

    # ------------------------------------------------------------------
    # Models that read columns by role (todo R-04)
    # ------------------------------------------------------------------
    MODEL_TITLES: Mapping[str, str] = {
        "contingency": "Contingency table",
        "two_way": "Two-way ANOVA",
        "survival": "Survival",
    }

    #: What each model looks for, said when a series does not have it.
    MODEL_NEEDS: Mapping[str, str] = {
        "contingency": "two categorical columns: the x and y roles (a Mosaic Plot's series), or columns named x and y",
        "two_way": "two factors and a response: the x, trace (or group) and y roles (an Interaction Plot's series)",
        "survival": "a time and an event column: the time and event roles (a Kaplan-Meier series), with an optional group",
    }

    def _format_table_models_html(self, result: SeriesStatsResult) -> str:
        """One section per checked series, for the models of FRAME_MODELS."""
        model = result.model
        parts: list[str] = []
        for table in result.tables:
            columns = model_columns(model, table.frame, table.roles)
            missing = [part for part in REQUIRED_COLUMNS[model] if part not in columns]
            title = html.escape(table.name)
            if missing:
                parts.append(report_html.section(table.name, report_html.note(
                    _("Not analysed: this model needs {needs}.").format(needs=_(self.MODEL_NEEDS[model]))
                )))
                continue
            try:
                body = {
                    "contingency": self._contingency_html,
                    "two_way": self._two_way_html,
                    "survival": self._survival_html,
                }[model](table.frame, columns, {part: table.label(column) for part, column in columns.items()})
            except NUMERICAL_FAILURES as exc:
                body = report_html.note(str(exc))
            parts.append(report_html.section(title, body))
        return report_html.document(_(self.MODEL_TITLES[model]), _("{count} series").format(count=len(result.tables)), *parts)

    @staticmethod
    def _count(value: Any) -> str:
        number = float(value)
        return f"{int(round(number)):,}" if number == round(number) else f"{number:,.4g}"

    def _contingency_html(self, frame: pd.DataFrame, columns: Mapping[str, str], labels: Mapping[str, str]) -> str:
        weights = to_numbers(frame[columns["weight"]]).to_numpy(dtype=float) if "weight" in columns else None
        table = st.contingency_table(frame[columns["rows"]], frame[columns["columns"]], weights)
        headers = [f"{labels['rows']} / {labels['columns']}", *(str(c) for c in table.columns), _("Total")]
        rows = [
            [html.escape(str(index)), *(self._count(value) for value in table.loc[index]),
             self._count(table.loc[index].sum())]
            for index in table.index
        ]
        rows.append([html.escape(_("Total")), *(self._count(table[c].sum()) for c in table.columns),
                     self._count(table.to_numpy().sum())])
        tests = [
            _html_test_row(labels["rows"] + " × " + labels["columns"], test, self._format_number, self._format_pvalue)
            for test in st.contingency_tests(table.to_numpy())
        ]
        out = self._table(headers, rows) + self._table(["Variables", "Test", "n", "Statistic", "p-value", "Note"], tests)
        measures = st.two_by_two_measures(table.to_numpy())
        if measures:
            out += report_html.section(
                _("Effect measures"),
                self._table(
                    ["Measure", "Estimate", "95% CI low", "95% CI high", "Note"],
                    [[html.escape(str(m["measure"])), self._format_number(m["estimate"]), self._format_number(m["low"]),
                      self._format_number(m["high"]), html.escape(m["note"])] for m in measures],
                ),
                report_html.note(_("Rows are the groups and the first column the outcome: the risk of a row is its "
                                   "first column's share.")),
            )
        if "strata" in columns:
            out += self._stratified_html(frame, columns, weights)
        return out

    def _stratified_html(self, frame: pd.DataFrame, columns: Mapping[str, str], weights: Any) -> str:
        """The Mantel-Haenszel analysis of the 2x2 table over the strata column's values."""
        try:
            result = st.stratified_2x2(frame[columns["rows"]], frame[columns["columns"]], frame[columns["strata"]], weights)
        except ValueError as exc:
            return report_html.section(_("Mantel-Haenszel over strata"), report_html.note(str(exc)))
        summary = [
            [_("Common odds ratio (Mantel-Haenszel)"), self._format_number(result["odds_ratio"]),
             f"{self._format_number(result['low'])} .. {self._format_number(result['high'])}", ""],
            [_("Cochran-Mantel-Haenszel test (odds ratio = 1)"), self._format_number(result["cmh"]), "",
             self._format_pvalue(result["cmh_p"])],
            [_("Breslow-Day test (same odds ratio in every stratum)"), self._format_number(result["breslow_day"]), "",
             self._format_pvalue(result["breslow_day_p"])],
        ]
        detail = [
            [html.escape(str(item["stratum"])), *(self._count(v) for v in item["table"].ravel()),
             self._format_number(item["odds_ratio"])]
            for item in result["strata"]
        ]
        r_levels, c_levels = result["row_levels"], result["column_levels"]
        cells = [f"{r}/{c}" for r in r_levels for c in c_levels]
        return report_html.section(
            _("Mantel-Haenszel over strata"),
            self._table(["", "Statistic", "95% CI", "p-value"], summary),
            self._table([html.escape(str(columns["strata"])), *(html.escape(str(c)) for c in cells), "Odds ratio"], detail),
            report_html.note(html.escape(result["note"]) + ". " + _("When the Breslow-Day test rejects, one pooled odds "
                                                                    "ratio hides a difference between the strata.")),
        )

    def _two_way_html(self, frame: pd.DataFrame, columns: Mapping[str, str], labels: Mapping[str, str]) -> str:
        first, second, response = frame[columns["first"]], frame[columns["second"]], to_numbers(frame[columns["response"]])
        anova = st.two_way_anova(first, second, response)
        names = {"A": labels["first"], "B": labels["second"], "A × B": f"{labels['first']} × {labels['second']}"}
        rows = [
            [html.escape(names.get(row["source"], _(row["source"]))), self._format_number(row["ss"]), str(row["df"]),
             self._format_number(row["ms"]), self._format_number(row["F"]), self._format_pvalue(row["pvalue"]),
             self._format_number(row["partial_eta2"])]
            for row in anova["table"]
        ]
        means = pd.DataFrame({"a": first.astype(str), "b": second.astype(str), "y": response}).dropna()
        cells = means.groupby(["a", "b"])["y"].agg(["count", "mean", "std"]).reset_index()
        cell_rows = [
            [html.escape(str(row.a)), html.escape(str(row.b)), str(int(row["count"])),
             self._format_number(row["mean"]), self._format_number(row["std"])]
            for _index, row in cells.iterrows()
        ]
        return (
            self._table(["Source", "Sum of squares", "df", "Mean square", "F", "p-value", "Partial eta²"], rows)
            + report_html.note(f"n = {anova['n']}; {anova['note']}.")
            + self._table([labels["first"], labels["second"], "n", "Mean", "Std"], cell_rows)
        )

    def _survival_html(self, frame: pd.DataFrame, columns: Mapping[str, str], labels: Mapping[str, str]) -> str:
        del labels  # the groups are named by their values
        groups = frame[columns["group"]] if "group" in columns else None
        summary = st.survival_summary(
            to_numbers(frame[columns["time"]]), to_numbers(frame[columns["event"]]), groups
        )
        rows = [
            [html.escape(row["group"] or _("All")), str(row["n"]), str(row["events"]), str(row["censored"]),
             self._format_number(row["median"]) if np.isfinite(row["median"]) else _("not reached"),
             f"{self._format_number(row['median_low'])} .. "
             + (self._format_number(row["median_high"]) if np.isfinite(row["median_high"]) else "∞")]
            for row in summary["groups"]
        ]
        html_out = self._table(["Group", "n", "Events", "Censored", "Median survival", "95% CI median"], rows)
        if summary["logrank"] is not None:
            html_out += self._table(
                ["Groups", "Test", "n", "Statistic", "p-value", "Note"],
                [_html_test_row(_("All groups"), summary["logrank"], self._format_number, self._format_pvalue)],
            )
        return html_out

    def _run_statistics(self, *, verb: str) -> bool:
        try:
            if self.reads_table():
                self._prepare_table_source()
            results = list(self.compute_results())
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False
            self.store_cached_results(results)
            formatted = self.format_results(results)
            self.publish_results(formatted)

            if self.chart_enabled("create_chart"):
                # On Preview as well, like every other operation dialog: a
                # preview you cannot see is not a preview.  Closing without
                # applying removes it again - see discard_operation_artifacts.
                self._create_model_axis(results[0])
                self.applied.emit()

            if verb == "Applied":
                self._applied = True
                self.results_published.emit(self.results_report_html(formatted, results))
                # A report and nothing written: kept in the history all the
                # same, with the series and settings it came from (todo R-03).
                self.record_applied([], formatted or "")
                self.applied.emit()
            return True
        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.error("Statistics failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # The chart the numbers describe
    # ------------------------------------------------------------------
    #: Model -> the renderer that shows what that model measures.  A test
    #: reports a number; the chart beside it is what makes the number
    #: believable, and which chart depends on what was tested.
    #: Rows of the distribution ranking to print per series.  Ten is enough
    #: to see whether the winner is clear or the top few are indistinguishable,
    #: which is the question a ranking answers.
    DISTRIBUTION_ROWS: int = 10

    #: Which charts each model offers, and what its axis is titled.  Not one
    #: chart per model but a list, because more than one picture answers the
    #: same question and which is clearer depends on the data: a histogram
    #: shows where the mass is, an ECDF shows the fit without a bin width to
    #: argue about.
    #:
    #: "all" is deliberately empty.  It prints every section at once, so no
    #: single chart illustrates it - and the controls are hidden rather than
    #: disabled, since a disabled control invites the user to look for what
    #: would enable it.
    CHARTS_BY_MODEL: Mapping[str, tuple[tuple[str, str], ...]] = {
        "all": (),
        "descriptive": (
            ("Histogram", "Descriptive statistics"),
            ("ECDF", "Descriptive statistics"),
        ),
        "one_sample": (
            ("Histogram", "One-sample tests"),
            ("ECDF", "One-sample tests"),
        ),
        # Normality is a claim about shape, so the picture is the sample
        # against the curve it is being tested for.
        "normality": (
            ("Histogram", "Normality"),
            ("ECDF", "Normality"),
            ("Q-Q Plot", "Normality"),
            ("P-P Plot", "Normality"),
        ),
        # Both of these are about two samples together, so the picture is one
        # plotted against the other - every pair of them.
        "paired": (("Scatter Plot", "Paired samples"),),
        # Groups side by side: a box per series is what the tests compare.
        "independent": (("Box Plot", "Independent samples"),),
        "groups": (("Box Plot", "Group comparison"),),
        "correlation": (("Scatter Plot", "Correlation"),),
        "distribution": (("Histogram", "Distribution fit"),),
        "contingency": (("Mosaic Plot", "Contingency table"),),
        "two_way": (("Interaction Plot", "Two-way ANOVA"),),
        "survival": (("Kaplan-Meier", "Survival"),),
    }

    #: Which distributions the chart may be asked to draw, per model.  The
    #: normality model offers exactly the two families its tests are about; the
    #: distribution model offers the ranked sweep and every curated name.
    DISTRIBUTIONS_BY_MODEL: Mapping[str, tuple[tuple[str, str], ...]] = {
        "normality": (("norm", "Normal"), ("lognorm", "Lognormal")),
        "distribution": (
            ("best", "Best fit"),
            ("top3", "Top 3 fits"),
            ("top5", "Top 5 fits"),
            *((name, name) for name in CURATED_DISTRIBUTIONS),
        ),
    }
    #: Everything else that can draw a fitted curve gets the ranked choices.
    DEFAULT_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
        ("", "None"),
        ("best", "Best fit"),
        ("top3", "Top 3 fits"),
    )

    #: The settings - the reference value and the tests' options, the distribution the chart draws and how the
    #: candidates are ranked - each shown for the models that use it; then the Charts tab: whether OK also draws
    #: a chart, and which. Off by default: the dialog is a report, and a chart appearing unasked in the figure is a
    #: side effect nobody requested. The charts and distributions on offer follow the model (_sync_model_controls).
    PARAMS = (
        FloatParam("popmean", "Reference value:", default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=8,
                   step=0.001, tooltip="Reference value for one-sample location tests."),
        ChoiceParam("alternative", "Alternative:", choices=(("Two-sided", "two-sided"), ("Greater", "greater"),
                                                            ("Less", "less")),
                    tooltip="Alternative hypothesis for supported tests."),
        IntParam("trim_percent", "Trimmed mean:", default_value=10, minimum=0, maximum=40, suffix=" %",
                 tooltip="Trim percentage used for trimmed mean."),
        ChoiceParam("anderson_method", "Anderson method:",
                    choices=(("Anderson p-value: interpolated", "interpolate"),
                             ("Anderson p-value: Monte Carlo", "monte_carlo")),
                    tooltip="Choose how Anderson-Darling p-values are calculated when supported by SciPy."),
        IntParam("monte_carlo_resamples", "MC resamples:", default_value=9999, minimum=100, maximum=1_000_000,
                 step=1000, tooltip="Number of Monte Carlo resamples for Anderson-Darling p-values."),
        IntParam("monte_carlo_batch", "MC batch:", default_value=0, minimum=0, maximum=1_000_000, step=1000,
                 special_text="Auto", tooltip="Batch size for Monte Carlo resampling. Auto lets SciPy choose."),
        # Ranking and the exhaustive sweep only exist to order the candidates: the one model that ranks them.
        ChoiceParam("rank_by", "Rank by:", choices=(("AIC (penalises free parameters)", "aic"),
                                                    ("BIC (penalises them harder)", "bic"),
                                                    ("KS statistic (closest curve)", "ks")),
                    visible_for={"model": ("distribution",)},
                    tooltip="How the candidates are ordered. AIC is the default because the KS statistic alone "
                            "always favours the distributions with the most free parameters, whatever the data "
                            "really is."),
        # Which distribution the report ranks first and the histogram draws: only where a curve is drawn.
        ChoiceParam("distribution", "Distribution:",
                    visible_for={"model": _models_with(CHARTS_BY_MODEL, DISTRIBUTIONS_BY_MODEL)},
                    tooltip="Which fitted distribution the histogram draws. Best fit follows whichever candidate "
                            "ranks first for the series."),
        BoolParam("exhaustive", "Try every SciPy distribution", visible_for={"model": ("distribution",)},
                  tooltip="Fit all ~100 continuous distributions instead of the 15 common ones. Considerably "
                          "slower - seconds per refresh - and many of the extra candidates fit no real "
                          "measurement."),
        BoolParam("create_chart", "Add a chart", tab="charts",
                  visible_for={"model": _models_with(CHARTS_BY_MODEL)},
                  tooltip="Add an axis to the current figure showing the analysed series. "
                          "Preview draws it; closing without applying removes it again."),
        ChoiceParam("chart_type", "Chart:", tab="charts",
                    visible_for={"model": _models_with(CHARTS_BY_MODEL), "create_chart": (True,)},
                    tooltip="Which chart to add. 'Match the model' picks the one that shows what the "
                            "selected statistics measure."),
    )

    def _axis_options(
        self, chart_type: str, result: SeriesStatsResult
    ) -> dict[str, Any]:
        """Return the axis options for the chart this run draws.

        The fitted curve is the point of charting most of these models, so the
        distribution the dialog resolved is passed to whichever renderer can
        draw it - the histogram over its bars, the ECDF over its steps.  Both
        read it through the same module the report ranks with, so the curve
        cannot contradict the table beside it.
        """
        options: dict[str, Any] = {"grid": True}
        if chart_type in ("Q-Q Plot", "P-P Plot"):
            # The family the normality tests are about; Q-Q's own default
            # otherwise.
            if result.distribution in ("norm", "lognorm"):
                options["distribution"] = result.distribution
            return options
        if chart_type not in ("Histogram", "ECDF") or not result.distribution:
            return options

        options["distribution_fit"] = result.distribution
        if chart_type == "Histogram":
            # Semi-transparent and outlined, so a curve crossing the bars stays
            # readable against them.
            options["histtype"] = "stepfilled"
            options["alpha"] = 0.55
        return options

    def _create_model_axis(self, result: SeriesStatsResult) -> None:
        """Add an axis showing what the selected model measured.

        Only when the user asked for a chart - the dialog's job is to report
        numbers, and an axis appearing unrequested is a side effect.

        Created on the first Preview and reused afterwards, so changing the
        model or a parameter redraws the same axis instead of stacking a new
        one each time.  Closing without Apply deletes it again, which is what
        keeps a preview a preview: see discard_operation_artifacts.

        The axis carries the same series the statistics were computed from, so
        the numbers and the picture cannot disagree.
        """
        charts = self.CHARTS_BY_MODEL.get(result.model, ())
        if not charts:
            # A model with nothing to draw: "all" prints every section, and no
            # one picture illustrates that.
            return

        # The chosen entry, or the model's first, which is the one the list was
        # ordered to put there.
        chosen = str(self.chart_value("chart_type") or "")
        chart_type, title = next(
            ((name, label) for name, label in charts if name == chosen), charts[0]
        )

        options = self._axis_options(chart_type, result)

        if (
            self._result_axis_id is not None
            and self._result_chart_type == chart_type
            and self._result_axis_options == options
        ):
            return

        # A different renderer needs different role columns, so the axis is
        # rebuilt rather than re-labelled.
        self._remove_result_axis()

        axis_id = self.create_result_axis(
            chart_type=chart_type,
            title=title,
            x_label="" if chart_type != "ECDF" else "value",
            y_label="",
            options=options,
        )
        self._result_axis_id = axis_id
        self._result_chart_type = chart_type
        self._result_axis_options = options
        rows = self._rows_for_statistics()
        if result.model in FRAME_MODELS:
            for row, table in zip(rows, result.tables):
                self._attach_table_series(axis_id, row, table, result.model, chart_type)
            return
        if result.model in ("paired", "correlation") and chart_type == "Scatter Plot":
            # Every pair, not every series.  These two models test one sample
            # against another, and the report already prints a row per pair -
            # so charting each series against its own x would show something
            # the statistics never looked at.
            for left, right in itertools.combinations(rows, 2):
                self._attach_series_pair(axis_id, left, right)
        else:
            for row in rows:
                self._attach_source_series(axis_id, row, chart_type)

    def _remove_result_axis(self) -> None:
        """Delete the axis this dialog added, if it added one."""
        axis_id = self._result_axis_id
        self._result_axis_id = None
        self._result_chart_type = ""
        self._result_axis_options = {}
        if axis_id is None:
            return
        try:
            self._repo.delete_axis(axis_id)
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to remove the statistics axis %s", axis_id)

    def discard_operation_artifacts(self) -> None:
        """Remove the previewed chart when Apply never happened.

        Creating an axis commits, so it is not covered by the preview
        savepoint and has to be undone by hand.
        """
        if self._applied:
            return
        self._remove_result_axis()

    def _attach_series_pair(self, axis_id: int, left: Any, right: Any) -> None:
        """Scatter one series' values against another's, as the tests read them.

        Joined on row order, which is what pairing means here and what
        ``_paired_values`` does when it computes the numbers: the nth
        observation of one series against the nth of the other.  A join on a
        shared key would be a different, defensible pairing - and a different
        statistic from the one printed above the chart, which is why it is not
        used.
        """
        left_sql = str(left["sql_query"] or "").strip()
        right_sql = str(right["sql_query"] or "").strip()
        if not left_sql or not right_sql:
            return

        left_y = str(parse_roles(row_value(left, "roles")).get("y", "y") or "y")
        right_y = str(parse_roles(row_value(right, "roles")).get("y", "y") or "y")
        left_name, right_name = str(left["name"]), str(right["name"])

        # ROW_NUMBER over the source order: SQLite has no positional join, and
        # rowid belongs to a table rather than to an arbitrary SELECT.
        sql = (
            "SELECT a.value AS x, b.value AS y FROM ("
            f"SELECT {quote_identifier(left_y)} AS value, "
            f"ROW_NUMBER() OVER () AS position FROM ({left_sql})) AS a "
            "JOIN ("
            f"SELECT {quote_identifier(right_y)} AS value, "
            f"ROW_NUMBER() OVER () AS position FROM ({right_sql})) AS b "
            "ON a.position = b.position"
        )

        self._repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=self._repo.next_series_index(axis_id),
            name=f"{left_name} vs {right_name} [statistics]",
            sql_query=sql,
            roles={"x": "x", "y": "y"},
            style={
                **dict(self.generated_style_filter),
                "label": f"{left_name} vs {right_name}",
                "marker": ".",
            },
        )

    def _attach_table_series(self, axis_id: int, row: Any, table: SeriesTable, model: str, chart_type: str) -> None:
        """Put a FRAME_MODELS series on its chart, its columns renamed to that chart's roles."""
        found = _table_series_select(model, model_columns(model, table.frame, table.roles))
        sql = str(row["sql_query"] or "").strip()
        if found is None or not sql:
            return
        select, roles = found
        self._repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=self._repo.next_series_index(axis_id),
            name=f"{table.name} [statistics]",
            sql_query=f"{select} FROM ({sql})",
            roles=roles,
            style={**dict(self.generated_style_filter), "label": table.name},
        )
        del chart_type

    def _attach_source_series(self, axis_id: int, row: Any, chart_type: str) -> None:
        """Put one analysed series onto the new axis, in that axis's roles.

        Each renderer names its columns differently - a box plot wants
        ``value``, a scatter wants x and y - so the same source SQL is aliased
        per chart type rather than copied per chart type.
        """
        roles = parse_roles(row_value(row, "roles"))
        sql = str(row["sql_query"] or "").strip()
        if not sql:
            return

        name = str(row["name"])
        y_column = str(roles.get("y", "y") or "y")
        x_column = str(roles.get("x", "x") or "x")

        if chart_type == "Scatter Plot":
            select = f"SELECT {quote_identifier(x_column)} AS x, {quote_identifier(y_column)} AS y"
            axis_roles: dict[str, Any] = {"x": "x", "y": "y"}
        elif chart_type in ("ECDF", "Q-Q Plot", "P-P Plot"):
            select = f"SELECT {quote_identifier(y_column)} AS value"
            axis_roles = {"value": "value"}
        elif chart_type == "Histogram":
            # The histogram splits on "dataset", not on "group".  Sent the
            # group column it would ignore it, pool every checked series into
            # one sample and fit a single distribution across all of them -
            # which is the one thing the ranking must not describe.
            select = (
                f"SELECT {quote_identifier(y_column)} AS value, "
                f"'{name}' AS dataset"
            )
            axis_roles = {"value": "value", "dataset": "dataset"}
        else:  # Box Plot and anything else grouped by series name
            select = (
                f"SELECT {quote_identifier(y_column)} AS value, "
                f"'{name}' AS \"group\""
            )
            axis_roles = {"value": "value", "group": "group"}

        self._repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=self._repo.next_series_index(axis_id),
            name=f"{name} [statistics]",
            sql_query=f"{select} FROM ({sql})",
            roles=axis_roles,
            style={**dict(self.generated_style_filter), "label": name, "marker": "."},
        )


def _table_series_select(model: str, columns: Mapping[str, str]) -> tuple[str, dict[str, str]] | None:
    """The SELECT list and roles that put a FRAME_MODELS series on its chart."""
    renames = {
        "contingency": {"rows": "x", "columns": "y", "weight": "weight"},
        "two_way": {"first": "x", "second": "trace", "response": "y"},
        "survival": {"time": "time", "event": "event", "group": "group"},
    }[model]
    if any(part not in columns for part in REQUIRED_COLUMNS[model]):
        return None
    pairs = [(columns[part], role) for part, role in renames.items() if part in columns]
    select = ", ".join(f"{quote_identifier(column)} AS {quote_identifier(role)}" for column, role in pairs)
    return f"SELECT {select}", {role: role for _column, role in pairs}


def _html_test_row(
    label: str,
    test: Mapping[str, Any],
    format_number,
    format_pvalue,
) -> list[str]:
    return [
        html.escape(str(label)),
        html.escape(str(test.get("test", ""))),
        str(test.get("n", "")),
        format_number(test.get("statistic")),
        format_pvalue(test.get("pvalue")),
        html.escape(str(test.get("note", ""))),
    ]
