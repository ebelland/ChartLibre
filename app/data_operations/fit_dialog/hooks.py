"""What the fit gives DataOperationDialogBase: results, result tables and series."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from app.plugin_api.data_operations.dialog_base import ResultSeriesSpec
from app.utils import report_html
from app.utils.i18n import _

from app.data_operations.fit_dialog.common import SeriesFitResult


class FitHooksMixin:
    """What the fit gives DataOperationDialogBase: results, result tables and series. A part of SeriesFitDialog (see dialog.py)."""

    if TYPE_CHECKING:  # StartingPointMixin's
        def _report_sections(self, result: SeriesFitResult) -> list[str]: ...
        def _results_html(self, result: SeriesFitResult) -> str: ...


    # ------------------------------------------------------------------
    # DataOperationDialogBase hooks
    # ------------------------------------------------------------------

    def compute_results(self) -> Sequence[SeriesFitResult]:
        """Evaluate the model at the parameters currently in the table.

        Preview and Apply both come through here, and neither optimises: that
        is what the Fit button is for.  Preview used to re-fit, which meant a
        hand-edited starting guess could never be seen - the optimiser
        overwrote it before anything was drawn.
        """
        self._last_result = None
        return super().compute_results()

    def result_table_name(self, axis_id: int, result: SeriesFitResult) -> str:
        del axis_id
        return result.output_table

    def result_series_spec(self, axis_id: int, table_name: str, result: SeriesFitResult) -> ResultSeriesSpec:
        del axis_id
        name = f"Fit: {result.source_table} [{result.model_name}]"

        if result.fit_mode == "2D":
            # Drawn with plot_trisurf (see TriSurfaceAxisRenderer), not
            # pivoted onto a rectangular grid: the source points a surface
            # fit runs on are rarely a complete x/y grid (that is exactly
            # what series_grid_xyz's own fallback to interpolation is for
            # elsewhere in this task), and triangulating the fitted z values
            # directly draws a continuous skin without inventing a grid the
            # data never supported.
            sql_query = f'SELECT x, y, z_fit AS z FROM "{table_name}"'
            roles = {"x": "x", "y": "y", "z": "z"}
            style = {
                "source_series": result.source_table,
                "source_x_col": result.x_col,
                "source_x2_col": result.x2_col or "",
                "source_z_col": result.target_col,
                "fit_model": result.model_name,
                "fit_mode": "2D",
                "generated_fit": True,
                "fit_dialog": "series_fit",
            }
        else:
            sql_query = f'SELECT x, y_fit AS y FROM "{table_name}" ORDER BY x'
            roles = {"x": "x", "y": "y"}
            style = {
                "linestyle": "--",
                "linewidth": 2.0,
                "marker": "",
                "source_series": result.source_table,
                "source_x_col": result.x_col,
                "source_y_col": result.target_col,
                "fit_model": result.model_name,
                "fit_mode": "1D",
                "generated_fit": True,
                "fit_dialog": "series_fit",
            }

        return ResultSeriesSpec(
            name=name,
            sql_query=sql_query,
            roles=roles,
            style=style,
        )

    def result_series_specs(
        self, axis_id: int, table_name: str, result: SeriesFitResult
    ) -> Sequence[ResultSeriesSpec]:
        """The fitted curve, plus the bands whose boxes are checked (1D only)."""
        specs = [self.result_series_spec(axis_id, table_name, result)]
        if result.fit_mode == "2D" or "ci_low" not in result.frame:
            return specs
        for check, kind, label, alpha in (
            (self._prediction_band_check, "pi", "95% prediction band", 0.12),
            (self._confidence_band_check, "ci", "95% confidence band", 0.25),
        ):
            if not check.isChecked():
                continue
            specs.append(
                ResultSeriesSpec(
                    name=f"{label}: {result.source_table} [{result.model_name}]",
                    # Aliased to x/y/y2 rather than mapped through roles:
                    # the axis's own chart type (a scatter) keeps only the
                    # roles it knows, and y2 is not one of them.
                    sql_query=(
                        f'SELECT x, {kind}_low AS y, {kind}_high AS y2 '
                        f'FROM "{table_name}" ORDER BY x'
                    ),
                    roles={"x": "x", "y": "y", "y2": "y2"},
                    style={
                        "draw_as": "band",
                        "alpha": alpha,
                        "label": label,
                        "generated_fit": True,
                        "fit_dialog": "series_fit",
                        "fit_model": result.model_name,
                    },
                )
            )
        return specs

    def format_results(self, results: Sequence[SeriesFitResult]) -> str:
        """Return the fit report as an HTML table."""
        if not results:
            return "<p>Run Fit, or press Preview to draw the current parameters.</p>"
        if len(results) == 1:
            return self._results_html(results[0])
        # One document for all the series, each under its own heading: a
        # report per series pasted one after the other nested one <html> and
        # <body> in the next.
        sections: list[str] = []
        for result in results:
            sections.append(report_html.section(f"{_('Series')}: {result.source_table}"))
            sections.extend(self._report_sections(result))
        return report_html.document(_("Fit"), results[0].model_name, *sections)
