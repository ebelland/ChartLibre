"""Accessory charts (residuals, measured vs. fit) and the parameter table's menu."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd
from PySide6.QtCore import QPoint
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QMenu, QTableWidgetItem

from app.logs.logger import applogger
from app.utils.messages import show_message

from app.utils.i18n import _

from app.data_operations.fit_dialog.common import SeriesFitResult


class AccessoryChartsMixin:
    """Accessory charts (residuals, measured vs. fit) and the parameter table's menu. A part of SeriesFitDialog (see dialog.py)."""

    if TYPE_CHECKING:  # the dialog's state and the base dialog's
        _accessory_table_names: set[str]

        def result_table_name(self, axis_id: int, result: Any) -> str: ...


    # ------------------------------------------------------------------
    # Optional accessory charts: residuals, and measured vs. fit
    # ------------------------------------------------------------------
    #
    # Both read straight off the output table _build_output_frame already
    # writes - x/residual always, plus y/y_fit (1D) or z/z_fit (2D) - so no
    # extra computation happens here, only a second axis and a query onto the
    # same table the overlay series already reads from.
    #
    # Each checkbox owns one axis, created on the first Preview or Apply that
    # has it checked and reused afterwards, the same way the spectral and
    # statistics dialogs reuse their own single result axis: an axis outside
    # the preview savepoint (creating one commits) would otherwise stack up
    # one per Preview click. Unchecking removes it immediately rather than
    # waiting for Close, since a chart nobody asked for any more should not
    # linger just because the dialog is still open.

    def _residual_axis_query(
        self, result: SeriesFitResult, table_name: str
    ) -> tuple[str, dict[str, str], str, str]:
        return (
            # Measured rows only: the curve's own rows (Fitted curve > Points) have no residual.
            f'SELECT x, residual AS y FROM "{table_name}" WHERE residual IS NOT NULL ORDER BY x',
            {"x": "x", "y": "y"},
            result.x_col or "x",
            _("residual"),
        )

    def _fit_vs_measured_axis_query(
        self, result: SeriesFitResult, table_name: str
    ) -> tuple[str, dict[str, str], str, str]:
        fit_col, measured_col = ("z_fit", "z") if self._is_2d_fit() else ("y_fit", "y")
        return (
            f'SELECT {fit_col} AS x, {measured_col} AS y FROM "{table_name}" WHERE {measured_col} IS NOT NULL',
            {"x": "x", "y": "y"},
            _("fit"),
            result.target_col or _("measured"),
        )

    def _sync_accessory_axis(
        self,
        *,
        wanted: bool,
        axis_id_attr: str,
        title: str,
        query_builder: Callable[[SeriesFitResult, str], tuple[str, dict[str, str], str, str]],
        results: Sequence[SeriesFitResult],
    ) -> None:
        if not wanted:
            self._remove_accessory_axis(axis_id_attr)
            return
        first = results[0]
        first_table = self.result_table_name(0, first)
        _query, _roles, x_label, y_label = query_builder(first, first_table)
        axis_id = getattr(self, axis_id_attr)
        if axis_id is not None and not self._axis_exists(int(axis_id)):
            # Gone with the figure or the Preview it was made in: a series added to it was "FOREIGN KEY
            # constraint failed", and nothing drawn.
            axis_id = None
        if axis_id is None:
            # Beside the curve: on the new figure when it went to one.
            axis_id = self.create_result_axis(
                chart_type="Scatter Plot", title=title, x_label=x_label,
                y_label=y_label, options={"grid": True}, figure_id=getattr(self, "_result_figure_id", None),
            )
            setattr(self, axis_id_attr, axis_id)
        for row in self._repo.get_series(axis_id) or []:
            self._repo.delete_series(int(row["id"]))
        for result in results:
            table_name = self.result_table_name(0, result)
            sql_query, roles, _x_label, _y_label = query_builder(result, table_name)
            self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=f"{title}: {result.source_table}",
                sql_query=sql_query, roles=roles,
                style={"marker": "o", "linestyle": "",
                       "source_series": result.source_table,
                       "generated_fit": True, "fit_dialog": "series_fit"},
            )
    def _remove_accessory_axis(self, axis_id_attr: str) -> None:
        axis_id = getattr(self, axis_id_attr, None)
        if axis_id is None:
            return
        setattr(self, axis_id_attr, None)
        try:
            self._repo.delete_axis(int(axis_id))
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to remove accessory axis (%s)", axis_id_attr)

    def _sync_accessory_charts(self, results: Sequence[SeriesFitResult]) -> None:
        if not results:
            return
        if not (self.chart_enabled("residual_chart") or self.chart_enabled("fit_vs_measured_chart")):
            self._remove_accessory_axis("_residual_axis_id")
            self._remove_accessory_axis("_fit_vs_measured_axis_id")
            return
        for result in results:
            table_name = self.result_table_name(0, result)
            self.write_result_table(table_name, result)
            self._accessory_table_names.add(table_name)
        self._sync_accessory_axis(
            wanted=self.chart_enabled("residual_chart"), axis_id_attr="_residual_axis_id",
            title=_("Residuals"), query_builder=self._residual_axis_query, results=results,
        )
        self._sync_accessory_axis(
            wanted=self.chart_enabled("fit_vs_measured_chart"), axis_id_attr="_fit_vs_measured_axis_id",
            title=_("Measured vs. fit"), query_builder=self._fit_vs_measured_axis_query, results=results,
        )
    def resolve_target_axis_id(
        self, selected_axis_id: int, results: Sequence[SeriesFitResult]
    ) -> int:
        """Keep the fit on its source axis, and sync the accessory axes here.

        The fit overlay itself belongs on the axis its data came from, so the
        selected axis is returned unchanged. The accessory axes are built in
        this hook - and *only* here - because ``_run_operation`` calls it
        before ``begin_preview_transaction()``, which is the one point where
        an axis can be created safely.

        Creating them inside the savepoint instead destroyed data: writing a
        result table goes through pandas' to_sql, which commits and so
        invalidates the savepoint, after which every later repository write
        has its own commit suppressed (see SqliteRepo._commit) and sits in an
        implicit transaction that the Close/Cancel cleanup then discards -
        taking the user's own series on the other axes with it. The spectral
        and statistics dialogs create their result axis in this same hook for
        exactly this reason.
        """
        # "Draw on": the source axis (the default), a new axis, or a new figure - the base's choice.
        target = super().resolve_target_axis_id(selected_axis_id, results)  # type: ignore[misc]
        if results:
            self._sync_accessory_charts(results)
        return target

    def apply_results_to_axis(self, axis_id: int, results: Sequence[SeriesFitResult]) -> None:
        super().apply_results_to_axis(axis_id, results)
        self._applied = True

    def discard_operation_artifacts(self) -> None:
        """Remove any accessory axis this dialog added, when Apply never ran.

        Creating an axis commits, so it is not covered by the preview
        savepoint and has to be undone by hand - see resolve_target_axis_id.
        """
        if self._applied:
            return
        self._remove_accessory_axis("_residual_axis_id")
        self._remove_accessory_axis("_fit_vs_measured_axis_id")

        table_names = set(self._accessory_table_names)
        self._accessory_table_names.clear()
        # And any axis that reads only those tables, its id lost or not: left behind, it was an empty
        # Residuals and an empty Measured vs. fit after Cancel.
        self._remove_axes_reading(table_names)
        super().discard_operation_artifacts()  # type: ignore[misc] - the new axis or figure "Draw on" made
        for table_name in table_names:
            try:
                self._repo.delete_table(table_name)
            except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
                applogger.exception(
                    "Failed to remove the accessory result table %s", table_name
                )

    def _axis_exists(self, axis_id: int) -> bool:
        figures = {self._figure_id, getattr(self, "_result_figure_id", None)} - {None}
        for figure_id in figures:
            descriptor = self._repo.load_figure_descriptor(int(figure_id))
            if descriptor is not None and any(int(axis.id) == axis_id for axis in descriptor.axes):
                return True
        return False

    def _remove_axes_reading(self, table_names: set[str]) -> None:
        """Delete the figure's axes whose every series reads one of *table_names*, then fit its grid to the rest."""
        if not table_names:
            return
        try:
            descriptor = self._repo.load_figure_descriptor(self._figure_id)
            doomed = [axis for axis in (descriptor.axes if descriptor else [])
                      if axis.series and all(any(f'"{name}"' in str(series.sql_query or "") for name in table_names)
                                             for series in axis.series)]
            for axis in doomed:
                self._repo.delete_axis(int(axis.id))
            if doomed:
                left = len(descriptor.axes) - len(doomed)
                columns = max(1, min(int(descriptor.ncols or 1), left))
                self._repo.set_figure_grid(self._figure_id, nrows=max(1, -(-left // columns)), ncols=columns)
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to remove the accessory axes left on the figure")

    def on_use_fit_results_as_initial(self) -> None:
        """Copy latest optimized fit parameters back to the Initial column."""
        if self._last_result is None:
            show_message(self, "series.no_fit_results")
            return
        # Every row of the table: in a global fit, more than one series' worth.
        params = np.asarray(getattr(self, "_last_table_params", self._last_result.params), dtype=float)
        if params.size == 0:
            show_message(self, "series.no_parameters")
            return
        self._spin_params.blockSignals(True)
        try:
            self._spin_params.setValue(int(params.size))
            self._ensure_params_rows(int(params.size))
        finally:
            self._spin_params.blockSignals(False)
        for row, value in enumerate(params):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))
            if self._params_table.item(row, 2) is None:
                self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            if self._params_table.item(row, 3) is None:
                self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            self._get_fix_checkbox(row).setChecked(False)
        applogger.info("Initial parameters updated from latest fit results.")

    def _cell_text(self, row: int, col: int, default: str = "") -> str:
        item = self._params_table.item(row, col)
        if item is None:
            return default
        return item.text()

    def _params_as_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in range(self._params_table.rowCount()):
            fix_widget = self._params_table.cellWidget(row, 4)
            fixed = bool(fix_widget.isChecked()) if isinstance(fix_widget, QCheckBox) else False
            rows.append(
                {
                    "parameter": self._cell_text(row, 0, f"p[{row}]"),
                    "initial": self._cell_text(row, 1),
                    "lower": self._cell_text(row, 2),
                    "upper": self._cell_text(row, 3),
                    "fixed": fixed,
                }
            )
        return pd.DataFrame(rows)

    def _copy_params_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self._params_as_frame().to_csv(index=False, sep="\t"))
        applogger.info("Parameter table copied to clipboard.")

    def _save_params_as_csv(self) -> None:
        default_name = f"{self._model_name().replace(' ', '_').lower()}_parameters.csv"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            _("Save parameters as CSV"),
            default_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self._params_as_frame().to_csv(path, index=False)
        applogger.info("Parameter table saved to %s", path)

    def _show_params_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self._params_table)
        copy_action = QAction(_("Copy parameters"), menu)
        copy_action.triggered.connect(self._copy_params_to_clipboard)
        menu.addAction(copy_action)

        use_fit_action = QAction(_("Use latest fitted parameters"), menu)
        use_fit_action.setEnabled(self._last_result is not None)
        use_fit_action.triggered.connect(self.on_use_fit_results_as_initial)
        menu.addAction(use_fit_action)

        reset_action = QAction(_("Reset parameters"), menu)
        reset_action.triggered.connect(self.on_reset_params)
        menu.addAction(reset_action)

        menu.addSeparator()
        save_csv_action = QAction(_("Save as CSV"), menu)
        save_csv_action.triggered.connect(self._save_params_as_csv)
        menu.addAction(save_csv_action)

        menu.exec(self._params_table.viewport().mapToGlobal(pos))

    def on_reset_params(self) -> None:
        self._populate_params_defaults()
        applogger.info("Parameter table reset to model defaults.")
