"""Where results are drawn: result tables, series, axes and figures; preview artefacts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import numpy as np

from app.logs.logger import applogger
from app.utils.coercion import to_numeric_axis
from app.plugin_api.data_operations.results import OperationResult
from app.plugin_api.data_operations.parameter_spec import ChoiceParam, Param

from app.plugin_api.data_operations.dialog_base.common import generated_table_name


class ResultsMixin:
    """Where results are drawn: result tables, series, axes and figures; preview artefacts. A part of DataOperationDialogBase (see dialog.py)."""

    # ------------------------------------------------------------------
    # Where results are drawn
    # ------------------------------------------------------------------

    # The three places a generated result can go (DEST_*) are OperationShell's: the same words in every
    # analysis window.

    @classmethod
    def destination_param(
        cls,
        *,
        name: str = "destination",
        label: str = "Draw on:",
        tooltip: str = "",
        default: str | None = None,
    ) -> Param:
        """Return the declared destination choice, worded once.

        Operations differ in *why* they default one way or another - a
        derivative needs its own scale, a drawn function usually belongs
        beside the data it is being compared with - so the default and the
        tooltip are arguments, while the options themselves are not.
        """
        return ChoiceParam(
            name,
            label,
            tooltip=tooltip,
            choices=(
                ("New axis in this figure", cls.DEST_NEW_AXIS),
                ("Same axis as the source", cls.DEST_SAME_AXIS),
                ("New figure", cls.DEST_NEW_FIGURE),
            ),
            default_value=default or cls.DEST_NEW_AXIS,
        )

    def resolve_destination_axis(
        self,
        selected_axis_id: int,
        *,
        chart_type: str,
        title: str = "",
        figure_name: str = "",
        options: Mapping[str, Any] | None = None,
        parameter: str = "destination",
    ) -> int:
        """Return the axis to draw on, honouring the declared destination.

        Creates the axis or figure once and reuses it, so adjusting a
        parameter repeatedly does not leave a trail of empty axes behind. The
        ids are remembered on the dialog so ``discard_result_target`` can undo
        them if Apply never happens.
        """
        destination = str(self.parameter_values().get(parameter, self.DEST_NEW_AXIS))
        if destination == self.DEST_SAME_AXIS:
            return selected_axis_id

        if getattr(self, "_result_axis_id", None) is None:
            if destination == self.DEST_NEW_FIGURE:
                self._result_figure_id, self._result_axis_id = self.create_result_figure(
                    name=figure_name or title or self.operation_label,
                    chart_type=chart_type,
                    title=title,
                    options=options,
                )
            else:
                self._result_figure_id = None
                self._result_axis_id = self.create_result_axis(
                    chart_type=chart_type,
                    title=title,
                    options=options,
                )

        # Falls back to the selected axis when creation failed: an
        # operation that cannot make its own axis should still draw
        # somewhere rather than raise on the way to the chart.
        return (
            int(self._result_axis_id)
            if self._result_axis_id is not None
            else selected_axis_id
        )

    def _forget_rolled_back_target(self) -> None:
        """Drop the remembered result axis or figure if the rollback removed it.

        resolve_destination_axis makes the target once and reuses it. Made
        during a Preview, it is inside the preview's savepoint, and undoing
        the Preview undoes it too - the next Preview then drew on an axis
        that no longer existed ("FOREIGN KEY constraint failed").
        """
        axis_id = getattr(self, "_result_axis_id", None)
        figure_id = getattr(self, "_result_figure_id", None)
        if axis_id is None and figure_id is None:
            return
        try:
            if figure_id is not None and self._repo.get_figure_descriptor(int(figure_id)) is None:
                self._result_axis_id = self._result_figure_id = None
                return
            descriptor = self._repo.load_figure_descriptor(
                int(figure_id) if figure_id is not None else self._figure_id
            )
            axis_ids = {int(axis.id) for axis in (descriptor.axes if descriptor else [])}
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Could not check the result target after a rollback")
            axis_ids = set()
        if axis_id is not None and int(axis_id) not in axis_ids:
            self._result_axis_id = None
            self._result_figure_id = None

    def discard_result_target(self) -> None:
        """Remove an axis or figure this dialog made, when Apply never ran.

        A whole figure when that is what was made: deleting only its axis
        would leave an empty chart tab behind, which is worse than the axis it
        was meant to clean up.
        """
        if getattr(self, "_applied", False):
            return

        axis_id = getattr(self, "_result_axis_id", None)
        figure_id = getattr(self, "_result_figure_id", None)
        if axis_id is None and figure_id is None:
            return

        self._result_axis_id = None
        self._result_figure_id = None
        try:
            if figure_id is not None:
                self._repo.delete_figure(int(figure_id))
                applogger.info("Discarded the unapplied figure %s.", figure_id)
            elif axis_id is not None:
                self._repo.delete_axis(int(axis_id))
                applogger.info("Discarded the unapplied axis %s.", axis_id)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to discard the unapplied result target")

    def create_result_figure(
        self,
        *,
        name: str,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Create a figure with one axis and return ``(figure_id, axis_id)``.

        For results that do not belong in the source chart at all.  A new axis
        (``create_result_axis``) keeps the result beside its input, which is
        right when the two are meant to be compared; a new figure is right when
        they are not - a derivative kept for its own sake, or a result destined
        for a different report.

        The dialog does NOT follow the new figure: ``self._figure_id`` stays
        put, because the selector is locked to it and the axis combo would
        otherwise start listing axes of a figure the user cannot see. The main
        window picks the new figure up when it reloads its tabs.
        """
        figure_id = int(self._repo.create_figure_descriptor(name=str(name), nrows=1, ncols=1))
        axis_id = int(
            self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=0,
                chart_type=chart_type,
                title=title,
                x_label=x_label,
                y_label=y_label,
                options=dict(options or {"grid": True}),
            )
        )
        applogger.info(
            "%s: created figure %s with axis %s (%s).",
            self.operation_label,
            figure_id,
            axis_id,
            chart_type,
        )
        return figure_id, axis_id

    def discard_operation_artifacts(self) -> None:
        """Undo anything the operation created outside the preview savepoint.

        Called from Close/Cancel.  The savepoint covers descriptor rows, but
        writing a result table commits (pandas' to_sql does), so a dialog that
        creates its own figure has to remove it here or an abandoned chart is
        left behind.

        By default, the axis or figure a result was drawn on when this dialog
        made it (see discard_result_target); nothing when Apply ran.
        """
        self.discard_result_target()

    def apply_results_to_axis(self, axis_id: int, results: Sequence[Any]) -> None:
        """Write every result as a table plus a chart series on *axis_id*.

        Every result, not the first: this runs inside the savepoint opened by
        ``_run_operation``, so anything raised here discards the whole batch.
        It used to end with ``optimize_db()``, whose VACUUM cannot run inside a
        transaction - the OperationalError aborted the operation part-way and
        left only the results already flushed to disk.  The caller optimizes
        after committing instead.
        """
        self.remove_previous_generated_series(axis_id)
        for result in results:
            if isinstance(result, OperationResult):
                result.apply(self, axis_id)
                continue
            # A user-written operation whose result predates OperationResult.
            table_name = self.result_table_name(axis_id, result)
            self.write_result_table(table_name, result)
            applogger.info(f"Saved result table: {table_name}")
            self.create_result_series(axis_id, table_name, result)

    @property
    def preview_style_filter(self) -> Mapping[str, Any]:
        return {**dict(self.generated_style_filter), "operation_preview": True, "preview_owner": self.__class__.__name__}

    def remove_previous_preview_series(self, axis_id: int) -> int:
        removed = 0
        expected = dict(self.preview_style_filter)
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except (TypeError, ValueError):
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        return removed

    def preview_table_name(self, axis_id: int, result: Any) -> str:
        """Return the table name a preview writes to.

        Derived from the final name so the two are recognisably a pair, and
        prefixed like it: a preview is even more clearly not the user's data.
        """
        base = self.result_table_name(axis_id, result)
        return generated_table_name(f"{base}_Preview", fallback="Series_Preview")

    def _delete_preview_table(self, table_name: str) -> None:
        self._repo.delete_table(str(table_name))

    def remove_preview_tables(self, table_names: Sequence[str] | None = None) -> None:
        names = {str(name) for name in (table_names or self._preview_table_names) if str(name).strip()}
        for table_name in names:
            self._delete_preview_table(table_name)
            applogger.info(f"Deleted preview table: {table_name}")
        self._preview_table_names.difference_update(names)

    def remove_preview_artifacts(self, axis_id: int | None = None) -> None:
        if axis_id is None:
            for preview_axis_id in list(self._preview_axis_ids):
                self.remove_previous_preview_series(int(preview_axis_id))
        else:
            self.remove_previous_preview_series(int(axis_id))
        self.remove_preview_tables()

    def create_preview_series(self, axis_id: int, table_name: str, result: Any) -> int | None:
        first_id: int | None = None
        for spec in self.result_series_specs(axis_id, table_name, result):
            style = dict(spec.style)
            style.update(self.preview_style_filter)
            series_id = self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=f"Preview: {spec.name}",
                sql_query=spec.sql_query,
                roles=dict(spec.roles),
                style=style,
            )
            if first_id is None and series_id is not None:
                first_id = int(series_id)
        return first_id

    def preview_results_to_axis(self, axis_id: int, results: Sequence[Any]) -> None:
        self.remove_preview_artifacts(axis_id)
        for result in results:
            if isinstance(result, OperationResult):
                result.preview(self, axis_id)
                continue
            table_name = self.preview_table_name(axis_id, result)
            self.write_result_table(table_name, result)
            self.remember_preview_table(table_name)
            self.create_preview_series(axis_id, table_name, result)

    def remember_preview_table(self, table_name: str) -> None:
        """Note a table a preview wrote, so closing the dialog removes it."""
        self._preview_table_names.add(table_name)
        # No optimize_db() here: a preview also runs inside the savepoint, and
        # VACUUM cannot run in a transaction (see apply_results_to_axis).
    # TO IMPLEMENT!!!!
    def get_data(
        self,
        sql_query: str,
        roles: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return finite X/Y values and source rowids for Hide-aware queries.

        ``query_series_frame_for_hide`` exposes normalized ``x``/``y`` columns
        plus ``__rowid__``.  Do not read ``z`` here: the outlier dialog is a
        1D X/Y operation, and normal series frames often have no Z column.
        """
        source_df = self._repo.query_series_frame_for_hide(
            sql_query=sql_query,
            roles=roles,
        )
        # Not pd.to_numeric: a timestamp x would become all-NaN and the
        # caller would see an empty series (see utils.coercion).
        raw_x = to_numeric_axis(source_df["x"])
        raw_y = to_numeric_axis(source_df["y"])
        raw_rowids = source_df["__rowid__"].to_numpy(dtype=int)

        finite_mask = np.isfinite(raw_x) & np.isfinite(raw_y)
        return raw_x[finite_mask], raw_y[finite_mask], raw_rowids[finite_mask]


    def _refresh_after_preview_state_change(self) -> None:
        """Notify the owner window that chart/data panes must refresh."""
        try:
            self.applied.emit()
            self.sync_preview_figure()
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to emit preview refresh signal.")

    def sync_preview_figure(self) -> None:
        """Tell the window about the figure a Preview made: shown when there is one, dropped when it went."""
        figure_id = getattr(self, "_result_figure_id", None)
        if figure_id is not None and self._repo.get_figure_descriptor(int(figure_id)) is None:
            figure_id = None
        shown = getattr(self, "_preview_figure_shown", None)
        if figure_id is not None and figure_id != shown:
            self._preview_figure_shown = int(figure_id)
            self.preview_drawn.emit(int(figure_id))
        elif figure_id is None and shown is not None:
            self._preview_figure_shown = None
            self.preview_drawn.emit(-1)

    def _reset_preview_state_flags(self) -> None:
        """Clear in-memory preview bookkeeping after commit or rollback."""
        self._preview_active = False
        self._preview_axis_ids.clear()
        self._preview_table_names.clear()

    def _capture_current_grid(self) -> tuple[int, int] | None:
        """Return the figure grid as it was before this dialog previews.

        Preview-only operations can temporarily create extra axes.  Some repo
        paths update the figure grid separately from the axis descriptors, so a
        rollback/cleanup must restore both pieces of state.
        """
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
            rows = int(getattr(descriptor, "nrows", 1) or 1)
            cols = int(getattr(descriptor, "ncols", 1) or 1)
            return max(1, rows), max(1, cols)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to capture original figure grid.")
            return None

    def _restore_original_grid(self) -> None:
        """Restore the figure grid captured when the dialog opened."""
        if self._original_grid is None:
            return
        rows, cols = self._original_grid
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
            current_rows = int(getattr(descriptor, "nrows", 1) or 1)
            current_cols = int(getattr(descriptor, "ncols", 1) or 1)
            if (current_rows, current_cols) != (rows, cols):
                self._repo.set_figure_grid(self._figure_id, nrows=rows, ncols=cols)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to restore original figure grid.")
