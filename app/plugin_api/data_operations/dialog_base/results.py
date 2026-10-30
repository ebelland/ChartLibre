"""Where results are drawn (result axes, figures, preview artefacts), and the preview transaction: running
Preview and Apply, recording them, and OK / Apply / Cancel."""
from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from app.data.data_source import parse_roles, row_value
from app.data.repo.operations import OPERATIONS_TABLE
from app.logs.logger import applogger
from app.plugin_api.data_operations.dialog_base.common import generated_table_name
from app.plugin_api.data_operations.results import OperationResult
from app.plugin_api.table_operations import next_steps
from app.utils.coercion import to_numeric_axis
from app.utils.dialog_state import dialog_entries
from app.utils.messages import show_message
from app.widgets.html_results import looks_like_html, plain_to_html


class ResultsMixin:
    """Where results are drawn, and the preview transaction. A part of DataOperationDialogBase."""


    # ------------------------------------------------------------------
    # Where results are drawn
    # ------------------------------------------------------------------

    # The three places a generated result can go (DEST_*) are OperationShell's: the same words in every
    # analysis window.

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
        wanted = (destination, str(chart_type), dict(options or {}))
        made_for = getattr(self, "_result_made_for", None)
        if getattr(self, "_result_destination", destination) != destination or (
                getattr(self, "_result_axis_id", None) is not None and made_for not in (None, wanted)):
            # Made for another choice - another place, or another chart: a Surface Plot asked for after a
            # Preview drew a Wireframe kept the Wireframe's axis - not reused, taken back
            # (destination_changed does it at once for the place).
            self.discard_result_target()
        if destination == self.DEST_SAME_AXIS:
            return selected_axis_id

        if getattr(self, "_result_axis_id", None) is None:
            self._result_destination = destination
            self._result_made_for = wanted
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

    def offer_destinations(self, destinations: Sequence[str], default: str | None = None) -> None:
        """Offer only *destinations* in "Draw on": a surface, say, cannot go on the axis of a curve.

        The choice is kept while it is still offered; otherwise *default*, or the first, and the change is
        followed as one the user made.
        """
        combo = getattr(self, "destination_combo", None)
        wanted = [str(value) for value in destinations]
        if combo is None or not wanted or [combo.itemData(i) for i in range(combo.count())] == wanted:
            return
        chosen = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            for value in wanted:
                combo.addItem(self.destination_label(value), value)
            target = chosen if chosen in wanted else (default if default in wanted else wanted[0])
            combo.setCurrentIndex(max(0, combo.findData(target)))
        finally:
            combo.blockSignals(False)
        if combo.currentData() != chosen:
            if self.chart_form is not None:
                self.chart_form.refresh_visibility()
            self.destination_changed()

    def destination_changed(self) -> None:
        """Draw on changed: the figure or axis drawn for the old choice - a new figure, say - is taken back at
        once, its Preview with it, and the window drops its tab; the next Preview draws where now asked."""
        made = getattr(self, "_result_axis_id", None) is not None or getattr(self, "_result_figure_id", None) is not None
        if made or getattr(self, "_preview_active", False):
            self.cancel_operation_changes(refresh=False)
            self.discard_result_target()
            self.sync_preview_figure()
        self.mark_results_stale()

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
        return self._remove_series_styled(axis_id, self.preview_style_filter)

    def preview_table_name(self, axis_id: int, result: Any) -> str:
        """Return the table name a preview writes to.

        Derived from the final name so the two are recognisably a pair, and
        prefixed like it: a preview is even more clearly not the user's data.
        """
        base = self.result_table_name(axis_id, result)
        return generated_table_name(f"{base}_Preview", fallback="Series_Preview")

    def remove_preview_tables(self, table_names: Sequence[str] | None = None) -> None:
        names = {str(name) for name in (table_names or self._preview_table_names) if str(name).strip()}
        for table_name in names:
            self._repo.delete_table(table_name)
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

    # ------------------------------------------------------------------
    # Preview transaction
    # ------------------------------------------------------------------
    def _finalize_successful_operation(
        self,
        *,
        results: Sequence[Any],
        verb: str,
    ) -> None:
        """Refresh UI and update the results pane after Preview/Apply."""
        self.store_cached_results(results)
        self._refresh_after_preview_state_change()
        formatted = self.results_report(results) or f"{verb}: {len(results)} result(s)."
        self.publish_results(formatted)

        # Applying is the point at which the report stops being scratch work,
        # so that is when it is handed to the chart it describes.  A preview is
        # deliberately excluded: it is undone on Close, and a note about
        # results that no longer exist is worse than no note.
        if verb == "Applied":
            self.results_published.emit(self.results_report_html(formatted, results))

    def results_report_html(self, formatted: str, results: Sequence[Any]) -> str:
        """Wrap one run's output in a titled block for the chart's notes pane.

        The heading is what makes several appended reports readable: without
        it, two runs of two different operations are one undifferentiated wall
        of text.
        """
        from datetime import datetime

        body = formatted if looks_like_html(formatted) else plain_to_html(formatted)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"<h3 style='margin:0 0 4px 0;'>{html.escape(self.operation_label)}</h3>"
            f"<p style='margin:0 0 8px 0;color:#666;font-size:9pt;'>"
            f"{stamp} - {len(results)} result(s)</p>"
            f"{body}"
        )

    def _run_operation(
        self, *, commit: bool, then: Callable[[bool], None] | None = None
    ) -> bool:
        """Run Preview or Apply through one simple pipeline.

        Preview first executes the same cleanup used by Close/Cancel, then writes
        new preview artifacts. Apply first removes any preview artifacts, then
        writes final artifacts and commits/exits through ``ok``.

        In the background (RUN_IN_BACKGROUND) the results are written once the
        calculation is back, and this returns True for "started"; *then* is
        told whether the operation went through, in either case, once it is
        over.
        """
        def finish(succeeded: bool) -> bool:
            if succeeded:
                self.operation_succeeded(commit=commit)
            if then is not None:
                then(succeeded)
            return succeeded

        try:
            if not self.confirm_run(commit=commit):
                return finish(False)
            if self.reads_table():
                self._prepare_table_source()
            axis_id_value = self.series_selector.selected_axis_id()
            if axis_id_value is None:
                # The catalogue holds the wording; the operation names the box.
                show_message(self, "series.no_axis_selected", title=self.operation_label)
                return finish(False)
            # Read now, not when the results are back: the window stays
            # editable while a background run computes.
            inputs = self.operation_inputs() if commit else None
            if self._runs_in_background():
                if self.evaluating:
                    return False
                started: list[bool] = []

                def deliver(results: Sequence[Any]) -> None:
                    started.append(True)
                    finish(self._deliver_results(list(results), int(axis_id_value), commit=commit, inputs=inputs))

                self.evaluate(deliver, background=True)
                if not self.evaluating and not started:
                    # Nothing to compute: the same answer the GUI thread gives.
                    show_message(self, "series.no_series_selected", title=self.operation_label)
                    return finish(False)
                return True
            results = list(self.compute_results())
        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            return finish(self._operation_failed(exc, commit=commit))
        return finish(self._deliver_results(results, int(axis_id_value), commit=commit, inputs=inputs))

    def _runs_in_background(self) -> bool:
        from app.plugin_api.data_operations.dialog_base.dialog import (
            DataOperationDialogBase,  # the class this is part of
        )

        return bool(
            self.RUN_IN_BACKGROUND
            and DataOperationDialogBase.BACKGROUND_ENABLED
            and (self._computes_per_series()
                 or type(self).prepare_job is not DataOperationDialogBase.prepare_job)
        )

    def _operation_failed(self, exc: Exception, *, commit: bool) -> bool:
        self.cancel_operation_changes(refresh=True)
        action = "Apply" if commit else "Preview"
        if commit:
            applogger.critical(f"{action} failed: {exc}")
        else:
            applogger.error(f"{action} failed: {exc}")
        self.show_failure(exc)
        return False

    def _deliver_results(
        self,
        results: list[Any],
        axis_id_value: int,
        *,
        commit: bool,
        inputs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Write computed results as a preview, or apply them; on the GUI thread.

        An Apply is also recorded in the project (``__operations__``, todo
        R-03) with the *inputs* read when it started.
        """
        try:
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False

            # Before resolve_target_axis_id, which may *create* the axis or
            # figure the results go on: a snapshot taken after it would have
            # nothing to say about that axis, and undo would leave an empty
            # one behind. The result tables are added to the same entry
            # below, once their names are known - they depend on which axis
            # this turned out to be.
            undo_entry: int | None = None
            # The last Preview's savepoint goes first, for OK and for a Preview alike: an axis made or
            # synced below (resolve_target_axis_id - a new figure, Fit's extra charts) inside it was
            # rolled back a line later, its id kept - the next series on it "FOREIGN KEY constraint
            # failed", or the chart just never showed. And a snapshot cannot be taken inside a
            # transaction; OK writes everything again anyway.
            self.cancel_operation_changes(refresh=False)
            if commit:
                undo_entry = self._repo.snapshot_for_undo(
                    self._repo.DESCRIPTOR_TABLES,
                    label=f"Apply {self.operation_label}",
                )

            # Most operations write back onto the axis their inputs came from.
            # One does not: see resolve_target_axis_id.
            axis_id = self.resolve_target_axis_id(axis_id_value, results)

            if commit:
                self._repo.snapshot_for_undo(
                    [self.result_table_name(axis_id, result) for result in results]
                    + [OPERATIONS_TABLE],
                    label=f"Apply {self.operation_label}",
                    entry_id=undo_entry,
                )
                self.cancel_operation_changes(refresh=False)
                self.begin_preview_transaction()
                self.apply_results_to_axis(axis_id, results)
                self.record_operation(axis_id, results, inputs or self.operation_inputs())
                # Release before optimize_db: VACUUM cannot run inside an open
                # transaction.
                self.commit_preview_transaction()
                self._repo.optimize_db()
                self._reset_preview_state_flags()
                self._finalize_successful_operation(
                    results=results,
                    verb="Applied",
                )
                return True

            self.begin_preview_transaction()
            self.preview_results_to_axis(axis_id, results)
            self._preview_axis_ids.add(axis_id)
            self._preview_active = True
            self.store_cached_results(results)
            self._refresh_after_preview_state_change()
            self.publish_results(
                self.results_report(results)
                or f"Preview updated: {len(results)} result(s)."
            )
            return True

        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            return self._operation_failed(exc, commit=commit)

    def operation_inputs(self) -> dict[str, Any]:
        """What an applied operation is recorded with: its parameters, entries and source series."""
        sources = []
        for row in self.selected_series():
            sources.append(
                {
                    "id": row_value(row, "id", default=None),
                    "name": str(row_value(row, "name", "series_name", default="")),
                    "sql_query": str(row_value(row, "sql_query", "query", "sql", default="")),
                    "roles": parse_roles(row_value(row, "roles", default={})),
                }
            )
        return {
            "parameters": self.parameter_values(),
            "entries": dialog_entries(self, inputs_only=True),
            "sources": sources,
        }

    def record_operation(self, axis_id: int, results: Sequence[Any], inputs: Mapping[str, Any]) -> None:
        """Add this Apply to the project's ``__operations__``, inside its transaction.

        A failure is logged, not raised: the record is about the result, and
        must never be the reason the result is lost.
        """
        try:
            written = [
                {
                    "table": self.result_table_name(axis_id, result),
                    "name": str(getattr(result, "result_name", "") or getattr(result, "output_name", "")),
                    "axis_id": int(axis_id),
                }
                for result in results
            ]
            report = self.results_report(results) or ""
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception(
                f"Could not record {self.operation_label} in the project's history",
            )
            return
        self.record_applied(written, report, inputs)

    def record_applied(
        self,
        written: Sequence[Mapping[str, Any]],
        report: str,
        inputs: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an Apply as it is: what it read, wrote and reported.

        The common path above writes result tables; Outlier (rows hidden or
        coloured in the source tables) and Statistics (a report, nothing
        written) call this themselves when their own Apply is kept. Logged,
        never raised, for the same reason as record_operation.
        """
        inputs = inputs if inputs is not None else self.operation_inputs()
        self.record_in_history(
            operation=self.operation_label,
            parameters=inputs.get("parameters", {}),
            entries=inputs.get("entries", {}),
            sources=list(inputs.get("sources", [])),
            results=written,
            report=report,
        )

    def confirm_run(self, *, commit: bool) -> bool:
        """Whether Preview or Apply goes ahead: an operation that knows a run is heavy asks here first."""
        del commit
        return True

    def operation_succeeded(self, *, commit: bool) -> None:
        """Called once a Preview (commit False) or an Apply has gone through."""

    def preview(self) -> bool:
        applogger.debug("Preview")
        return self._run_operation(commit=False)

    def ok(self) -> None:
        self.apply(then=lambda applied: self.accept() if applied else None)

    def _follow_link(self, url: str) -> None:
        """A "What next?" link clicked in Results: this result is kept (OK), then the next analysis opens."""
        step = next_steps.parse_step(url)
        if step is None:
            return
        self.next_request = step
        if self._last_results:
            self.ok()
        else:
            self.reject()

    def cancel(self) -> None:
        self.cancel_operation_changes()
        self._discard_table_figure()
        super().reject()

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Remove temporary Preview artifacts.

        This is intentionally the same cleanup used before creating a new
        Preview, so clicking Preview repeatedly behaves like Close followed by a
        fresh Preview.
        """
        rolled_back = self.rollback_preview_transaction()

        has_preview = bool(
            self._preview_active
            or self._preview_axis_ids
            or self._preview_table_names
        )
        if not rolled_back and not has_preview:
            return

        # The savepoint already undid everything it covered.  The artifact
        # sweep still runs when it did not, so a stale savepoint (or a preview
        # written before this dialog opened one) is not left behind.
        if not rolled_back:
            self.remove_preview_artifacts()
        else:
            self._forget_rolled_back_target()

        # Keep the grid consistent with the rolled-back axis descriptors.
        # This fixes preview flows that grow the grid (for example 1x1 -> 2x1)
        # and then cancel back to the original axis count.
        self._restore_original_grid()

        self._reset_preview_state_flags()

        if refresh:
            self._refresh_after_preview_state_change()

    def apply(self, then: Callable[[bool], None] | None = None) -> bool:
        """Run the operation and commit it; remembers the entries either way.

        *then* is told whether it was applied, once it has been - later, when
        the calculation runs in the background.
        """
        self.remember_window()

        def done(applied: bool) -> None:
            self._applied = self._applied or applied
            if then is not None:
                then(applied)

        return self._run_operation(commit=True, then=done)

    def discard_unapplied(self) -> None:
        """Close/Cancel: the Preview's changes rolled back, the axis or figure it made, the table's drawing."""
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        self._discard_table_figure()
        self.sync_preview_figure()
