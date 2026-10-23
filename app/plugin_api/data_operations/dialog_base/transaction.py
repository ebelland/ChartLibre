"""The preview transaction, running the operation, recording it, OK / Apply / Cancel."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import html
from typing import Any

from PySide6.QtGui import QCloseEvent

from app.data.data_source import parse_roles, row_value
from app.data.repo.operations import OPERATIONS_TABLE
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.utils.dialog_state import dialog_entries
from app.widgets.html_results import looks_like_html, plain_to_html


class TransactionMixin:
    """The preview transaction, running the operation, recording it, OK / Apply / Cancel. A part of DataOperationDialogBase (see dialog.py)."""

    # ------------------------------------------------------------------
    # Preview transaction
    # ------------------------------------------------------------------
    def begin_preview_transaction(self) -> None:
        """Open the SAVEPOINT that makes a Preview undoable.

        Why a database savepoint and not just artifact bookkeeping: removing the
        preview series and temp tables only undoes what the *generated-table*
        operations create.  Clustering writes a ClusterId column into the user's
        own source table and rewrites the selected series' SQL, and outlier
        removal flips hide flags on real rows - none of which the artifact
        cleanup can reach.  With every preview write inside one savepoint,
        Close rolls back whatever the operation did, whatever that was.

        SqliteRepo._commit() suppresses commits while the savepoint is open, so
        repository helpers that normally commit after each write stay inside it.
        """
        if self._repo.preview_savepoint_active:
            return
        self._repo.begin_preview_savepoint(
            f"preview_{self.__class__.__name__}"
        )

    def rollback_preview_transaction(self) -> bool:
        """Undo every database change made since the preview started."""
        if not self._repo.preview_savepoint_active:
            return False
        try:
            return bool(self._repo.rollback_preview_savepoint())
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to roll back the preview savepoint")
            return False

    def commit_preview_transaction(self) -> None:
        """Make the preview changes permanent."""
        if not self._repo.preview_savepoint_active:
            return
        try:
            self._repo.release_preview_savepoint()
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to commit the preview savepoint")

    def _finalize_successful_operation(
        self,
        *,
        results: Sequence[Any],
        verb: str,
    ) -> None:
        """Refresh UI and update the results pane after Preview/Apply."""
        self.store_cached_results(results)
        self._refresh_after_preview_state_change()
        formatted = self.format_results(results) or f"{verb}: {len(results)} result(s)."
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
        from app.plugin_api.data_operations.dialog_base.dialog import DataOperationDialogBase  # the class this is part of

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

            self.cancel_operation_changes(refresh=False)
            self.begin_preview_transaction()
            self.preview_results_to_axis(axis_id, results)
            self._preview_axis_ids.add(axis_id)
            self._preview_active = True
            self.store_cached_results(results)
            self._refresh_after_preview_state_change()
            self.publish_results(
                self.format_results(results)
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
            report = self.format_results(results) or ""
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

    def operation_succeeded(self, *, commit: bool) -> None:
        """Called once a Preview (commit False) or an Apply has gone through."""

    def preview(self) -> bool:
        applogger.debug("Preview")
        return self._run_operation(commit=False)

    def ok(self) -> None:
        self.apply(then=lambda applied: self.accept() if applied else None)

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

    def reject(self) -> None:
        """Close/Cancel rejects temporary Preview changes."""
        self.stop()
        self.remember_window()
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        self._discard_table_figure()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Window close button also rejects temporary Preview changes."""
        self.stop()
        self.remember_window()
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        self._discard_table_figure()
        super().closeEvent(event)
