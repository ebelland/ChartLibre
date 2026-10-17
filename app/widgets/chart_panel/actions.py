"""What the panel does to its chart: copy, save, export, templates, delete."""
from __future__ import annotations

import csv
from io import BytesIO
from typing import Any

import PySide6.QtGui
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

import numpy as np

from app.data.sqlite_repo import DatabaseError
from app.logs.logger import applogger
from app.utils.messages import ask, show_message
from app.utils.config import get_value
from app.utils.i18n import _

from app.widgets.chart_panel.constants import CONFIG_COPY_DPI


class ActionsMixin:
    """What the panel does to its chart: copy, save, export, templates, delete. A part of ChartPanel (see panel.py)."""

    # ------------------------------------------------------------------
    # Local actions
    # ------------------------------------------------------------------
    def _canvas_to_qimage(self) -> PySide6.QtGui.QImage | None:
        """Render the current Matplotlib figure into a detached QImage."""
        self._canvas.draw()
        buffer = BytesIO()
        copy_dpi:float = self._config.get(CONFIG_COPY_DPI,600.0)
        
        try:
            self._figure.savefig(
                buffer,
                format="png",
                dpi=copy_dpi,
                bbox_inches=None,
                facecolor=self._figure.get_facecolor(),
                edgecolor=self._figure.get_edgecolor(),
            )
            data = buffer.getvalue()
        finally:
            buffer.close()

        image = PySide6.QtGui.QImage.fromData(data)
        if image.isNull():
            applogger.error("Failed to build QImage from in-memory PNG data.")
            return None
        return image

    def copy_chart_to_clipboard(self) -> None:
        """Copy the current chart image to the system clipboard."""
        try:
            image = self._canvas_to_qimage()
            if image is None:
                return
            PySide6.QtGui.QGuiApplication.clipboard().setImage(image)
            applogger.info("Chart copied to clipboard (figure_id=%s)", self._figure_id)
        except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
            applogger.exception(
                "Failed to copy chart to clipboard (figure_id=%s)",
                self._figure_id,
            )

    def save_chart_as(self) -> None:
        """Save the chart to an image or vector format via a file dialog."""
        # The configured default decides the suggested extension and which
        # filter the dialog opens on.  Both were hard-coded to PNG while
        # config.json carried a ``save_format`` that nothing read, so the
        # preference existed and did nothing.
        from app.dialogs.settings_dialog import (
            CONFIG_SAVE_FORMAT,
            SAVE_FORMAT_FILTERS,
            normalized_save_format,
        )

        preferred = normalized_save_format(get_value(CONFIG_SAVE_FORMAT))
        extension, preferred_filter = SAVE_FORMAT_FILTERS[preferred]
        default_name = f"chart_{self._figure_id}.{extension}"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            _("Save chart as"),
            default_name,
            ";;".join(label for _unused, label in SAVE_FORMAT_FILTERS.values()),
            preferred_filter,
        )
        if not file_path:
            return

        try:
            # Not self._figure.dpi: in FIXED mode that now holds the on-screen
            # rendering dpi (FIXED_MODE_SCREEN_DPI times zoom), which has
            # nothing to do with export quality.  _fixed_figure_dpi is always
            # the dpi actually configured for this figure, screen state aside.
            self._figure.savefig(
                file_path, dpi=self._fixed_figure_dpi, bbox_inches="tight"
            )
            applogger.info(
                "Chart saved (figure_id=%s, path=%s, filter=%s)",
                self._figure_id,
                file_path,
                selected_filter,
            )
        except Exception:  # noqa: BLE001 - a chart interaction or layout pass: never breaks the window
            applogger.exception("Failed to save chart (figure_id=%s)", self._figure_id)
            show_message(self, "chart.save_failed")

    def export_report(self) -> None:
        """The project report of this chart alone: its picture and notes."""
        from app.dialogs.project_report_dialog import ProjectReportDialog

        ProjectReportDialog(self._repo, self, figure_id=self._figure_id).exec()

    def save_as_template(self) -> None:
        """Save this figure's look as a named graph template."""
        from app.utils import graph_templates

        names = graph_templates.list_templates()
        name, accepted = QInputDialog.getItem(
            self, _("Save as template"), _("Template name (an existing name is replaced):"),
            names or [self._repo.get_figure_title(self._figure_id) or _("My template")], 0, True,
        )
        if not accepted or not str(name).strip():
            return
        descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
        if descriptor is None:
            return
        try:
            path = graph_templates.save_template(str(name), graph_templates.template_from_figure(descriptor))
        except OSError:
            applogger.exception("Could not save the template %r", name)
            show_message(self, "chart.save_failed")
            return
        applogger.info("Graph template %r saved: %s", name, path)

    def apply_saved_template(self) -> None:
        """Give this figure the look of a saved template. Undoable."""
        from app.utils import graph_templates

        names = graph_templates.list_templates()
        if not names:
            QMessageBox.information(self, _("Apply a template"), _("No template saved yet: save one from a figure first."))
            return
        name, accepted = QInputDialog.getItem(self, _("Apply a template"), _("Template:"), names, 0, False)
        if not accepted:
            return
        try:
            template = graph_templates.load_template(str(name))
            self._repo.snapshot_for_undo(self._repo.DESCRIPTOR_TABLES, label=_("Apply template"))
            changed = graph_templates.apply_template(self._repo, self._figure_id, template)
        except (OSError, ValueError, DatabaseError):
            applogger.exception("Could not apply the template %r", name)
            QMessageBox.warning(self, _("Apply a template"), _("The template could not be applied; see the log."))
            return
        applogger.info("Graph template %r applied to figure %s (%d items).", name, self._figure_id, changed)
        self.reload()
        self.figure_edited.emit()

    def delete_saved_template(self) -> None:
        from app.utils import graph_templates

        names = graph_templates.list_templates()
        if not names:
            QMessageBox.information(self, _("Delete a template"), _("No template saved yet."))
            return
        name, accepted = QInputDialog.getItem(self, _("Delete a template"), _("Template:"), names, 0, False)
        if accepted:
            graph_templates.delete_template(str(name))
            applogger.info("Graph template %r deleted.", name)

    def delete_chart(self) -> None:
        """Ask for confirmation and close/delete the current chart panel."""
        if ask(self, "chart.confirm_delete"):
            self.close()

    def export_view_as_csv(self) -> None:
        """Save the rows actually on screen - after zoom and downsampling -
        as one CSV. A sibling of copy/save-chart, for the data rather than
        the picture: what a peak-fit or a "why does this look off" question
        wants is the numbers behind what is visible, not the whole series.
        """
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            _("Export view as CSV"),
            f"chart_{self._figure_id}_view.csv",
            _("CSV files (*.csv)"),
        )
        if not file_path:
            return

        rows = self._current_view_rows()
        if not rows:
            show_message(self, "chart.export_view_empty")
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as handle:
                fieldnames: list[str] = ["axis", "series", "x", "y"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            applogger.info(
                "Chart view exported (figure_id=%s, path=%s, rows=%d)",
                self._figure_id,
                file_path,
                len(rows),
            )
        except OSError:
            applogger.exception("Failed to export chart view (figure_id=%s)", self._figure_id)
            show_message(self, "chart.save_failed")

    def _current_view_rows(self) -> list[dict[str, Any]]:
        """Return every plotted point currently inside its own axes' zoom.

        Each chart axes keeps its own xlim/ylim, so "the current view" is
        computed per axes rather than once for the whole figure - a
        zoomed-in panel next to an untouched one only exports the zoomed
        rows from the first.
        """
        rows: list[dict[str, Any]] = []
        for axes in self._figure.axes:
            axis_id = getattr(axes, "_dhub_axis_id", None)
            if axis_id is None:
                continue
            x_min, x_max = axes.get_xlim()
            y_min, y_max = axes.get_ylim()

            for artist in (*axes.lines, *axes.collections):
                label = str(artist.get_label() or "").strip()
                if not label or label.startswith("_"):
                    continue

                if hasattr(artist, "get_offsets"):
                    points = np.asarray(artist.get_offsets())
                    if points.size == 0:
                        continue
                    xs, ys = points[:, 0].astype(float), points[:, 1].astype(float)
                else:
                    x_data, y_data = artist.get_data()
                    try:
                        xs = np.asarray(x_data, dtype=float)
                        ys = np.asarray(y_data, dtype=float)
                    except (TypeError, ValueError):
                        continue

                mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
                for x_value, y_value in zip(xs[mask], ys[mask]):
                    rows.append(
                        {"axis": int(axis_id), "series": label, "x": float(x_value), "y": float(y_value)}
                    )
        return rows
