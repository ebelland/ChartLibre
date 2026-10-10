"""Project report: the whole project as one HTML or PDF file (todo R-09).

The report is built by app/utils/project_report.py; this asks what to put in
it and writes it. HTML is one self-contained file, its pictures inside it -
SVG keeps them vector. The PDF is the same page laid out by Qt's own rich
text engine on A4 pages, with the pictures as PNG.
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QMarginsF, QSizeF, QUrl
from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QFormLayout, QLabel, QSpinBox, QWidget

from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.logs.logger import applogger
from app.dialogs.two_panels_dialog_base import PanelDialog
from app.styles.style import TitledCard, load_icon, stdSizeAndlayout
from app.utils.dialog_state import restore_dialog_state, save_dialog_state
from app.utils.i18n import _
from app.utils.project_report import ProjectReport, build_project_report, write_report_html

_STATE_KEY = "project_report_dialog"

#: Points per millimetre, for the PDF page margins.
_MARGIN_MM = 15.0


def write_report_pdf(report: ProjectReport, path: Path | str) -> Path:
    """Lay *report* out on A4 pages and write it as a PDF."""
    target = Path(path)
    if target.suffix.lower() != ".pdf":
        target = target.with_name(target.name + ".pdf")
    writer = QPdfWriter(str(target))
    writer.setResolution(300)
    writer.setTitle(report.title)
    writer.setCreator("ChartLibre")
    writer.setPageLayout(QPageLayout(
        QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait,
        QMarginsF(_MARGIN_MM, _MARGIN_MM, _MARGIN_MM, _MARGIN_MM), QPageLayout.Unit.Millimeter,
    ))
    document = QTextDocument()
    for name, data in report.images.items():
        image = QImage.fromData(data)
        if not image.isNull():
            document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(name), image)
    document.setHtml(report.html)
    # The page's own width in points, so the HTML's 620-pixel pictures fit it.
    page = writer.pageLayout().paintRect(QPageLayout.Unit.Point)
    document.setPageSize(QSizeF(page.width(), page.height()))
    document.print_(writer)
    return target


class ProjectReportDialog(PanelDialog):
    """Ask what the report holds and in which format, then write it.

    With *figure_id*, the report of that one chart - its picture and notes -
    which is the chart menu's "Export report": the same dialog and writer,
    without the project's tables and history.
    """

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None, *, figure_id: int | None = None) -> None:
        super().__init__(parent, title=_("Export report") if figure_id is not None else _("Project report"),
                         icon=load_icon("save"))
        self._repo = repo
        self._figure_id = figure_id
        self.written: Path | None = None
        root = self.body_layout
        # Titled frames as in the rest of the application: the title above
        # its box, not inside it.
        report = TitledCard(self, _("Report"), "projectReportCard")
        form = QFormLayout()
        stdSizeAndlayout(form)
        report.card.layout().addLayout(form)

        self._format_combo = QComboBox(report.card)
        self._format_combo.addItem(_("HTML - one file, opens in any browser"), "html")
        self._format_combo.addItem(_("PDF - A4 pages"), "pdf")
        form.addRow(_("Format"), self._format_combo)

        self._image_combo = QComboBox(report.card)
        self._image_combo.addItem(_("PNG"), "png")
        self._image_combo.addItem(_("SVG - sharp at any zoom"), "svg")
        self._image_combo.setToolTip(_("The PDF always uses PNG."))
        form.addRow(_("Pictures"), self._image_combo)

        self._dpi_spin = QSpinBox(report.card)
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setSingleStep(25)
        self._dpi_spin.setValue(150)
        self._dpi_spin.setSuffix(" dpi")
        form.addRow(_("Resolution"), self._dpi_spin)

        self._status = QLabel(
            _("The chart is drawn again, with its notes.") if figure_id is not None
            else _("Every figure is drawn again, with its notes."),
            report.card,
        )
        self._status.setProperty("muted", True)
        self._status.setWordWrap(True)
        report.card.layout().addWidget(self._status)
        root.addWidget(report, 0)

        contents = TitledCard(self, _("Contents"), "projectReportContentsCard")
        self._tables_check = QCheckBox(_("The tables, their size and notes"), contents.card)
        self._tables_check.setChecked(True)
        self._history_check = QCheckBox(_("The operation history"), contents.card)
        self._history_check.setChecked(True)
        for check in (self._tables_check, self._history_check):
            contents.card.layout().addWidget(check)
        root.addWidget(contents, 0)
        # One chart: its picture and notes; the project's tables and history
        # are not about it.
        contents.setVisible(figure_id is None)
        root.addStretch(1)

        self.add_action("save", self._export,
                        presentation=(load_icon("save"), _("Export…"), _("Choose where to save the report")))
        self.add_action("close", self.reject)

        restore_dialog_state(self, _STATE_KEY)
        self._format_combo.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args: object) -> None:
        self._image_combo.setEnabled(self._format_combo.currentData() == "html")

    def _export(self) -> None:
        kind = str(self._format_combo.currentData() or "html")
        stem = Path(str(self._repo.db_path)).stem if self._repo.db_path else "report"
        if self._figure_id is not None:
            title = self._repo.get_figure_title(self._figure_id) or f"figure_{self._figure_id}"
            stem = re.sub(r"[^\w\-]+", "_", title).strip("_") or "figure"
        file_filter = _("HTML page (*.html)") if kind == "html" else _("PDF Document (*.pdf)")
        path, _filter = QFileDialog.getSaveFileName(self, _("Project report"), f"{stem}.{kind}", file_filter)
        if not path:
            return
        # Said in the dialog rather than with a wait cursor: setOverrideCursor
        # crashed the application on macOS 27 (Qt builds that cursor from an
        # image, and CoreGraphics refused the image).
        self._status.setText(_("Drawing every figure and writing the report…"))
        self._status.repaint()
        try:
            report = build_project_report(
                self._repo,
                image_format=str(self._image_combo.currentData() or "png") if kind == "html" else "png",
                dpi=self._dpi_spin.value(),
                include_tables=self._figure_id is None and self._tables_check.isChecked(),
                include_history=self._figure_id is None and self._history_check.isChecked(),
                figure_ids=None if self._figure_id is None else [self._figure_id],
            )
            self.written = write_report_html(report, path) if kind == "html" else write_report_pdf(report, path)
        except (OSError, ValueError, DatabaseError) as exc:
            applogger.exception("Project report failed")
            self._status.setText(_("Could not write the report: {reason}").format(reason=exc))
            return
        for name, reason in report.failures:
            applogger.warning("Report: figure %r could not be drawn - %s", name, reason)
        applogger.info("Project report written: %s (%d figures)", self.written, len(report.images))
        save_dialog_state(self, _STATE_KEY)
        self.accept()
