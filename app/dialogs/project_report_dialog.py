"""Project report: the whole project as one HTML or PDF file (todo R-09).

The report is built by app/utils/project_report.py; this asks what to put in
it and writes it. HTML is one self-contained file, its pictures inside it -
SVG keeps them vector. The PDF is the same page laid out by Qt's own rich
text engine on A4 pages, with the pictures as PNG.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMarginsF, QSizeF, QUrl
from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
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


class ProjectReportDialog(QDialog):
    """Ask what the report holds and in which format, then write it."""

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.written: Path | None = None
        self.setWindowTitle(_("Project report"))
        self.setWindowIcon(load_icon("save"))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")
        card = CardFrame(self, "projectReportCard")
        card_layout = card.layout()
        form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(create_section_title(_("Report"), card))

        self._format_combo = QComboBox(card)
        self._format_combo.addItem(_("HTML - one file, opens in any browser"), "html")
        self._format_combo.addItem(_("PDF - A4 pages"), "pdf")
        form.addRow(_("Format"), self._format_combo)

        self._image_combo = QComboBox(card)
        self._image_combo.addItem(_("PNG"), "png")
        self._image_combo.addItem(_("SVG - sharp at any zoom"), "svg")
        self._image_combo.setToolTip(_("The PDF always uses PNG."))
        form.addRow(_("Pictures"), self._image_combo)

        self._dpi_spin = QSpinBox(card)
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setSingleStep(25)
        self._dpi_spin.setValue(150)
        self._dpi_spin.setSuffix(" dpi")
        form.addRow(_("Resolution"), self._dpi_spin)

        form.addRow(create_section_title(_("Contents"), card))
        self._data_check = QCheckBox(_("Each figure's series and their SQL"), card)
        self._data_check.setChecked(True)
        self._tables_check = QCheckBox(_("The tables, their size and notes"), card)
        self._tables_check.setChecked(True)
        self._history_check = QCheckBox(_("The operation history"), card)
        self._history_check.setChecked(True)
        for check in (self._data_check, self._tables_check, self._history_check):
            form.addRow("", check)
        card_layout.addLayout(form)

        self._status = QLabel(_("Every figure is drawn again, with its notes."), card)
        self._status.setProperty("muted", True)
        self._status.setWordWrap(True)
        card_layout.addWidget(self._status)
        card_layout.addStretch(1)
        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="save", action=self._export, layout=action_row,
            presentation=(load_icon("save"), _("Export…"), _("Choose where to save the report")),
        )
        create_action_button(parent=self, action_id="close", action=self.reject, layout=action_row)
        root.addLayout(action_row, 0)

        restore_dialog_state(self, _STATE_KEY)
        self._format_combo.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args: object) -> None:
        self._image_combo.setEnabled(self._format_combo.currentData() == "html")

    def _export(self) -> None:
        kind = str(self._format_combo.currentData() or "html")
        stem = Path(str(self._repo.db_path)).stem if self._repo.db_path else "report"
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
                include_data=self._data_check.isChecked(),
                include_tables=self._tables_check.isChecked(),
                include_history=self._history_check.isChecked(),
            )
            self.written = write_report_html(report, path) if kind == "html" else write_report_pdf(report, path)
        except (OSError, ValueError, DatabaseError) as exc:
            applogger.exception("Project report failed")
            self._status.setText(_("Could not write the report: {reason}").format(reason=exc))
            return
        for name, reason in report.failures:
            applogger.warning("Report: figure %r could not be drawn - %s", name, reason, show_dialog=False, raise_error=False)
        applogger.info("Project report written: %s (%d figures)", self.written, len(report.images))
        save_dialog_state(self, _STATE_KEY)
        self.accept()
