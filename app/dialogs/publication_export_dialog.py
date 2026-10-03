"""Export a figure for a journal: column width, font size, vector format (todo R-09).

The settings and the arithmetic are in app/utils/publication.py; this asks
for them. The width starts on a journal column - 85, 120 or 180 mm - and the
height on the figure's own proportions, so a figure designed on screen comes
out the same shape, only at the printed size, with its text at the size the
journal asks for.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
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
from app.utils.publication import (
    JOURNAL_WIDTHS_MM,
    PUBLICATION_FORMATS,
    VECTOR_FORMATS,
    PublicationSettings,
    export_publication_figure,
)

_STATE_KEY = "publication_export_dialog"
_CUSTOM = -1.0


class PublicationExportDialog(QDialog):
    """Ask for a journal figure's size, text size and format, then write it."""

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        *,
        aspect: float = 0.75,
        default_name: str = "figure",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo = repo
        self._figure_id = figure_id
        self._aspect = aspect if aspect > 0 else 0.75
        self._default_name = default_name
        self.written: Path | None = None
        self.setWindowTitle(_("Export for publication"))
        self.setWindowIcon(load_icon("save"))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")
        card = CardFrame(self, "publicationExportCard")
        card_layout = card.layout()
        form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(create_section_title(_("Size"), card))

        self._width_preset_combo = QComboBox(card)
        for label, width in JOURNAL_WIDTHS_MM:
            self._width_preset_combo.addItem(_("{label} ({width:g} mm)").format(label=_(label), width=width), width)
        self._width_preset_combo.addItem(_("Custom"), _CUSTOM)
        form.addRow(_("Width"), self._width_preset_combo)

        self._width_spin = self._millimetres(card, JOURNAL_WIDTHS_MM[0][1])
        form.addRow(_("Width (mm)"), self._width_spin)
        self._height_spin = self._millimetres(card, round(JOURNAL_WIDTHS_MM[0][1] * self._aspect, 1))
        self._height_spin.setToolTip(_("Starts at the figure's own proportions for the width chosen."))
        form.addRow(_("Height (mm)"), self._height_spin)

        form.addRow(create_section_title(_("Text and file"), card))
        self._font_size_spin = QDoubleSpinBox(card)
        self._font_size_spin.setRange(5.0, 14.0)
        self._font_size_spin.setSingleStep(0.5)
        self._font_size_spin.setDecimals(1)
        self._font_size_spin.setSuffix(" pt")
        self._font_size_spin.setValue(8.0)
        self._font_size_spin.setToolTip(_("Text size at the printed size. Journals ask for 7 to 9 pt."))
        form.addRow(_("Font size"), self._font_size_spin)

        self._format_combo = QComboBox(card)
        for name in PUBLICATION_FORMATS:
            self._format_combo.addItem(name, name)
        self._format_combo.setToolTip(
            _("PDF, EPS and SVG stay sharp at any size and keep their text as text, with the fonts "
              "embedded. EPS has no transparency: see-through areas come out solid.")
        )
        form.addRow(_("Format"), self._format_combo)

        self._dpi_spin = QSpinBox(card)
        self._dpi_spin.setRange(72, 2400)
        self._dpi_spin.setSingleStep(50)
        self._dpi_spin.setValue(600)
        self._dpi_spin.setSuffix(" dpi")
        self._dpi_spin.setToolTip(_("Resolution of a TIFF or PNG: 300 for photographs, 600 or more for line art."))
        form.addRow(_("Resolution"), self._dpi_spin)
        card_layout.addLayout(form)

        self._summary = QLabel(card)
        self._summary.setProperty("muted", True)
        self._summary.setWordWrap(True)
        card_layout.addWidget(self._summary)
        card_layout.addStretch(1)
        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="save", action=self._export, layout=action_row,
            presentation=(load_icon("save"), _("Export…"), _("Choose where to save the figure")),
        )
        create_action_button(parent=self, action_id="close", action=self.reject, layout=action_row)
        root.addLayout(action_row, 0)

        restore_dialog_state(self, _STATE_KEY)
        # The width, text and format are remembered; the height is this figure's.
        self._height_spin.setValue(round(self._width_spin.value() * self._aspect, 1))
        self._width_preset_combo.currentIndexChanged.connect(self._on_preset)
        self._width_spin.valueChanged.connect(self._on_width)
        for signal in (self._height_spin.valueChanged, self._font_size_spin.valueChanged,
                       self._format_combo.currentIndexChanged, self._dpi_spin.valueChanged):
            signal.connect(self._refresh)
        self._on_preset()

    @staticmethod
    def _millimetres(parent: QWidget, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(parent)
        spin.setRange(20.0, 500.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" mm")
        spin.setValue(value)
        return spin

    # ------------------------------------------------------------------
    def settings(self) -> PublicationSettings:
        return PublicationSettings(
            width_mm=self._width_spin.value(),
            height_mm=self._height_spin.value(),
            font_size_pt=self._font_size_spin.value(),
            format=str(self._format_combo.currentData() or "PDF"),
            dpi=self._dpi_spin.value(),
        )

    def _on_preset(self, *_args: object) -> None:
        width = float(self._width_preset_combo.currentData() or _CUSTOM)
        custom = width == _CUSTOM
        self._width_spin.setEnabled(custom)
        if not custom and abs(self._width_spin.value() - width) > 1e-9:
            self._width_spin.setValue(width)  # _on_width follows
        self._refresh()

    def _on_width(self, width: float) -> None:
        self._height_spin.setValue(round(width * self._aspect, 1))
        self._refresh()

    def _refresh(self, *_args: object) -> None:
        settings = self.settings()
        raster = settings.format not in VECTOR_FORMATS
        self._dpi_spin.setEnabled(raster)
        size = _("{width:g} × {height:g} mm, text {font:g} pt").format(
            width=settings.width_mm, height=settings.height_mm, font=settings.font_size_pt
        )
        if raster:
            pixels = _("{w} × {h} pixels").format(
                w=int(settings.width_mm / 25.4 * settings.dpi), h=int(settings.height_mm / 25.4 * settings.dpi)
            )
            self._summary.setText(f"{size} - {pixels}")
        else:
            self._summary.setText(f"{size} - " + _("vector, fonts embedded"))

    def _export(self) -> None:
        settings = self.settings()
        extension, file_filter = PUBLICATION_FORMATS[settings.format]
        path, _filter = QFileDialog.getSaveFileName(
            self, _("Export for publication"), f"{self._default_name}.{extension}", _(file_filter)
        )
        if not path:
            return
        # Said in the dialog, not with a wait cursor: setOverrideCursor crashed
        # the application on macOS 27 (see project_report_dialog).
        self._summary.setText(_("Writing the figure…"))
        self._summary.repaint()
        try:
            self.written = export_publication_figure(self._repo, self._figure_id, Path(path), settings)
        except (OSError, ValueError, RuntimeError) as exc:
            applogger.exception("Publication export failed (figure_id=%s)", self._figure_id)
            self._summary.setText(_("Could not export the figure: {reason}").format(reason=exc))
            return
        applogger.info("Figure %s exported for publication: %s (%s)", self._figure_id, self.written, settings)
        save_dialog_state(self, _STATE_KEY)
        self.accept()
