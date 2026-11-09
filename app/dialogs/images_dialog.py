"""The project's pictures (__images__): see them, add them from a file or the clipboard, rename, remove.

Pictures are what Marker > Picture draws (and, later, bars and pictograms). Each keeps where it came from and
its licence, shown here and editable, so a chart that uses one can credit it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import _

_THUMB = 48


def _pixmap(kind: str, data: bytes, side: int) -> QPixmap:
    if kind == "svg":
        from PySide6.QtGui import QPainter
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(QByteArray(data))
        image = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        if renderer.isValid():
            painter = QPainter(image)
            size = renderer.defaultSize().scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio)
            renderer.render(painter, _rect(side, size))
            painter.end()
        return QPixmap.fromImage(image)
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def _rect(side: int, size: QSize):
    from PySide6.QtCore import QRectF

    return QRectF((side - size.width()) / 2, (side - size.height()) / 2, size.width(), size.height())


class ImagesDialog(QDialog):
    """The pictures of *repo*, with their source and licence."""

    def __init__(self, repo: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.setWindowTitle(_("Pictures"))
        self.resize(560, 420)

        root = QVBoxLayout(self)
        intro = QLabel(_("Pictures kept in the project, for markers drawn as pictures. SVG stays sharp at any "
                         "size; a single-colour icon takes the series' colour."), self)
        intro.setWordWrap(True)
        root.addWidget(intro)

        body = QHBoxLayout()
        self._list = QListWidget(self)
        self._list.setIconSize(QSize(_THUMB, _THUMB))
        self._list.currentItemChanged.connect(lambda *_a: self._show_current())
        body.addWidget(self._list, 1)

        side = QVBoxLayout()
        self._preview = QLabel(self)
        self._preview.setFixedSize(160, 160)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side.addWidget(self._preview)
        details = QFormLayout()
        self._kind = QLabel(self)
        self._source = QLineEdit(self)
        self._license = QLineEdit(self)
        self._license.setPlaceholderText(_("e.g. MIT, CC BY 4.0, own work"))
        self._source.editingFinished.connect(self._save_details)
        self._license.editingFinished.connect(self._save_details)
        details.addRow(_("Type"), self._kind)
        details.addRow(_("Source"), self._source)
        details.addRow(_("Licence"), self._license)
        side.addLayout(details)
        side.addStretch(1)
        body.addLayout(side)
        root.addLayout(body, 1)

        buttons = QHBoxLayout()
        for text, slot in ((_("Add from file…"), self._add_files), (_("Paste"), self._paste),
                           (_("Rename…"), self._rename), (_("Remove"), self._remove)):
            button = QPushButton(text, self)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        root.addLayout(buttons)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close.rejected.connect(self.reject)
        root.addWidget(close)
        self._fill()

    # ------------------------------------------------------------------ list
    def _fill(self, select: str | None = None) -> None:
        current = select or (self._list.currentItem().text() if self._list.currentItem() else None)
        self._list.clear()
        for name in self._repo.image_names():
            stored = self._repo.get_image(name)
            item = QListWidgetItem(QIcon(_pixmap(stored.kind, stored.data, _THUMB)), name)
            self._list.addItem(item)
            if name == current:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._show_current()

    def _current(self) -> Any:
        item = self._list.currentItem()
        return self._repo.get_image(item.text()) if item is not None else None

    def _show_current(self) -> None:
        stored = self._current()
        for widget in (self._source, self._license):
            widget.setEnabled(stored is not None)
        if stored is None:
            self._preview.clear()
            self._kind.clear()
            self._source.clear()
            self._license.clear()
            return
        self._preview.setPixmap(_pixmap(stored.kind, stored.data, 150))
        self._kind.setText(stored.kind.upper())
        self._source.setText(stored.source)
        self._license.setText(stored.license)

    # ------------------------------------------------------------------ actions
    def _save_details(self) -> None:
        stored = self._current()
        if stored is None:
            return
        source, license_ = self._source.text().strip(), self._license.text().strip()
        if (source, license_) != (stored.source, stored.license):
            self._repo.set_image_details(stored.name, source=source, license=license_)

    def _add(self, name: str, data: bytes, source: str) -> str | None:
        if name in self._repo.image_names():
            answer = QMessageBox.question(self, _("Pictures"),
                                          _("A picture named \"{name}\" is already kept. Replace it?").format(name=name))
            if answer != QMessageBox.StandardButton.Yes:
                return None
        try:
            return self._repo.add_image(name, data, source=source)
        except ValueError as error:
            QMessageBox.warning(self, _("Pictures"), str(error))
            return None

    def _add_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, _("Add pictures"), "",
                                                      _("Pictures (*.svg *.png *.jpg *.jpeg)"))
        added = None
        for path in paths:
            try:
                data = Path(path).read_bytes()
            except OSError as error:
                QMessageBox.warning(self, _("Pictures"), str(error))
                continue
            added = self._add(Path(path).stem, data, Path(path).name) or added
        if added:
            self._fill(added)

    def _paste(self) -> None:
        """An image or SVG text on the clipboard, kept under a name asked for."""
        mime = QGuiApplication.clipboard().mimeData()
        data = b""
        if mime is not None and mime.hasFormat("image/svg+xml"):
            data = bytes(mime.data("image/svg+xml").data())
        elif mime is not None and mime.hasText() and "<svg" in mime.text()[:4096]:
            data = mime.text().encode("utf-8")
        elif mime is not None and mime.hasImage():
            image = QGuiApplication.clipboard().image()
            buffer_bytes = QByteArray()
            buffer = QBuffer(buffer_bytes)
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            image.save(buffer, "PNG")
            data = bytes(buffer_bytes.data())
        if not data:
            QMessageBox.information(self, _("Pictures"), _("The clipboard holds no picture."))
            return
        name, ok = QInputDialog.getText(self, _("Paste picture"), _("Name of the picture:"))
        if ok and name.strip():
            added = self._add(name.strip(), data, _("Clipboard"))
            if added:
                self._fill(added)

    def _rename(self) -> None:
        stored = self._current()
        if stored is None:
            return
        name, ok = QInputDialog.getText(self, _("Rename picture"), _("New name:"), text=stored.name)
        name = name.strip()
        if not ok or not name or name == stored.name:
            return
        if name in self._repo.image_names():
            QMessageBox.warning(self, _("Pictures"), _("A picture named \"{name}\" is already kept.").format(name=name))
            return
        self._repo.rename_image(stored.name, name)
        self._fill(name)

    def _remove(self) -> None:
        stored = self._current()
        if stored is None:
            return
        answer = QMessageBox.question(
            self, _("Pictures"),
            _("Remove the picture \"{name}\"? Series that draw it go back to their style's marker; "
              "Undo brings it back.").format(name=stored.name))
        if answer == QMessageBox.StandardButton.Yes:
            self._repo.remove_image(stored.name)
            self._fill()
