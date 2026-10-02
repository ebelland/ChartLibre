"""Pick one of the shipped demo projects, and load it on request.

The set itself - what each project shows, which figures and tables it needs -
is app/data/demos.py's; this dialog only presents the choice. Loading
it, unlike "New" or "Open", asks nothing about where: see
MainWindow._on_load_demo for why a fixed, well-known name needs no dialog of
its own.

This is the only way in. A first run used to offer the whole set before the
window was even up; it now starts on an empty database and says nothing, so
the demo is something you go and get rather than something you decline.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.data.demos import DEMO_PROJECTS, DemoProject
from app.styles.style import (
    apply_dialog_shell,
    CardFrame,
    create_action_button,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


class LoadDemoDialog(QDialog):
    """Let the user choose one demo project; ``chosen`` holds the result.

    Modal, and read through ``chosen`` rather than a signal: the caller wants
    one answer before it goes on to load it, not an ongoing conversation.
    """

    #: Lines of summary shown below the picture at least; a longer one scrolls.
    _SUMMARY_LINES: int = 7

    #: The picture of the selected demo (demo/previews, 640 x 400), shown
    #: at this size: what it looks like is most of what a choice needs.
    PREVIEW_SIZE: QSize = QSize(400, 250)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle(_("Load demo"))
        self.setWindowIcon(load_icon("plot"))
        self.chosen: DemoProject | None = None

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        card = CardFrame(self, "loadDemoCard")
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_("Load demo"), card))

        self._list = QListWidget(card)
        mark_editor_panel(self._list)
        for demo in DEMO_PROJECTS:
            item = QListWidgetItem(_(demo.file_name), self._list)
            item.setToolTip(_(demo.summary))
            item.setData(Qt.ItemDataRole.UserRole, demo)
        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._update_summary)
        body = QHBoxLayout()
        stdSizeAndlayout(body)
        body.addWidget(self._list, 1)
        details = QVBoxLayout()
        stdSizeAndlayout(details)
        self._preview = QLabel(card)
        self._preview.setFixedSize(self.PREVIEW_SIZE)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details.addWidget(self._preview, 0)

        self._summary = QLabel("", card)
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        self._summary.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # A box of the picture's width, scrolling if a summary is longer: a
        # word-wrapped label grows and shrinks with its width, and the layout
        # around it used to move every time the dialog was resized.
        summary_box = QScrollArea(card)
        summary_box.setWidget(self._summary)
        summary_box.setWidgetResizable(True)
        summary_box.setFrameShape(QFrame.Shape.NoFrame)
        summary_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        summary_box.viewport().setAutoFillBackground(False)
        self._summary.setAutoFillBackground(False)
        # At least this many lines, and whatever height the list beside it has.
        summary_box.setMinimumHeight(self._summary.fontMetrics().lineSpacing() * self._SUMMARY_LINES + 4)
        summary_box.setFixedWidth(self.PREVIEW_SIZE.width())
        details.addWidget(summary_box, 1)
        body.addLayout(details, 0)
        card_layout.addLayout(body, 1)
        self._update_summary(0)

        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="apply", action=self._confirm, layout=action_row
        )
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        root.addLayout(action_row, 0)

    def _update_summary(self, row: int) -> None:
        item = self._list.item(row)
        demo: DemoProject | None = item.data(Qt.ItemDataRole.UserRole) if item else None
        self._summary.setText(_(demo.summary) if demo else "")
        self._preview.setPixmap(self._picture(demo))

    def _picture(self, demo: DemoProject | None) -> QPixmap:
        """The demo's preview at the label's size, sharp on a high-density screen;
        empty when there is none - a clone that has not built them."""
        if demo is None or not demo.preview_path.exists():
            return QPixmap()
        picture = QPixmap(str(demo.preview_path))
        if picture.isNull():
            return picture
        ratio = self.devicePixelRatioF()
        scaled = picture.scaled(
            self.PREVIEW_SIZE * ratio, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        scaled.setDevicePixelRatio(ratio)
        return scaled

    def _confirm(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        self.chosen = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

