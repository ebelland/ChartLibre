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
    QComboBox,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QWidget,
)

from app.dialogs.two_panels_dialog_base import SettingsTableDialog
from app.data.demos import DEMO_CATEGORIES, DEMO_PROJECTS, DemoProject
from app.styles.style import load_icon, mark_editor_panel
from app.utils.config import get_value, set_value
from app.utils.i18n import _


class LoadDemoDialog(SettingsTableDialog):
    """Let the user choose one demo project; ``chosen`` holds the result.

    Modal, and read through ``chosen`` rather than a signal: the caller wants
    one answer before it goes on to load it, not an ongoing conversation.
    """

    #: Lines of summary shown below the picture at least; a longer one scrolls.
    _SUMMARY_LINES: int = 7

    #: The picture of the selected demo (the preview its project names), shown
    #: at this size: what it looks like is most of what a choice needs.
    PREVIEW_SIZE: QSize = QSize(400, 250)

    def __init__(self, parent: QWidget | None = None) -> None:
        # The demos on the left; the one chosen, pictured, on the right.
        super().__init__(
            parent,
            title=_("Load demo"),
            icon=load_icon("load_demo"),
            settings_title=_("Demo projects"),
            content_title=_("Preview"),
            size="medium",
            settings_width=380,
        )
        self.chosen: DemoProject | None = None
        card = self.content_frame

        # By category (JMP's platform names where a demo is about one): all of
        # them, under a heading each, or one category's.
        self.category = QComboBox(self.settings_frame)
        self.category.addItem(_("All demos"), "")
        for name in DEMO_CATEGORIES:
            count = sum(1 for demo in DEMO_PROJECTS if demo.category == name)
            if count:
                self.category.addItem(f"{_(name)} ({count})", name)
        self.category.setToolTip(_("Show the demos of one kind"))
        self.settings_layout.addWidget(self.category)

        self._list = QListWidget(self.settings_frame)
        # Long names end in "…" rather than scroll sideways; the tooltip says the rest.
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        mark_editor_panel(self._list)
        self._list.currentRowChanged.connect(self._update_summary)
        self.settings_layout.addWidget(self._list, 1)
        self.category.currentIndexChanged.connect(lambda _index: self._fill_list())
        details = self.content_layout
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
        summary_box.setMinimumWidth(self.PREVIEW_SIZE.width())
        details.addWidget(summary_box, 1)
        remembered = self.category.findData(str(get_value(self.CATEGORY_KEY, "") or ""))
        self.category.setCurrentIndex(max(remembered, 0))
        self._fill_list()

        self.add_action("apply", self._confirm, default=True)
        self.add_action("close", self.reject)

    #: The category shown last, in user.json.
    CATEGORY_KEY = "load_demo_category"

    def _fill_list(self) -> None:
        """The chosen category's demos - or all, each category under its heading."""
        chosen = str(self.category.currentData() or "")
        set_value(self.CATEGORY_KEY, chosen)
        self._list.blockSignals(True)
        self._list.clear()
        for name in ([chosen] if chosen else DEMO_CATEGORIES):
            demos = [demo for demo in DEMO_PROJECTS if demo.category == name]
            if not demos:
                continue
            if not chosen:
                heading = QListWidgetItem(_(name).upper(), self._list)
                heading.setFlags(Qt.ItemFlag.NoItemFlags)
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
            for demo in demos:
                item = QListWidgetItem(_(demo.file_name), self._list)
                item.setToolTip(f"<b>{_(demo.file_name)}</b><br>{_(demo.description)}")
                item.setData(Qt.ItemDataRole.UserRole, demo)
        self._list.blockSignals(False)
        first = next((row for row in range(self._list.count())
                      if self._list.item(row).data(Qt.ItemDataRole.UserRole) is not None), -1)
        self._list.setCurrentRow(first)
        self._update_summary(first)

    def _update_summary(self, row: int) -> None:
        item = self._list.item(row)
        demo: DemoProject | None = item.data(Qt.ItemDataRole.UserRole) if item else None
        # From the built project itself: its notes are the demo's summary
        # (translated through the catalogue, which holds the same text).
        self._summary.setText(_(demo.description) if demo else "")
        self._preview.setPixmap(self._picture(demo))

    def _picture(self, demo: DemoProject | None) -> QPixmap:
        """The demo's preview at the label's size, sharp on a high-density screen;
        empty when there is none - a clone that has not built them."""
        png = demo.preview_bytes if demo is not None else None
        if not png:
            return QPixmap()
        picture = QPixmap()
        picture.loadFromData(png)
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

