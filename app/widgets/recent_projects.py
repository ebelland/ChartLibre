"""The File panel's project lists: the demos and the projects opened lately.

Both tabs share one view - a list, the selected project's picture and
description under it, Open (and, for the recent projects, Clear list) under
that. A click shows a project, read from the file without opening it; Open,
or a double click, opens it. The recent list comes from user.json and is
rebuilt on every :meth:`refresh`: it changes between visits to the panel.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.data.demos import DEMO_CATEGORIES, DEMO_PROJECTS, DemoProject
from app.styles.style import (
    action_presentation,
    apply_fusion_for_item_view_styling,
    colored_icon,
    create_action_button,
    mark_destructive_button,
    stdSizeAndlayout,
)
from app.utils.config import get_recent_databases
from app.utils.i18n import _

#: The selected project's picture, over its description.
PREVIEW_SIZE = QSize(112, 70)

#: Lines the description keeps in view; a longer one scrolls.
DESCRIPTION_LINES = 5


class ProjectListView(QWidget):
    """A list of projects, the selected one's picture and description, Open and (optionally) Clear list.

    What a row holds, and what it shows, is the subclass's: :meth:`refresh`
    fills the list, :meth:`describe` returns the picture and the text.
    """

    #: Open was pressed, or a row double-clicked; the row's data.
    open_requested = Signal(object)
    #: Clear list was pressed; the owner forgets the list, then calls refresh().
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, clearable: bool = False,
                 empty_text: str = "") -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._list = QListWidget(self)
        self._list.setObjectName("projectList")
        apply_fusion_for_item_view_styling(self._list)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setIconSize(QSize(20, 20))
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        # Ignored vertically: it takes the room left, but its own preferred
        # height no longer counts towards the panel's, which made it scroll.
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._list.setMinimumHeight(60)
        self._list.currentItemChanged.connect(self._show_details)
        self._list.itemDoubleClicked.connect(self._open_item)
        layout.addWidget(self._list, 1)

        self._placeholder = QLabel(empty_text, self)
        self._placeholder.setProperty("muted", True)
        self._placeholder.setVisible(False)
        layout.addWidget(self._placeholder)

        self._details = QWidget(self)
        # The picture over its description, which has the panel's whole width.
        details = QVBoxLayout(self._details)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(6)
        self._preview = QLabel(self._details)
        self._preview.setFixedSize(PREVIEW_SIZE)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details.addWidget(self._preview, 0, Qt.AlignmentFlag.AlignLeft)
        self._info = QLabel(self._details)
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.TextFormat.RichText)
        self._info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._info.setAutoFillBackground(False)
        # A box of fixed height, scrolling if the text is longer: a wrapping
        # label asks for height by width, and the panel grew a scroll bar.
        box = QScrollArea(self._details)
        box.setWidget(self._info)
        box.setWidgetResizable(True)
        box.setFrameShape(QFrame.Shape.NoFrame)
        box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        box.viewport().setAutoFillBackground(False)
        box.setFixedHeight(self._info.fontMetrics().lineSpacing() * DESCRIPTION_LINES + 4)
        details.addWidget(box, 1)
        layout.addWidget(self._details)

        buttons = QHBoxLayout()
        stdSizeAndlayout(buttons)
        self._open_button = create_action_button(
            parent=self, action_id="open", action=self._open_selected, layout=buttons,
        )
        clear_icon, _clear_text, _clear_tooltip = action_presentation("clear")
        self._clear_button = create_action_button(
            parent=self,
            action_id="clear",
            action=self.clear_requested.emit,
            layout=buttons,
            presentation=(clear_icon, _("Clear list"), _("Forget the list of recently opened projects")),
        )
        # Red: the one button here a misclick cannot undo.
        mark_destructive_button(self._clear_button)
        self._clearable = clearable
        self._clear_button.setVisible(clearable)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    @property
    def buttons(self) -> list[QPushButton]:
        """Open and Clear list, for the panel to size like its own buttons."""
        return [self._open_button, self._clear_button]

    def refresh(self) -> None:
        """Fill the list again; the subclass's."""

    def describe(self, data: object) -> tuple[bytes | None, str, bool]:
        """The picture (PNG), the description (rich text) and whether it can be opened, for a row's *data*."""
        raise NotImplementedError

    def _finish_refresh(self) -> None:
        has_any = any(self._list.item(row).data(Qt.ItemDataRole.UserRole) is not None
                      for row in range(self._list.count()))
        for widget in (self._list, self._open_button, self._details):
            widget.setVisible(has_any)
        self._clear_button.setVisible(self._clearable and has_any)
        self._placeholder.setVisible(not has_any)
        first = next((row for row in range(self._list.count())
                      if self._list.item(row).data(Qt.ItemDataRole.UserRole) is not None), -1)
        self._list.setCurrentRow(first)
        if first < 0:
            self._show_details(None)

    def _open_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is not None:
            self.open_requested.emit(data)

    def _open_selected(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self._open_item(item)

    def _show_details(self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        data = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        self._preview.clear()
        if data is None:
            self._info.clear()
            self._open_button.setEnabled(False)
            return
        png, text, openable = self.describe(data)
        self._open_button.setEnabled(openable)
        self._info.setText(text)
        picture = QPixmap()
        if png:
            picture.loadFromData(png)
        if picture.isNull():
            self._preview.setText(_("No preview") if openable else "")
            self._preview.setProperty("muted", True)
        else:
            ratio = self.devicePixelRatioF()
            scaled = picture.scaled(PREVIEW_SIZE * ratio, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(ratio)
            self._preview.setPixmap(scaled)


class DemoProjectsView(ProjectListView):
    """The shipped demos, each category under its heading; Open loads a fresh copy."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, clearable=False)
        self.refresh()

    def refresh(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for name in DEMO_CATEGORIES:
            demos = [demo for demo in DEMO_PROJECTS if demo.category == name]
            if not demos:
                continue
            heading = QListWidgetItem(_(name).upper(), self._list)
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            for demo in demos:
                item = QListWidgetItem(_(demo.file_name), self._list)
                item.setToolTip(_(demo.file_name))
                item.setData(Qt.ItemDataRole.UserRole, demo)
        self._list.blockSignals(False)
        self._finish_refresh()

    def describe(self, data: object) -> tuple[bytes | None, str, bool]:
        demo: DemoProject = data  # type: ignore[assignment]
        return demo.preview_bytes, html.escape(_(demo.description)), True


class RecentProjectsView(ProjectListView):
    """The projects opened lately, from user.json; Clear list forgets them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, clearable=True, empty_text=_("No recent projects"))
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list from user.json: a project per row, its name over its folder."""
        recent = get_recent_databases()
        self._list.blockSignals(True)
        self._list.clear()
        # A document, in the accent blue, as Finder draws a file.
        icon = colored_icon(
            action_presentation("recent_project")[0],
            QApplication.palette().color(QPalette.ColorRole.Highlight),
            20,
        )
        home = str(Path.home())
        for path in recent:
            folder = str(path.parent)
            if folder.startswith(home):
                folder = "~" + folder[len(home):]
            item = QListWidgetItem(icon, f"{path.name}\n{folder}")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, Path(path))
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._finish_refresh()

    def describe(self, data: object) -> tuple[bytes | None, str, bool]:
        """Picture and information of the project at *data*, read without opening it."""
        from app.data.repo.project_info import parse_references, read_project_info
        from app.dialogs.project_info_dialog import readable_date
        from app.utils.project_preview import project_preview_bytes

        path = Path(str(data))
        if not path.is_file():
            return None, html.escape(_("This file is no longer there: {path}").format(path=path)), False
        info = read_project_info(path)
        png = project_preview_bytes(path, info.get("preview_path"))
        stat = path.stat()
        rows = []
        if info.get("author"):
            rows.append(html.escape(info["author"]))
        if info.get("created"):
            rows.append(html.escape(_("Created {date}").format(date=readable_date(info["created"])[:10])))
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        rows.append(html.escape(_("Saved {date} · {size:.1f} MB").format(date=modified, size=stat.st_size / 1e6)))
        references = parse_references(info.get("references"))
        if references:
            rows.append(html.escape(_("{count} reference(s)").format(count=len(references))))
        notes = " ".join(info.get("notes", "").split())
        if notes:
            rows.append(f"<i>{html.escape(notes)}</i>")
        return png, "<br>".join(rows), True


class ProjectsTabs(QTabWidget):
    """Demo | Recent: the two lists of the File panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fileProjectsTabs")
        self.demos = DemoProjectsView(self)
        self.recent = RecentProjectsView(self)
        self.addTab(self.demos, _("Demo"))
        self.addTab(self.recent, _("Recent"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def buttons(self) -> list[QPushButton]:
        return self.demos.buttons + self.recent.buttons
