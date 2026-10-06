"""The File panel's Open recent: the projects opened lately, and the selected one's details.

A click shows a project - its preview picture, author, dates, references and
notes, read from the file without opening it - and Open, or a double click,
opens it, so a project can be looked at before the one open now is left for
it. The list comes from user.json and is rebuilt on every :meth:`refresh`:
it changes between visits to the panel.
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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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

#: The selected project's picture, beside its details.
PREVIEW_SIZE = QSize(112, 70)

#: Rows the list keeps even on a short window.
MINIMUM_ROWS = 1


class RecentProjectsView(QWidget):
    """The recent projects, the selected one's details, Open and Clear list."""

    #: Open was pressed, or a project double-clicked.
    open_requested = Signal(Path)
    #: Clear list was pressed; the owner forgets the list, then calls refresh().
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget(self)
        self._list.setObjectName("recentProjectsList")
        apply_fusion_for_item_view_styling(self._list)
        self._list.setAlternatingRowColors(True)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setUniformItemSizes(True)
        self._list.setIconSize(QSize(20, 20))
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        # Ignored, not Expanding, vertically: it still takes the room left,
        # but its own preferred height (a list asks for ~190 px) no longer
        # counts towards the panel's, which made the File panel scroll.
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._list.currentItemChanged.connect(self._show_details)
        self._list.itemDoubleClicked.connect(self._open_item)

        self._placeholder = QLabel(_("No recent projects"), self)
        self._placeholder.setProperty("muted", True)
        layout.addWidget(self._placeholder)

        # Side by side, small, in short lines: the File panel has to fit a
        # laptop's height without scrolling.
        self._details = QWidget(self)
        details = QHBoxLayout(self._details)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(8)
        self._preview = QLabel(self._details)
        self._preview.setFixedSize(PREVIEW_SIZE)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details.addWidget(self._preview, 0, Qt.AlignmentFlag.AlignTop)
        self._info = QLabel(self._details)
        # Not wrapped: a wrapping label asks the scroll area for height by
        # width, and the File panel grew a scroll bar it did not need.
        self._info.setWordWrap(False)
        self._info.setTextFormat(Qt.TextFormat.RichText)
        self._info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.addWidget(self._info, 1)
        layout.addWidget(self._details)

        buttons = QHBoxLayout()
        stdSizeAndlayout(buttons)
        self._open_button = create_action_button(
            parent=self, action_id="open", action=self._open_selected, layout=buttons,
        )
        buttons.addStretch(1)
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
        layout.addLayout(buttons)
        # The list last, taking the room left: the selected project's picture
        # and its Open button stay in view on a short screen.
        layout.addWidget(self._list, 1)

        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list from user.json: a project per row, its name over its folder."""
        recent = get_recent_databases()
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
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._list.addItem(item)

        has_any = bool(recent)
        for widget in (self._list, self._clear_button, self._open_button, self._details):
            widget.setVisible(has_any)
        self._placeholder.setVisible(not has_any)
        if not has_any:
            self._show_details(None)
            return
        self._list.setCurrentRow(0)
        row_height = max(self._list.sizeHintForRow(0), 1)
        self._list.setMinimumHeight(row_height * min(len(recent), MINIMUM_ROWS) + 2)

    def _open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.open_requested.emit(Path(str(path)))

    def _open_selected(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self._open_item(item)

    def _show_details(self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        """Picture and information of the project *item* names, read without opening it."""
        from app.data.repo.project_info import parse_references, read_project_info
        from app.dialogs.project_info_dialog import readable_date
        from app.utils.project_preview import resolve_preview

        path = Path(str(item.data(Qt.ItemDataRole.UserRole))) if item is not None else None
        self._open_button.setEnabled(path is not None and path.is_file())
        self._preview.clear()
        if path is None:
            self._info.clear()
            return
        if not path.is_file():
            self._info.setText(html.escape(_("This file is no longer there: {path}").format(path=path)))
            return
        info = read_project_info(path)
        picture_path = resolve_preview(path, info.get("preview_path"))
        picture = QPixmap(str(picture_path)) if picture_path is not None else QPixmap()
        if picture.isNull():
            self._preview.setText(_("No preview"))
            self._preview.setProperty("muted", True)
        else:
            ratio = self.devicePixelRatioF()
            scaled = picture.scaled(PREVIEW_SIZE * ratio, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(ratio)
            self._preview.setPixmap(scaled)
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
            rows.append(f"<i>{html.escape(notes if len(notes) <= 60 else notes[:57] + '...')}</i>")
        self._info.setText("<br>".join(rows))
        # The whole notes, which the line above cuts short.
        self._info.setToolTip(info.get("notes", ""))
