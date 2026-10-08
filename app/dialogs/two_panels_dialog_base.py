"""Shared two-column shell for settings-and-preview dialogs."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QDialog, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_compact_section_title,
    stdSizeAndlayout,
)


class SettingsTableDialog(QDialog):
    """Standard dialog with fixed settings on the left and content on the right."""

    SETTINGS_WIDTH = 420

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str,
        icon=None,
        settings_title: str = "",
        content_title: str = "",
        size: str = "medium",
        settings_width: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        if icon is not None:
            self.setWindowIcon(icon)

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size=size)
        columns = QHBoxLayout()
        stdSizeAndlayout(columns)

        width = settings_width or self.SETTINGS_WIDTH
        self.settings_frame = CardFrame(self, "dialogSettingsCard")
        self.settings_frame.setFixedWidth(width)
        self.settings_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.settings_layout = self.settings_frame.layout()
        if settings_title:
            self.settings_layout.addWidget(
                create_compact_section_title(settings_title, self.settings_frame)
            )

        self.content_frame = CardFrame(self, "dialogContentCard")
        self.content_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_layout = self.content_frame.layout()
        if content_title:
            self.content_layout.addWidget(
                create_compact_section_title(content_title, self.content_frame)
            )

        columns.addWidget(self.settings_frame, 0)
        columns.addWidget(self.content_frame, 1)
        root.addLayout(columns, 1)

        self.action_row = QHBoxLayout()
        stdSizeAndlayout(self.action_row)
        self.action_row.addStretch(1)
        root.addLayout(self.action_row)

    def add_action(self, action_id: str, action: Callable[[], None], *, default: bool = False):
        button = create_action_button(
            parent=self, action_id=action_id, action=action, layout=self.action_row
        )
        if default:
            button.setDefault(True)
        return button
