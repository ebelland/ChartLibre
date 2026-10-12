"""The dialog shells: one column of titled frames, or settings beside content.

``PanelDialog`` is the base: the shared margins and size, a column of
titled frames (``add_card``) and the row of buttons under them
(``add_action``). ``SettingsTableDialog`` puts two frames side by side in
that column - fixed settings on the left, the content on the right.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QDialog, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from app.styles.style import (
    CardFrame,
    TitledCard,
    apply_dialog_shell,
    create_action_button,
    create_compact_section_title,
    stdSizeAndlayout,
)


class PanelDialog(QDialog):
    """A dialog of titled frames, one under the other, over a row of buttons."""

    def __init__(self, parent: QWidget | None = None, *, title: str, icon=None, size: str | None = "small") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        if icon is not None:
            self.setWindowIcon(icon)
        self.root_layout = QVBoxLayout(self)
        apply_dialog_shell(self, self.root_layout, size=size)
        #: Where the frames go: add_card, or any widget or layout.
        self.body_layout = QVBoxLayout()
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.addLayout(self.body_layout, 1)
        self.action_row = QHBoxLayout()
        stdSizeAndlayout(self.action_row)
        self.action_row.addStretch(1)
        self.root_layout.addLayout(self.action_row)

    def add_card(self, title: str, object_name: str | None = None, *, stretch: int = 0) -> CardFrame:
        """A frame titled *title* (the title above its box); its box, to fill."""
        titled = TitledCard(self, title, object_name)
        self.body_layout.addWidget(titled, stretch)
        if stretch:
            titled.layout().setStretchFactor(titled.card, 1)
        return titled.card

    def add_action(self, action_id: str, action: Callable[[], None], *, default: bool = False, presentation=None,
                   left: bool = False):
        """A button in the bottom row, from the action catalogue; *left* puts it apart, at the left."""
        kwargs = {} if presentation is None else {"presentation": presentation}
        button = create_action_button(parent=self, action_id=action_id, action=action, layout=self.action_row,
                                      **kwargs)
        if left:
            self.action_row.removeWidget(button)
            self.action_row.insertWidget(0, button)
        if default:
            button.setDefault(True)
        return button


class SettingsTableDialog(PanelDialog):
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
        super().__init__(parent, title=title, icon=icon, size=size)
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
        self.body_layout.addLayout(columns, 1)

