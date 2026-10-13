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
        sections: bool = False,
    ) -> None:
        super().__init__(parent, title=title, icon=icon, size=size)
        columns = QHBoxLayout()
        stdSizeAndlayout(columns)

        width = settings_width or self.SETTINGS_WIDTH
        if sections:
            # Each side a column of titled frames (add_settings_card,
            # add_content_card): a title inside a frame, over part of it,
            # is what this replaces.
            self._settings_column, self._content_column = QWidget(self), QWidget(self)
            for column in (self._settings_column, self._content_column):
                column_layout = QVBoxLayout(column)
                column_layout.setContentsMargins(0, 0, 0, 0)
            self._settings_column.setFixedWidth(width)
            self._settings_column.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            self._content_column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            columns.addWidget(self._settings_column, 0)
            columns.addWidget(self._content_column, 1)
            self.body_layout.addLayout(columns, 1)
            return
        # A titled panel is a standard frame: its title above the box, not in it.
        settings_widget = self._panel(settings_title, "dialogSettingsCard")
        self.settings_frame = settings_widget.card if isinstance(settings_widget, TitledCard) else settings_widget
        settings_widget.setFixedWidth(width)
        settings_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.settings_layout = self.settings_frame.layout()

        content_widget = self._panel(content_title, "dialogContentCard")
        self.content_frame = content_widget.card if isinstance(content_widget, TitledCard) else content_widget
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_layout = self.content_frame.layout()

        columns.addWidget(settings_widget, 0)
        columns.addWidget(content_widget, 1)
        self.body_layout.addLayout(columns, 1)

    def add_settings_card(self, title: str, object_name: str | None = None, *, stretch: int = 0) -> CardFrame:
        """With sections=True: a titled frame in the settings column; its box, to fill."""
        return self._section_card(self._settings_column, title, object_name, stretch)

    def add_content_card(self, title: str, object_name: str | None = None, *, stretch: int = 0) -> CardFrame:
        """With sections=True: a titled frame in the content column; its box, to fill."""
        return self._section_card(self._content_column, title, object_name, stretch)

    def _section_card(self, column: QWidget, title: str, object_name: str | None, stretch: int) -> CardFrame:
        titled = TitledCard(column, title, object_name)
        column.layout().addWidget(titled, stretch)
        if stretch:
            titled.layout().setStretchFactor(titled.card, 1)
            titled.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return titled.card

    def _panel(self, title: str, object_name: str) -> QWidget:
        """A frame for one side: titled (the title above the box) when *title* is given."""
        if not title:
            return CardFrame(self, object_name)
        titled = TitledCard(self, title, object_name)
        titled.layout().setStretchFactor(titled.card, 1)
        titled.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return titled
