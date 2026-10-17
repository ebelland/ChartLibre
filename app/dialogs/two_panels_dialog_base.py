"""The dialog shells: a dialog, a dialog of N columns, a dialog of settings beside content.

``PanelDialog`` is the base: the shared margins and size, a column of
titled frames (``add_card``) and the row of buttons under them
(``add_action``). ``ColumnsDialog`` lays N columns side by side - each of a
fixed width or stretching, optionally with handles to drag between them -
and titled frames go into any of them (``add_column_card``).
``SettingsTableDialog`` is the two-column case: fixed settings on the
left, the content on the right.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QSizePolicy, QSplitter, QVBoxLayout, QWidget

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


@dataclass(frozen=True, slots=True)
class Column:
    """One column of a ColumnsDialog: a fixed *width*, or None to share what is left by *stretch*."""

    width: int | None = None
    stretch: int = 1
    #: The narrowest a stretching column may become.
    minimum: int = 0


class ColumnsDialog(PanelDialog):
    """A dialog of columns side by side, each a stack of titled frames, over the row of buttons.

    *resizable* puts handles between the columns (a QSplitter, on the
    window's own ground) instead of a fixed layout.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str,
        icon=None,
        size: str | None = "medium",
        columns: Sequence[Column] = (Column(), Column()),
        resizable: bool = False,
    ) -> None:
        super().__init__(parent, title=title, icon=icon, size=size)
        self._columns: list[QWidget] = []
        if resizable:
            container = QSplitter(Qt.Orientation.Horizontal, self)
            # The handles on the window's grey between the frames, not white strips (B-12).
            container.setStyleSheet("QSplitter::handle { background: transparent; }")
            self.splitter: QSplitter | None = container
        else:
            self.splitter = None
            row = QHBoxLayout()
            stdSizeAndlayout(row)
        for index, spec in enumerate(columns):
            column = QWidget(self)
            layout = QVBoxLayout(column)
            layout.setContentsMargins(0, 0, 0, 0)
            if spec.width is not None:
                column.setFixedWidth(spec.width)
                column.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            else:
                column.setMinimumWidth(spec.minimum)
                column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            stretch = 0 if spec.width is not None else spec.stretch
            if resizable:
                container.addWidget(column)
                container.setStretchFactor(index, stretch)
            else:
                row.addWidget(column, stretch)
            self._columns.append(column)
        if resizable:
            self.body_layout.addWidget(container, 1)
        else:
            self.body_layout.addLayout(row, 1)

    def column(self, index: int) -> QWidget:
        """Column *index*: its layout (a QVBoxLayout) takes any widget."""
        return self._columns[index]

    def add_column_card(self, index: int, title: str, object_name: str | None = None, *,
                        stretch: int = 0) -> CardFrame:
        """A titled frame at the bottom of column *index*; its box, to fill."""
        column = self._columns[index]
        titled = TitledCard(column, title, object_name)
        column.layout().addWidget(titled, stretch)
        if stretch:
            titled.layout().setStretchFactor(titled.card, 1)
            titled.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return titled.card


class SettingsTableDialog(ColumnsDialog):
    """Two columns: fixed settings on the left, the content on the right."""

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
        width = settings_width or self.SETTINGS_WIDTH
        super().__init__(parent, title=title, icon=icon, size=size, columns=(Column(width=width), Column()))
        self._settings_column, self._content_column = self.column(0), self.column(1)
        if sections:
            # Each side a column of titled frames (add_settings_card, add_content_card).
            return
        # One frame per side: a titled panel is a standard frame, its title above the box.
        settings_widget = self._panel(settings_title, "dialogSettingsCard")
        self.settings_frame = settings_widget.card if isinstance(settings_widget, TitledCard) else settings_widget
        self.settings_layout = cast(QVBoxLayout, self.settings_frame.layout())
        content_widget = self._panel(content_title, "dialogContentCard")
        self.content_frame = content_widget.card if isinstance(content_widget, TitledCard) else content_widget
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_layout = cast(QVBoxLayout, self.content_frame.layout())
        self._settings_column.layout().addWidget(settings_widget, 1)
        self._content_column.layout().addWidget(content_widget, 1)

    def add_settings_card(self, title: str, object_name: str | None = None, *, stretch: int = 0) -> CardFrame:
        """With sections=True: a titled frame in the settings column; its box, to fill."""
        return self.add_column_card(0, title, object_name, stretch=stretch)

    def add_content_card(self, title: str, object_name: str | None = None, *, stretch: int = 0) -> CardFrame:
        """With sections=True: a titled frame in the content column; its box, to fill."""
        return self.add_column_card(1, title, object_name, stretch=stretch)

    def _panel(self, title: str, object_name: str) -> QWidget:
        """A frame for one side: titled (the title above the box) when *title* is given."""
        if not title:
            return CardFrame(self, object_name)
        titled = TitledCard(self, title, object_name)
        titled.layout().setStretchFactor(titled.card, 1)
        titled.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return titled
