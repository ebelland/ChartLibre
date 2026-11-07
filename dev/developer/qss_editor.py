"""Qt style editor: one of app/styles/*.qss, edited and tried live on a sample of the app's widgets.

Apply installs the edited sheet on the whole application - this window, the sample and the launcher - with the
light or the dark palette's @TOKENS@, as apply_platform_style would; closing puts the saved style back. Save
writes the file and keeps the previous one beside it as .qss.bak.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.styles.palettes import PALETTES, themed_qss
from app.styles.style import TitledCard, apply_platform_style, load_icon
from app.styles.style.icons import _substitute_qss_symbols, substitute_ui_font

STYLES_DIR = Path(__file__).resolve().parents[2] / "app" / "styles"
TOKEN = re.compile(r"@([A-Z][A-Z0-9_]*)@")


class QssEditorDialog(QDialog):
    """Pick a sheet, edit it, Apply to see it, Save to keep it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Qt style editor")
        self._path: Path | None = None
        self._applied = False

        top = QHBoxLayout()
        self._file = QComboBox(self)
        for path in sorted(STYLES_DIR.glob("*.qss")):
            self._file.addItem(path.name, str(path))
        self._file.currentIndexChanged.connect(self._load)
        top.addWidget(QLabel("Sheet", self))
        top.addWidget(self._file)
        self._palette = QComboBox(self)
        for key in PALETTES:
            self._palette.addItem(key, key)
        top.addWidget(QLabel("Palette", self))
        top.addWidget(self._palette)
        self._find = QLineEdit(self)
        self._find.setPlaceholderText("Find (Return: next)")
        self._find.returnPressed.connect(self._find_next)
        top.addWidget(self._find, 1)

        self._text = QPlainTextEdit(self)
        self._text.setFont(QFont("Menlo", 11))
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.textChanged.connect(self._list_tokens)
        self._tokens = QListWidget(self)
        self._tokens.setToolTip("The palette's tokens this sheet uses, with the value the chosen palette gives.")
        self._palette.currentIndexChanged.connect(self._list_tokens)

        split = QSplitter(Qt.Orientation.Horizontal, self)
        left = QSplitter(Qt.Orientation.Vertical, split)
        left.addWidget(self._text)
        left.addWidget(self._tokens)
        left.setSizes([520, 140])
        split.addWidget(left)
        split.addWidget(self._sample())
        split.setSizes([620, 380])

        buttons = QHBoxLayout()
        self._note = QLabel("", self)
        self._note.setProperty("muted", True)
        buttons.addWidget(self._note, 1)
        for text, slot in (("Apply", self._apply), ("Restore the app's style", self._restore), ("Save", self._save)):
            button = QPushButton(text, self)
            button.clicked.connect(slot)
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        layout.addLayout(buttons)
        self.resize(1100, 720)
        self._load()

    def _sample(self) -> QWidget:
        """A few of the app's widgets, with the object names its sheets style."""
        page = QWidget(self)
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        column = QVBoxLayout(page)
        card = TitledCard(page, "A titled card")
        inner = card.card.layout()
        row = QHBoxLayout()
        row.addWidget(QPushButton("Button", card.card))
        combo = QComboBox(card.card)
        combo.addItems(["Combo box", "Second"])
        row.addWidget(combo)
        inner.addLayout(row)
        inner.addWidget(QCheckBox("Check box", card.card))
        inner.addWidget(QLineEdit("Line edit", card.card))
        muted = QLabel("A muted sentence under a button.", card.card)
        muted.setProperty("muted", True)
        inner.addWidget(muted)
        column.addWidget(card)

        pill = QFrame(page)
        pill.setObjectName("chartNavPill")
        pill_row = QHBoxLayout(pill)
        pill_row.setContentsMargins(3, 2, 3, 2)
        pill_row.setSpacing(1)
        for action_id in ("add", "delete", "chart_from_columns", "edit", "table_more"):
            button = QToolButton(pill)
            button.setObjectName("chartToolButton")
            button.setIcon(load_icon(action_id))
            button.setIconSize(QSize(16, 16))
            button.setFixedSize(QSize(28, 28))
            button.setAutoRaise(True)
            pill_row.addWidget(button)
        column.addWidget(pill, 0, Qt.AlignmentFlag.AlignLeft)

        table = QTableWidget(4, 3, page)
        table.setObjectName("dataTable")
        table.setHorizontalHeaderLabels(["x (F)", "y (F)", "group (S)"])
        for r in range(4):
            for c, value in enumerate((r, r * r / 3, "ab"[r % 2])):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        column.addWidget(table, 1)
        return page

    # ------------------------------------------------------------------

    def _load(self) -> None:
        path = Path(str(self._file.currentData() or ""))
        if not path.is_file():
            return
        self._path = path
        self._text.setPlainText(path.read_text(encoding="utf-8"))
        self._note.setText(f"{path.name}: {len(self._text.toPlainText().splitlines())} lines.")

    def _list_tokens(self) -> None:
        palette = PALETTES[str(self._palette.currentData() or "light")]
        values = palette.tokens()
        used = sorted(set(TOKEN.findall(self._text.toPlainText())))
        self._tokens.clear()
        for name in used:
            self._tokens.addItem(f"@{name}@  =  {values.get(name, '(not a palette token)')}")

    def _find_next(self) -> None:
        if not self._text.find(self._find.text()):
            self._text.moveCursor(QTextCursor.MoveOperation.Start)
            self._text.find(self._find.text(), QTextDocument.FindFlag(0))

    def themed(self) -> str:
        """The edited sheet as the app would install it: tokens, font and symbols filled in."""
        qss, _palette = themed_qss(self._text.toPlainText(), str(self._palette.currentData() or "light"))
        return _substitute_qss_symbols(substitute_ui_font(qss))

    def _apply(self) -> None:
        app = QApplication.instance()
        palette = PALETTES[str(self._palette.currentData() or "light")]
        app.setPalette(palette.qpalette())
        app.setStyleSheet(self.themed())
        self._applied = True
        self._note.setText("Applied to the whole window - not saved.")

    def _restore(self) -> None:
        apply_platform_style(QApplication.instance())
        self._applied = False
        self._note.setText("The app's own style again.")

    def _save(self) -> None:
        if self._path is None:
            return
        shutil.copy2(self._path, self._path.with_suffix(".qss.bak"))
        self._path.write_text(self._text.toPlainText(), encoding="utf-8")
        self._note.setText(f"Saved {self._path.name} (the previous one: {self._path.name}.bak).")

    def done(self, result: int) -> None:  # noqa: D401 - Qt's name
        if self._applied:
            self._restore()
        super().done(result)
