"""Manual demo of the macOS look, with a live-editable stylesheet.

Not a pytest module: run it directly (``python3 app/tests/manual_macos_style_demo.py``)
to inspect macos_native.qss and the frameless CustomTitleBar against a real
window, without needing a database or the rest of MainWindow. Forces
macos_native.qss + the frameless/traffic-lights title bar regardless of the
host platform.

The right-hand panel is the actual macos_native.qss text, editable in
place: "Applica" re-applies whatever is in the box (themed the same way
apply_platform_style themes it) without touching the file on disk - the
fast loop for trying something out. "Ricarica dal file" re-reads
macos_native.qss, discarding unsaved edits in the box. "Salva su file"
writes the box back to macos_native.qss - only once something in it is
actually worth keeping.
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import (
    CardFrame,
    _load_qss,
    apply_platform_style,
    create_action_button,
    create_section_title,
    load_icon,
    mark_icon_only,
    stdSizeAndlayout,
    themed_qss,
)
from app.utils.i18n import _
from app.widgets.custom_title_bar import CustomTitleBar

QSS_NAME = "macos_native.qss"


class StyleDemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChartLibre | style demo")
        self.resize(1040, 720)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        host = QWidget(self)
        host.setObjectName("windowFrame")
        host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        self._title_bar = CustomTitleBar(self, is_macos=True)
        outer.addWidget(self._title_bar, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, host)
        outer.addWidget(splitter, 1)
        self.setCentralWidget(host)

        splitter.addWidget(self._preview_panel(splitter))
        splitter.addWidget(self._editor_panel(splitter))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self._reload_from_file()

    # ------------------------------------------------------------------
    # Preview: the widgets under review
    # ------------------------------------------------------------------
    def _preview_panel(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)
        layout.addWidget(self._menu_card(panel))
        layout.addWidget(self._buttons_card(panel))
        layout.addWidget(self._controls_card(panel))
        layout.addStretch(1)
        return panel

    def _menu_card(self, parent: QWidget) -> QWidget:
        """A QMenu opened from a plain button - "menu con voci bianche e
        bordi doppi" was reported against this exact control."""
        card = CardFrame(parent, "demoMenuCard")
        layout = card.layout()
        layout.addWidget(create_section_title(_("Menu"), card))

        menu = QMenu(card)
        for label in (_("New"), _("Open"), _("Save")):
            menu.addAction(label)
        menu.addSeparator()
        sub = menu.addMenu(_("Recent"))
        for label in ("one.dhub", "two.dhub", "three.dhub"):
            sub.addAction(label)

        button = create_action_button(parent=card, action_id="open", action=lambda: None)
        button.setText(_("Open recent"))
        button.setMenu(menu)
        row = QHBoxLayout()
        stdSizeAndlayout(row)
        row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _buttons_card(self, parent: QWidget) -> QWidget:
        card = CardFrame(parent, "demoButtonsCard")
        layout = card.layout()
        layout.addWidget(create_section_title(_("Buttons"), card))

        row = QHBoxLayout()
        stdSizeAndlayout(row)
        plain = create_action_button(parent=card, action_id="open", action=lambda: None)
        plain.setText(_("Plain"))
        primary = create_action_button(parent=card, action_id="save", action=lambda: None)
        primary.setText(_("Apply"))
        primary.setProperty("primary", True)
        icon_only = create_action_button(parent=card, action_id="delete", action=lambda: None)
        mark_icon_only(icon_only)
        row.addWidget(plain)
        row.addWidget(primary)
        row.addWidget(icon_only)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _controls_card(self, parent: QWidget) -> QWidget:
        card = CardFrame(parent, "demoControlsCard")
        layout = card.layout()
        layout.addWidget(create_section_title(_("Controls"), card))

        combo = QComboBox(card)
        combo.addItems(["Line", "Scatter", "Bar", "Area"])
        layout.addWidget(combo)

        checkbox = QCheckBox(_("Show grid"), card)
        checkbox.setChecked(True)
        layout.addWidget(checkbox)

        checkbox_off = QCheckBox(_("Show legend"), card)
        layout.addWidget(checkbox_off)

        return card

    # ------------------------------------------------------------------
    # Editor: the actual macos_native.qss text, live-editable
    # ------------------------------------------------------------------
    def _editor_panel(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)
        layout.addWidget(create_section_title(QSS_NAME, panel))

        self._editor = QPlainTextEdit(panel)
        self._editor.setFont(QFont("Menlo", 11))
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._editor, 1)

        self._status_label = QLabel(panel)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        row = QHBoxLayout()
        stdSizeAndlayout(row)
        create_action_button(
            parent=panel, action_id="reload", action=self._reload_from_file, layout=row,
            presentation=(load_icon("reload"), _("Ricarica dal file"), _("Discard edits and re-read macos_native.qss")),
        )
        row.addStretch(1)
        create_action_button(
            parent=panel, action_id="run", action=self._apply_editor_text, layout=row,
            presentation=(load_icon("run"), _("Applica"), _("Re-apply the text above, without touching the file")),
        )
        create_action_button(
            parent=panel, action_id="save", action=self._save_to_file, layout=row,
            presentation=(load_icon("save"), _("Salva su file"), _("Write the text above back to macos_native.qss")),
        )
        layout.addLayout(row)
        return panel

    def _reload_from_file(self) -> None:
        qss, path = _load_qss(QSS_NAME)
        self._editor.setPlainText(qss or "")
        self._status_label.setText(_("Loaded from {path}").format(path=path))
        self._apply_editor_text()

    def _apply_editor_text(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        try:
            themed, _palette = themed_qss(self._editor.toPlainText(), "light")
            app.setStyleSheet(themed)
        except Exception as exc:  # noqa: BLE001 - shown in the demo, not raised
            self._status_label.setText(_("Could not apply: {error}").format(error=exc))
            return
        self._status_label.setText(_("Applied ({count} characters)").format(count=len(themed)))

    def _save_to_file(self) -> None:
        _qss, path = _load_qss(QSS_NAME)
        if path is None:
            self._status_label.setText(_("No file to save to."))
            return
        path.write_text(self._editor.toPlainText(), encoding="utf-8")
        self._status_label.setText(_("Saved to {path}").format(path=path))


def main() -> int:
    app = QApplication(sys.argv)
    apply_platform_style(app, preference="macos_native")
    window = StyleDemoWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
