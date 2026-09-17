"""Manual demo of the macOS look, with a live-editable stylesheet.

Not a pytest module: run it directly (``python3 app/tests/manual_macos_style_demo.py``)
to inspect macos_native.qss and the frameless CustomTitleBar against a real
window, without needing a database or the rest of MainWindow. Forces
macos_native.qss + the frameless/traffic-lights title bar regardless of the
host platform.

The left column is a stand-in for the real app's #leftPanelCard: a narrow
#activityRail sidebar (grey, hairline on its own right edge - Tahoe's
System Settings sidebar) with the traffic lights embedded at its own top
(as Finder/Mail/System Settings place them, inside the sidebar column
itself, rather than in a separate window-wide title strip), and a single
Workspace toggle. #leftPanelCard itself is white with its own right
hairline - the "list column" beside a grey sidebar, Mail/Finder-style -
and starts hidden, same as "no left panels" is the real app's own default
reading. Toggling it in reveals a stack of cards whose section-title
labels sit above their card, never inside its border.

The right side simulates the real ChartPanel: a QTabWidget using the same
tab-bar language the real chart tabs use (macos_native.qss styles QTabBar
generically, not by object name), next to the QSS editor panel.

The right-hand panel is the actual macos_native.qss text, editable in
place: "Applica" re-applies whatever is in the box (themed the same way
apply_platform_style themes it) without touching the file on disk - the
fast loop for trying something out. "Ricarica dal file" re-reads
macos_native.qss, discarding unsaved edits in the box. "Salva su file"
writes the box back to macos_native.qss - only once something in it is
actually worth keeping.
"""
from __future__ import annotations

from typing import Callable
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import (
    SPACING_LOOSE,
    SPACING_TIGHT,
    CardFrame,
    _load_qss,
    apply_platform_style,
    apply_rounded_window_mask,
    create_action_button,
    create_section_title,
    icon_from_svg_source,
    load_icon,
    mark_icon_only,
    stdSizeAndlayout,
    themed_qss,
)
from app.utils.i18n import _
from app.widgets.custom_title_bar import CustomTitleBar

QSS_NAME = "macos_native.qss"

_SIDEBAR_WIDTH = 200
_LEFT_CONTENT_WIDTH = 280
_ROW_HEIGHT = 30

#: Small inline glyph, same idiom as nav_bar.py's own icons - kept local
#: rather than imported so this demo has no dependency on NavigationBar's
#: MainWindow-shaped constructor.
_COMPONENTS_ICON = (
    '<circle cx="12" cy="12" r="3"/>'
    '<path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8'
    'M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/>'
)


class StyleDemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChartLibre | style demo")
        self.resize(1180, 720)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        host = QWidget(self)
        host.setObjectName("windowFrame")
        host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QHBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(host)

        outer.addWidget(self._left_panel(host), 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, host)
        outer.addWidget(splitter, 1)
        splitter.addWidget(self._main_content(splitter))
        splitter.addWidget(self._editor_panel(splitter))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self._reload_from_file()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Keep the rounded-corner mask sized to the window - see
        apply_rounded_window_mask's own docstring, and MainWindow's
        identical override, which this one is meant to match."""
        super().resizeEvent(event)
        if self.isMaximized():
            self.clearMask()
        else:
            apply_rounded_window_mask(self)

    # ------------------------------------------------------------------
    # Left panel: #activityRail (grey, own right edge) + hidden-by-default
    # content column (white, own right edge), matching macos_native.qss's
    # #leftPanelCard/#activityRail rules.
    # ------------------------------------------------------------------
    def _left_panel(self, parent: QWidget) -> QWidget:
        panel = QFrame(parent)
        panel.setObjectName("leftPanelCard")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._nav_sidebar(panel), 0)

        self._left_content = self._componenti_page(panel)
        self._left_content.setFixedWidth(_LEFT_CONTENT_WIDTH)
        self._left_content.setVisible(False)
        layout.addWidget(self._left_content, 0)
        return panel

    def _nav_sidebar(self, parent: QWidget) -> QWidget:
        sidebar = QFrame(parent)
        sidebar.setObjectName("activityRail")
        sidebar.setFixedWidth(_SIDEBAR_WIDTH)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 10)
        layout.setSpacing(4)

        # Not a separate window-wide strip: sized to the sidebar's own
        # width, the way a real Mac app's traffic lights sit inside the
        # sidebar column rather than spanning the whole window.
        self._title_bar = CustomTitleBar(self, is_macos=True)
        layout.addWidget(self._title_bar)
        layout.addSpacing(6)

        self._workspace_button = QToolButton(sidebar)
        self._workspace_button.setObjectName("navigationItem")
        self._workspace_button.setAutoRaise(False)
        self._workspace_button.setIcon(icon_from_svg_source(_COMPONENTS_ICON, size=16))
        self._workspace_button.setIconSize(QSize(16, 16))
        self._workspace_button.setText(_("Workspace"))
        self._workspace_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._workspace_button.setFixedHeight(_ROW_HEIGHT)
        self._workspace_button.setCheckable(True)
        self._workspace_button.setToolTip(_("Show the left panel"))
        self._workspace_button.toggled.connect(self._toggle_workspace)
        layout.addWidget(self._workspace_button)
        layout.addStretch(1)
        return sidebar

    def _toggle_workspace(self, checked: bool) -> None:
        self._left_content.setVisible(checked)
        self._workspace_button.setToolTip(
            _("Hide the left panel") if checked else _("Show the left panel")
        )

    # ------------------------------------------------------------------
    # Main content: the ChartPanel stand-in — a plain QTabWidget using the
    # same tab-bar language the real chart tabs use.
    # ------------------------------------------------------------------
    def _main_content(self, parent: QWidget) -> QWidget:
        content = QWidget(parent)
        # Same "white page the cards float on" convention the real
        # properties/toolbox pages already use (see macos_native.qss).
        content.setProperty("toolboxPage", True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(SPACING_LOOSE)

        tabs = QTabWidget(content)
        tabs.setTabsClosable(True)
        tabs.setMovable(True)
        for label in (_("Line"), _("Scatter"), _("Bar")):
            tab_page = QWidget(tabs)
            tab_layout = QVBoxLayout(tab_page)
            tab_layout.addWidget(
                QLabel(_('Contenuto di "{label}"').format(label=label), tab_page)
            )
            tab_layout.addStretch(1)
            tabs.addTab(tab_page, label)
        layout.addWidget(tabs, 1)
        return content

    def _componenti_page(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        stdSizeAndlayout(layout)
        layout.addWidget(self._titled_card(page, _("Menu"), self._fill_menu_card))
        layout.addWidget(self._titled_card(page, _("Buttons"), self._fill_buttons_card))
        layout.addWidget(self._titled_card(page, _("Controls"), self._fill_controls_card))
        layout.addStretch(1)
        return page

    def _titled_card(
        self, parent: QWidget, title: str, fill: Callable[[CardFrame], None]
    ) -> QWidget:
        """A section title *above* its card, never inside the card's own
        border - "macOS" / "Schermi" sit outside the grouped box in
        Impostazioni di Sistema, not as a title baked into its frame."""
        wrapper = QWidget(parent)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(SPACING_TIGHT)
        wrapper_layout.addWidget(create_section_title(title, wrapper))

        card = CardFrame(wrapper, f"demo{title}Card")
        fill(card)
        wrapper_layout.addWidget(card)
        return wrapper

    def _fill_menu_card(self, card: CardFrame) -> None:
        """A QMenu opened from a plain button - "menu con voci bianche e
        bordi doppi" was reported against this exact control."""
        layout = card.layout()
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

    def _fill_buttons_card(self, card: CardFrame) -> None:
        layout = card.layout()
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

    def _fill_controls_card(self, card: CardFrame) -> None:
        layout = card.layout()
        combo = QComboBox(card)
        combo.addItems(["Line", "Scatter", "Bar", "Area"])
        layout.addWidget(combo)

        checkbox = QCheckBox(_("Show grid"), card)
        checkbox.setChecked(True)
        layout.addWidget(checkbox)

        checkbox_off = QCheckBox(_("Show legend"), card)
        layout.addWidget(checkbox_off)

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
