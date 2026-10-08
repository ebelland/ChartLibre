"""Text boxes in the property panels apply on Enter or on leaving the box.

Applying on every keystroke redrew the chart while the user typed and
reloaded the form, which took the focus out of the box mid-word.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QVBoxLayout

from app.widgets.chart_properties.base_properties import BaseProperties


class _Panel(BaseProperties):
    def __init__(self) -> None:
        super().__init__()
        self.applied = 0
        layout = QVBoxLayout(self)
        self.line = QLineEdit(self)
        self.sql = QPlainTextEdit(self)
        self.other = QLineEdit(self)
        for widget in (self.line, self.sql, self.other):
            layout.addWidget(widget)
        self._install_auto_apply(self._apply)
        self._apply_on_commit(self.line)
        self._apply_on_commit(self.sql)
        self._figure_id = 1

    def _apply(self) -> None:
        self.applied += 1


@pytest.fixture
def panel(qapp: QApplication):
    widget = _Panel()
    widget.show()
    widget.activateWindow()
    qapp.processEvents()
    yield widget
    widget.close()


def test_typing_does_not_apply_and_keeps_the_focus(qapp: QApplication, panel: _Panel) -> None:
    panel.line.setFocus()
    QTest.keyClicks(panel.line, "Pressure")
    qapp.processEvents()
    QTest.qWait(400)  # longer than the auto-apply delay
    assert panel.applied == 0
    assert panel.line.text() == "Pressure"


def test_enter_applies_once(qapp: QApplication, panel: _Panel) -> None:
    panel.line.setFocus()
    QTest.keyClicks(panel.line, "Pressure")
    QTest.keyClick(panel.line, Qt.Key.Key_Return)
    assert panel.applied == 1
    QTest.keyClick(panel.line, Qt.Key.Key_Return)  # nothing changed since
    assert panel.applied == 1


def test_leaving_a_box_applies_but_only_when_its_text_changed(panel: _Panel) -> None:
    panel.line.setText("set by a reload")  # programmatic: not an edit
    panel._commit_text_edit(panel.line)
    assert panel.applied == 0

    panel.line.setFocus()
    QTest.keyClicks(panel.line, " and typed")
    panel.line.editingFinished.emit()  # what leaving the box sends
    assert panel.applied == 1


def test_a_multi_line_box_applies_on_ctrl_enter_or_when_it_loses_the_focus(
    qapp: QApplication, panel: _Panel
) -> None:
    panel.sql.setFocus()
    QTest.keyClicks(panel.sql, "SELECT x, y")
    QTest.keyClick(panel.sql, Qt.Key.Key_Return)  # a new line, not an apply
    assert panel.applied == 0 and "\n" in panel.sql.toPlainText()

    QTest.keyClick(panel.sql, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert panel.applied == 1

    QTest.keyClicks(panel.sql, " FROM t")
    panel._commit_text_edit(panel.sql)  # what losing the focus does
    assert panel.applied == 2


def test_nothing_is_applied_while_the_form_is_being_reloaded(panel: _Panel) -> None:
    panel.line.setFocus()
    QTest.keyClicks(panel.line, "x")
    with panel._reloading_controls():
        panel._commit_text_edit(panel.line)
    assert panel.applied == 0
