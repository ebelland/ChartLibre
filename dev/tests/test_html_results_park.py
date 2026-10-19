"""A closed window's web view is deleted by the event loop, not by Python's collector later.

The collector can run inside Chromium's own call into Python (a page's
acceptNavigationRequest); a web view destroyed there aborts the process
(SIGTRAP in ~WebContentsAdapter) - it did in the full test run.
"""
from __future__ import annotations

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout

from app.widgets import html_results
from app.widgets.html_results import HtmlResultsView


def test_hidden_window_gives_up_its_web_view_and_gets_a_new_one_when_shown(qapp) -> None:
    dialog = QDialog()
    QVBoxLayout(dialog).addWidget(pane := HtmlResultsView(dialog))
    if not hasattr(pane._viewer, "page"):
        pytest.skip("QtWebEngine is not installed: QTextBrowser needs none of this")
    pane.setHtml("<p>report</p>")
    dialog.show()
    qapp.processEvents()
    old = pane._viewer

    dialog.hide()
    assert pane._parked and old.parent() is None and old in html_results._PARKED
    QCoreApplication.sendPostedEvents(None, int(QEvent.Type.DeferredDelete))
    assert not shiboken6.isValid(old)

    pane.setHtml("<p>later</p>")  # while parked: kept for the next view
    dialog.show()
    qapp.processEvents()
    assert not pane._parked and pane._viewer is not old and pane.text() == "<p>later</p>"
    dialog.close()
