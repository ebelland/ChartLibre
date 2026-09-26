"""hide_native_macos_window_title() must never crash the process.

IS_MACOS only checks the OS, not the Qt platform plugin: under "offscreen"
winId() is a synthetic handle, and handing it to objc.objc_object()
segfaulted the interpreter (EXC_BAD_ACCESS in objc_opt_self, from
showEvent). The function checks platformName() == "cocoa" first, and
refuses a null handle; both refusals are pinned here. A regression shows
up as pytest itself dying, not as a failed assertion.
"""
from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from app.styles.style import hide_native_macos_window_title


def test_off_the_native_platform_it_refuses_instead_of_crashing(qapp, monkeypatch) -> None:
    monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda: "offscreen"))
    widget = QWidget()
    try:
        assert hide_native_macos_window_title(widget) is False
    finally:
        widget.deleteLater()


def test_a_zero_window_id_is_refused_before_any_objc_call(qapp, monkeypatch) -> None:
    monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda: "cocoa"))
    widget = QWidget()
    monkeypatch.setattr(type(widget), "winId", lambda self: 0)
    try:
        assert hide_native_macos_window_title(widget) is False
    finally:
        widget.deleteLater()
