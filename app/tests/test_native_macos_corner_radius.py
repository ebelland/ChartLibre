"""apply_native_macos_corner_radius() must never crash the process.

It used to: IS_MACOS only checks the OS, not the active Qt platform plugin,
so under "offscreen" (this whole test suite's own platform, and any headless
script) window.winId() returns a synthetic handle with no relation to a real
NSView. Handing that to objc.objc_object() and calling .window() on the
result read unowned memory and segfaulted the interpreter - a crash no
Python try/except can catch, reproduced with a real MainWindow under
QT_QPA_PLATFORM=offscreen and confirmed against the resulting macOS crash
report (EXC_BAD_ACCESS / SIGSEGV in objc_opt_self, called from showEvent).

The fix checks QGuiApplication.platformName() == "cocoa" before ever
touching pyobjc, which is false under offscreen. This suite's own qapp
fixture runs under a real "cocoa" platform on this machine (there is a
real display), so the offscreen condition is simulated by monkeypatching
platformName() rather than relied on from the ambient environment - the
same crash reproduces with a real MainWindow under
QT_QPA_PLATFORM=offscreen, which a plain unpatched QWidget here would not
exercise.
"""
from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from app.styles.style import apply_native_macos_corner_radius


def test_it_returns_false_instead_of_crashing_off_the_native_platform(
    qapp, monkeypatch
) -> None:
    monkeypatch.setattr(
        QGuiApplication, "instance", staticmethod(lambda: _FakeOffscreenApp())
    )
    widget = QWidget()
    try:
        # This process is still alive to make this assertion at all - the
        # crash this guards against kills the interpreter outright, so a
        # regression here would show up as pytest itself dying, not as a
        # failed assertion.
        assert apply_native_macos_corner_radius(widget) is False
    finally:
        widget.deleteLater()


class _FakeOffscreenApp:
    @staticmethod
    def platformName() -> str:
        return "offscreen"


def test_a_zero_window_id_is_refused_before_any_objc_call(qapp, monkeypatch) -> None:
    """Belt and braces: even if the platform check above is ever bypassed
    (mocked platformName, a future refactor), a null handle alone must not
    reach objc.objc_object()."""
    widget = QWidget()
    monkeypatch.setattr(
        QGuiApplication, "instance", staticmethod(lambda: _FakeCocoaApp())
    )
    monkeypatch.setattr(type(widget), "winId", lambda self: 0)
    try:
        assert apply_native_macos_corner_radius(widget) is False
    finally:
        widget.deleteLater()


class _FakeCocoaApp:
    @staticmethod
    def platformName() -> str:
        return "cocoa"
