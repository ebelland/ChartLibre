"""Run a long calculation off the GUI thread and hand the result back to it.

A fit, a spectrum or a clustering on a large series can take seconds; run
on the GUI thread, the window freezes and cannot even be told to stop. The
work here is plain Python and NumPy - nothing touches a widget - so it can
run on a worker thread, and the result comes back through a Qt signal,
which Qt delivers on the thread the receiver lives on: the GUI thread.

Rules, all of them for crash safety:
- the function must not touch any Qt object;
- the task object is created on the GUI thread and is kept alive in
  ``_ACTIVE`` until its result has been delivered, so neither Python nor Qt
  can collect it while the worker is still running;
- connect ``finished``/``failed`` to methods of a QObject (the dialog): if
  that object is destroyed first, Qt drops the connection instead of
  calling into a deleted widget.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from app.logs.logger import applogger


class Cancelled(Exception):
    """Raised inside a task's function when ``cancel()`` was asked for."""


class _Runner(QRunnable):
    def __init__(self, task: "BackgroundTask") -> None:
        super().__init__()
        self._task = task
        # Deleted by us, not by the pool: the wrapper is held in the task.
        self.setAutoDelete(False)

    def run(self) -> None:  # worker thread
        task = self._task
        try:
            result = task.function(task.cancel_event)
        except BaseException as exc:  # noqa: BLE001 - reported to the GUI thread
            task.failed.emit(exc)
        else:
            task.finished.emit(result)


class BackgroundTask(QObject):
    """One calculation on the thread pool.

    ``function`` receives a ``threading.Event``; a long loop checks it (or
    raises :class:`Cancelled` when it is set) so Stop takes effect.
    """

    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, function: Callable[[threading.Event], Any]) -> None:
        super().__init__()
        self.function = function
        self.cancel_event = threading.Event()
        self._runner = _Runner(self)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        # Connected here, after the caller's receivers: slots run in the
        # order they were connected, so the task is released only once the
        # result has been handed over. Delivered on the GUI thread, which is
        # the thread this object lives on.
        self.finished.connect(self._release)
        self.failed.connect(self._release)
        self._running = True
        _ACTIVE.add(self)
        QThreadPool.globalInstance().start(self._runner)

    def cancel(self) -> None:
        """Ask the function to stop at its next check."""
        self.cancel_event.set()

    @Slot(object)
    def _release(self, _payload: object) -> None:
        self._running = False
        _ACTIVE.discard(self)


#: Tasks still running or not yet delivered. See the module docstring.
_ACTIVE: set[BackgroundTask] = set()


def run_in_background(
    function: Callable[[threading.Event], Any],
    on_finished: Callable[[Any], None],
    on_failed: Callable[[BaseException], None],
) -> BackgroundTask:
    """Start *function* on the thread pool and return its task."""
    task = BackgroundTask(function)
    task.finished.connect(on_finished)
    task.failed.connect(on_failed)
    task.start()
    applogger.debug("Background task started: %s", getattr(function, "__name__", function))
    return task
