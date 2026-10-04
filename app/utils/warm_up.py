"""Import the scientific libraries ahead of time, off the main thread.

The first series operation of a session used to take two to three seconds
to open: SciPy's signal and optimisation modules, scikit-learn, statsmodels,
scikit-image and PyWavelets are imported when an operation's module first
loads, and that is a second or more of a frozen window each. Imported here
instead, in a background thread a few seconds after the window appears, so
by the time an operation is opened they are already in memory.

Only libraries with no Qt in them: a module that creates Qt objects must be
imported on the main thread. A failure is logged and otherwise ignored - the
operation that needs the library imports it again, and reports the error
where it belongs.
"""
from __future__ import annotations

import importlib
import threading

from app.logs.logger import applogger

#: The modules the series operations import, slowest first.
MODULES: tuple[str, ...] = (
    "sklearn.ensemble",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.manifold",
    "sklearn.gaussian_process",
    "sklearn.preprocessing",
    "sklearn.linear_model",
    "sklearn.isotonic",
    "sklearn.neighbors",
    "sklearn.svm",
    "sklearn.covariance",
    "statsmodels.api",
    "statsmodels.formula.api",
    "scipy.signal",
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.stats",
    "scipy.cluster.hierarchy",
    "skimage.restoration",
    "pywt",
)

_started = threading.Event()


def _import_all() -> None:
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reported, then left to the operation that needs it
            applogger.debug("Warm-up: %s could not be imported: %s", name, exc)


def warm_up_in_background() -> threading.Thread | None:
    """Start importing MODULES in a daemon thread, once per process."""
    if _started.is_set():
        return None
    _started.set()
    thread = threading.Thread(target=_import_all, name="chartlibre-warm-up", daemon=True)
    thread.start()
    return thread
