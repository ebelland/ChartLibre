"""Shared pytest fixtures and options for the ChartLibre test suite.

Artifacts (databases, saved figures) go to the directory chosen by
``_artifacts_root``; set ``DHUB_TEST_ARTIFACTS`` to redirect them.
"""
# dev/tests/conftest.py
from __future__ import annotations

import json

from collections.abc import Iterator

import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import pytest

from app.data.sqlite_repo import SqliteRepo

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _operations_compute_in_place(request: pytest.FixtureRequest):
    """Preview and Apply finish before they return, unless a test asks otherwise.

    The operations that compute on a worker thread would otherwise hand
    their results back only once the event loop runs, and every test that
    reads the chart straight after preview() would read it too early. The
    tests of the background path itself are marked ``background``.
    """
    from app.series_operations.dialog_base import SeriesOperationDialogBase

    SeriesOperationDialogBase.BACKGROUND_ENABLED = request.node.get_closest_marker("background") is not None
    yield
    SeriesOperationDialogBase.BACKGROUND_ENABLED = True


@pytest.fixture(autouse=True, scope="session")
def _private_user_config(tmp_path_factory: pytest.TempPathFactory):
    """Point user.json at a copy for the whole run.

    Dialogs and the main window remember their state in user.json when they
    close - splitter sizes, the last connection, the last import - and the
    tests close a great many of them. Against the real file that left the
    developer's own window with its panels 47 px wide after a test run.
    A copy, so tests still start from realistic settings.
    """
    from app.utils import config

    real = config.USER_CONFIG_PATH
    copy = tmp_path_factory.mktemp("user_config") / "user.json"
    if real.exists():
        copy.write_bytes(real.read_bytes())
        # The developer's own view, not a setting: with the rail collapsed
        # (Workspace) every layout test measures a window nobody tests.
        try:
            settings = json.loads(copy.read_text(encoding="utf-8"))
            settings.get("main_window", {}).pop("navigation_compact", None)
            copy.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except (OSError, ValueError, AttributeError):
            pass
    config.USER_CONFIG_PATH = copy
    try:
        yield copy
    finally:
        config.USER_CONFIG_PATH = real


def _artifacts_root() -> Path:
    """Return the directory that receives test artifacts.

    Resolution order:
      1. ``DHUB_TEST_ARTIFACTS`` environment variable, when set;
      2. ``C:/bin/dhub`` on Windows, the fixed location used on the dev machine;
      3. an in-repo ``dev/test_results`` folder;
      4. the system temp directory.

    Why: the previous implementation referenced ``root`` before assignment and
    only worked because the resulting NameError fell through to the fallback.
    """
    candidates: list[Path] = []

    override = os.environ.get("DHUB_TEST_ARTIFACTS", "").strip()
    if override:
        candidates.append(Path(override))
    if sys.platform.startswith("win"):
        candidates.append(Path("C:/bin/dhub"))
    candidates.append(Path(__file__).resolve().parents[1] / "test_results")
    candidates.append(Path(tempfile.gettempdir()) / "dhub_test_results")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Why: a mounted read-only folder passes mkdir(exist_ok=True) but
            # fails on the first write, so probe an actual create/unlink cycle.
            probe = candidate / ".write_probe"
            probe.touch()
            probe.unlink()
        except Exception:
            continue
        return candidate
    raise RuntimeError("No writable directory available for test artifacts")


@pytest.fixture(scope="session")
def test_results_dir() -> Path:
    base = _artifacts_root() / "dhub_tests"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def plots_dir(test_results_dir: Path) -> Path:
    path = test_results_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def tmp_db_path(test_results_dir: Path, request: pytest.FixtureRequest) -> Path:
    name = (request.node.name or "test").replace("/", "_").replace("\\", "_")
    return test_results_dir / f"{name}.dhub"


@pytest.fixture
def repo(tmp_db_path: Path) -> Iterator[SqliteRepo]:
    """A fresh, empty SqliteRepo at this test's own tmp_db_path.

    The base shape most test modules were redefining by hand: open, yield,
    close. ``tmp_db_path`` names its file after the test rather than a new
    temp path every run, so a database - and its WAL/SHM siblings - left
    over from an earlier local run has to be cleared before this test
    builds its own, or a stale table or link from that run leaks into this
    one's assertions. A module that needs more than this - seed data, a
    non-default constructor argument - defines its own ``repo`` fixture,
    which shadows this one for that module only; nothing here changes for
    every module that already does.
    """
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "background: Preview and Apply compute on a worker thread, as in the app"
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    # Keep show-plots available for other tests.
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=True,
        help="Save rendered matplotlib figures to the plots dir",
    )


@pytest.fixture(scope="session")
def show_plots(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--show-plots"))




@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session.

    Qt allows exactly one, and creating a second aborts the process, so every
    test that needs widgets - or only a palette and a device pixel ratio -
    shares this rather than making its own.  Two modules used to define their
    own copy of this fixture, which shadowed it and duplicated the check below.

    Some environments ship a PySide6 whose widget constructors exist but reject
    their arguments; there the whole group is skipped rather than failing one
    test at a time with an error that says nothing about the cause.
    """
    qt = pytest.importorskip("PySide6.QtWidgets")

    try:
        app = qt.QApplication.instance() or qt.QApplication([])
        qt.QLineEdit(qt.QWidget())
    except Exception:  # pragma: no cover - depends on the installed PySide6
        pytest.skip("PySide6 widgets are not usable in this environment")

    yield app


@pytest.fixture(autouse=True, scope="module")
def _close_leftover_windows():
    """After each test module, close and delete the windows it left open.

    Tests build dialogs and panels and rarely destroy them; hundreds piled up
    over a run, and every stylesheet applied later repolished all of them -
    20 s for one apply_platform_style by the time the window-chrome tests ran.
    """
    yield
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    # Deleted, not closed: close() runs closeEvent, where dialogs remember
    # their entries in user.json - and the next module's dialogs would open
    # with this module's choices.
    for widget in app.topLevelWidgets():
        widget.hide()
        widget.deleteLater()
    app.processEvents()
    app.sendPostedEvents(None, 0)  # the DeferredDelete events
    app.processEvents()


@pytest.fixture(autouse=True, scope="session")
def _qt_messages_to_stderr_at_exit():
    """At the end of the run, Qt's messages go back to its own handler.

    Qt warns while the process shuts down (a web page released with its
    profile), after pytest has closed the streams the application's logger
    writes to - which printed "--- Logging error ---" tracebacks after the
    summary for every one of them.
    """
    yield
    from PySide6.QtCore import qInstallMessageHandler

    qInstallMessageHandler(None)
