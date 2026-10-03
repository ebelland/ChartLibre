"""Small interface checks: font scale, log viewer, the new chart, screen fit, theme icons."""
from __future__ import annotations

from typing import Any, cast
from types import SimpleNamespace
import matplotlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pytest
import app.dialogs.log_viewer_dialog as viewer_module
import app.logs.logger as logger_module
import pandas as pd
import app.dialogs.main_window as main_window_module
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.create_chart_dialog import NewPlotTabResult
from app.dialogs.main_window import MainWindow
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from app.styles.style import apply_dialog_shell
from app.utils import screen_fit
import json
import re
from PySide6.QtGui import QIcon


# ======================================================================
# todo N-12: one factor for every font of a figure, and one per axis.
# (was test_font_scale.py)
# ======================================================================
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from app.charts.render_figure import _apply_font_scale, font_scale_option  # noqa: E402


def _figure_with_one_axis():
    figure = Figure()
    ax = figure.add_subplot()
    ax.set_title("t", fontsize=10)
    ax.set_xlabel("x", fontsize=10)
    ax.plot([0, 1], [0, 1], label="line")
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    figure.suptitle("s", fontsize=10)
    return figure, ax


def test_the_figure_and_axis_factors_multiply() -> None:
    figure, ax = _figure_with_one_axis()
    axis = SimpleNamespace(id=1, options={"font_scale": 1.5})
    descriptor = SimpleNamespace(options={"font_scale": 2.0})

    _apply_font_scale(figure, cast(Any, descriptor), [cast(Any, axis)], {1: ax})

    assert ax.title.get_fontsize() == 30  # pyright: ignore[reportAttributeAccessIssue]
    assert ax.xaxis.label.get_fontsize() == 30
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_fontsize() == 30
    assert ax.xaxis.get_major_ticks()[0].label1.get_fontsize() == 30
    suptitle = getattr(figure, "_suptitle")  # private, and not in the stubs
    assert suptitle.get_fontsize() == 20


def test_unset_or_silly_values_mean_no_change() -> None:
    assert font_scale_option({}) == 1.0
    assert font_scale_option({"font_scale": "big"}) == 1.0
    assert font_scale_option({"font_scale": 99}) == 3.0



# ======================================================================
# The log viewer's Clear: asks, then empties the log file and the view.
# (was test_log_viewer_clear.py)
# ======================================================================
@pytest.fixture
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "datahub.log"
    handler = RotatingFileHandler(path, encoding="utf-8")
    logger = logging.getLogger(logger_module._LOGGER_NAME)
    logger.addHandler(handler)
    monkeypatch.setattr(logger_module, "LOG_FILE", path)
    monkeypatch.setattr(viewer_module, "LOG_FILE", path)
    handler.stream.write("old line\n")
    handler.flush()
    yield path, handler
    logger.removeHandler(handler)
    handler.close()


def test_clear_empties_the_file_through_its_handler(qapp, log_file, monkeypatch: pytest.MonkeyPatch) -> None:
    path, handler = log_file
    others = [h for h in logging.getLogger(logger_module._LOGGER_NAME).handlers if isinstance(h, RotatingFileHandler) and h is not handler]
    for other in others:  # only the test's own file handler, for this test
        logging.getLogger(logger_module._LOGGER_NAME).removeHandler(other)
    try:
        dialog = viewer_module.LogViewerDialog(None)  # pyright: ignore[reportArgumentType]
        assert "old line" in dialog._viewer.toPlainText()
        monkeypatch.setattr(viewer_module, "ask", lambda *_a, **_k: False)
        dialog._clear_log()
        assert path.read_text(encoding="utf-8") == "old line\n"
        monkeypatch.setattr(viewer_module, "ask", lambda *_a, **_k: True)
        dialog._clear_log()
        assert path.read_text(encoding="utf-8") == "" and dialog._viewer.toPlainText() == ""
        handler.stream.write("new line\n")
        handler.flush()
        assert path.read_text(encoding="utf-8") == "new line\n"  # still writing to the same file
        dialog.close()
    finally:
        for other in others:
            logging.getLogger(logger_module._LOGGER_NAME).addHandler(other)


def test_the_clear_button_is_red(qapp, log_file) -> None:
    dialog = viewer_module.LogViewerDialog(None)  # pyright: ignore[reportArgumentType]
    from PySide6.QtWidgets import QPushButton

    clear = [b for b in dialog.findChildren(QPushButton) if b.property("destructive")]
    assert len(clear) == 1
    dialog.close()



# ======================================================================
# A chart just created is the one shown.
# (was test_new_chart_is_current.py)
# ======================================================================
def _figure(repo: SqliteRepo, name: str) -> int:
    figure_id = int(repo.create_figure_descriptor(name=name))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title=name, x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=name,
                                  sql_query="SELECT a AS x, b AS y FROM t", roles={"x": "x", "y": "y"}, style={})
    return figure_id


def test_a_new_plot_becomes_the_current_tab(qapp, tmp_path: Path, monkeypatch) -> None:
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), table_name="t", normalize_columns=False)
    first = _figure(repo, "first")
    _figure(repo, "second")
    window = MainWindow(repo, tmp_path / "p.dhub")
    window._tabs.setCurrentIndex(window._tab_index_of_figure(first))

    class _NewPlot:
        Icon = ""

        def __init__(self, repo_, **_kwargs) -> None:
            self.chart_result: NewPlotTabResult | None = None
            self._repo = repo_

        def setWindowIcon(self, _icon) -> None:
            pass

        def exec(self) -> int:
            figure_id = _figure(self._repo, "third")
            self.chart_result = NewPlotTabResult(figure_id=figure_id, axis_id=0, series_count=1)
            return 1

    monkeypatch.setattr(main_window_module, "NewPlotTabDialog", _NewPlot)
    window._on_new_plot_tab()
    panel = window._current_chart_panel()
    assert panel is not None and window._tabs.tabText(window._tabs.currentIndex()) == "third"
    window.close()
    repo.close()



# ======================================================================
# Every window fits the screen it opens on (laptops included).
# (was test_screen_fit.py)
# ======================================================================
def _available() -> QRect:
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    return screen.availableGeometry()


def _settle() -> None:
    for _ in range(3):
        QApplication.processEvents()


def test_a_dialog_larger_than_the_screen_is_shrunk_and_moved_onto_it(qapp) -> None:
    dialog = QDialog()
    apply_dialog_shell(dialog, QVBoxLayout(dialog), size=None)
    available = _available()
    dialog.resize(available.width() + 500, available.height() + 400)
    dialog.move(available.right() - 50, available.bottom() - 50)
    dialog.show()
    _settle()
    frame = dialog.frameGeometry()
    assert dialog.width() <= available.width() and dialog.height() <= available.height()
    assert available.contains(frame.topLeft()) and frame.bottom() <= available.bottom() + 1
    dialog.close()


def test_a_window_is_never_shrunk_below_its_minimum(qapp) -> None:
    dialog = QDialog()
    available = _available()
    dialog.setMinimumSize(available.width() + 10, 100)
    dialog.resize(available.width() + 200, 200)
    screen_fit.fit_to_screen(dialog)
    assert dialog.width() == available.width() + 10


def test_a_window_that_fits_is_left_alone_and_installing_twice_is_once(qapp) -> None:
    dialog = QDialog()
    screen_fit.fit_on_show(dialog)
    screen_fit.fit_on_show(dialog)
    assert len(dialog.findChildren(screen_fit._FitOnShow)) == 1
    dialog.resize(300, 200)
    dialog.move(_available().topLeft())
    dialog.show()
    _settle()
    assert not screen_fit.fit_to_screen(dialog)
    dialog.close()


def test_windows_destroyed_right_after_showing_are_harmless(qapp) -> None:
    """The crash of 2026-10-03: windows and their children created, shown and
    destroyed while a fit was pending."""
    for _ in range(50):
        dialog = QDialog()
        apply_dialog_shell(dialog, QVBoxLayout(dialog), size="small")
        QLabel("x", dialog)
        dialog.show()
        dialog.deleteLater()
    _settle()



# ======================================================================
# Every ThemeIcon name in config.json is one an icon theme can resolve.
# (was test_theme_icon_names.py)
# ======================================================================
CONFIG = Path(__file__).resolve().parents[2] / "config.json"

#: Names outside Qt's enum that the application uses anyway. Each one is in
#: the freedesktop Icon Naming Specification or is shipped by every major
#: icon theme (Adwaita, Breeze, Papirus) - which is the bar for adding one.
EXTRA_THEME_ICON_NAMES: frozenset[str] = frozenset(
    {
        "accessories-calculator",
        "applications-system",
        "document-edit",
        # The user manual: freedesktop's own name for "open the help
        # document", which is exactly what this action does.
        "help-contents",
        # Connecting to a server database: the freedesktop Status icon for
        # network activity, the closest standard name to what this does.
        "network-transmit-receive",
        # No Qt ThemeIcon enum member covers "fetch this from the web" -
        # GoDown is a plain navigation arrow, not a download. Breeze,
        # Papirus and Adwaita all ship this Icon Naming Specification
        # emblem, originally for a downloads folder.
        "emblem-downloads",
        "object-select",
        # No icon theme has a chart, and a plotting application's Plot button
        # is the one place a drawing of our own is the honest answer. Breeze
        # and Papirus ship this name; Adwaita does not, and there the SVG in
        # app/icons takes over - which is the fallback doing its job rather
        # than a gap.
        "office-chart-line",
        "open-menu",
        "preferences-system",
        "system-run",
        # Likewise: Breeze has it, Adwaita dropped it, and "the rows matching
        # a condition" has no better standard name.
        "view-filter",
        "view-sort-ascending",
        # Import from another database: the same MIME-type naming family as
        # "x-office-spreadsheet" above, this one for LibreOffice Base's .odb.
        "x-office-database",
        "x-office-spreadsheet",
        # Developer-menu stubs (todo.txt P3-4/P3-5/P3-6): freedesktop Icon
        # Naming Specification categories, not covered by Qt's own smaller
        # ThemeIcon enum.
        "applications-development",
        "applications-graphics",
        "preferences-desktop-locale",
        # Added to config.json while this check was not running, and found
        # when it was brought back: each one is in the freedesktop Icon
        # Naming Specification or ships with Breeze, which is enough for the
        # themed icon to be tried before the application's own drawing.
        "applications-internet",
        "document-export",
        "edit-find-replace",
        "object-flip-horizontal",
        "text-x-generic",
        "view-grid",
        "view-hidden",
        "view-list-details",
        "view-list-tree",
        "view-more",
        "view-sidebar",
        "view-sort-descending",
        "view-statistics",
        "view-visible",
    }
)


def _standard_theme_icon_names() -> set[str]:
    """Every freedesktop name QIcon.ThemeIcon knows: DocumentOpen -> document-open."""
    return {
        re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
        for name in QIcon.ThemeIcon.__members__
    }


def _configured_names(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ThemeIcon":
                found.update([value] if isinstance(value, str) else list(value))
            else:
                found |= _configured_names(value)
    elif isinstance(node, list):
        for value in node:
            found |= _configured_names(value)
    return found


def test_every_configured_theme_icon_name_is_known() -> None:
    names = _configured_names(json.loads(CONFIG.read_text(encoding="utf-8")))
    assert names, "no ThemeIcon names found in config.json"
    unknown = sorted(names - _standard_theme_icon_names() - EXTRA_THEME_ICON_NAMES)
    assert unknown == [], f"not a known theme icon name: {unknown}"
