"""Regenerate the user manual's figures, in the macOS native look.

Run it from the project root::

    python docs/manual/make_screenshots.py

and it rewrites the ``screenshot_*.png`` files beside ``user_manual.typ``.
Nothing else reads them; the manual does, by name.

Why a script rather than a person with a screen-capture key: the figures
had drifted several versions behind the application, which is what happens
to a screenshot taken by hand - there is no way to tell a stale one from a
current one by looking at it, and no cheap way to redo the set after a UI
change. This takes about a minute and is the same every time.

**Run it on a Mac when you can.** It forces ``macos_native.qss`` and the
macOS layout branches on whatever host it runs on, so it produces the right
*structure* anywhere - the sidebar rows, the traffic lights, the cards. Two
things it cannot conjure off a Mac:

* *SF Symbols*, which need AppKit through pyobjc. The macOS nav rail draws
  five of its six icons from its own inline SVG (identical everywhere), so
  only "Series Operations" is affected - it falls back to a freedesktop
  theme icon, or to nothing where no icon theme is installed.
* *The system font*. macos_native.qss names no font-family at all, so the
  text is whatever Qt calls the UI font: SF Pro on a Mac, and typically
  DejaVu Sans elsewhere, which is wide enough to elide labels that fit on
  a Mac. Off a Mac this substitutes Inter, the closest widely-packaged
  face to SF Pro's metrics, so the figures show the eliding the
  application really has rather than the font's.

Everything else - layout, colour, spacing, the charts themselves - is the
real application, driven through its real entry points.

On a headless Linux box it needs a display for Matplotlib's Qt backend and
an offscreen platform for Qt itself::

    QT_QPA_PLATFORM=offscreen xvfb-run -a python docs/manual/make_screenshots.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Before QApplication exists, or it is read too late to matter. The figures
# the manual has always carried were 2x Retina captures (2400x1600 for a
# 1200x800 window), and an A4 page at 100% width is about 280 dpi of that -
# a 1x grab prints visibly soft. On a Mac with a Retina display this is
# already what the screen does; this makes every host match it.
import os  # noqa: E402

os.environ.setdefault("QT_SCALE_FACTOR", "2")

# Before *anything* imports these names: the macOS branches are chosen by
# module-level constants read at import time, so flipping them afterwards
# would leave half the window built the other way.
import app.styles.style as style  # noqa: E402

style._IS_MACOS = True
style.IS_MACOS = True

import platform  # noqa: E402

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

import app.dialogs.main_window as main_window_module  # noqa: E402
from app.data.demo_project import DEMO_PROJECTS, copy_demo_project  # noqa: E402
from app.data.sqlite_repo import SqliteRepo  # noqa: E402
from app.logs.logger import applogger  # noqa: E402

main_window_module.IS_MACOS = True

OUTPUT_DIR = Path(__file__).resolve().parent

#: Logical size. QT_SCALE_FACTOR above makes the saved file twice this, so
#: this is the window as a person would size it, not the pixel count.
#: 1440x900 is a MacBook's own default scaled resolution, and it is the
#: narrowest that shows the table list's four columns without pushing them
#: under a horizontal scrollbar.
WINDOW_SIZE = (1440, 900)

#: The demo to open. "Getting started" carries a table for every chart
#: family, so the table list looks like a real project rather than an empty
#: one.
DEMO_INDEX = 0

#: Which of that demo's ~50 figures to show. Matched on the tab's own text.
#: Wanted: one axis (a multi-axis figure shrinks to nothing once it is
#: fitted to the panel) and a title short enough not to be elided, so the
#: figure shows a chart rather than a demonstration of clipping.
COVER_FIGURE = "Defect causes"


#: What to stand in for the macOS system font off a Mac, best first.
#: macos_native.qss names no font-family at all - it takes whatever Qt
#: calls the UI font, which on macOS is SF Pro. Elsewhere that is usually
#: DejaVu Sans, which is a good deal wider, and the difference is not
#: cosmetic here: the operation tiles and the nav rail size their text to
#: fit SF Pro, so a wider face elides them ("Smoothing" as "moothing") and
#: the figure then documents a clipping bug that macOS users do not have.
#: Inter is the closest widely-packaged face to SF Pro's metrics.
_SF_PRO_STAND_INS = ("Inter", "Helvetica Neue", "Liberation Sans")


def _use_a_mac_like_font(app: QApplication) -> None:
    """On a Mac, leave the system font alone; elsewhere, get close to it."""
    if platform.system() == "Darwin":
        return
    families = set(QFontDatabase.families())
    for family in _SF_PRO_STAND_INS:
        if family in families:
            font = QFont(app.font())
            font.setFamily(family)
            app.setFont(font)
            print(f"  (substituting {family} for the macOS system font)")
            return
    print("  ! no SF Pro stand-in installed; labels may be elided")


def _settle(app: QApplication, rounds: int = 12) -> None:
    """Let deferred work finish before the pixels are read.

    ChartPanel renders on a timer rather than during construction, so a grab
    taken too early catches an empty canvas.
    """
    for _unused in range(rounds):
        app.processEvents()


def _quieten_status_bar(window) -> None:
    """Stop the log writing into the status bar, and leave it idle.

    AppLogger routes every warning at or above INFO to the status bar, so
    whatever happened to be logged while the window was building would
    otherwise be frozen into the figure - including messages about the
    offscreen platform plugin that say nothing about the application.
    """
    applogger.set_status_bar(None)
    window.statusBar().clearMessage()
    window._set_status_state("normal")


def _hide_in_window_menu_bar(window) -> None:
    """Drop the menu bar from the window itself.

    On a real Mac the menu bar is the system's, at the top of the screen,
    and is never drawn inside the window. Qt only falls back to an in-window
    bar because this is not macOS - leaving it in would put a strip in the
    figure that no macOS user ever sees.
    """
    bar = window.menuBar()
    if bar is not None and not bar.isNativeMenuBar():
        bar.setVisible(False)


def _open_demo(tmp_dir: Path):
    demo = DEMO_PROJECTS[DEMO_INDEX]
    path = copy_demo_project(demo, tmp_dir / f"{demo.file_name}.dhub")
    repo = SqliteRepo(db_path=path)
    window = main_window_module.MainWindow(repo=repo, db_path=path)
    window.resize(*WINDOW_SIZE)
    window.show()
    return window, repo


def _grab(app: QApplication, widget: QWidget, name: str) -> Path:
    _settle(app)
    target = OUTPUT_DIR / name
    if not widget.grab().save(str(target)):
        raise RuntimeError(f"could not write {target}")
    print(f"  wrote {target.relative_to(PROJECT_ROOT)}")
    return target


def main() -> int:
    app = QApplication(sys.argv)
    style.ensure_icon_theme()
    style.apply_platform_style(app, preference="macos_native")
    _use_a_mac_like_font(app)

    scratch = tempfile.TemporaryDirectory(prefix="chartlibre-shots-")
    window, repo = _open_demo(Path(scratch.name))
    _hide_in_window_menu_bar(window)
    _settle(app)
    _quieten_status_bar(window)

    print("Writing figures:")

    # 1. The main window, on the Tables page: the list, a preview and a
    #    finished chart, which is what the manual's own caption promises.
    window._set_nav_index(0)
    _lay_out_data_page(window)
    _select_figure(window, COVER_FIGURE)
    _settle(app)
    _fit_chart(window)
    _settle(app)
    _quieten_status_bar(window)
    _grab(app, window, "screenshot_main_window.png")

    # 2. Chart Options: the figure/axis/series property panels.
    window._set_nav_index(1)
    _settle(app)
    _quieten_status_bar(window)
    _grab(app, window, "screenshot_chart_options.png")

    # 3. Series Operations: the grid of operations and its hint bar.
    window._set_nav_index(2)
    _settle(app)
    _quieten_status_bar(window)
    _grab(app, window, "screenshot_series_operations.png")

    window.close()
    repo.close()
    applogger.set_status_bar(None)
    scratch.cleanup()

    # The window is gone; nothing below should be kept alive by a timer.
    QTimer.singleShot(0, app.quit)
    return 0


def _select_figure(window, needle: str) -> bool:
    """Switch the chart panel to the first tab whose name contains *needle*."""
    tabs = window._tabs
    for index in range(tabs.count()):
        if needle.casefold() in tabs.tabText(index).casefold():
            tabs.setCurrentIndex(index)
            return True
    print(f"  ! no chart tab matching {needle!r}; keeping the first")
    return False


def _fit_chart(window) -> None:
    """Zoom the visible chart to its viewport, as the toolbar's own button does.

    A figure is drawn at a fixed pixel size and shown at 100% by default, so
    one wider than the panel it landed in is simply cut off at the edge -
    which in a figure looks like a rendering fault rather than a scroll
    position. This is the application's own "Zoom to fit", called rather
    than reimplemented.
    """
    panel = window._tabs.currentWidget()
    fit = getattr(panel, "_zoom_best_fit", None)
    if not callable(fit):
        return
    # Twice, with the layout settled in between. The first call sizes the
    # figure to the viewport as it is *now*; that can be wide enough to
    # bring up a vertical scrollbar, which takes width back off the
    # viewport and leaves the figure overflowing by exactly a scrollbar.
    # The second call sees the viewport it actually ended up with.
    fit()
    app = QApplication.instance()
    if app is not None:
        _settle(app)
    fit()

    # Then back off a little from the exact fit. The canvas is blitted from
    # a buffer resized on its own timer, and a figure sized to the viewport
    # to the pixel leaves its own edge sitting exactly where a stale slice
    # of that buffer shows through - which lands in the figure as a torn
    # fragment down the right-hand side. A few percent of margin puts the
    # drawing clear of that edge, and reads as a chart in a panel rather
    # than a chart wedged into one.
    current = getattr(panel, "_zoom_percent", None)
    setter = getattr(panel, "set_zoom_percent", None)
    if isinstance(current, int) and callable(setter):
        setter(int(current * 0.86))

    sync = getattr(panel, "_sync_canvas_geometry", None)
    if callable(sync):
        sync(redraw=True)
    if app is not None:
        _settle(app)
    panel.repaint()


def _lay_out_data_page(window) -> None:
    """Put the left column where the manual's caption says it is.

    Two separate splitters decide this and both default to whatever the
    machine that last saved a layout happened to use:

    * ``_main_split`` (rail+panel | charts). Too narrow and the table
      list's four columns spill under a horizontal scrollbar, so the figure
      shows a clipped "attribute_chart..." and a header reading ".in" -
      which reads as a bug in the application rather than as a splitter
      position.
    * ``_data_split`` (table list above | data preview below). Collapsed,
      the figure shows no preview at all, and the caption promising one is
      then simply wrong.
    """
    # The table list's own columns are 180 + 40 + 240 + 260 logical pixels
    # wide (see TableListPanel._configure_headers), so anything under ~740
    # puts the last of them under a scrollbar.
    total = max(window.width(), 1)
    left = max(int(total * 0.52), 740)
    window._main_split.setSizes([left, max(total - left, 1)])

    height = max(window._data_split.height(), 1)
    window._data_split.setSizes([int(height * 0.45), height - int(height * 0.45)])


if __name__ == "__main__":
    raise SystemExit(main())
