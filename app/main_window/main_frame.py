"""The main window's frame: its client area, its edges, and files dropped on it.

On Windows the window is frameless (the caption strip is drawn by
CustomTitleBar), so its edges are made resizable here by hand; on every
platform a project or a data file dropped on the window is opened or
imported. A mixin of MainWindow.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QMainWindow, QSizePolicy, QSplitter, QVBoxLayout, QWidget

from app.dialogs.import_data_dialog import ImportDataDialog
from app.logs.logger import applogger
from app.styles.style import IS_WINDOWS
from app.styles.style import IS_MACOS, stdSizeAndlayout
from app.utils.data_sources import is_importable
from app.utils.messages import show_message
from app.main_window.custom_title_bar import CustomTitleBar


if TYPE_CHECKING:
    from app.data.sqlite_repo import SqliteRepo

    class _MainWindowFrameBase(QMainWindow):
        """Static contract supplied by the composed MainWindow class."""

        _main_split: QSplitter
        _central_host: QWidget
        _repo: SqliteRepo
        _table_panel: Any
        _own_title_bar: CustomTitleBar | None

        def _switch_database(self, db_path: Path) -> None: ...
        def _reload_tabs(self, select_figure_id: int | None = None) -> None: ...
else:
    class _MainWindowFrameBase:
        """Runtime-neutral base: MainWindow supplies the actual Qt base."""


class MainWindowFrame(_MainWindowFrameBase):
    """A part of MainWindow; ``self`` is the window."""

    #: How close to an edge, in pixels, counts as "grab this edge to resize".
    _RESIZE_MARGIN: int = 6

    #: Cursor shape for each edge/corner combination _resize_edge_at can
    #: return. Absent from here (the interior, or a maximized window) means
    #: the ordinary arrow.
    _RESIZE_CURSORS: dict[Qt.Edge, Qt.CursorShape] = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.TopEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.TopEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeBDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeBDiagCursor,
    }

    def _create_central_host(self) -> QWidget:
        """Wrap the main splitter in a neutral central widget.

        This prevents the top-level QMainWindow from taking an over-constrained
        size hint directly from the splitter tree.
        """
        host = QWidget(self)
        host.setMinimumSize(0, 0)
        host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(host)
        stdSizeAndlayout(layout)
        if IS_MACOS:
            # Flush to the window's own edge: #leftPanelCard and
            # #activityRail draw their own hairlines, and the NSWindow its
            # own border and corners. The traffic lights sit in the rail's
            # title strip.
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        else:
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)
        if IS_WINDOWS or IS_MACOS:
            # FramelessWindowHint (see __init__) strips every bit of native
            # chrome - border, corner, drop shadow - so without this the
            # window has no visible edge at all against whatever is behind
            # it. Each platform QSS draws a plain 1px outline on this object
            # name (fluent_win11.qss, macos_native.qss); WA_StyledBackground
            # is what makes a QWidget paint a QSS border at all rather than
            # silently ignoring it. Keep this a plain fill with no border
            # radius of its own on macOS: the window's real corners are
            # rounded by its NSWindow layer (apply_native_macos_corner_
            # radius), and a second curve painted here never lands on the
            # same pixels as that one - which is the doubled edge along the
            # top that the strip below used to show.
            #
            # No drop shadow here, unlike the elevated internal cards this
            # replaced: a shadow effect needs room *outside* the widget it
            # is attached to in order to render, and host fills the client
            # area exactly - the OS window edge sits at host's own edge, so
            # the shadow's bleed would just be clipped away. Giving it that
            # room means making the window translucent and inset from its
            # real (now larger, transparent) bounds, which also strands
            # QMainWindow's own statusBar() - built by Qt outside
            # centralWidget entirely - outside the inset border with no
            # background of its own. Worth doing, not safely from here
            # without a way to check it on an actual Windows/Mac build.
            host.setObjectName("windowFrame")
            host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            if IS_MACOS:
                # The NSWindow draws its own edge and corners now; this only
                # paints the fill beneath the translucent rail.
                host.setProperty("nativeRounded", True)

        if IS_WINDOWS:
            # Windows only: a caption strip across the whole window, which
            # is what a Windows 11 window has - icon and title on the left,
            # min/max/close on the right, all of it draggable.
            #
            # macOS deliberately has none. Its traffic lights live at the
            # top of the navigation rail instead (NavigationBar.__init__),
            # so each column keeps its own background all the way to the
            # window's top edge - grey for the rail, white for the panels
            # beside it - the way Finder and System Settings look. A strip
            # here would cut across all of them.
            self._own_title_bar = CustomTitleBar(cast(Any, self), is_macos=False)
            layout.addWidget(self._own_title_bar, 0)
        layout.addWidget(self._main_split, 1)
        return host

    def _resize_edge_at(self, pos: QPoint) -> Qt.Edge:
        """Return which edge(s) of _central_host *pos* is within the margin of.

        *pos* is in _central_host's own coordinates - the widget
        eventFilter below watches, not the window's (self never sees a
        mouse event of its own to filter; see __init__'s comment on why).
        """
        rect = self._central_host.rect()
        margin = self._RESIZE_MARGIN
        edges = Qt.Edge(0)
        if pos.x() <= margin:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= rect.width() - margin:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= margin:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= rect.height() - margin:
            edges |= Qt.Edge.BottomEdge
        return edges

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Give the frameless Windows window edge-drag resizing.

        FramelessWindowHint (see __init__) leaves the OS with no resize
        handles of its own to offer - this is the replacement, the same
        margin-hit-test-plus-startSystemResize technique
        CustomTitleBar.mousePressEvent already uses for the move case.
        A maximized window is left alone: its edges are the screen's own
        edges, and dragging those would resize the display area a user
        grabbing what looks like a window border did not mean to touch.
        """
        if not (
            IS_WINDOWS
            and watched is self._central_host
            and isinstance(event, QMouseEvent)
            and not self.isMaximized()
        ):
            return super().eventFilter(watched, event)

        edges = self._resize_edge_at(event.position().toPoint())
        if event.type() == QEvent.Type.MouseMove:
            cursor = self._RESIZE_CURSORS.get(edges, Qt.CursorShape.ArrowCursor)
            self._central_host.setCursor(cursor)
        elif (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and edges
        ):
            handle = self.windowHandle()
            if handle is not None:
                handle.startSystemResize(edges)
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def dropped_paths(event: Any) -> list[Path]:
        """Return the local files carried by a drag, in the order dropped.

        Local files only: a drag from a browser carries a URL that names a
        file on a web server, and ``toLocalFile`` returns an empty string for
        it rather than something ``open()`` would fail on later.
        """
        data = event.mimeData()
        if data is None or not data.hasUrls():
            return []
        return [
            Path(local)
            for url in data.urls()
            if (local := url.toLocalFile())
        ]

    @classmethod
    def accepted_drop(cls, paths: list[Path]) -> tuple[str, list[Path]]:
        """Classify a drop as ``("database"|"import"|"", paths)``.

        A ``.dhub`` is a project to open, anything the import readers know is
        data to import, and everything else is refused - refused *before* the
        cursor changes, so the window says no by not offering to accept it
        rather than by a box after the fact.

        A database wins over data files dropped with it, and only one at a
        time: opening two projects at once has no meaning, and importing into
        a database that is about to be closed has less.
        """
        databases = [path for path in paths if path.suffix.lower() == ".dhub"]
        if len(databases) == 1:
            return "database", databases

        importable = [path for path in paths if is_importable(path)]
        if importable and not databases:
            return "import", importable

        return "", []

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        """Accept a drag only when the drop would actually do something."""
        kind, _paths = self.accepted_drop(self.dropped_paths(event))
        if kind:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:  # noqa: N802
        """Keep the acceptance while the cursor moves over the window."""
        self.dragEnterEvent(event)

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        """Open a dropped project, or import dropped data files.

        Several data files are handled one dialog at a time, in the order they
        were dropped, and cancelling one stops the rest: cancel means "not
        this", and the natural reading of it on the second of four files is
        "stop", not "carry on with the next one".
        """
        kind, paths = self.accepted_drop(self.dropped_paths(event))
        if not kind:
            event.ignore()
            return

        event.acceptProposedAction()

        if kind == "database":
            applogger.info("Opening dropped database: %s", paths[0])
            try:
                self._switch_database(paths[0])
            except Exception as exc:  # noqa: BLE001
                applogger.exception("Failed to open the dropped database: %s", exc)
                show_message(self, "database.open_failed", error=exc)
            return

        for path in paths:
            applogger.info("Importing dropped file: %s", path)
            if not self._import_one_dropped_file(path):
                break

    def _import_one_dropped_file(self, path: Path) -> bool:
        """Import one dropped file; False when the user cancelled."""
        dialog = ImportDataDialog(self._repo, parent=self)
        dialog.load_file(path)
        if not dialog.exec():
            return False
        self._table_panel.reload()
        self._reload_tabs()
        return True
