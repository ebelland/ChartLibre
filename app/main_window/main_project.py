"""The project's side of the main window: the file it is, and what is done to it as a whole.

New, Open, Open recent, Load demo, Save and Save as; switching the window
to another database; Import; the project's info, report and history;
Optimize; and Undo, with the snapshots it goes back to. A mixin of
MainWindow: the menus (main_menus.py) and the File panel call these.
"""
from __future__ import annotations

import gc
import re
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable, cast

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QWidget

from app.data.demos import PROJECTS_DIR, copy_demo_project
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.connect_database_dialog import ConnectDatabaseDialog
from app.dialogs.import_data_dialog import ImportDataDialog
from app.dialogs.load_demo_dialog import LoadDemoDialog
from app.dialogs.query_builder_dialog import QueryBuilderDialog
from app.logs.logger import applogger
from app.utils.config import clear_recent_databases, get_constant, set_last_database
from app.utils.data_sources import SERVER_DATABASE_QUERY_READERS, SERVER_DATABASE_READERS
from app.utils.i18n import _
from app.utils.messages import show_message
from app.utils.startup import PROJECT_FILE_FILTER

# A run of same-label descriptor snapshots inside this many seconds is
# treated as one edit and recorded once. The auto-applying property panels
# fire _snapshot_descriptors on every debounced change; without this a
# single slider drag would leave a stack of identical "Figure properties"
# undo entries in front of the state worth going back to.
SNAPSHOT_COALESCE_SECONDS: float = get_constant("snapshot_coalesce_seconds", 2.0)


if TYPE_CHECKING:
    class _MainWindowProjectBase(QMainWindow):
        _repo: SqliteRepo
        _db_path: Path
        _database_info_panel: Any
        _table_panel: Any
        _preview: Any
        _tabs: Any
        _recent_view: Any
        _last_snapshot_label: str
        _last_snapshot_at: float

        def _refresh_undo_item(self) -> None: ...
        def _reload_tabs(self, select_figure_id: int | None = None) -> None: ...
        def _update_properties_for_current_chart(self) -> None: ...
        def _build_app_menu(self) -> None: ...
        def _update_window_title(self) -> None: ...
else:
    class _MainWindowProjectBase:
        pass


class MainWindowProject(_MainWindowProjectBase):
    """A part of MainWindow; ``self`` is the window."""

    #: The project metadata panel is supplied by the composed MainWindow.
    #: Declaring it here keeps the mixin type-checkable without changing the
    #: runtime construction order of the window's widgets.
    _database_info_panel: Any

    #: How many database-check problems the message box lists inline before
    #: it stops and points at Show Details, which holds all of them. Enough
    #: that the usual report is complete on sight, few enough that a database
    #: with hundreds of dangling references does not make a box taller than
    #: the screen.
    MAX_PROBLEMS_SHOWN: int = get_constant("max_problems_shown", 20)

    def _on_project_info(self) -> None:
        """Author, creation date and notes of the open project."""
        if self._repo is None:
            return
        from app.dialogs.project_info_dialog import ProjectInfoDialog

        if ProjectInfoDialog(self._repo, cast(QWidget, self)).exec():
            self._database_info_panel.set_repo(self._repo)
            self.statusBar().showMessage(_("Project info saved."), 4_000)

    def _on_project_report(self) -> None:
        from app.dialogs.project_report_dialog import ProjectReportDialog

        dialog = ProjectReportDialog(self._repo, self)
        if dialog.exec() and dialog.written is not None:
            self.statusBar().showMessage(_("Report written: {path}").format(path=dialog.written), 10_000)

    def _on_project_history(self) -> None:
        from app.dialogs.operation_history_dialog import OperationHistoryDialog

        OperationHistoryDialog(self._repo, None, self).exec()

    def _snapshot_descriptors(self, label: str) -> None:
        """Record the chart settings before an edit changes them.

        Every descriptor table, not the one this edit will touch: they hold
        a few rows each, so working out which is more expensive than
        copying all four - and getting that wrong is an undo that restores
        half of a change (todo.txt P2-11).

        A run of identical labels inside SNAPSHOT_COALESCE_SECONDS is one
        edit: the auto-applying property panels call this on every debounced
        change, and only the first call - the one taken before the edit
        began - has a pre-edit state worth keeping.
        """
        if self._repo is None:
            return

        now = monotonic()
        if (
            label == self._last_snapshot_label
            and now - self._last_snapshot_at < SNAPSHOT_COALESCE_SECONDS
        ):
            self._last_snapshot_at = now
            return
        self._last_snapshot_label = label
        self._last_snapshot_at = now
        self._repo.snapshot_for_undo(self._repo.DESCRIPTOR_TABLES, label=label)
        self._refresh_undo_item()

    def _on_undo(self) -> None:
        """Take back the last recorded change, and show the result."""
        entry = self._repo.undo_last()
        if entry is None:
            applogger.info("There is nothing to undo.")
            self._refresh_undo_item()
            return

        applogger.info("Undid: %s", entry.describe())
        self._table_panel.reload()
        self._preview.clear()
        self._reload_tabs()
        self._update_properties_for_current_chart()
        # The entry just consumed - update the menu to name the next one (or
        # disable it). On macOS this rebuilds the bar; safe here because Cocoa
        # has already closed the menu before dispatching this.
        self._refresh_undo_item()

    def _on_open_recent(self, db_path: Path) -> None:
        """Open one remembered database."""
        if not db_path.exists():
            # Between building the menu and clicking it - or a file on a
            # volume that has since been unmounted.
            applogger.warning(
                "That database is no longer there: %s",
                db_path,
            )
            show_message(self, "database.open_failed", error=db_path)
            self._build_app_menu()
            self._recent_view.refresh()
            return

        applogger.info("Opening recent database: %s", db_path)
        try:
            self._switch_database(db_path)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_clear_recent(self) -> None:
        """Forget the list. The database currently open is not affected."""
        clear_recent_databases()
        self._build_app_menu()
        self._recent_view.refresh()

    def _check_project_sql(self) -> None:
        """Check the opened project's saved queries and series SQL, once.

        Anything that would write is blocked (see SqliteRepo.scan_user_sql)
        and listed in a warning shown once the window is up. Nothing is
        changed in the file: fixing or deleting the query is the person's
        call. user.json "sql_guard": false switches this off.
        """
        try:
            found = self._repo.scan_user_sql()
        except Exception:
            applogger.exception("The project SQL check failed.")
            return
        if not found:
            return
        listed = "\n".join(f"\u2022 {line}" for line in found[:12])
        more = len(found) - 12
        if more > 0:
            listed += "\n" + _("...and {count} more.").format(count=more)
        text = _(
            "This project contains SQL that would change the database. It has been "
            "blocked and will not run until it is edited:\n\n{items}"
        ).format(items=listed)
        QTimer.singleShot(0, lambda: QMessageBox.warning(self, _("SQL blocked"), text))

    def _switch_database(
        self, db_path: Path, prepare: Callable[[], object] | None = None
    ) -> None:
        """Switch the UI to another database and rebind all dependent widgets.

        *prepare* runs once the current database is closed and before
        *db_path* is opened - the moment a file that may be the open one can
        safely be replaced (Load demo, reloading the demo already open). If
        it fails, the database that was open is opened again and the error
        goes to the caller.
        """
        applogger.info("Switching database to %s", db_path)
        # Updates are held on the window's contents, never on the window
        # itself: on macOS, turning a top-level window's updates off and on
        # again around the first switch of a session left it not painting at
        # all - the new project's charts stayed blank (or showed the old
        # ones) until the window was resized.
        contents = self.centralWidget() or self
        contents.setUpdatesEnabled(False)
        try:
            self._tabs.clear()
            self._preview.clear()
            QApplication.processEvents()

            self._repo.close()
            gc.collect()

            QApplication.processEvents()

            if prepare is not None:
                try:
                    prepare()
                except Exception:
                    self._repo = SqliteRepo(db_path=self._db_path)
                    self._table_panel.set_repo(self._repo)
                    self._database_info_panel.set_repo(self._repo)
                    self._table_panel.reload()
                    self._reload_tabs()
                    raise

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            from app.dialogs.project_info_dialog import default_author

            if default_author():
                # A new project carries the author this computer remembers.
                self._repo.set_project_info({"author": default_author()})
            self._table_panel.set_repo(self._repo)
            self._database_info_panel.set_repo(self._repo)
            self._update_window_title()
            set_last_database(db_path)

            self._check_project_sql()
            self._table_panel.reload()
            self._reload_tabs()
            self._update_properties_for_current_chart()
            # set_last_database above has just put this file at the top of
            # the recent list; the menu showing that list has to be rebuilt
            # or it goes on showing the order from before the switch - same
            # for the File page's own list.
            self._build_app_menu()
            self._recent_view.refresh()
        finally:
            contents.setUpdatesEnabled(True)

    def _on_new_file(self) -> None:
        """Create a new database, safely replacing an existing file if possible."""
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getSaveFileName(
            self,
            _("New database"),
            base_dir,
            PROJECT_FILE_FILTER,
        )
        if not file_path:
            return

        db_path = SqliteRepo.ensure_dhub_extension(Path(file_path))
        applogger.info("Creating new database: %s", db_path)

        self._tabs.clear()
        self._preview.clear()
        QApplication.processEvents()

        if self._repo:
            self._repo.close()

        gc.collect()
        QApplication.processEvents()

        try:
            if db_path.exists():
                db_path.unlink()

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            self._table_panel.set_repo(self._repo)
            self._database_info_panel.set_repo(self._repo)
            self._update_window_title()
            set_last_database(db_path)
            self._table_panel.reload()
            self._reload_tabs()
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to create database: %s", exc)
            show_message(self, "database.create_failed", error=exc)
            return
        # A new project says who made it and what it is for from the start.
        self._project_info_asked = getattr(self, "_project_info_asked", set()) | {str(db_path)}
        self._on_project_info()

    def _on_load_demo(self) -> None:
        """Copy one of the shipped, pre-built demo projects, and open it.

        No "where to save it" dialog: that was the one step between picking a
        demo and seeing it, for a file whose name already says what it is and
        that nobody is expected to keep - the same reasoning that lets a first
        run open straight into a fixed, well-known path with no dialog of its
        own (see app.utils.startup.DEFAULT_DATABASE_NAME). The copy lands in
        ``projects/`` beside the application (demos.PROJECTS_DIR), not
        loose in the home directory: the copies are throwaway and one folder
        holds all of them. Loading the same demo again overwrites its previous
        copy in place, which is the point: it puts back the pristine version
        rather than asking what to call a second one. Someone who wants to
        keep a demo under its own name has Save As for that once it is open,
        same as any other project.
        """
        picker = LoadDemoDialog(self)
        if not picker.exec() or picker.chosen is None:
            return
        demo = picker.chosen

        target = SqliteRepo.ensure_dhub_extension(PROJECTS_DIR / demo.path_name)
        applogger.info("Loading demo project %r into %s", demo.file_name, target)
        # The copy is made with the current project closed: it may be this
        # very file - loading the open demo again, to start over - and
        # replacing a database under its own open connection left that
        # connection's pending changes to be replayed onto the fresh copy.
        try:
            self._switch_database(target, prepare=lambda: copy_demo_project(demo, target))
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to load demo project: %s", exc)
            show_message(self, "demo.build_failed", error=exc)

    def _on_open_database(self) -> None:
        """Open an existing database and switch the current UI."""
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getOpenFileName(
            self,
            _("Open database"),
            base_dir,
            PROJECT_FILE_FILTER,
        )
        if not file_path:
            return

        db_path = Path(file_path)
        applogger.info("Opening database: %s", db_path)

        try:
            self._switch_database(db_path)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_save(self) -> None:
        """Fold the WAL into the .dhub file on disk, right now.

        Not "there is unsaved work": every change already committed through
        SQLite's WAL as it happened. But Cmd+S/Ctrl+S is a reflex, and this
        is the honest thing an already-live database can do under it -
        guarantee the file on disk is not waiting on a WAL checkpoint SQLite
        would otherwise fold in on its own schedule. See SqliteRepo.checkpoint.
        """
        if self._repo is None:
            return
        self._ask_for_missing_project_info()
        self._refresh_project_preview()
        try:
            self._repo.checkpoint()
            self.statusBar().showMessage(_("Database saved."), 4_000)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Checkpoint failed: %s", exc)

    def _ask_for_missing_project_info(self) -> None:
        """Open Project info when the project has no author or no notes.

        Once per project and session: closing the window without filling it
        in means "not now", and the next Save should not ask again.
        """
        if self._repo is None:
            return
        asked = getattr(self, "_project_info_asked", set())
        key = str(self._repo.db_path)
        info = self._repo.project_info()
        if key in asked or (info.get("author") and info.get("notes")):
            return
        asked.add(key)
        self._project_info_asked = asked
        self._on_project_info()

    def _refresh_project_preview(self) -> None:
        """Draw the project's first figure again into its preview picture."""
        if self._repo is None or self._repo.db_path is None:
            return
        try:
            from app.utils.project_preview import write_project_preview

            write_project_preview(self._repo)
        except Exception as exc:  # noqa: BLE001 - a preview must never stop a save
            applogger.warning("Could not draw the project preview: %s", exc)

    def _on_save_as(self) -> None:
        """Save a copy of the current database under a new name, and switch to it.

        ``SqliteRepo.save_as`` uses VACUUM INTO rather than copying the .dhub
        file: with WAL mode active, the file on disk is not the whole
        database until its -wal side file is checkpointed into it, so a
        plain filesystem copy could silently miss recent writes.
        """
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getSaveFileName(
            self,
            _("Save database as"),
            base_dir,
            PROJECT_FILE_FILTER,
        )
        if not file_path:
            return

        target = SqliteRepo.ensure_dhub_extension(Path(file_path))
        if self._db_path is not None and target == self._db_path:
            return

        applogger.info("Saving database as: %s", target)
        self._ask_for_missing_project_info()
        try:
            saved_path = self._repo.save_as(target)
            self._switch_database(saved_path)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to save database as: %s", exc)
            show_message(self, "database.save_as_failed", error=exc)
            return
        # The copy has its own name, so its own preview.
        self._refresh_project_preview()

    def _on_import_data(self, source_path: Path | None = None) -> None:
        """Open the import dialog and refresh UI if import succeeds.

        ``source_path`` is the file a drop arrived with; the dialog then opens
        already showing it, with its preview and its default table name, which
        is the whole point of dropping it rather than browsing for it.
        """
        dlg = ImportDataDialog(self._repo, parent=self)
        if source_path is not None:
            dlg.load_file(source_path)
        if dlg.exec():
            self._table_panel.reload()
            self._reload_tabs()

    def _on_database_table_import(self) -> None:
        """Import a table - or what a query returns - from another database, as a linked table."""
        picker = ConnectDatabaseDialog(cast(QWidget, self))
        if not picker.exec() or picker.connection is None:
            return
        if not picker.table and not picker.query:
            return
        connection = picker.connection
        settings = connection.to_link_settings()
        try:
            if picker.query:
                settings["query"] = picker.query
                frame = SERVER_DATABASE_QUERY_READERS[connection.kind](
                    connection, picker.query, skiprows=0, skipfooter=0
                )
            else:
                settings["table"] = picker.table
                _list_tables, read_table = SERVER_DATABASE_READERS[connection.kind]
                frame = read_table(connection, picker.table, skiprows=0, skipfooter=0)
        except Exception as exc:  # noqa: BLE001 - any driver's error, told to the user
            applogger.exception("Database import failed")
            show_message(cast(QWidget, self), "import.database_failed", error=exc)
            return
        if frame is None or frame.empty:
            applogger.warning("The database returned no rows to import.")
            return

        # Named for the source table, as Import Data names a file's. A free
        # name - "readings_2" when "readings" exists: import_into_sqlite
        # would replace a table of the same name. "sqlite_" starts SQLite's
        # own names, which it refuses.
        base = re.sub(r"[^0-9A-Za-z_]+", "_", picker.table or f"{connection.display_name()}_query").strip("_")
        if not base or base.lower().startswith("sqlite_") or base[0].isdigit():
            base = f"db_{base}"
        table_name = self._repo.free_table_name(base)
        types = {str(column): ImportDataDialog._guess_sqlite_type(frame[column]) for column in frame.columns}
        self._repo.snapshot_for_undo([table_name], label=f"Import '{table_name}'")
        self._repo.import_into_sqlite(table_name, frame, types)
        applogger.info("Imported %s from %s.", table_name, connection.display_name())
        link_settings = {
            "source": settings,
            "read": {"skiprows": 0, "skip_last": 0, "header": True, "delimiter": None, "encoding": None},
            "destination": {"table": table_name, "normalize_columns": False},
            "columns": {"types": types},
        }
        if not self._repo.upsert_link(
            table_name=table_name,
            source_path=f"{connection.display_name()}#{picker.table or picker.query}",
            settings=link_settings,
        ):
            applogger.warning("Failed to create link for imported table '%s'", table_name)
        self._table_panel.reload()

    def _on_create_query_table(self) -> None:
        dialog = QueryBuilderDialog(self._repo, parent=self)
        dialog.exec()
        self._table_panel.reload()

    def _on_optimize_db(self) -> None:
        """Check the database, report what it found, then compact it.

        The findings go in the box twice on purpose: the problems inline, so
        that what is wrong is readable without clicking anything, and the whole
        grouped report - unreferenced tables included - behind Show Details,
        where it scrolls and can be copied into a bug report.  The status bar
        keeps the count, which is all it has room for.
        """
        report = self._repo.optimize_db()
        self.statusBar().showMessage(report.summary(), 10_000)

        if report.is_healthy:
            return

        shown = report.problems[: self.MAX_PROBLEMS_SHOWN]
        lines = [report.summary(), "", *shown]
        remaining = len(report.problems) - len(shown)
        if remaining > 0:
            lines.append(
                _("...and {count} more, under Show Details.").format(count=remaining)
            )

        show_message(
            self,
            "database.check_found_problems",
            report="\n".join(lines),
            details=report.details(),
        )
