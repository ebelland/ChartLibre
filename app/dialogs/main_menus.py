"""The main window's menus: the macOS menu bar, the application menu, Undo and Open recent.

A mixin of MainWindow, kept apart because it is a world of its own - which
action sits in which menu, the roles Cocoa moves items by, the native items
renamed in the app menu - and was a sixth of main_window.py. The actions
themselves stay in the window (``_on_save``, ``_on_undo``...); this only
says where they appear and keeps Undo and Open recent current.
"""
from __future__ import annotations

from functools import partial

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QMenuBar

from app.logs.logger import applogger
from app.styles.style import (
    IS_MACOS,
    MenuItem,
    _pyobjc_core_is_safe_to_import,
    action_menu_item,
    action_presentation,
    create_menu,
    create_menu_item,
)
from app.utils.config import get_recent_databases
from app.utils.i18n import _


class MainWindowMenus:
    """Menu building for MainWindow; ``self`` is the window."""

    #: Actions moved into the real macOS menu bar via QAction.MenuRole -
    #: Preferences and About are relocated by Cocoa itself into the
    #: application menu next to the apple, wherever they were declared, so
    #: they read here exactly the way they read in the rail's popup menu.
    _MACOS_MENU_ROLES: dict[str, QAction.MenuRole] = {
        "settings": QAction.MenuRole.PreferencesRole,
        "credits": QAction.MenuRole.AboutRole,
    }

    def _app_menu_items(self) -> list[tuple[str, list[MenuItem | None]]]:
        """Return the app menu's contents, grouped the way macOS expects.

        One list of (title, items) groups, shared by both places the menu
        appears: flattened into one popup for the activity rail's "Menu"
        button and every non-Mac platform (see _flatten_menu_groups), or
        built into one QMenu per group for the real macOS menu bar (see
        _build_macos_menu_bar). One source means the two cannot drift apart
        the way a hand-kept second copy would.
        """
        return [
            (
                _("File"),
                [
                    action_menu_item("new", self._on_new_file),
                    action_menu_item("open", self._on_open_database),
                    self._recent_databases_item(),
                    action_menu_item("import", self._on_import_data),
                    None,
                    action_menu_item("save", self._on_save, shortcut="Ctrl+S"),
                    action_menu_item("save_as", self._on_save_as),
                    None,
                    action_menu_item("project_report", self._on_project_report),
                ],
            ),
            (
                _("Edit"),
                [
                    self._undo_item(),
                    None,
                    action_menu_item("copy", self._on_copy_chart),
                ],
            ),
            (
                _("Database"),
                [
                    action_menu_item("query_builder", self._on_query_builder),
                    action_menu_item("optimize_db", self._on_optimize_db),
                    None,
                    action_menu_item("database_info", self._on_database_info),
                ],
            ),
            (
                _("Help"),
                [
                    # Only on macOS: PreferencesRole (_MACOS_MENU_ROLES) is
                    # what actually pulls this out into the native apple
                    # menu there, so it has to be declared somewhere for
                    # Cocoa to relocate - "Help" is as good a place as any
                    # a user never actually sees it listed under. Everywhere
                    # else it would just be a second, redundant way to reach
                    # what the rail's own dedicated Settings tile already
                    # does one click away.
                    *(
                        [action_menu_item("settings", self._on_settings)]
                        if IS_MACOS else []
                    ),
                    action_menu_item("user_manual", self._on_user_manual),
                    None,
                    action_menu_item("credits", self._on_credits),
                ],
            ),
        ]

    @staticmethod
    def _flatten_menu_groups(
        groups: list[tuple[str, list[MenuItem | None]]]
    ) -> list[MenuItem | None]:
        """One flat list for the popup: every group's items, a separator
        between groups - the shape the popup already had before the real
        macOS menu bar split it into File/Edit/Database/Help."""
        flat: list[MenuItem | None] = []
        for index, (_title, items) in enumerate(groups):
            if index:
                flat.append(None)
            flat.extend(items)
        return flat

    def _undo_item(self) -> MenuItem:
        """The Undo entry, naming the change it will take back.

        "Undo" alone would not say whether it is about to bring back a
        column, a table or a chart - and this undoes one recorded action
        against the database, not the last thing the user touched
        anywhere. Naming it is what keeps the two from being confused.
        """
        entries = self._repo.undo_entries() if self._repo is not None else []
        latest = entries[0] if entries else None
        _icon, label, tooltip = action_presentation("undo")
        text = (
            _("Undo: {what}").format(what=latest.label)
            if latest is not None
            else label
        )
        return MenuItem(
            text=text,
            tooltip=tooltip,
            icon="undo",
            shortcut="Ctrl+Z",
            callback=self._on_undo,
            action_id="undo",
            enabled=latest is not None,
        )

    def _undo_actions(self) -> list[QAction]:
        """Every Undo entry on screen: the rail's popup, and the macOS bar."""
        menus: list[QMenu | None] = [self._app_menu]
        if IS_MACOS:
            menus.extend(
                menu
                for action in self.menuBar().actions()
                if isinstance(menu := action.menu(), QMenu)
            )

        found: list[QAction] = []
        for menu in menus:
            if menu is None:
                continue
            try:
                actions = menu.actions()
            except RuntimeError:
                # Rebuilding the macOS menu bar deletes the QMenu behind the
                # File entry, and a shell of it can outlive that on the Python
                # side. Nothing to refresh in a menu that is already gone.
                continue
            found.extend(
                action for action in actions if str(action.data() or "") == "undo"
            )
        return found

    def _undo_item_state(self) -> tuple[str, bool]:
        """The Undo entry's text and whether it should be enabled, read fresh."""
        entries = self._repo.undo_entries() if self._repo is not None else []
        latest = entries[0] if entries else None
        _icon, label, _tooltip = action_presentation("undo")
        text = (
            _("Undo: {what}").format(what=latest.label)
            if latest is not None
            else label
        )
        return text, latest is not None

    def _sync_undo_item(self) -> None:
        """Push the current Undo state onto the QAction objects on screen.

        Enough on its own for the rail's popup, which is a live QMenu and is
        wired to call this from ``aboutToShow``. Not enough for the native
        macOS menu bar - see :meth:`_refresh_undo_item`.
        """
        text, enabled = self._undo_item_state()
        for action in self._undo_actions():
            action.setText(text)
            action.setEnabled(enabled)

    def _refresh_undo_item(self) -> None:
        """Bring the Undo entry up to date after the stack changed.

        Called straight after anything that records or consumes an undo entry
        - the few UI refresh points every change funnels through
        (_snapshot_descriptors, refresh, _on_chart_panel_deleted,
        ChartPanel.figure_edited) and _on_undo - not every handler.

        The native macOS menu bar makes this more than a property poke: it
        caches each item's text and enabled state from the last time the menu
        was built, never emits ``aboutToShow`` for its items, and so never
        re-reads the QAction - _sync_undo_item's changes simply do not land
        there. A full rebuild does land, and is the same hammer the
        Open-recent list already swings on every change (_build_app_menu).
        """
        self._sync_undo_item()
        if IS_MACOS:
            self._build_app_menu()



    def _recent_databases_item(self) -> MenuItem:
        """The Open recent submenu, built from user.json's own list.

        Every entry names one file, and the tooltip carries the folder it
        is in: two projects called "analysis.dhub" in different places are
        the normal case, and a menu of identical names is a menu of
        guesses. Disabled rather than hidden when the list is empty, so
        the menu keeps its shape between the first and the second launch.
        """
        recent = get_recent_databases()
        items: list[MenuItem | None] = [
            MenuItem(
                text=path.name,
                tooltip=str(path.parent),
                icon="open",
                callback=partial(self._on_open_recent, path),
            )
            for path in recent
        ]
        if items:
            items.append(None)
            items.append(
                MenuItem(text=_("Clear menu"), callback=self._on_clear_recent)
            )

        return MenuItem(
            text=_("Open recent"),
            icon="open",
            submenu=items,
            enabled=bool(recent),
        )



    def _build_app_menu(self) -> None:
        """Create the app menu, and place it where each platform expects it.

        Everywhere else this is the popup the activity rail's "Menu" button
        opens - there is no equivalent slot to move it into. On macOS it is
        a real QMenuBar instead, which is what turns the menu bar next to the
        apple from the bare running-process name into an actual menu: with no
        QMenuBar at all, Cocoa still draws that bar, showing only the
        process's own name (here, "Python", since this is not a signed .app
        bundle) and a default Quit.

        Rebuilt wholesale on every call - after Settings, since language and
        theme are what changes underneath these, and after every undo-stack
        change on macOS, whose native menu bar only reads the item list at
        build time - rather than patched in place, which is simpler and is
        exactly what already happened when this was only ever the popup.
        """
        previous = getattr(self, "_app_menu", None)
        groups = self._app_menu_items()
        # No self._file_menu: it only ever existed for the rail's own File
        # popup button, which nav_file (a page now, not a popup) replaced.
        self._help_menu = create_menu(self, list(groups[-1][1]))
        self._help_menu.setTitle(_("Help & About"))
        # The rail's Help tile holds whichever menu object existed when it
        # was built, so a rebuild has to hand it the new one - otherwise
        # the tile keeps popping up the menu from before the language or
        # the undo stack changed. Guarded because the first build runs
        # before the rail exists.
        self._app_menu = create_menu(self, self._flatten_menu_groups(groups))
        if previous is not None and IS_MACOS:
            # It is parented to this window, so replacing the attribute is not
            # enough to free it - and on macOS this runs on every undo-stack
            # change, often enough for the leak to matter. Only macOS: off it,
            # the activity-rail button still holds this exact object.
            previous.deleteLater()
        # The live popup can just re-read on open; _sync, not _refresh, so it
        # never triggers the macOS menu-bar rebuild from a show handler.
        self._app_menu.aboutToShow.connect(self._sync_undo_item)

        if IS_MACOS:
            self._build_macos_menu_bar(groups)
            # Cocoa rebuilds its own native menu items - overwriting any
            # rename - at least once more after this call returns, somewhere
            # between menu construction and the window's first activation.
            # Measured, not documented anywhere: a rename applied immediately
            # is gone again within 500ms, while one applied at 2s already
            # survives. Rather than guess the exact moment, this reapplies a
            # few times over the first two seconds and then stops.
            for delay_ms in (0, 150, 400, 800, 1500, 2500):
                QTimer.singleShot(delay_ms, self._rename_macos_native_app_menu_items)

    def _build_macos_menu_bar(
        self, groups: list[tuple[str, list[MenuItem | None]]]
    ) -> None:
        """Populate the real menu bar with one QMenu per group.

        Settings and Credits do not stay wherever this puts them: Cocoa
        pulls any action carrying MenuRole.PreferencesRole/AboutRole out
        into the native application menu - the one already showing next to
        the apple - wherever in the menu bar it was declared. Window is not
        one of the groups: it holds no reusable MenuItem, only two Cocoa
        window operations nothing else needs, so it is built directly here
        instead, between the app's own menus and Help - the usual place on
        a Mac.
        """
        menu_bar = self.menuBar()
        menu_bar.clear()

        # menu_bar.clear() above only lets go of menus it was actually
        # holding - the hidden-but-alive File/Database menus below are
        # never added to it, so without this every rebuild (every
        # undo-stack change; see this method's own docstring) would pile up
        # another orphaned QMenu plus another copy of the same shortcut
        # re-registered on self, instead of replacing the previous one.
        for action in getattr(self, "_macos_hidden_menu_actions", ()):
            self.removeAction(action)
        for menu in getattr(self, "_macos_hidden_menus", ()):
            menu.deleteLater()
        self._macos_hidden_menus: list[QMenu] = []
        self._macos_hidden_menu_actions: list[QAction] = []

        # File and Database are both covered by the nav rail's own pages
        # now (_create_file_page/_create_developer_page) - a second, visible
        # menu for the same actions would just be redundant chrome. Their
        # items are still built (below, into a menu that is never added to
        # the bar) and their shortcuts kept alive on the window itself
        # (self.addAction), so Cmd+S and friends do not stop working just
        # because the menu that used to carry them is gone from view.
        hidden_titles = {_("File"), _("Database")}

        for index, (title, items) in enumerate(groups):
            if index == len(groups) - 1:
                self._build_macos_window_menu(menu_bar)

            hidden = title in hidden_titles
            if hidden:
                menu = QMenu(self)
                menu.setTitle(title)
            else:
                menu = menu_bar.addMenu(title)
            # Cocoa rarely delivers this for a menu-bar menu, and rebuilding
            # the bar from inside a show handler would clear the menu
            # mid-display - so _sync (a plain property poke), never _refresh.
            menu.aboutToShow.connect(self._sync_undo_item)
            for item in items:
                if item is None:
                    menu.addSeparator()
                    continue

                if item.submenu is not None:
                    # The same helper the popup uses, so Open recent is one
                    # list of files rendered twice rather than two lists that
                    # can disagree.
                    child = create_menu(self, item.submenu)
                    child.setTitle(item.text)
                    child.setEnabled(item.enabled)
                    menu.addMenu(child)
                    continue

                create_menu_item(
                    parent=self,
                    menu=menu,
                    icon=item.icon,
                    checkable=item.checkable,
                    text=item.text,
                    tooltip=item.tooltip,
                    key=item.shortcut,
                    action=item.callback,
                    action_id=item.action_id,
                    checked=item.checked,
                    enabled=item.enabled,
                )

            for action in menu.actions():
                role = self._MACOS_MENU_ROLES.get(str(action.data() or ""))
                if role is not None:
                    action.setMenuRole(role)

            if hidden:
                # menu itself keeps every action in it alive regardless
                # (append below); addAction is only for the ones whose
                # shortcut has to keep firing with no visible menu left to
                # carry it - Cmd+S and the rest of File/Database's own.
                self._macos_hidden_menus.append(menu)
                for action in menu.actions():
                    if not action.shortcut().isEmpty():
                        self.addAction(action)
                        self._macos_hidden_menu_actions.append(action)

    def _build_macos_window_menu(self, menu_bar: QMenuBar) -> None:
        """Build the Window menu: Minimize and Zoom.

        Neither is a reusable MenuItem - nothing else in the app ever needs
        "minimize this window" - so, unlike every other menu, this one is
        built directly against the QMenuBar rather than through
        _app_menu_items/create_menu_item.
        """
        menu = menu_bar.addMenu(_("Window"))
        create_menu_item(
            parent=self,
            menu=menu,
            icon=None,
            checkable=False,
            text=_("Minimize"),
            tooltip=_("Minimize"),
            key="Ctrl+M",
            action=self.showMinimized,
        )
        create_menu_item(
            parent=self,
            menu=menu,
            icon=None,
            checkable=False,
            text=_("Zoom"),
            tooltip=_("Zoom"),
            key=None,
            action=self._on_zoom,
        )

    #: Cocoa's own selectors for Hide and Quit - reliable regardless of
    #: title, since neither is backed by a QAction of ours and both keep
    #: these selectors on every Mac, in every language.  About is *not* in
    #: here: it is backed by our own "credits" QAction, so Qt dispatches it
    #: through the same generic "qt_itemFired:" selector as Preferences,
    #: indistinguishable from it by selector alone.  It is matched by
    #: position instead - see _rename_macos_native_app_menu_items.
    _MACOS_APP_MENU_SELECTORS: dict[str, str] = {
        "hide:": "Hide",
        "terminate:": "Close",
    }

    def _rename_macos_native_app_menu_items(self) -> None:
        """Strip the app name Cocoa insists on adding to About/Hide/Quit.

        `QAction.MenuRole` moves an action into the native application menu,
        but Cocoa then *replaces* its text with its own "About {name}" /
        "Hide {name}" / "Quit {name}" template - not read from the QAction at
        all, so nothing on the Qt side can change it.  The assumption in
        main.py that QApplication.setApplicationName() controls "{name}" does
        not hold here: measured directly, Cocoa's template uses
        NSProcessInfo.processName (the running interpreter, "Python" for an
        unsigned script) instead, which no public Qt or Cocoa API renames
        without an actual .app bundle - only reaching past Qt into the
        NSMenuItem itself can.

        So this does not try to make the name correct - it removes it,
        landing on the bare "About" / "Hide" / "Close" the request asked for.
        Hide and Quit are matched by their native Cocoa action selector,
        stable regardless of title or language.  About cannot be: it is
        backed by our own "credits" QAction, so Qt dispatches it through the
        same generic selector as Preferences.  Apple's own Human Interface
        Guidelines fix the application menu's layout on every Mac - About
        first, when present - which is the stable signal used here instead.

        Deferred one event-loop turn past menu construction: Cocoa builds the
        native menu from Qt's QMenuBar lazily, so nothing is there to rename
        yet in the same call that builds it.
        """
        if not _pyobjc_core_is_safe_to_import():
            return

        # Offscreen (tests, scripts) there is no native menu bar to rename.
        if QApplication.platformName() != "cocoa":
            return

        try:
            import AppKit  # type: ignore[import-not-found]
        except Exception:
            return

        try:
            app = getattr(AppKit, "NSApp", None)
            if app is None:
                ns_application = getattr(AppKit, "NSApplication")
                app = ns_application.sharedApplication()

            app_menu = app.mainMenu().itemAtIndex_(0).submenu()
            if app_menu is None or app_menu.numberOfItems() == 0:
                return

            about_item = app_menu.itemAtIndex_(0)
            if str(about_item.action() or "") == "qt_itemFired:":
                about_item.setTitle_(_("About"))

            for index in range(app_menu.numberOfItems()):
                item = app_menu.itemAtIndex_(index)
                selector = str(item.action()) if item.action() is not None else ""
                bare_text = self._MACOS_APP_MENU_SELECTORS.get(selector)
                if bare_text is not None:
                    item.setTitle_(_(bare_text))
        except Exception:
            applogger.exception("Failed to rename the native macOS app menu items.")
