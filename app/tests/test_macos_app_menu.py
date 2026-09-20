"""The app menu, on macOS - and everywhere else.

There was no real menu bar at all: with none built, Cocoa still draws the bar
next to the apple, showing only the running process's own name ("Python",
since this is not a signed .app bundle) and nothing beneath it. "Menu" lived
only as the activity rail's popup button - not visible near the apple at all,
and not where a Mac user looks for it.

On macOS this now also fills a real QMenuBar; the popup keeps working
everywhere else. Settings and Credits carry explicit MenuRole values, which
is what actually moves them into the native application menu next to the
apple - Cocoa relocates a role-marked action wherever in the menu bar it was
declared, so they read the way "Preferences..." and "About" read in every
other Mac app, in the one slot this application's menu structure cannot
otherwise reach.

Qt object lifetime note for whoever edits this file: every intermediate
QAction/QMenu returned along a chain (menuBar().actions()[0].menu(), and so
on) has to be kept in a named variable for as long as it is used. Losing the
reference to an intermediate wrapper - even briefly, inside one chained
expression - lets PySide6 tear down the C++ object under it, which surfaces
later as "Internal C++ object already deleted" on a completely unrelated
line. Every test below keeps each step named for exactly this reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
import app.dialogs.main_window as main_window_module
from app.dialogs.main_window import MainWindow


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


@pytest.fixture
def make_window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    """Build a MainWindow, and guarantee applogger loses track of its status
    bar again afterwards.

    MainWindow.__init__ points the (process-wide singleton) logger at
    self.statusBar(). Nothing it does undoes that on close, so a later test -
    any later test, anywhere in the suite, that happens to log a message at a
    level the status bar shows - was handed a reference to an already-deleted
    C++ QStatusBar and crashed with "Internal C++ object already deleted", on
    a line that has nothing to do with this file. One MainWindow built here
    without this fixture was enough to take the rest of the suite down.
    """
    built_windows: list[MainWindow] = []

    def factory() -> MainWindow:
        built = MainWindow(repo=repo, db_path=tmp_db_path)
        built_windows.append(built)
        return built

    yield factory

    for built in built_windows:
        built.close()
    applogger.set_status_bar(None)


@pytest.fixture
def window(make_window):
    return make_window()


def _rail_tooltips(window: MainWindow) -> list[str]:
    buttons = window._left_rail.findChildren(type(window._nav_buttons[0]))
    return [button.toolTip() for button in buttons]


# ----------------------------------------------------------------------
# The macOS menu bar
#
# _menu_named returns (top_actions, menu): top_actions is menu_bar.actions(),
# a real list the caller must keep assigned to a name for as long as menu is
# used. menu is one QAction inside that list's own .menu() - not a fresh,
# unanchored list of its own - so losing the name "top_actions" would free
# the list, and with it the QAction menu is a child of. See the file
# docstring on Qt object lifetime; this is that trap by another route.
# ----------------------------------------------------------------------
def _menu_named(window: MainWindow, title: str):
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    top_action = next(action for action in top_actions if action.text() == title)
    return top_actions, top_action.menu()


def test_the_menu_bar_carries_one_menu_per_group(window: MainWindow) -> None:
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    titles = [action.text() for action in top_actions]

    # No File or Database: both are covered by the nav rail's own pages now
    # (_create_file_page/_create_database_page) - a second, visible menu
    # for the same actions would be redundant chrome. No Developer either:
    # unlike File/Database it is not even hidden-but-built, it is gone from
    # _app_menu_items entirely (see _create_developer_page) - none of its
    # four actions carries a shortcut worth keeping alive, so there was
    # nothing left for a hidden menu to do. Window sits between the app's
    # own menus and Help - the usual place on a Mac - even though it holds
    # no MenuItem of its own (see _build_macos_window_menu).
    assert titles == ["Edit", "Window", "Help"]


def test_file_and_database_are_hidden_but_still_alive(window: MainWindow) -> None:
    """Not in the visible bar (see test_the_menu_bar_carries_one_menu_per_
    group), but still built - _MACOS_MENU_ROLES applies to their items same
    as any other group's, and File's own shortcuts (Cmd+S...) still have to
    fire with no menu left to carry them."""
    hidden_ids_by_title = {
        menu.title(): [a.data() for a in menu.actions() if not a.isSeparator()]
        for menu in window._macos_hidden_menus
    }
    assert hidden_ids_by_title["File"] == [
        "new", "open", None, "import", "save", "save_as"
    ]
    assert hidden_ids_by_title["Database"] == [
        "query_builder", "optimize_db", "database_info"
    ]


def test_the_file_shortcuts_still_fire_with_no_menu_to_carry_them(
    window: MainWindow,
) -> None:
    shortcuts = {
        action.shortcut().toString()
        for action in window.actions()
        if not action.shortcut().isEmpty()
    }
    assert "Ctrl+S" in shortcuts


def test_edit_holds_undo_and_copy(window: MainWindow) -> None:
    _top_actions, menu = _menu_named(window, "Edit")
    items = menu.actions()
    ids = [action.data() for action in items if not action.isSeparator()]
    assert ids == ["undo", "copy"]


def test_help_holds_the_help_group(window: MainWindow) -> None:
    """No load_demo here any more either - it moved to the File page's own
    Workspace section (see main_window._create_file_page), same reasoning
    as Developer: starting from a demo is starting a workspace, not
    documentation."""
    _top_actions, menu = _menu_named(window, "Help")
    items = menu.actions()
    ids = [action.data() for action in items if not action.isSeparator()]
    assert ids == ["settings", "log_viewer", "user_manual", "credits"]


def test_window_holds_minimize_and_zoom(window: MainWindow) -> None:
    _top_actions, menu = _menu_named(window, "Window")
    items = menu.actions()
    texts = [action.text() for action in items]
    assert texts == ["Minimize", "Zoom"]


def test_settings_carries_the_preferences_role(window: MainWindow) -> None:
    """This, not the menu it sits in, is what actually moves it next to the
    apple - Cocoa relocates a role-marked action wherever it was declared."""
    _top_actions, menu = _menu_named(window, "Help")
    items = menu.actions()
    settings_action = next(action for action in items if action.data() == "settings")
    assert settings_action.menuRole() == QAction.MenuRole.PreferencesRole


def test_credits_carries_the_about_role(window: MainWindow) -> None:
    _top_actions, menu = _menu_named(window, "Help")
    items = menu.actions()
    credits_action = next(action for action in items if action.data() == "credits")
    assert credits_action.menuRole() == QAction.MenuRole.AboutRole


def test_nothing_else_is_given_a_role_it_did_not_ask_for(window: MainWindow) -> None:
    """Only Settings and Credits are meant to be pulled out of any menu."""
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()

    ordinary = []
    for top_action in top_actions:
        menu = top_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.isSeparator() or action.data() in ("settings", "credits"):
                continue
            ordinary.append(action)

    assert ordinary
    assert all(
        action.menuRole() == QAction.MenuRole.TextHeuristicRole for action in ordinary
    )


# ----------------------------------------------------------------------
# The rail no longer offers a redundant popup
# ----------------------------------------------------------------------
def test_the_rail_has_no_menu_button_on_macos(window: MainWindow) -> None:
    assert "Show main menu" not in _rail_tooltips(window)


def test_page_switching_no_longer_needs_an_offset(window: MainWindow) -> None:
    """The button group used to reserve id 0 for the popup and shift every
    real page by one; with the popup gone from the group entirely, id and
    stack-page index are the same number."""
    window._set_nav_index(0)
    assert window._left_stack.currentIndex() == 0
    assert window._nav_buttons[0].isChecked()

    window._set_nav_index(2)
    assert window._left_stack.currentIndex() == 2
    assert window._nav_buttons[2].isChecked()


def test_startup_selects_and_highlights_the_default_page(window: MainWindow) -> None:
    """Under the old offset, the startup call to _set_nav_index addressed
    the popup's own reserved slot and did nothing: the data page showed
    only because QStackedWidget already defaults to index 0, and its rail
    button was never actually checked.

    The default page is the tables, addressed by key rather than by the
    literal 0 it used to be - File sits above them in the rail now (see
    nav_bar.DEFAULT_PAGES), and opening onto it would hide the data the
    window was just opened on."""
    index = window._left_rail.action_ids.index("nav_data")
    assert window._left_stack.currentIndex() == index
    assert window._nav_buttons[index].isChecked()


# ----------------------------------------------------------------------
# Everywhere else, unchanged
# ----------------------------------------------------------------------
def test_off_macos_page_switching_is_unaffected(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window_module, "IS_MACOS", False)

    built = make_window()

    built._set_nav_index(1)
    assert built._left_stack.currentIndex() == 1
    assert built._nav_buttons[1].isChecked()


def test_off_macos_the_help_group_drops_the_redundant_settings_item(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings has its own rail tile off macOS (NavigationBar.settings_button)
    - listing it again in the Help popup would be a second, redundant way to
    reach the same thing. On macOS it stays: PreferencesRole is what actually
    pulls it into the native apple menu, and nothing else declares it."""
    monkeypatch.setattr(main_window_module, "IS_MACOS", False)

    built = make_window()

    ids = [action.data() for action in built._help_menu.actions() if not action.isSeparator()]
    assert "settings" not in ids
    assert ids == ["log_viewer", "user_manual", "credits"]


# ----------------------------------------------------------------------
# Developer menu: each tool opens its own dialog (see todo.txt P3-4/5/6)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "handler_name, dialog_attr",
    [
        ("_on_edit_localization", "EditLocalizationDialog"),
        ("_on_series_operation_builder", "SeriesOperationBuilderDialog"),
        ("_on_function_creator", "FunctionCreatorDialog"),
        ("_on_renderer_helper", "RendererHelperDialog"),
    ],
)
def test_each_dev_tool_opens_its_own_dialog(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, handler_name: str, dialog_attr: str
) -> None:
    opened: list[QWidget] = []

    class _FakeDialog:
        def __init__(self, parent: QWidget) -> None:
            opened.append(parent)

        def exec(self) -> None:
            return None

    monkeypatch.setattr(main_window_module, dialog_attr, _FakeDialog)

    getattr(window, handler_name)()

    assert opened == [window]


# ----------------------------------------------------------------------
# One list feeds both
# ----------------------------------------------------------------------
def test_the_popup_and_the_menu_bar_share_one_item_list(window: MainWindow) -> None:
    """They cannot drift apart the way two hand-kept copies would: the popup
    is every group's items flattened into one list (_flatten_menu_groups),
    and the bar is the same groups split back out into one QMenu each - plus
    File and Database, which are built the same way but kept off the visible
    bar (see test_file_and_database_are_hidden_but_still_alive)."""
    popup_ids = [
        action.data()
        for action in window._app_menu.actions()
        if not action.isSeparator()
    ]

    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()

    bar_ids = []
    for top_action in top_actions:
        # Window is not one of the shared groups - it holds no MenuItem the
        # popup could show, only two Cocoa window operations built directly
        # against the bar (see _build_macos_window_menu).
        if top_action.text() == "Window":
            continue
        menu = top_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if not action.isSeparator():
                bar_ids.append(action.data())

    hidden_ids = [
        action.data()
        for menu in window._macos_hidden_menus
        for action in menu.actions()
        if not action.isSeparator()
    ]

    assert sorted(popup_ids, key=str) == sorted(bar_ids + hidden_ids, key=str)


def test_rebuilding_the_menu_does_not_duplicate_the_bar(window: MainWindow) -> None:
    """_build_app_menu runs again after Settings closes - language or theme
    may have changed. The menu bar must be replaced, not added to - and the
    hidden File/Database menus (and the shortcut actions standing in for
    them) must not pile up either."""
    window._build_app_menu()
    window._build_app_menu()

    menu_bar = window.menuBar()
    assert len(menu_bar.actions()) == 3
    assert len(window._macos_hidden_menus) == 2
    assert len(window.actions()) == len(set(window.actions()))
