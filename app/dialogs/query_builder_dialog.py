"""Dialog for writing, validating and saving SQL queries.

A saved query is stored as text in ``__queries__`` and executed every time it is
read. It is deliberately *not* a SQLite VIEW: a view would freeze the statement
into the schema, would need a migration to change, and would make a query
indistinguishable from a table in ``sqlite_master`` - which is exactly the
distinction the table list has to show.

The dialog does three things and keeps them separate: build the text, prove it
runs, and give it a name. Validation is a ``LIMIT 0`` probe, so it is instant
even against a table with millions of rows; Run is what actually reads data,
and only on demand.
"""
from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QFormLayout,
    QPlainTextEdit,
    QSizePolicy,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.data.data_source import quote_identifier
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    create_action_button,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.config import get_constant
from app.utils.dialog_state import (
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
)
from app.utils.messages import ask, show_message
from app.widgets.table_preview import DataFrameTableModel
from app.utils.i18n import _
from app.dialogs.query_operations import FORMS, OperationForm
from app.dialogs.two_panels_dialog_base import SettingsTableDialog

# Rows fetched by Run.  Enough to see whether the query is right, few enough
# that pressing Run on a full-table select is not a mistake.
PREVIEW_ROW_LIMIT: int = get_constant("query_builder_preview_row_limit", 500)

# config.json key for the remembered entries and geometry of this dialog.
STATE_KEY: str = "query_builder_dialog"

# Snippet id -> (button label, tooltip).  The order here is the order of the
# buttons, which runs from the most common to the least.
# Each builder receives the already-quoted table name.  Placeholders are
# written in <angle brackets>: they are invalid SQL, so a snippet left
# unfinished fails validation instead of running against the wrong column.
# Join, union and the summary are in the Build frame now, with the other
# Tables operations (app.dialogs.query_operations).
_SNIPPET_IDS = ("select", "order", "filter")


class QueryBuilderDialog(SettingsTableDialog):
    """Create and edit saved queries, and run them to check the result."""

    def __init__(
        self,
        repo: SqliteRepo,
        *,
        query_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Open the builder, optionally on an existing saved query."""
        super().__init__(
            parent,
            title=_("Query Builder"),
            icon=load_icon("data"),
            size="large",
            settings_width=340,
            sections=True,
        )
        self._repo = repo
        self._current_name: str | None = None
        # False while the name is one this dialog picked for a draft nobody
        # has saved yet, which is what makes Save ask for it once and never
        # again. Loading a saved query sets it: that name is already the
        # user's.
        self._name_confirmed: bool = False


        self._build_ui()
        self._reload_queries()
        self._reload_tables()

        # Restore after the lists are populated so combo selections exist.
        restore_window_geometry(self, STATE_KEY)
        restore_dialog_state(self, STATE_KEY)

        if query_name:
            self.load_query(query_name)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Build the standard settings/editor two-column shell."""
        self._build_side_panel()
        self._build_editor_panel()

        run_button = create_action_button(
            parent=self, action_id="run", action=self.run, layout=None
        )
        self.action_row.insertWidget(0, run_button)
        table_button = self.add_action("save_as_table", self.save_as_table)
        table_button.setText(_("Save as table"))
        table_button.setToolTip(_("Write what the statement returns as a new table: fixed rows, "
                                  "not read again like a saved query"))
        self.add_action("apply", self.accept, default=True)
        self.add_action("close", self.reject)

    def _build_side_panel(self) -> None:
        """The settings column: saved queries, table and field, snippets - a titled frame each."""
        card = self.add_settings_card(_("Saved queries"))
        layout = card.layout()
        self._query_combo = QComboBox(card)
        self._query_combo.setToolTip(_("Pick a saved query."))
        self._query_combo.currentTextChanged.connect(self._on_saved_query_changed)
        # The combo and its two buttons on one row: the Build frame below needs the height.
        add_row = QWidget(card)
        add_layout = QHBoxLayout(add_row)
        stdSizeAndlayout(add_layout)
        self._query_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._query_combo.setMinimumContentsLength(8)
        self._query_combo.setPlaceholderText(_("No saved query"))
        add_layout.addWidget(self._query_combo, 1)
        create_action_button(parent=add_row, action_id="add", action=self._new_query, layout=add_layout)
        create_action_button(
            parent=add_row, action_id="delete",
            action=self._delete_current_query, layout=add_layout
        )
        layout.addWidget(add_row)

        self._build_operations_card()

        card = self.add_settings_card(_("Table and field"))
        form = QFormLayout()
        stdSizeAndlayout(form)
        self._table_combo = QComboBox(card)
        self._table_combo.currentTextChanged.connect(self._reload_fields)
        form.addRow(_("Table"), self._table_combo)
        self._field_combo = QComboBox(card)
        self._field_combo.setToolTip(_("The field the SQL buttons below act on."))
        form.addRow(_("Field"), self._field_combo)
        card.layout().addLayout(form)

        card = self.add_settings_card(_("Snippets"))
        card.layout().addWidget(self._build_snippet_rows(card))
        self._settings_column.layout().addStretch(1)

    def _build_operations_card(self) -> None:
        """Build: an operation of the Tables menu (join, stack, split...), its form, and Write SQL."""
        card = self.add_settings_card(_("Build"))
        layout = card.layout()
        self._operation_combo = QComboBox(card)
        self._operation_description = QLabel("", card)
        self._operation_description.setWordWrap(True)
        self._operation_description.setProperty("muted", True)
        self._operation_stack = QStackedWidget(card)
        self._operation_forms: list[OperationForm] = []
        for form_class in FORMS:
            self._operation_combo.addItem(_(form_class.NAME))
            form = form_class(self._repo, self._operation_stack)
            self._operation_forms.append(form)
            self._operation_stack.addWidget(form)
        self._operation_combo.currentIndexChanged.connect(self._on_operation_changed)
        layout.addWidget(self._operation_combo)
        layout.addWidget(self._operation_description)
        layout.addWidget(self._operation_stack)
        write = create_action_button(parent=card, action_id="sql_write", action=self.write_operation_sql,
                                     layout=None)
        write.setText(_("Write SQL"))
        write.setToolTip(_("Put this operation's statement in the editor and run it"))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(write)
        layout.addLayout(row)
        self._on_operation_changed(0)

    def _on_operation_changed(self, index: int) -> None:
        self._operation_stack.setCurrentIndex(index)
        self._operation_description.setText(_(FORMS[index].DESCRIPTION))
        # Only the shown form takes height: the others would size the stack to the tallest.
        for position, form in enumerate(self._operation_forms):
            form.setSizePolicy(QSizePolicy.Policy.Preferred,
                               QSizePolicy.Policy.Preferred if position == index else QSizePolicy.Policy.Ignored)
        self._operation_stack.adjustSize()

    def write_operation_sql(self) -> bool:
        """The chosen operation's statement into the editor, then run."""
        form = self._operation_forms[self._operation_combo.currentIndex()]
        try:
            sql = form.sql()
        except ValueError as exc:
            self._status.setText(f"Error - {exc}")
            return False
        self._editor.setPlainText(sql)
        return self.run()

    def save_as_table(self) -> bool:
        """Write the statement's rows as a new table, under a name asked for."""
        if not self.validate():
            show_message(self, "query.invalid", error=self._status.text())
            return False
        suggested = self._repo.free_table_name((self._current_name or "query_table").replace(" ", "_"))
        name, accepted = QInputDialog.getText(self, _("Save as table"), _("Table name:"), text=suggested)
        if not accepted or not name.strip():
            return False
        try:
            created = self._repo.create_table_from_query(name.strip(), self.sql)
        except Exception as exc:  # noqa: BLE001 - shown, the dialog stays open
            applogger.exception("Failed to save the query as a table")
            self._status.setText(f"Error - {exc}")
            return False
        self._status.setText(_("Table '{name}' created.").format(name=created))
        applogger.info("Created table '%s' from a query.", created)
        return True

    def _build_snippet_rows(self, parent: QWidget) -> QWidget:
        """Build the SQL command buttons in a row."""
        container = QWidget(parent)
        outer = QVBoxLayout(container)
        stdSizeAndlayout(outer)
        for first in (0,):
            row = QWidget(container)
            row_layout = QHBoxLayout(row)
            stdSizeAndlayout(row_layout)
            for snippet_id in _SNIPPET_IDS[first:first + 3]:
                create_action_button(
                    parent=row,
                    action_id=f"sql_{snippet_id}",
                    action=lambda _checked=False, key=snippet_id: self._apply_command(key),
                    layout=row_layout,
                )
            row_layout.addStretch(1)
            outer.addWidget(row)
        return container

    def _build_editor_panel(self) -> None:
        """The content column: the SQL editor over the result preview, a titled frame each."""
        card = self.add_content_card(_("SQL"), stretch=2)
        self._editor = QPlainTextEdit(card)
        self._editor.setPlaceholderText(_("SELECT x, y FROM my_table WHERE ..."))
        self._editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._editor.textChanged.connect(self._on_text_changed)
        mark_editor_panel(self._editor)
        card.layout().addWidget(self._editor, 1)

        self._status = QLabel("", card)
        self._status.setWordWrap(True)
        self._status.setProperty("muted", True)
        card.layout().addWidget(self._status, 0)

        card = self.add_content_card(_("Result preview"), stretch=3)
        self._result_view = QTableView(card)
        self._result_view.setAlternatingRowColors(True)
        mark_editor_panel(self._result_view)
        card.layout().addWidget(self._result_view, 1)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def sql(self) -> str:
        """Return the SQL currently in the editor."""
        return self._editor.toPlainText().strip()


    def _reload_queries(self) -> None:
        """Refill the saved-query drop-down without synthetic entries."""
        previous = self._current_name
        self._query_combo.blockSignals(True)
        try:
            self._query_combo.clear()
            selected = -1
            for saved in self._repo.list_queries():
                self._query_combo.addItem(saved.name, saved.name)
                index = self._query_combo.count() - 1
                self._query_combo.setItemData(index, saved.sql, Qt.ItemDataRole.ToolTipRole)
                if saved.name == previous:
                    selected = index
            self._query_combo.setCurrentIndex(selected)
        finally:
            self._query_combo.blockSignals(False)

    def _reload_tables(self) -> None:
        """Refill the table and field drop-downs."""
        self._table_combo.blockSignals(True)
        try:
            self._table_combo.clear()
            self._table_combo.addItems(self._repo.list_table_names())
        except Exception:
            applogger.exception("Failed to list tables for the query builder")
        finally:
            self._table_combo.blockSignals(False)
        self._reload_fields()

    def _reload_fields(self, _table: str = "") -> None:
        """Show the fields belonging to the selected table."""
        self._field_combo.clear()
        table = self._current_table()
        if table:
            self._field_combo.addItems(self._repo.get_columns(table))

    def load_query(self, name: str) -> None:
        """Load a saved query into the editor."""
        saved = self._repo.get_query(name)
        if saved is None:
            applogger.warning("Saved query '%s' not found.", name)
            return
        self._current_name = saved.name
        self._name_confirmed = True
        self._editor.setPlainText(saved.sql)
        self._reload_queries()
        self.validate()

    def _on_saved_query_changed(self, _text: str) -> None:
        name = self._query_combo.currentData()
        if name:
            self.load_query(str(name))

    def _delete_current_query(self) -> None:
        """Delete the saved query the combo is on, after confirmation.

        A saved query is not just a row: charts read through it, so deleting
        one leaves them with no data - which is why this asks first, and why
        the question says so.
        """
        name = str(self._query_combo.currentData() or "").strip()
        if not name:
            self._status.setText(_("Pick a saved query."))
            return

        if not ask(self, "query.confirm_delete", name=name):
            return

        self._repo.delete_query(name)
        applogger.info("Deleted query '%s'.", name)

        if self._current_name == name:
            # The row is gone, so its name is free again and no longer the
            # user's confirmed choice: whatever they type next gets asked
            # about like any other new draft.
            self._current_name = None
            self._name_confirmed = False
            self._editor.clear()
            self._result_view.setModel(None)
        self._reload_queries()

    def _new_query(self) -> None:
        """Start an empty draft. Nothing is asked, and nothing is written.

        New query used to open a name prompt first and ask about overwriting
        an existing query second - two decisions about storing something,
        taken before there was anything to store. A statement that does not
        validate can never be saved, so both questions belong at Save, where
        :meth:`_resolve_name_for_save` now asks them.

        The repository is still what names the draft: ``new_query`` picks the
        first free ``Query <n>`` and writes no row, so the editor opens on a
        blank statement under a name that is already known not to clash.
        """
        draft = self._repo.new_query()
        self._current_name = draft.name
        self._name_confirmed = False
        self._editor.setPlainText(draft.sql)
        self._result_view.setModel(None)
        self._status.setText(_("New query, not saved yet."))
        self._query_combo.blockSignals(True)
        try:
            self._query_combo.setCurrentIndex(-1)
        finally:
            self._query_combo.blockSignals(False)
        self._editor.setFocus()

    def _on_text_changed(self) -> None:
        self._status.setText("")

    def _current_table(self) -> str:
        return self._table_combo.currentText().strip()

    def _current_field(self) -> str:
        return self._field_combo.currentText().strip()

    def _require_table_field(self) -> tuple[str, str] | None:
        table, field = self._current_table(), self._current_field()
        if not table or not field:
            self._status.setText(_("Pick a table and a field."))
            return None
        return table, field

    def _apply_command(self, command: str) -> None:
        handlers = {
            "select": self._select_field,
            "order": self._order_by_field,
            "filter": self._filter_by_field,
        }
        handler = handlers.get(command)
        if handler:
            handler()

    def _select_field(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        table, field = selected
        self._editor.setPlainText(
            f"SELECT {quote_identifier(field)}\nFROM {quote_identifier(table)}"
        )

    @staticmethod
    def _remove_clause(sql: str, clause: str, following: str) -> str:
        pattern = rf"(?is)\s+{clause}\b.*?(?=\s+(?:{following})\b|$)"
        return re.sub(pattern, "", sql).strip().rstrip(";")

    def _order_by_field(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        _table, field = selected
        sql = self.sql or f"SELECT * FROM {quote_identifier(self._current_table())}"
        sql = self._remove_clause(sql, r"ORDER\s+BY", r"LIMIT|OFFSET")
        self._editor.setPlainText(f"{sql}\nORDER BY {quote_identifier(field)} ASC")

    def _filter_by_field(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        table, field = selected
        sql = self.sql or f"SELECT * FROM {quote_identifier(table)}"
        sql = self._remove_clause(
            sql, "WHERE", r"GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET"
        )
        match = re.search(r"(?is)\s+(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b", sql)
        predicate = f"WHERE {quote_identifier(field)} = "
        if match:
            sql = f"{sql[:match.start()].rstrip()}\n{predicate}\n{sql[match.start():].lstrip()}"
        else:
            sql = f"{sql.rstrip()}\n{predicate}"
        self._editor.setPlainText(sql)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def validate(self) -> bool:
        """Check the statement without reading rows; report either way."""
        ok, message = self._repo.validate_query(self.sql)
        self._status.setText(("OK - " if ok else "Error - ") + message)
        return ok

    def run(self) -> bool:
        """Execute the statement and show the first rows."""
        if not self.validate():
            return False

        try:
            frame = self._repo.query_df(
                f"SELECT * FROM ({self.sql.rstrip(';')}) AS _preview "
                f"LIMIT {PREVIEW_ROW_LIMIT}"
            )
        except Exception as exc:
            applogger.exception("Query preview failed")
            self._status.setText(f"Error - {exc}")
            return False

        self._result_view.setModel(DataFrameTableModel(frame, parent=self._result_view))
        self._status.setText(
            f"{len(frame)} row(s) shown"
            + (f" (limited to {PREVIEW_ROW_LIMIT})" if len(frame) >= PREVIEW_ROW_LIMIT else "")
        )
        return True

    def save(self) -> bool:
        """Validate, settle the name, then write the row. In that order.

        Validation first, because a statement that does not run is not worth
        naming - and because the alternative, asking for a name and then
        refusing to save under it, is the flow this dialog used to have.
        """
        if not self.validate():
            show_message(self, "query.invalid", error=self._status.text())
            return False

        clean = self._resolve_name_for_save()
        if clean is None:
            return False

        try:
            self._repo.save_query(clean, self.sql)
        except Exception as exc:
            applogger.exception("Failed to save query '%s'", clean)
            show_message(self, "query.save_failed", error=exc)
            return False

        self._current_name = clean
        self._name_confirmed = True
        self._reload_queries()
        self._status.setText(f"Saved as '{clean}'.")
        applogger.info("Saved query '%s'.", clean)
        return True

    def _resolve_name_for_save(self) -> str | None:
        """Return the name to save under, asking for it once, or None to stop.

        A draft is named the first time it is saved and never asked about
        again, so editing a saved query and pressing Apply repeatedly does not
        re-ask. The two checks that used to guard New query live here instead,
        where they are about something real: a name a table already owns can
        never be selected back (the table always wins), and a name another
        query owns will replace it.
        """
        suggested = (self._current_name or "").strip()
        if self._name_confirmed and suggested:
            return suggested

        name, accepted = QInputDialog.getText(
            self, _("Save query"), _("Query name:"), text=suggested
        )
        if not accepted:
            return None

        clean = name.strip()
        if not clean:
            show_message(self, "query.needs_a_name")
            return None
        if self._repo.check_if_table_exists(clean):
            show_message(self, "query.name_is_a_table", name=clean)
            return None
        if self._repo.get_query(clean) is not None and not ask(
            self, "query.confirm_overwrite", name=clean
        ):
            return None
        return clean

    def accept(self) -> None:  # noqa: D102 - Qt override
        """Save, and close only if it was saved: a statement that does not
        run, or a name prompt cancelled, leaves the dialog open to fix it."""
        self._remember_state()
        if self.save():
            super().accept()

    def reject(self) -> None:  # noqa: D102 - Qt override
        self._remember_state()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802, D102 - Qt override
        self._remember_state()
        super().closeEvent(event)

    def _remember_state(self) -> None:
        """Persist the selected controls and window geometry."""
        save_dialog_state(self, STATE_KEY)
        save_window_geometry(self, STATE_KEY)

    def settings(self) -> dict[str, Any]:
        """Return the dialog state, for callers that want to reopen it."""
        return {"name": self._current_name, "sql": self.sql}
