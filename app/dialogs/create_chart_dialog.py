"""Dialog for creating a chart and its first axes and series.

Chart types are discovered dynamically from the renderer scanner, so a renderer
dropped into ``app/charts`` appears here with its name, documentation link and
role requirements without any registration step.

Renderers deliberately have no icon.  Every chart type had one, each needed a
file in up to three platform folders, and none of them told the user anything
the name did not - the picker is a list of names, and a 32-pixel glyph of a
scatter plot next to the words "Scatter Plot" is decoration, not information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QMessageBox, QListWidget, QListWidgetItem, QSizePolicy, QWidget, QHBoxLayout, QLabel, QRadioButton, QFormLayout, QLineEdit, QComboBox, QPlainTextEdit, QScrollArea, QToolBox

from app.scanners.axis_renderer_scanner import (
    get_renderer,
    import_class_from_file,
    renderers,
)
from app.plugin_api.charts.base import BaseAxisRenderer
from app.logs.logger import applogger
from app.data.repo._common import ensure_read_only_select
from app.data.sqlite_repo import SqliteRepo
from app.data.data_source import DataSource, quote_identifier
from app.utils.messages import show_message
from app.styles.style import (
    apply_toolbox_header_metrics,
    create_doc_link,
    set_doc_link,
    CardFrame,
    TitledCard,
    create_action_button,
    load_icon,
    mark_editor_panel,
    stdPlainTextEdit,
    stdSizeAndlayout,
)
from app.utils.i18n import _, tr
from app.dialogs.two_panels_dialog_base import Column, ColumnsDialog
from app.widgets.function_source_panel import FunctionSourcePanel
from app.utils.config import get_constant


@dataclass(slots=True)
class SeriesDraft:
    """In-memory series configuration edited in the dialog."""
    name: str
    sql: str
    roles: dict[str, str | None] = field(default_factory=dict)
    user_named: bool = False
    #: A column to split the series by: one series per value it takes.
    group_by: str | None = None


@dataclass(slots=True)
class NewPlotTabResult:
    """Result returned by the dialog."""
    figure_id: int
    axis_id: int
    series_count: int


class NewPlotTabDialog(ColumnsDialog):
    """Dialog used to create a figure/axis and one or more plot series.

    Role selection is renderer-driven: the scrollable roles panel is rebuilt
    from ``renderer.required`` and ``renderer.optional`` whenever the selected
    renderer changes.  Each role can be mapped to a table/query column and is
    persisted through ``SqliteRepo.create_series_descriptor(..., roles=...)``.
    """
    Icon = """
        <path d="M4 19.5h16"/>
        <path d="M4.5 19V5"/>
        <path d="M7 15l3.2-3.2 2.5 2.5L18 8"/>
        <path d="M15.5 8H18v2.5"/>
        """
    def __init__(
        self,
        repo: SqliteRepo,
        *,
        current_figure_id: int | None = None,
        current_table: str | None = None,
        preferred_columns: Sequence[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # The shell - title, icon, padding, three columns with handles between
        # them, the row of buttons - is ColumnsDialog's.
        super().__init__(parent, title=_("New Chart"), icon=load_icon("new_plot"), size="large", resizable=True,
                         columns=(Column(stretch=0, minimum=220), Column(minimum=380), Column(minimum=260)))

        self._repo = repo
        #: Columns picked in the table preview, mapped onto x, y and z in
        #: that order the first time the role combos are filled.
        self._preferred_columns: list[str] = [str(c) for c in (preferred_columns or [])]
        self._current_figure_id = current_figure_id
        self._current_table = current_table
        self._result: NewPlotTabResult | None = None
        self._series_by_item: dict[int, SeriesDraft] = {}
        self.renderer: dict[str, object] | None = None
        self._renderer_class: type[BaseAxisRenderer] | None = None
        self._figure_name_user_modified = False
        self._axis_name_user_modified = False
        #: The stored function chosen as the data source (its __sympy__ row), or None for a table or a query.
        self._function_data: dict[str, Any] | None = None

        self.setModal(False)

        # ------------------------------------------------------------------
        # Left: renderer list.
        # ------------------------------------------------------------------
        self._search_types = QLineEdit()
        self._search_types.setPlaceholderText(_("Search chart types..."))
        self._search_types.setToolTip(
            _("Filter by name or category. Sections with no match are hidden.")
        )
        # The clear button is the standard search affordance on both platforms,
        # and it is one fewer widget than a button beside the field.
        self._search_types.setClearButtonEnabled(True)

        # One accordion section per Matplotlib category, each holding the
        # renderers of that family.  Same widget as the properties panels, so
        # the two sides of the window behave the same way.
        self._types_toolbox = QToolBox()
        self._lists_by_category: dict[str, QListWidget] = {}
        self._fill_renderer_toolbox()

        self._select_first_renderer()
        selected = self._current_item()
        if selected is not None:
            self.renderer = get_renderer(str(selected.data(Qt.ItemDataRole.UserRole)))
            self._renderer_class = self._load_renderer_class()

        # ------------------------------------------------------------------
        # Left: selected renderer details.
        # ------------------------------------------------------------------
        self._renderer_name = QLabel()
        self._renderer_name.setProperty("role", "subtitle")
        self._renderer_name.setStyleSheet("font-weight: 600;")
        self._renderer_name.setWordWrap(True)

        self._renderer_description = QLabel()
        self._renderer_description.setWordWrap(True)
        self._renderer_description.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._renderer_link = create_doc_link()
        self._renderer_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )

        renderer_details = CardFrame(self, "rendererDetailsCard")
        renderer_details_layout = renderer_details.layout()
        renderer_details_layout.addWidget(self._renderer_name)

        renderer_details_layout.addWidget(self._renderer_link)
        self._renderer_details = renderer_details

        # ------------------------------------------------------------------
        # Left: target figure/axis.
        # ------------------------------------------------------------------
        self._rb_new_figure = QRadioButton(_("Create new figure (new tab)"))
        self._rb_current_figure = QRadioButton(_("Use current figure"))
        self._rb_new_figure.setChecked(True)

        if self._current_figure_id is None:
            self._rb_current_figure.setEnabled(False)
            self._rb_current_figure.setToolTip(_("No current figure selected."))

        self._edit_figure_name = QLineEdit()
        self._edit_figure_name.setPlaceholderText(_("Figure name"))

        # Titles above the boxes, as in every other window (TitledCard).
        fig_group = TitledCard(self, _("Target"), "plotTargetCard")
        fig_main = fig_group.card.layout()

        fig_form = QWidget(fig_group.card)
        fig_layout = QFormLayout(fig_form)
        stdSizeAndlayout(fig_layout)
        fig_layout.addRow(self._rb_new_figure)
        fig_layout.addRow(self._rb_current_figure)
        fig_layout.addRow(_("Name:"), self._edit_figure_name)
        fig_main.addWidget(fig_form)

        self._rb_new_axis = QRadioButton(_("Add new axis"))
        self._rb_existing_axis = QRadioButton(_("Use existing axis"))
        self._rb_new_axis.setChecked(True)

        self._combo_axes = QComboBox()
        self._combo_axes.setEnabled(False)

        self._edit_axis_name = QLineEdit()
        self._edit_axis_name.setPlaceholderText(_("Axis name"))

        axis_form = QWidget(fig_group.card)
        axis_layout = QFormLayout(axis_form)
        stdSizeAndlayout(axis_layout)
        axis_layout.addRow(self._rb_new_axis)
        axis_layout.addRow(self._rb_existing_axis)
        axis_layout.addRow(_("Axis name:"), self._edit_axis_name)
        # One axis per value of a column (B-13): the same series on each,
        # filtered to that value - small multiples, one panel per group.
        self._combo_axes_by = QComboBox()
        self._combo_axes_by.addItem(_("(none)"), None)
        self._combo_axes_by.setToolTip(_(
            "One axis per value of this column - one per species, per batch, per site - "
            "each with every series, filtered to that value."
        ))
        self._combo_axes_by.currentIndexChanged.connect(lambda _index: self._sync_controls())
        axis_layout.addRow(_("Separate axes by:"), self._combo_axes_by)
        self._axis_form_layout = axis_layout

        axis_row = QWidget(fig_group.card)
        axis_row_layout = QHBoxLayout(axis_row)
        stdSizeAndlayout(axis_row_layout)
        axis_row_layout.addWidget(QLabel(_("Axis:")))
        axis_row_layout.addWidget(self._combo_axes, 1)
        axis_layout.addRow(axis_row)
        fig_main.addWidget(axis_form)

        # ------------------------------------------------------------------
        # Series list.
        # ------------------------------------------------------------------
        self._series_list = QListWidget()
        self._series_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._series_list.setAlternatingRowColors(True)
        self._series_list.setUniformItemSizes(True)
        self._series_list.setSpacing(1)
        mark_editor_panel(self._series_list)
        self._series_list.setMinimumWidth(180)
        self._series_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        series_list_group = TitledCard(self, _("Series"), "plotSeriesListCard")
        series_list_group.setMinimumWidth(220)
        series_list_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        series_list_layout = series_list_group.card.layout()

        series_header = QWidget(series_list_group.card)
        series_header_layout = QHBoxLayout(series_header)
        stdSizeAndlayout(series_header_layout)
        series_header_layout.addStretch(1)
        create_action_button(
            parent=series_header,
            action_id="add_series",
            action=self._on_add_series,
            layout=series_header_layout,
        )
        create_action_button(
                                      parent=series_header,
                                      action_id="delete",
                                      action=self._on_remove_series,
                                      layout=series_header_layout,
                                  )

        # Says, when it applies, that the entries below are not going to end
        # up on one axes together.  Hidden the rest of the time: a note that
        # is always there is a note nobody reads.
        self._series_split_note = QLabel(series_list_group.card)
        self._series_split_note.setWordWrap(True)
        self._series_split_note.setProperty("muted", True)
        self._series_split_note.hide()

        series_list_layout.addWidget(series_header)
        series_list_layout.addWidget(self._series_split_note)
        series_list_layout.addWidget(self._series_list, 1)


        # Series editor.
        self._combo_table = QComboBox()
        self._combo_table.setEditable(False)
        # A stored function as the source: its parameters and range, as this chart draws it.
        self._function_panel = FunctionSourcePanel(self)

        self._edit_series_name = QLineEdit()
        self._edit_series_name.setPlaceholderText(_("Series name"))

        self._edit_sql = QPlainTextEdit()
        stdPlainTextEdit(self._edit_sql)     
        self._edit_sql.setPlaceholderText(_("SQL query (SELECT / WITH ...)"))


        self._role_combos: dict[str, QComboBox] = {}
        self._role_labels: dict[str, QLabel] = {}
        self._roles_panel = QWidget()
        self._roles_layout = QFormLayout(self._roles_panel)
        stdSizeAndlayout(self._roles_layout)

        self._roles_scroll = QScrollArea()
        self._roles_scroll.setObjectName("plotRolesScroll")
        self._roles_scroll.setWidgetResizable(True)
        self._roles_scroll.setMinimumHeight(110)
        stdSizeAndlayout(self._roles_scroll)
        # On the Series card's own white, like the fields above it - not a
        # grey editor panel set into it (B-12).
        self._roles_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._roles_scroll.setStyleSheet(
            "QScrollArea#plotRolesScroll, QScrollArea#plotRolesScroll > QWidget > QWidget { background: transparent; }"
        )
        self._roles_scroll.setWidget(self._roles_panel)
        self._rebuild_role_combos()

        self._combo_group = QComboBox()
        self._combo_group.addItem(_("(none)"), None)
        self._combo_group.setToolTip(_(
            "Split the series into one per value of this column - one per "
            "species, per batch, per site - each with its own colour and "
            "legend entry. The columns the chart already uses are not offered."
        ))
        self._combo_group.currentIndexChanged.connect(self._on_group_changed)

        editor_group = TitledCard(self, _("Series"), "plotSeriesEditorCard")
        editor_layout = editor_group.card.layout()

        editor_form = QWidget(editor_group.card)
        editor_form_layout = QFormLayout(editor_form)
        stdSizeAndlayout(editor_form_layout)
        editor_form_layout.addRow(_("Data source:"), self._combo_table)
        self._function_panel.insert_rows(editor_form_layout, editor_form_layout.rowCount())
        editor_form_layout.addRow(_("Series name:"), self._edit_series_name)
        editor_form_layout.addRow(_("Query:"), self._edit_sql)
        editor_form_layout.addRow(self._roles_scroll)
        editor_form_layout.addRow(_("Group by:"), self._combo_group)
        self._editor_form_layout = editor_form_layout
        self._show_group_by()
        editor_layout.addWidget(editor_form, 1)

        # Dialog buttons and layout.
        #
        # A card like the other two columns, not a plain QWidget: it used to
        # be the one column with no white "card" surface of its own, so it
        # showed the dialog's own grey background straight through - the
        # "grigio e bianco" (grey and white) the panels looked inconsistent
        # for was this column against the other two, not a shade mismatch
        # within any one of them.
        renderer_panel = TitledCard(self, _("Renderers"), "plotRendererCard")
        renderer_panel.setMinimumWidth(220)
        renderer_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        renderer_layout = renderer_panel.card.layout()
        renderer_layout.addWidget(self._search_types, 0)
        renderer_layout.addWidget(self._types_toolbox, 1)
        renderer_layout.addWidget(self._renderer_details, 0)

        editor_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.column(0).layout().addWidget(renderer_panel, 1)
        self.column(1).layout().addWidget(editor_group, 1)
        self.column(2).layout().addWidget(fig_group, 0)
        self.column(2).layout().addWidget(series_list_group, 1)
        assert self.splitter is not None
        self.splitter.setSizes([230, 430, 330])
        self.add_action("apply", self._on_accept)
        self.add_action("close", self.reject)
    
        # ------------------------------------------------------------------
        # Wiring.
        # ------------------------------------------------------------------

        self._rb_current_figure.toggled.connect(self._sync_controls)
        self._rb_new_axis.toggled.connect(self._sync_controls)
        self._rb_existing_axis.toggled.connect(self._sync_controls)
        self._search_types.textChanged.connect(self._filter_renderer_toolbox)
        for renderer_list in self._lists_by_category.values():
            renderer_list.currentItemChanged.connect(self._on_chart_type_changed)
        self._combo_table.currentIndexChanged.connect(self._on_table_changed)
        self._function_panel.changed.connect(self._update_sql)
        self._series_list.currentItemChanged.connect(self._on_series_selected)

        self._edit_figure_name.textEdited.connect(self._on_figure_name_edited)
        self._edit_axis_name.textEdited.connect(self._on_axis_name_edited)
        self._edit_series_name.textEdited.connect(self._on_series_name_changed)
        self._edit_sql.textChanged.connect(self._on_sql_changed)

        self._reload_tables()
        self._reload_axes()
        self._sync_controls()
        self._set_default_figure_name(force=True)
        self._set_default_axis_name(force=True)
        self._update_renderer_details()

        # Start with one ready-to-edit series so the dialog is usable by
        # default and the generated SQL is visible immediately.
        self._on_add_series()
        self._sync_series_mode()

    @property
    def chart_result(self) -> NewPlotTabResult | None:
        """Return the dialog result after accept."""
        return self._result

    # Matplotlib's own order, from https://matplotlib.org/stable/plot_types/.
    # Kept rather than sorted alphabetically so the list reads the same way as
    # the documentation the renderers wrap, and so "Pairwise data" - the plots
    # people reach for most - stays at the top.
    CATEGORY_ORDER: tuple[str, ...] = (
        "Pairwise data",
        "Statistical distributions",
        "Diagnostic plots",
        "Gridded data",
        "Irregularly gridded data",
        "3D and volumetric data",
    )

    def _fill_renderer_toolbox(self) -> None:
        """Build one accordion section per category, each a list of renderers.

        A list per section rather than one list with heading rows: the sections
        collapse, which is the point of an accordion on a panel this narrow, and
        it means the search can hide a whole family in one move.
        """
        for category in self._categories_in_order():
            renderer_list = QListWidget()
            renderer_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            renderer_list.setUniformItemSizes(True)
            mark_editor_panel(renderer_list)

            for renderer in self._renderers_in(category):
                # Shown translated, stored raw: the name in UserRole is the
                # chart_type written to the database and matched against
                # elsewhere, so only the label a user reads goes through tr().
                item = QListWidgetItem(tr(str(renderer["value"])))
                item.setData(Qt.ItemDataRole.UserRole, renderer["value"])
                item.setToolTip(tr(str(renderer.get("description") or "")))
                renderer_list.addItem(item)

            # The list scrolls, filling its accordion page; asking for next to
            # no height, it never makes the page scroll too - one scroll bar,
            # not two side by side (B-12).
            renderer_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            renderer_list.setMinimumHeight(40)
            renderer_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
            self._lists_by_category[category] = renderer_list
            self._types_toolbox.addItem(renderer_list, tr(category))

        apply_toolbox_header_metrics(self._types_toolbox)

    def _categories_in_order(self) -> list[str]:
        """Known categories first, in Matplotlib's order, then any others.

        A renderer with a category nobody has heard of still appears - in its
        own section, at the end - rather than vanishing from the dialog.
        """
        present = {str(renderer.get("category") or "") for renderer in renderers}
        known = [name for name in self.CATEGORY_ORDER if name in present]
        return known + sorted(present - set(known))

    def _renderers_in(self, category: str) -> list[dict]:
        return [r for r in renderers if str(r.get("category") or "") == category]

    # ------------------------------------------------------------------
    # Selection across several lists
    # ------------------------------------------------------------------
    def _current_item(self) -> QListWidgetItem | None:
        """Return the selected renderer item, from the active toolbox page.

        Only one category is ever visible at a time - the toolbox shows one
        page - so the current item is unambiguously that page's own list's
        ``currentItem()``. This used to scan every list for the first whose
        ``currentItem()`` also had ``isSelected()`` true, which broke two
        ways: on a real mouse click - as opposed to a programmatic
        ``setCurrentRow()`` - Qt does not guarantee ``isSelected()`` is
        already true at the exact moment ``currentItemChanged`` fires, so a
        freshly clicked item briefly read back as unselected and the details
        panel fell back to a hardcoded "Axis"; and a list that is not even
        visible can still be carrying a *stale* current item from before the
        user switched categories, which - iterated first, dict order being
        what it is - won the read over the section actually on screen.
        """
        current_widget = self._types_toolbox.currentWidget()
        if isinstance(current_widget, QListWidget):
            return current_widget.currentItem()
        return None

    def _clear_other_selections(self, keep: QListWidget) -> None:
        """Keep the selection exclusive across sections.

        Qt makes each list exclusive on its own, so without this two sections
        could each show a highlighted row and the dialog would look as if two
        chart types were chosen.
        """
        for renderer_list in self._lists_by_category.values():
            if renderer_list is not keep:
                renderer_list.clearSelection()
                # setCurrentRow(-1) rather than setCurrentItem(None): both
                # clear the current item, but only this one is typed for it.
                renderer_list.setCurrentRow(-1)

    #: Selected on open when present, ahead of whatever sits first in
    #: Matplotlib's own category order. Scatter is the chart almost every new
    #: figure turns out to be, and the general "just plot the data" case
    #: users reach for.
    DEFAULT_RENDERER: str = "Scatter Plot"

    def _select_first_renderer(self) -> None:
        """Select the default renderer, or the first one otherwise."""
        if self._select_renderer_by_name(self.DEFAULT_RENDERER):
            return

        for index, renderer_list in enumerate(self._lists_by_category.values()):
            if renderer_list.count() == 0 or renderer_list.isHidden():
                continue
            for row in range(renderer_list.count()):
                if not renderer_list.item(row).isHidden():
                    # Toolbox page first: _current_item() reads the active
                    # page's own currentItem(), so setting the row before the
                    # page is switched would have it looking at the wrong
                    # list for as long as that order was reversed.
                    self._types_toolbox.setCurrentIndex(index)
                    renderer_list.setCurrentRow(row)
                    self._clear_other_selections(renderer_list)
                    return

    def select_chart_type(self, name: str) -> bool:
        """Choose the chart type *name* (the search opens New plot on one); whether it exists."""
        return self._select_renderer_by_name(name)

    def _select_renderer_by_name(self, name: str) -> bool:
        """Select the renderer *name* if it exists in any section.

        Returns whether it was found, so a caller can fall back when it is
        not - a renderer this dialog no longer ships must not be a reason to
        select nothing at all.
        """
        for index, renderer_list in enumerate(self._lists_by_category.values()):
            if renderer_list.isHidden():
                continue
            for row in range(renderer_list.count()):
                item = renderer_list.item(row)
                # Not a match while an active search has hidden it: forcing
                # it into view would silently override what the user typed,
                # and calling this again after every keystroke - it runs
                # whenever the current selection drops out of the filter -
                # would make the default win the search outright.
                if item.isHidden():
                    continue
                if str(item.data(Qt.ItemDataRole.UserRole)) == name:
                    # Toolbox page first - see the matching comment in
                    # _select_first_renderer.
                    self._types_toolbox.setCurrentIndex(index)
                    renderer_list.setCurrentRow(row)
                    self._clear_other_selections(renderer_list)
                    return True
        return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _filter_renderer_toolbox(self, text: str) -> None:
        """Show only renderers matching *text*, and only sections that have any.

        Matching on the category as well as the name, so typing "stat" brings
        up the whole statistical family - the categories are half of why the
        list is worth searching.

        Both spellings of every name are matched - the English one stored in
        UserRole and the translated one on screen - so a search works whether
        the word that came to mind was the one being read or the one in the
        Matplotlib documentation it came from.
        """
        needle = str(text or "").strip().lower()

        # Indexed by position, not by header text: the header gains a match
        # count while a search is active, so it is not a stable key.
        for index, category in enumerate(self._lists_by_category):
            renderer_list = self._lists_by_category[category]
            translated_category = tr(category)
            category_matches = (
                needle in category.lower() or needle in translated_category.lower()
            )

            visible = 0
            for row in range(renderer_list.count()):
                item = renderer_list.item(row)
                matches = (
                    not needle
                    or category_matches
                    or needle in str(item.data(Qt.ItemDataRole.UserRole)).lower()
                    or needle in item.text().lower()
                )
                item.setHidden(not matches)
                visible += int(matches)

            # An empty section is disabled rather than removed: removing pages
            # renumbers every index after it, and the numbering is what maps a
            # section back to its list.
            self._types_toolbox.setItemEnabled(index, visible > 0)
            self._types_toolbox.setItemText(
                index,
                translated_category
                if not needle
                else f"{translated_category}  ({visible})",
            )

        # Bound once: two calls could disagree, and the second one is what the
        # type checker cannot see is non-None.
        current = self._current_item()
        if current is None or current.isHidden():
            self._select_first_renderer()

    def _load_renderer_class(self) -> type[BaseAxisRenderer] | None:
        """Import and return the selected renderer class."""
        if self.renderer is None:
            return None
        renderer_class = import_class_from_file(self.renderer)
        return cast(type[BaseAxisRenderer] | None, renderer_class)

    def _current_renderer_class(self) -> type[BaseAxisRenderer] | None:
        """Return the cached renderer class for the selected renderer dict."""
        if self._renderer_class is None:
            self._renderer_class = self._load_renderer_class()
        return self._renderer_class

    # ------------------------------------------------------------------
    # Renderer metadata.
    # ------------------------------------------------------------------
    def _update_renderer_details(self) -> None:
        """Refresh the name and the documentation link from the renderer class."""
        renderer_class = self._current_renderer_class()
        if renderer_class is None:
            self._renderer_name.clear()
            self._renderer_description.clear()
            set_doc_link(self._renderer_link, "", "")
            self._renderer_link.hide()
            return

        renderer_name = self._current_renderer_name()
        link = str(renderer_class.Link or "").strip()

        self._renderer_name.setText(renderer_name or renderer_class.__name__)
        self._renderer_description.clear()

        # set_doc_link() clears the tooltip along with the text when link is
        # empty - going through it unconditionally is what keeps a renderer
        # with no documentation from showing the previous renderer's URL as
        # its tooltip, which set_doc_link()-only-when-truthy used to leave
        # behind: the label's own text emptied, but setToolTip("") never ran.
        set_doc_link(self._renderer_link, _("Documentation"), link)
        self._renderer_link.setVisible(bool(link))

    # ------------------------------------------------------------------
    # Load UI data.
    # ------------------------------------------------------------------
    def _current_table_name(self) -> str:
        """Return the selected data source table name."""
        table = self._combo_table.currentData()
        if isinstance(table, str) and table.strip():
            return table.strip()
        return ""

    def _current_renderer_name(self) -> str:
        """Return the selected renderer's name.

        From the item's data, not its text: the text is indented so that the
        renderer sits under its category heading, and stripping the indent back
        off would be one more place that has to agree about the layout.
        """
        item = self._current_item()
        if item is None:
            return "Axis"

        return str(item.data(Qt.ItemDataRole.UserRole) or "Axis")

    def _default_figure_name(self) -> str:
        """Default figure name is the selected data source."""
        return self._current_table_name() or "Figure"

    def _default_axis_name(self) -> str:
        """Default axis name is the selected renderer."""
        return self._current_renderer_name()

    def _role_value_from_mapping(
        self,
        roles: dict[str, str | None],
        role_name: str,
    ) -> str | None:
        """Return one role value from a mapping, matching keys case-insensitively."""
        if role_name in roles:
            return roles.get(role_name)
        role_name_lower = role_name.lower()
        for key, value in roles.items():
            if key.lower() == role_name_lower:
                return value
        return None

    def _default_series_name_from_roles(self, roles: dict[str, str | None]) -> str:
        """Default series name is '<Data source> - <Y/Value role>'."""
        table_name = self._current_table_name()
        value_name = (
            self._role_value_from_mapping(roles, "y")
            or self._role_value_from_mapping(roles, "value")
            or self._role_value_from_mapping(roles, "values")
            or self._role_value_from_mapping(roles, "z")
            or "Series"
        )
        if table_name:
            return f"{table_name} - {value_name}"
        return str(value_name)

    def _set_default_figure_name(self, *, force: bool = False) -> None:
        """Set the figure name while preserving manual edits."""
        if self._figure_name_user_modified and not force:
            return
        self._edit_figure_name.blockSignals(True)
        try:
            self._edit_figure_name.setText(self._default_figure_name())
        finally:
            self._edit_figure_name.blockSignals(False)
        self._figure_name_user_modified = False

    def _set_default_axis_name(self, *, force: bool = False) -> None:
        """Set the axis name while preserving manual edits."""
        if self._axis_name_user_modified and not force:
            return
        self._edit_axis_name.blockSignals(True)
        try:
            self._edit_axis_name.setText(self._default_axis_name())
        finally:
            self._edit_axis_name.blockSignals(False)
        self._axis_name_user_modified = False

    def _set_default_series_name(
        self,
        item: QListWidgetItem,
        draft: SeriesDraft,
        *,
        force: bool = False,
    ) -> None:
        """Set the series name while preserving manual edits."""
        if draft.user_named and not force:
            return
        draft.name = self._default_series_name_from_roles(draft.roles)
        item.setText(draft.name)
        if item is self._current_series_item():
            self._edit_series_name.blockSignals(True)
            try:
                self._edit_series_name.setText(draft.name)
            finally:
                self._edit_series_name.blockSignals(False)
        draft.user_named = False

    def _on_figure_name_edited(self, _text: str) -> None:
        self._figure_name_user_modified = True

    def _on_axis_name_edited(self, _text: str) -> None:
        self._axis_name_user_modified = True

    def _reload_tables(self) -> None:
        """Load every data source - tables and saved queries - into the combo.

        Saved queries are marked in the label but stored under the same plain
        name in the item data, so everything downstream keeps working with a
        single string and resolves the kind through the repository.
        """
        self._combo_table.clear()
        df = self._repo.list_data_sources(functions=True)
        if df is None or df.empty:
            return

        for row in df.itertuples(index=False):
            name = str(row.Table)
            kind = str(getattr(row, "kind", "table"))
            self._combo_table.addItem(f"{name}  ({kind})" if kind in ("query", "function") else name, name)

        if self._combo_table.count() <= 0:
            return

        index = 0
        if self._current_table:
            found = self._combo_table.findData(self._current_table)
            if found < 0:
                found = self._combo_table.findText(self._current_table)
            if found >= 0:
                index = found

        self._combo_table.setCurrentIndex(index)
        self._on_table_changed(index)

    def _reload_axes(self) -> None:
        """Load existing axes for the current figure."""
        self._combo_axes.clear()
        if self._current_figure_id is None:
            return

        axes = self._repo.list_axes_for_figure(self._current_figure_id)
        for axis_id, axis_index, title in axes:
            label = f"{axis_index}: {title}" if title else f"{axis_index}: (no title)"
            self._combo_axes.addItem(label, int(axis_id))

        if self._combo_axes.count() == 0:
            self._rb_existing_axis.setEnabled(False)

    def _sync_controls(self) -> None:
        """Enable/disable target controls based on figure/axis choices."""
        using_current = self._rb_current_figure.isChecked()
        self._edit_figure_name.setEnabled(not using_current)

        # An existing axis is one axis: it cannot hold a set of series the
        # renderer wants spread over several, so the choice is withdrawn
        # rather than accepted and then refused on Create.
        splitting = len(self._axis_groups(self._ordered_drafts())) > 1
        can_use_existing_axis = (
            using_current and self._combo_axes.count() > 0 and not splitting
        )
        self._rb_existing_axis.setEnabled(can_use_existing_axis)
        self._combo_axes.setEnabled(
            can_use_existing_axis and self._rb_existing_axis.isChecked()
        )
        self._rb_existing_axis.setToolTip(
            _("One axis cannot hold this many series of this chart type.")
            if splitting
            else ""
        )
        # Each axis of a split takes its own series' name, so there is nothing
        # for one axis-name field to name.
        separate = self._rb_new_axis.isChecked() and self._combo_axes_by.currentData() is not None
        self._edit_axis_name.setEnabled(self._rb_new_axis.isChecked() and not splitting and not separate)
        # Separate axes are new axes: nothing to choose with an existing one.
        self._combo_axes_by.setEnabled(self._rb_new_axis.isChecked())

        if not can_use_existing_axis:
            self._rb_new_axis.setChecked(True)

    # ------------------------------------------------------------------
    # How many series fit on one axes.
    # ------------------------------------------------------------------
    def _max_series_per_axis(self) -> int | None:
        """Return the selected renderer's series limit, or None for no limit.

        Read from the renderer class rather than from a list of chart-type
        names kept here: the renderers that draw one series already say so in
        their own code, and a second list in the dialog would be a second
        place to forget a new one.
        """
        limit = getattr(self._current_renderer_class(), "MaxSeries", None)
        if limit is None:
            return None
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            applogger.warning(
                "Renderer %r declares an unusable MaxSeries=%r; treating it as "
                "unlimited.",
                self._current_renderer_name(),
                limit,
            )
            return None
        return limit if limit > 0 else None

    def _ordered_drafts(self) -> list[SeriesDraft]:
        """Return the drafts in the order they are shown in the list.

        Which axis a series lands on now depends on this order, so it is read
        from the widget rather than from the insertion order of the map.
        """
        return [
            self._draft_for_item(item)
            for item in (
                self._series_list.item(row) for row in range(self._series_list.count())
            )
            if item is not None
        ]

    def _axis_groups(self, drafts: list[SeriesDraft]) -> list[list[SeriesDraft]]:
        """Split *drafts* into the groups that each share one axes."""
        limit = self._max_series_per_axis()
        if limit is None or not drafts:
            return [list(drafts)] if drafts else []
        return [drafts[start : start + limit] for start in range(0, len(drafts), limit)]

    def _sync_series_mode(self) -> None:
        """Say what will happen to the series list, and re-sync the target."""
        limit = self._max_series_per_axis()
        groups = self._axis_groups(self._ordered_drafts())

        if limit is None or len(groups) <= 1:
            self._series_split_note.clear()
            self._series_split_note.hide()
        else:
            chart_type = self._current_renderer_name()
            self._series_split_note.setText(
                _(
                    "{chart} draws one series per axis: the {count} below will "
                    "each get an axis of their own."
                ).format(chart=chart_type, count=len(groups))
                if limit == 1
                else _(
                    "{chart} draws up to {limit} series per axis: the {count} "
                    "below will be split across {axes} axes."
                ).format(
                    chart=chart_type,
                    limit=limit,
                    count=len(self._ordered_drafts()),
                    axes=len(groups),
                )
            )
            self._series_split_note.show()
        self._sync_controls()

    # ------------------------------------------------------------------
    # Series list management.
    # ------------------------------------------------------------------
    def _set_series_editor_enabled(self, enabled: bool) -> None:
        """Enable/disable all series editor widgets."""
        widgets = (
            self._edit_series_name,
            self._edit_sql,
            self._roles_scroll,
            *self._role_combos.values(),
        )
        for widget in widgets:
            widget.setEnabled(enabled)

    def _current_series_item(self) -> QListWidgetItem | None:
        """Return the selected series item."""
        return self._series_list.currentItem()

    def _draft_for_item(self, item: QListWidgetItem) -> SeriesDraft:
        """Return the draft associated with a QListWidgetItem."""
        key = id(item)
        draft = self._series_by_item.get(key)
        if draft is None:
            draft = SeriesDraft(name=item.text(), sql="", user_named=True)
            self._series_by_item[key] = draft
        return draft

    def _on_add_series(self, _checked: bool = False) -> None:
        """Add a new series draft and select it."""
        roles = self._current_role_mapping()
        name = self._default_series_name_from_roles(roles)
        item = QListWidgetItem(name)
        self._series_list.addItem(item)
        self._series_by_item[id(item)] = SeriesDraft(
            name=name,
            sql=self._default_sql_for_current_table(),
            roles=roles,
            user_named=False,
        )
        self._series_list.setCurrentItem(item)
        self._sync_series_mode()

    def _on_remove_series(self, _checked: bool = False) -> None:
        """Remove the selected series draft."""
        item = self._current_series_item()
        if item is None:
            return

        row = self._series_list.row(item)
        self._series_list.takeItem(row)
        self._series_by_item.pop(id(item), None)

        if self._series_list.count() > 0:
            self._series_list.setCurrentRow(max(0, row - 1))
        else:
            self._set_series_editor_enabled(False)
        self._sync_series_mode()

    # ------------------------------------------------------------------
    # Series editor bindings.
    # ------------------------------------------------------------------
    def _on_series_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Load the selected series draft into the editor."""
        if current is None:
            self._edit_series_name.clear()
            self._edit_sql.clear()
            self._set_series_editor_enabled(False)
            return

        self._set_series_editor_enabled(True)
        self._load_draft_into_editor(self._draft_for_item(current))

    def _load_draft_into_editor(self, draft: SeriesDraft) -> None:
        """Load a draft into the editor without recursive updates."""
        self._edit_series_name.blockSignals(True)
        self._edit_sql.blockSignals(True)
        try:
            self._edit_series_name.setText(draft.name)
            self._edit_sql.setPlainText(draft.sql)
        finally:
            self._edit_series_name.blockSignals(False)
            self._edit_sql.blockSignals(False)

        self._populate_role_columns(self._columns_from_table())
        self._set_role_selection(draft)
        self._populate_group_columns()

    def _set_role_selection(self, draft: SeriesDraft) -> None:
        """Select role combo values from the given draft."""
        for key, combo in self._role_combos.items():
            value = draft.roles.get(key)
            combo.blockSignals(True)
            try:
                if value is None:
                    combo.setCurrentIndex(0)
                    continue
                index = combo.findData(value)
                combo.setCurrentIndex(index if index >= 0 else 0)
            finally:
                combo.blockSignals(False)

    def _current_role_mapping(self) -> dict[str, str | None]:
        """Return the current role-to-column selection."""
        roles: dict[str, str | None] = {}
        for key, combo in self._role_combos.items():
            data = combo.currentData()
            roles[key] = str(data) if isinstance(data, str) and data else None
        return roles

    def _persist_editor_to_draft(self) -> None:
        """Copy editor state into the selected draft."""
        item = self._current_series_item()
        if item is None:
            return

        draft = self._draft_for_item(item)
        fallback_name = item.text()
        draft.name = (
            (self._edit_series_name.text() or fallback_name).strip()
            or fallback_name
        )
        draft.sql = (self._edit_sql.toPlainText() or "").strip()

        draft.roles = self._current_role_mapping()
        data = self._combo_group.currentData()
        draft.group_by = str(data) if isinstance(data, str) and data else None

    def _on_series_name_changed(self, text: str) -> None:
        """Keep the list item and draft name synchronized."""
        item = self._current_series_item()
        if item is None:
            return
        item.setText(text)
        draft = self._draft_for_item(item)
        draft.user_named = True
        self._persist_editor_to_draft()

    def _on_sql_changed(self) -> None:
        """Persist edited SQL into the current draft."""
        self._persist_editor_to_draft()

    def _on_roles_changed(self, _index: int = 0) -> None:
        """Persist role changes and refresh SQL automatically."""
        self._persist_editor_to_draft()
        self._update_sql()
        self._populate_group_columns()

    def _on_group_changed(self, _index: int = 0) -> None:
        self._persist_editor_to_draft()

    def _populate_group_columns(self) -> None:
        """Offer every column of the source the chart's roles do not use."""
        used = {value for value in self._current_role_mapping().values() if value}
        item = self._current_series_item()
        wanted = self._draft_for_item(item).group_by if item is not None else self._combo_group.currentData()
        self._combo_group.blockSignals(True)
        try:
            self._combo_group.clear()
            self._combo_group.addItem(_("(none)"), None)
            for column in self._columns_from_table():
                if column not in used:
                    self._combo_group.addItem(column, column)
            index = self._combo_group.findData(wanted) if wanted else 0
            self._combo_group.setCurrentIndex(max(index, 0))
        finally:
            self._combo_group.blockSignals(False)
        if item is not None:
            data = self._combo_group.currentData()
            self._draft_for_item(item).group_by = str(data) if isinstance(data, str) and data else None
        self._populate_axes_by_columns()

    def _populate_axes_by_columns(self) -> None:
        """Every column of the source, for Separate axes by; the choice kept when it is still there."""
        combo = getattr(self, "_combo_axes_by", None)
        if combo is None:
            return
        wanted = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(_("(none)"), None)
            for column in self._columns_from_table():
                combo.addItem(column, column)
            index = combo.findData(wanted) if wanted else 0
            combo.setCurrentIndex(max(index, 0))
        finally:
            combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Columns / roles.
    # ------------------------------------------------------------------
    def _current_source(self) -> DataSource | None:
        """The selected source; a stored function at the parameters and range chosen for this chart."""
        name = self._combo_table.currentData()
        if not isinstance(name, str) or not name:
            return None
        source = self._repo.get_data_source(name)
        if source is not None and source.is_function and self._function_data is not None:
            return DataSource.function(source.name, self._function_panel.source_sql())
        return source

    def _sync_function_source(self) -> None:
        """A stored function chosen: its parameters and range shown, from the store; hidden for the rest."""
        name = self._combo_table.currentData()
        source = self._repo.get_data_source(name) if isinstance(name, str) and name else None
        data = self._repo.get_sympy_function(source.name) if source is not None and source.is_function else None
        self._function_data = data
        if data is not None:
            self._function_panel.load(data)
        self._function_panel.set_shown(data is not None)

    def _columns_from_table(self) -> list[str]:
        """Return the columns of the selected source: a table, a saved query or a stored function."""
        source = self._current_source()
        if source is None:
            return []
        return [str(column) for column in self._repo.data_source_columns(source)]

    def _default_sql_for_current_table(self) -> str:
        """Generate a default SELECT using the selected renderer role aliases.

        The FROM fragment comes from the data source, so a saved query becomes
        an inline subquery and a table becomes a quoted name.  The series SQL
        that gets stored is therefore self-contained: it keeps working whatever
        the source was, and a saved query is re-executed on every render rather
        than frozen into a table.
        """
        source = self._current_source()
        if source is None:
            return ""

        select_parts: list[str] = []
        for alias, combo in self._role_combos.items():
            column = combo.currentData()
            if isinstance(column, str) and column:
                select_parts.append(
                    f"{quote_identifier(column)} AS {quote_identifier(alias)}"
                )

        projection = ", ".join(select_parts) if select_parts else "*"
        return f"SELECT {projection} FROM {source.from_clause()}"

    def _renderer_role_names(self, kind: str) -> list[str]:
        """Return renderer role names from the renderer class."""
        renderer_class = self._current_renderer_class()
        if renderer_class is None:
            return []

        if kind == "required":
            return list(renderer_class.RequiredRoles)
        if kind == "optional":
            return list(renderer_class.OptionalRoles)
        return []

    def _all_renderer_role_names(self) -> list[str]:
        """Return all renderer roles, required first and then optional."""
        roles: list[str] = []
        for kind in ("required", "optional"):
            for role in self._renderer_role_names(kind):
                if role not in roles:
                    roles.append(role)
        return roles

    def _role_value(self, draft: SeriesDraft, role: str) -> str | None:
        """Return a draft role value, matching role names case-insensitively."""
        if role in draft.roles:
            return draft.roles.get(role)

        role_lower = role.lower()
        for key, value in draft.roles.items():
            if key.lower() == role_lower:
                return value
        return None

    def _role_is_required(self, role: str) -> bool:
        """Return True when the role belongs to renderer.required."""
        required = {
            name.lower()
            for name in self._renderer_role_names("required")
        }
        return role.lower() in required

    def _role_label_widget(self, role: str) -> QLabel:
        """Build a styled label for one role row."""
        label = QLabel(f"{role}:", self._roles_panel)
        if self._role_is_required(role):
            label.setProperty("requiredRole", True)
        return label

    def _show_group_by(self) -> None:
        """Group by only for a chart without a group role of its own: one way to ask for groups (B-12)."""
        layout = getattr(self, "_editor_form_layout", None)
        combo = getattr(self, "_combo_group", None)
        if layout is None or combo is None:
            return
        own = "group" in self._role_combos or self._roles_automatic()
        if own and combo.currentIndex() > 0:
            combo.setCurrentIndex(0)
        layout.setRowVisible(combo, not own)

    def _roles_automatic(self) -> bool:
        """A stored function's x, y (and z) go to the roles of the same name, with nothing else to choose.

        Unless the chart needs a role the function has no column for - a histogram's values, a surface's z
        from f(x): then the roles are shown.
        """
        if self._function_data is None:
            return False
        columns = {column.lower() for column in self._columns_from_table()}
        return all(role.lower() in columns for role in self._renderer_role_names("required"))

    def _show_roles(self) -> None:
        """The roles, Group by and Separate axes by: hidden for a stored function, whose columns place themselves."""
        layout = getattr(self, "_editor_form_layout", None)
        if layout is None:
            return
        automatic = self._roles_automatic()
        if automatic:
            for role, combo in self._role_combos.items():
                if role.lower() not in ("x", "y", "z") and combo.currentIndex() > 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
        layout.setRowVisible(self._roles_scroll, not automatic)
        self._show_group_by()
        if self._function_data is not None and self._combo_axes_by.currentIndex() > 0:
            self._combo_axes_by.setCurrentIndex(0)
        self._axis_form_layout.setRowVisible(self._combo_axes_by, self._function_data is None)

    def _rebuild_role_combos(self) -> None:
        """Rebuild the scrollable role panel for the selected renderer."""
        previous = {
            key: combo.currentData()
            for key, combo in self._role_combos.items()
        }

        while self._roles_layout.rowCount():
            self._roles_layout.removeRow(0)
        self._role_combos.clear()
        self._role_labels.clear()

        for role in self._all_renderer_role_names():
            combo = QComboBox(self._roles_panel)
            combo.addItem(_("(none)"), None)

            previous_value = previous.get(role)
            if isinstance(previous_value, str):
                combo.addItem(previous_value, previous_value)
                combo.setCurrentIndex(1)

            combo.currentIndexChanged.connect(self._on_roles_changed)
            label = self._role_label_widget(role)
            self._role_combos[role] = combo
            self._role_labels[role] = label
            self._roles_layout.addRow(label, combo)

        if not self._role_combos:
            self._roles_layout.addRow(
                QLabel(_("No renderer roles declared."), self._roles_panel)
            )
        self._show_group_by()

    def _populate_role_columns(self, cols: list[str]) -> None:
        """Fill every role combo with available table or query columns."""
        for _role, combo in self._role_combos.items():
            current = combo.currentData()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItem(_("(none)"), None)
                for column in cols:
                    combo.addItem(str(column), str(column))
                if isinstance(current, str):
                    index = combo.findData(current)
                    if index >= 0:
                        combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)

        # Keep convenient defaults for common X/Y role names; a function's x, y (and z) on its roles.
        lower_to_role = {role.lower(): role for role in self._role_combos}
        preferred = cols if self._function_data is not None else self._preferred_columns
        for role_name, column in zip(("x", "y", "z"), preferred):
            role = lower_to_role.get(role_name)
            if role and column in cols and self._role_combos[role].currentData() is None:
                self._role_combos[role].setCurrentIndex(self._role_combos[role].findData(column))
        if cols:
            x_role = lower_to_role.get("x")
            if x_role and self._role_combos[x_role].currentData() is None:
                self._role_combos[x_role].setCurrentIndex(1)

            y_role = lower_to_role.get("y")
            if (
                y_role
                and self._role_combos[y_role].currentData() is None
                and len(cols) > 1
            ):
                self._role_combos[y_role].setCurrentIndex(2)
        self._show_roles()

    def _on_chart_type_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Refresh role rows after the renderer selection changes."""
        self._persist_editor_to_draft()
        item = current or self._current_item()
        # A category heading carries no data; it is unselectable, but the
        # signal can still arrive with one during a rebuild.
        if item is not None and item.data(Qt.ItemDataRole.UserRole):
            chart_type = str(item.data(Qt.ItemDataRole.UserRole))
            self.renderer = get_renderer(chart_type)
            self._renderer_class = self._load_renderer_class()

        self._update_renderer_details()
        self._set_default_axis_name()
        self._rebuild_role_combos()
        self._populate_role_columns(self._columns_from_table())

        series_item = self._current_series_item()
        if series_item is not None:
            draft = self._draft_for_item(series_item)
            self._set_role_selection(draft)
            draft.roles = self._current_role_mapping()
            self._set_default_series_name(series_item, draft)
        self._update_sql()
        self._populate_group_columns()
        # Last: the note and the target radios depend on the renderer that
        # has just been selected.
        self._sync_series_mode()

    def _on_table_changed(self, _index: int = 0) -> None:
        """Refresh role choices, SQL and automatic names when the data source changes."""
        self._sync_function_source()
        self._populate_role_columns(self._columns_from_table())
        self._set_default_figure_name()

        item = self._current_series_item()
        if item is None:
            return

        draft = self._draft_for_item(item)
        draft.roles = self._current_role_mapping()
        self._update_sql()
        self._set_default_series_name(item, draft)
        self._populate_group_columns()

    def _update_sql(self) -> None:
        """Regenerate SQL from the current data source and role choices."""
        item = self._current_series_item()
        if item is None:
            return

        draft = self._draft_for_item(item)
        draft.roles = self._current_role_mapping()
        draft.sql = self._default_sql_for_current_table()
        self._set_default_series_name(item, draft)
        self._load_draft_into_editor(draft)

    # ------------------------------------------------------------------
    # Persist.
    # ------------------------------------------------------------------
    def _selected_chart_type(self) -> str:
        """Return selected renderer/chart type and cache the renderer object."""
        item = self._current_item()
        if item is None:
            return "Scatter Plot"

        key = str(item.data(Qt.ItemDataRole.UserRole))
        renderer = get_renderer(key)
        if renderer != self.renderer:
            self.renderer = renderer
            self._renderer_class = self._load_renderer_class()
        return key

    def _validate_series(self, chart_type: str, draft: SeriesDraft) -> bool:
        """Validate SQL and required renderer roles for one draft."""
        if not draft.sql.strip():
            show_message(self, "chart.series_empty_sql", series=draft.name)
            return False

        renderer_class = self._current_renderer_class()
        if renderer_class is None:
            show_message(self, "chart.renderer_not_found", series=draft.name)
            return False

        for role in self._renderer_role_names("required"):
            if not self._role_value(draft, role):
                show_message(self, "chart.role_missing", series=draft.name, role=role)
                return False
        return True

    def _series_roles(self, draft: SeriesDraft) -> dict[str, str]:
        """Return only assigned renderer roles for descriptor persistence."""
        return {
            role: value
            for role, value in draft.roles.items()
            if isinstance(value, str) and value
        }

    #: Chart types with no meaningful x/y scale of their own - a table has
    #: nothing to tick or draw a frame around. The renderer's own
    #: ax.axis("off") call is undone again by _apply_axis_runtime_options's
    #: unconditional set_axis_on() afterwards, so the descriptor has to ask
    #: for it explicitly through the one option that survives that: without
    #: this, a newly created Table axis showed a bare 0-1 frame around it.
    CHART_TYPES_WITHOUT_AXIS_CHROME: frozenset[str] = frozenset({"Table"})

    #: Chart types that need a 3D Matplotlib axes rather than the ordinary
    #: 2D one every other renderer draws on. render_figure.py's
    #: _subplot_kwargs_for_axis already reads a generic "projection" option
    #: for every axis, so this is the only place that had to name which
    #: renderers want it.
    CHART_TYPES_NEEDING_3D_AXES: frozenset[str] = frozenset(
        {
            "Surface Plot",
            "Surface Plot (Scattered)",
            "Scatter Plot (3D)",
            "3D Line Plot",
            "3D Bar Chart",
        }
    )

    def _default_axis_options(self, chart_type: str) -> dict[str, Any] | None:
        if chart_type in self.CHART_TYPES_WITHOUT_AXIS_CHROME:
            return {"hide_axis": True}
        if chart_type in self.CHART_TYPES_NEEDING_3D_AXES:
            return {"projection": "3d"}
        return None

    def _axis_title_from_series(self, drafts: list[SeriesDraft]) -> str:
        """Infer a readable axis title from selected series names."""
        names = [draft.name.strip() for draft in drafts if draft.name.strip()]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:3]) + ("..." if len(names) > 3 else "")

    def _axis_label_for_role(self, drafts: list[SeriesDraft], role_name: str) -> str:
        """Infer an axis label from the first mapped role value."""
        for draft in drafts:
            value = self._role_value(draft, role_name)
            if value:
                return value
        return ""

    def _on_accept(self) -> None:
        """Validate and persist the figure/axis/series descriptors."""
        self._persist_editor_to_draft()
        chart_type = self._selected_chart_type()

        drafts = self._ordered_drafts()
        if not drafts:
            show_message(self, "chart.no_series")
            return

        for draft in drafts:
            if not self._validate_series(chart_type, draft):
                return
        # SQL typed here by hand: the SQL guard's moment (todo N-05).
        for draft in drafts:
            sql = str(draft.sql or "").strip()
            if " " not in sql:
                continue
            try:
                ensure_read_only_select(sql)
            except ValueError as exc:
                QMessageBox.warning(self, _("SQL blocked"), f"{draft.name}: {exc}")
                return

        drafts = self._split_into_groups(drafts)
        if not drafts:
            return

        if self._rb_current_figure.isChecked():
            if self._current_figure_id is None:
                return
            figure_id = int(self._current_figure_id)
        else:
            name = (self._edit_figure_name.text() or "").strip()
            if not name:
                self._edit_figure_name.setFocus(Qt.FocusReason.OtherFocusReason)
                return
            figure_id = self._repo.create_figure_descriptor(
                name=name,
                nrows=1,
                ncols=1,
            )

        # One group per axes.  A renderer that draws a single series - a
        # surface, a pie, a contour map - gets one axis each rather than one
        # axis that keeps the first series and logs the others away.
        groups = self._axis_groups(drafts)
        titles: list[str] | None = None
        axes_by = self._combo_axes_by.currentData() if self._rb_new_axis.isChecked() else None
        if axes_by:
            split = self._split_into_axes(drafts, str(axes_by))
            if not split:
                return
            titles = [title for title, _group in split]
            groups = [group for _title, group in split]

        if self._rb_existing_axis.isChecked() and self._rb_current_figure.isChecked():
            axis_id_data = self._combo_axes.currentData()
            if axis_id_data is None:
                return
            axis_id = int(axis_id_data)
            if not self._existing_axis_has_room(axis_id, len(drafts), chart_type):
                return
            self._repo.update_axis_chart_type(axis_id=axis_id, chart_type=chart_type)
            axis_ids = [axis_id]
            groups = [drafts]
        else:
            axis_ids = self._create_axes_for_groups(figure_id, chart_type, groups, titles=titles)

        created = 0
        for axis_id, group in zip(axis_ids, groups):
            for draft in group:
                series_index = self._repo.next_series_index(axis_id)
                self._repo.create_series_descriptor(
                    axis_id=axis_id,
                    series_index=series_index,
                    name=draft.name,
                    sql_query=draft.sql,
                    roles=self._series_roles(draft),
                    style={"label": draft.name},
                )
                created += 1

        self._result = NewPlotTabResult(
            figure_id=int(figure_id),
            # The first axis, so the caller selects the one holding the series
            # the person was editing when they pressed Create.
            axis_id=int(axis_ids[0]),
            series_count=int(created),
        )
        self.accept()

    #: Most series one "Group by" may make: past this the column is an
    #: identifier or a measurement, not a grouping, and the legend unreadable.
    MAX_GROUPS: int = get_constant("chart_max_groups", 50)

    def _split_into_groups(self, drafts: list[SeriesDraft]) -> list[SeriesDraft]:
        """Each draft with a Group by column becomes one draft per value of it.

        Every copy is the draft's own query with ``WHERE "column" = value``
        added (and the Hide filter, when the table has one), the same roles,
        and the value as its name. An empty list means it could not be done,
        and the person has been told why.
        """
        source = self._current_table_name()
        result: list[SeriesDraft] = []
        for draft in drafts:
            if not draft.group_by:
                result.append(draft)
                continue
            try:
                values = self._repo.distinct_values(source, draft.group_by, limit=self.MAX_GROUPS)
            except Exception as exc:  # noqa: BLE001 - shown, not swallowed
                show_message(self, "chart.group_failed", column=draft.group_by, reason=str(exc))
                return []
            if len(values) > self.MAX_GROUPS:
                show_message(self, "chart.too_many_groups", column=draft.group_by, limit=self.MAX_GROUPS)
                return []
            hide = self._repo.source_has_hide_column(source)
            for value in values:
                try:
                    sql = self._repo.sql_with_value_filter(draft.sql, draft.group_by, value)
                    if hide:
                        sql = self._repo.sql_with_hide_filter(sql)
                except ValueError as exc:
                    show_message(self, "chart.group_failed", column=draft.group_by, reason=str(exc))
                    return []
                label = _("(empty)") if self._repo.sql_literal(value) == "NULL" else str(value)
                name = label if len(drafts) == 1 else f"{draft.name}: {label}"
                result.append(SeriesDraft(name=name, sql=sql, roles=dict(draft.roles), user_named=True))
        return result

    def _existing_axis_has_room(
        self,
        axis_id: int,
        wanted: int,
        chart_type: str,
    ) -> bool:
        """True when *wanted* more series fit on an existing axis.

        Checked here rather than left to the renderer: the renderer's own
        guard runs at draw time, by which point the series are in the
        database and the chart has silently lost them.
        """
        limit = self._max_series_per_axis()
        if limit is None:
            return True

        existing = int(self._repo.next_series_index(axis_id))
        if existing + wanted <= limit:
            return True

        show_message(
            self,
            "chart.axis_series_limit",
            chart=chart_type,
            limit=limit,
            existing=existing,
            wanted=wanted,
        )
        return False

    #: Most axes Separate axes by may make: past this a column is an identifier, not a grouping.
    MAX_AXES: int = get_constant("chart_max_separate_axes", 24)

    def _split_into_axes(self, drafts: list[SeriesDraft], column: str) -> list[tuple[str, list[SeriesDraft]]]:
        """(title, drafts) per value of *column*: every draft filtered to that value (B-13).

        The axes keep the renderer's own limit: a chart of one series per axis
        still gets one axis per series within each value. Empty when it could
        not be done, and the person has been told why.
        """
        source = self._current_table_name()
        try:
            values = self._repo.distinct_values(source, column, limit=self.MAX_AXES)
        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            show_message(self, "chart.group_failed", column=column, reason=str(exc))
            return []
        if len(values) > self.MAX_AXES:
            show_message(self, "chart.too_many_groups", column=column, limit=self.MAX_AXES)
            return []
        hide = self._repo.source_has_hide_column(source)
        split: list[tuple[str, list[SeriesDraft]]] = []
        for value in values:
            label = _("(empty)") if self._repo.sql_literal(value) == "NULL" else str(value)
            filtered = []
            for draft in drafts:
                try:
                    sql = self._repo.sql_with_value_filter(draft.sql, column, value)
                    if hide and '"Hide"' not in sql:
                        sql = self._repo.sql_with_hide_filter(sql)
                except ValueError as exc:
                    show_message(self, "chart.group_failed", column=column, reason=str(exc))
                    return []
                filtered.append(SeriesDraft(name=draft.name, sql=sql, roles=dict(draft.roles), user_named=True))
            for group in self._axis_groups(filtered):
                split.append((f"{column} = {label}", group))
        return split

    def _create_axes_for_groups(
        self,
        figure_id: int,
        chart_type: str,
        groups: list[list[SeriesDraft]],
        *,
        titles: list[str] | None = None,
    ) -> list[int]:
        """Create one axis per group and return their ids, in order.

        The grid is grown once for the whole set rather than per axis, so a
        figure gaining four surfaces is one capacity change and not four.
        """
        first_index = int(self._repo.next_axis_index(figure_id))
        self._repo.ensure_figure_grid_capacity(
            figure_id=figure_id,
            needed_axes=first_index + len(groups),
        )

        # A single axis keeps the name the person typed.  Several cannot: one
        # field names one axis, so each takes the title of the series on it.
        typed_name = (self._edit_axis_name.text() or "").strip()
        axis_ids: list[int] = []
        for offset, group in enumerate(groups):
            title = titles[offset] if titles else self._axis_title_from_series(group)
            if len(groups) == 1 and typed_name and not titles:
                title = typed_name
            axis_ids.append(
                int(
                    self._repo.create_axis_descriptor(
                        figure_id=figure_id,
                        axis_index=first_index + offset,
                        chart_type=chart_type,
                        title=title,
                        x_label=self._axis_label_for_role(group, "x"),
                        y_label=self._axis_label_for_role(group, "y"),
                        z_label=self._axis_label_for_role(group, "z"),
                        options=self._default_axis_options(chart_type),
                    )
                )
            )
        return axis_ids
