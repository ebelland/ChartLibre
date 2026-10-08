"""Cast a table's columns into the roles an operation needs: Y, factors, a block...

The table operations' counterpart of the series operations' axis/series
selector, after JMP's "Cast Selected Columns into Roles": the columns on
the left, each with its modelling type, and one box per role on the right.
Select columns, press a role's button, and they move into it; double-click
one in a role to take it out again. A column has at most one role.

The modelling type says how a column enters a model: *continuous* (a number
that varies - a temperature, a yield) or *nominal* (a label - a catalyst, an
operator, a batch). It is guessed from the data and can be changed: a
number used as a category (a machine number, a 1/2/3 level) is nominal.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import _

CONTINUOUS = "continuous"
NOMINAL = "nominal"

#: The modelling types, as the type combo lists them, and the mark a column
#: carries for its type in the lists.
KINDS: tuple[tuple[str, str], ...] = ((CONTINUOUS, "Continuous"), (NOMINAL, "Nominal"))
_KIND_MARKS: dict[str, str] = {CONTINUOUS: "▲", NOMINAL: "●"}

#: A numeric column with at most this many distinct values, all whole
#: numbers, reads as levels rather than measurements when guessing.
_LEVEL_LIMIT = 6

_NAME_ROLE = Qt.ItemDataRole.UserRole


@dataclass(frozen=True, slots=True)
class ColumnRole:
    """A role an operation needs columns in.

    ``kinds`` are the modelling types the role accepts; ``minimum`` how many
    columns it must have before the operation can run, ``maximum`` how many
    it may have (None: any number).
    """

    key: str
    label: str
    tooltip: str = ""
    kinds: tuple[str, ...] = (CONTINUOUS, NOMINAL)
    minimum: int = 0
    maximum: int | None = None


@dataclass(slots=True)
class ColumnCasting:
    """The columns' types and roles, as the widget holds them."""

    kinds: dict[str, str] = field(default_factory=dict)
    roles: dict[str, list[str]] = field(default_factory=dict)

    def columns(self, role: str) -> list[str]:
        return list(self.roles.get(role, []))

    def of_kind(self, role: str, kind: str) -> list[str]:
        return [name for name in self.roles.get(role, []) if self.kinds.get(name) == kind]


def guess_kind(values: pd.Series) -> str:
    """The modelling type *values* suggest: text is nominal, so are a few whole-number levels."""
    numbers = pd.to_numeric(values, errors="coerce")
    present = values.dropna()
    if present.empty or numbers.notna().sum() < len(present):
        return NOMINAL
    distinct = numbers.dropna().unique()
    whole = bool(((distinct % 1) == 0).all())
    if whole and len(distinct) <= min(_LEVEL_LIMIT, max(len(present) // 3, 2)) and len(present) > len(distinct):
        return NOMINAL
    return CONTINUOUS


class ColumnRolesWidget(QWidget):
    """The table's columns on the left, a box per role on the right."""

    #: The roles or a type changed.
    changed = Signal()

    def __init__(self, roles: Sequence[ColumnRole], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._roles = tuple(roles)
        self._kinds: dict[str, str] = {}
        self._columns: list[str] = []
        self._boxes: dict[str, QListWidget] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(4)
        left.addWidget(self._heading(_("Columns")))
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Find a column"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter_columns)
        left.addWidget(self._search)
        self._column_list = QListWidget(self)
        self._column_list.setObjectName("tableColumnsList")
        self._column_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._column_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._column_list.setMinimumWidth(150)
        self._column_list.itemSelectionChanged.connect(self._sync_kind_combo)
        left.addWidget(self._column_list, 1)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel(_("Type:"), self))
        self._kind_combo = QComboBox(self)
        self._kind_combo.setToolTip(_("How the selected columns enter a model: as numbers, or as labels."))
        for value, label in KINDS:
            self._kind_combo.addItem(f"{_KIND_MARKS[value]} {_(label)}", value)
        self._kind_combo.activated.connect(self._apply_kind)
        type_row.addWidget(self._kind_combo, 1)
        left.addLayout(type_row)
        layout.addLayout(left, 2)

        side = QVBoxLayout()
        side.setSpacing(6)
        right = QGridLayout()
        right.setHorizontalSpacing(6)
        right.setVerticalSpacing(6)
        right.addWidget(self._heading(_("Roles")), 0, 0, 1, 2)
        for row, role in enumerate(self._roles, start=1):
            button = QPushButton(_(role.label), self)
            button.setToolTip(_(role.tooltip) if role.tooltip else "")
            button.clicked.connect(lambda _checked=False, key=role.key: self.assign_selected(key))
            right.addWidget(button, row, 0, Qt.AlignmentFlag.AlignTop)
            box = QListWidget(self)
            box.setObjectName(f"roleBox_{role.key}")
            box.setToolTip(_("Double-click a column to take it out of this role."))
            box.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            rows = 1 if role.maximum == 1 else 3
            box.setFixedHeight(10 + rows * max(box.fontMetrics().height() + 8, 22))
            box.itemDoubleClicked.connect(lambda item, key=role.key: self.unassign(key, [item.data(_NAME_ROLE)]))
            right.addWidget(box, row, 1)
            self._boxes[role.key] = box
        right.setColumnStretch(1, 1)
        side.addLayout(right)
        #: Where the operation adds its own inputs, under the roles: a model's
        #: effects, its parameters.
        self.side_layout = side
        layout.addLayout(side, 3)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("muted", True)
        return label

    # -- Content ---------------------------------------------------------

    @property
    def roles(self) -> tuple[ColumnRole, ...]:
        return self._roles

    def set_columns(self, frame: pd.DataFrame, kinds: Mapping[str, str] | None = None) -> None:
        """Show *frame*'s columns, typed from *kinds* or guessed; every role empties."""
        self._columns = [str(name) for name in frame.columns]
        given = dict(kinds or {})
        self._kinds = {name: given.get(name) or guess_kind(frame[name]) for name in self._columns}
        for box in self._boxes.values():
            box.clear()
        self._refresh_columns()
        self.changed.emit()

    def casting(self) -> ColumnCasting:
        """The columns' types and what each role holds, in order."""
        return ColumnCasting(
            kinds=dict(self._kinds),
            roles={key: [box.item(i).data(_NAME_ROLE) for i in range(box.count())] for key, box in self._boxes.items()},
        )

    def set_casting(self, roles: Mapping[str, Iterable[str]], kinds: Mapping[str, str] | None = None) -> None:
        """Put columns into roles (and set their types); names not in the table are skipped."""
        for name, kind in (kinds or {}).items():
            if name in self._kinds and kind in _KIND_MARKS:
                self._kinds[name] = kind
        for box in self._boxes.values():
            box.clear()
        for key, names in roles.items():
            if key in self._boxes:
                self._add(key, [name for name in names if name in self._kinds])
        self._refresh_columns()
        self.changed.emit()

    def problems(self) -> list[str]:
        """What keeps the casting from being usable: a role short of columns, a column of the wrong type."""
        found = []
        casting = self.casting()
        for role in self._roles:
            names = casting.columns(role.key)
            if len(names) < role.minimum:
                found.append(_("{role}: choose at least {count} column(s).").format(role=_(role.label), count=role.minimum))
            wrong = [name for name in names if casting.kinds.get(name) not in role.kinds]
            if wrong:
                found.append(_("{role} takes {kinds} columns only: {columns}.").format(
                    role=_(role.label),
                    kinds=" / ".join(_(label) for value, label in KINDS if value in role.kinds).lower(),
                    columns=", ".join(wrong),
                ))
        return found

    def selected_columns(self) -> list[str]:
        """The columns selected on the left, in the table's order."""
        chosen = {item.data(_NAME_ROLE) for item in self._column_list.selectedItems()}
        return [name for name in self._columns if name in chosen]

    def _filter_columns(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self._column_list.count()):
            item = self._column_list.item(row)
            item.setHidden(bool(needle) and needle not in str(item.data(_NAME_ROLE)).casefold())

    # -- Moving columns ----------------------------------------------------

    def assign_selected(self, key: str) -> None:
        """Move the columns selected on the left into the role *key*."""
        names = [item.data(_NAME_ROLE) for item in self._column_list.selectedItems()]
        if names:
            self.assign(key, names)

    def assign(self, key: str, names: Sequence[str]) -> None:
        """Move *names* into the role *key*, out of any role they had."""
        for other in self._boxes:
            self._remove(other, names)
        self._add(key, names)
        self._refresh_columns()
        self.changed.emit()

    def unassign(self, key: str, names: Sequence[str]) -> None:
        self._remove(key, names)
        self._refresh_columns()
        self.changed.emit()

    def _add(self, key: str, names: Sequence[str]) -> None:
        role = next(role for role in self._roles if role.key == key)
        box = self._boxes[key]
        present = {box.item(i).data(_NAME_ROLE) for i in range(box.count())}
        for name in names:
            if name in present:
                continue
            if role.maximum is not None and box.count() >= role.maximum:
                # A one-column role takes the newest choice.
                box.takeItem(0)
            item = QListWidgetItem(self._label(name), box)
            item.setData(_NAME_ROLE, name)

    def _remove(self, key: str, names: Sequence[str]) -> None:
        box = self._boxes[key]
        for row in reversed(range(box.count())):
            if box.item(row).data(_NAME_ROLE) in names:
                box.takeItem(row)

    # -- Types ---------------------------------------------------------------

    def _apply_kind(self, _index: int) -> None:
        kind = str(self._kind_combo.currentData())
        for item in self._column_list.selectedItems():
            self._kinds[item.data(_NAME_ROLE)] = kind
        self._refresh_columns()
        self.changed.emit()

    def _sync_kind_combo(self) -> None:
        selected = {self._kinds.get(item.data(_NAME_ROLE)) for item in self._column_list.selectedItems()}
        self._kind_combo.setEnabled(bool(selected))
        if len(selected) == 1:
            self._kind_combo.setCurrentIndex(max(0, self._kind_combo.findData(selected.pop())))

    def _label(self, name: str) -> str:
        return f"{_KIND_MARKS.get(self._kinds.get(name, ''), '')} {name}"

    def _refresh_columns(self) -> None:
        """Relabel every list with the current types; dim the columns that already have a role."""
        selected = {item.data(_NAME_ROLE) for item in self._column_list.selectedItems()}
        cast = {box.item(i).data(_NAME_ROLE) for box in self._boxes.values() for i in range(box.count())}
        self._column_list.clear()
        for name in self._columns:
            item = QListWidgetItem(self._label(name), self._column_list)
            item.setData(_NAME_ROLE, name)
            if name in cast:
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setForeground(self.palette().placeholderText())
            item.setSelected(name in selected)
        for box in self._boxes.values():
            for i in range(box.count()):
                box.item(i).setText(self._label(box.item(i).data(_NAME_ROLE)))
        self._filter_columns(self._search.text())
        self._sync_kind_combo()
