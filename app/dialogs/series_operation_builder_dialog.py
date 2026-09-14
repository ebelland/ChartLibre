"""Series Operation Builder: scaffold a new series-operation dialog file
(todo.txt P3-6).

A series operation lives entirely in one file under app/series_operations/
(built in) or user/series_operations/ (user-authored, kept out of the
app's own source tree - see app.utils.config.USER_SERIES_OPERATIONS_DIR),
as a class that directly subclasses SeriesOperationDialogBase -
series_operation_scanner.py AST-scans both folders at import time and
picks it up with no registration step. This dialog is a short form for
Name and Description that writes a new, immediately-importable file with
those fields already filled in, a working (if trivial) compute_results
that passes each selected series' data through unchanged, and the rest of
the dialog shell (model selector, parameter form, preview pane) left at
their base-class defaults for the user to override where their operation
actually needs one. See SeriesOperationDialogBase's own docstring
(app/series_operations/dialog_base.py) for the full contract, and
app/series_operations/function_dialog.py for the smallest complete,
hand-written dialog to read alongside it.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.dev_tools_common import (
    class_name as _class_name,
    discover_both_roots,
    import_check as _generic_import_check,
    open_in_editor,
    slug as _slug,
)
from app.logs.logger import applogger
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils.config import USER_SERIES_OPERATIONS_DIR
from app.utils.i18n import _
from app.utils.messages import show_message

#: Where series_operation_scanner.py itself looks for the built-in
#: operations - used here only to check a new Name against them, never
#: written to.
BUILTIN_OPERATIONS_DIR: Path = (
    Path(__file__).resolve().parent.parent / "series_operations"
)

#: Where this dialog writes - a user-authored operation, kept separate
#: from the app's own shipped source. Created on first use.
OPERATIONS_DIR: Path = USER_SERIES_OPERATIONS_DIR


def render_stub_source(*, class_name: str, name: str, description: str, slug: str) -> str:
    """Return the full source text of the scaffolded operation file."""
    return f'''"""{name} series operation.

TODO: describe what this operation computes and when to reach for it, the
way every dialog under app/series_operations/ does at the top of its own
file - app/series_operations/function_dialog.py is the smallest complete,
hand-written example to read alongside this one, and
SeriesOperationDialogBase's own docstring (app/series_operations/
dialog_base.py) documents every hook below plus the ones left out here
(build_model_selector, PARAMS/build_parameter_selector, ...) because their
base-class defaults already do something reasonable - override one only
once this operation actually needs it.

This file was scaffolded by the Series Operation Builder (Developer menu).
It is already a valid, discoverable operation - series_operation_scanner.py
picks up any class in app/series_operations/ that directly subclasses
SeriesOperationDialogBase, with no registration step - and compute_results
below already runs: it passes each selected series' data through
unchanged. Replace the marked line with the real computation, then delete
this paragraph.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)


@dataclass(slots=True)
class {class_name}Result:
    """One computed result for one source series."""

    source_name: str
    model: str
    x: Any
    y: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({{"x": self.x, "y": self.y}})


class {class_name}(SeriesOperationDialogBase):
    """TODO: one line describing this operation."""

    Name: str = {name!r}
    Description: str = {description!r}

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: Any | None = None,
    ) -> None:
        self._last_results: list[{class_name}Result] = []
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title={name!r},
            parent=parent,
            width=720,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self.refresh_results()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self.set_results_text(f"Error:\\n{{exc}}")
            return
        self._last_results = list(results)
        self.set_results_text(self.format_results(results))

    def compute_results(self) -> Sequence[{class_name}Result]:
        results: list[{class_name}Result] = []
        errors: list[str] = []
        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                # TODO: replace this line with the real computation -
                # x_values/y_values are already read, coerced to numeric
                # (datetime x included), and remembered for the write-back
                # below. This passes them through unchanged.
                result_x, result_y = x_values, y_values
                results.append(
                    {class_name}Result(
                        source_name=name, model="identity", x=result_x, y=result_y
                    )
                )
            except Exception as exc:
                errors.append(f"{{name}}: {{exc}}")
        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)
        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def result_to_frame(self, result: {class_name}Result) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self, axis_id: int, table_name: str, result: {class_name}Result
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=f"{{result.source_name}} - {name}",
            sql_query=f'SELECT x, y FROM "{{table_name}}" ORDER BY x',
            roles={{"x": "x", "y": "y"}},
            style={{
                "generated_{slug}": True,
                "{slug}_dialog": {slug!r},
                "source_name": result.source_name,
                "model": result.model,
            }},
        )

    @property
    def generated_style_filter(self) -> dict[str, Any]:
        return {{"generated_{slug}": True, "{slug}_dialog": {slug!r}}}

    @property
    def operation_label(self) -> str:
        return {name!r}
'''


class SeriesOperationBuilderDialog(QDialog):
    """Scaffold a new SeriesOperationDialogBase file from a short form."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Series Operation Builder"))
        self.setWindowIcon(load_icon("series_operation_builder"))
        self.setModal(True)

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        card = CardFrame(self, "seriesOperationBuilderCard")
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_("Series Operation Builder"), card))

        hint = QLabel(
            _(
                "Writes a new file under user/series_operations/ with a class "
                "already filled in from these fields - no registration step, "
                "the scanner picks it up as soon as the file exists. "
                "compute_results already passes each selected series through "
                "unchanged; replace it with the real computation."
            ),
            card,
        )
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._name_edit = QLineEdit(card)
        self._name_edit.setPlaceholderText(_("e.g. Envelope Detector"))
        self._name_edit.textEdited.connect(self._on_name_edited)
        form.addRow(_("Name:"), self._name_edit)

        self._file_edit = QLineEdit(card)
        self._file_edit.setPlaceholderText(_("e.g. envelope_detector_dialog.py"))
        self._file_edit.textEdited.connect(self._on_file_edited_by_user)
        form.addRow(_("File name:"), self._file_edit)

        self._description_edit = QLineEdit(card)
        self._description_edit.setPlaceholderText(
            _("One line, shown in the Series Operations panel.")
        )
        form.addRow(_("Description:"), self._description_edit)

        card_layout.addLayout(form)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        create_action_button(
            parent=self,
            action_id="create_series_operation",
            action=self._on_create,
            layout=action_row,
        )
        card_layout.addLayout(action_row)

        root.addWidget(card, 1)

        self._file_edited_by_user = False

    # ------------------------------------------------------------------
    # Auto-deriving the file name from the operation's Name
    # ------------------------------------------------------------------
    def _on_name_edited(self, text: str) -> None:
        if self._file_edited_by_user:
            return
        self._file_edit.setText(f"{_slug(text)}_dialog.py" if text.strip() else "")

    def _on_file_edited_by_user(self, _text: str) -> None:
        self._file_edited_by_user = True

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def _on_create(self) -> None:
        problem = self._validate()
        if problem:
            show_message(self, "dev.validation_error", detail=problem)
            return

        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        class_name = _class_name(name, suffix="Dialog", fallback="Custom")
        description = self._description_edit.text().strip()
        slug = _slug(name) or "operation"

        source = render_stub_source(
            class_name=class_name, name=name, description=description, slug=slug
        )

        target = OPERATIONS_DIR / file_name
        try:
            OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        except OSError as exc:
            applogger.exception("Could not write the new series operation file.")
            show_message(self, "dev.validation_error", detail=str(exc))
            return

        error = self._import_check(target, class_name)
        if error:
            show_message(
                self,
                "dev.series_operation_create_failed",
                path=str(target),
                error=error,
            )
            return

        applogger.info(
            "Series Operation Builder: wrote %s (%s, Name=%r).", target, class_name, name
        )
        show_message(self, "dev.series_operation_created", name=name, path=str(target))
        open_in_editor(target)
        self.accept()

    def _validate(self) -> str:
        name = self._name_edit.text().strip()
        file_name = self._file_edit.text().strip()
        description = self._description_edit.text().strip()

        if not name:
            return _("Name cannot be empty.")
        if not description:
            return _("Description cannot be empty.")
        if not file_name.endswith(".py") or not _slug(file_name[:-3]):
            return _(
                "File name must be a valid Python file name, e.g. my_operation_dialog.py."
            )
        if (OPERATIONS_DIR / file_name).exists():
            return _(
                "A file named \"{file}\" already exists under user/series_operations/."
            ).format(file=file_name)
        if self._operation_name_taken(name):
            return _(
                "\"{name}\" is already used by an existing series operation."
            ).format(name=name)
        return ""

    @staticmethod
    def _operation_name_taken(name: str) -> bool:
        entries = discover_both_roots(
            builtin_root=BUILTIN_OPERATIONS_DIR,
            user_root=OPERATIONS_DIR,
            base_class_name="SeriesOperationDialogBase",
        )
        return any(entry["value"] == name for entry in entries)

    @staticmethod
    def _import_check(path: Path, class_name: str) -> str:
        """Import *path* fresh and confirm it discovers as an operation."""
        error = _generic_import_check(path, class_name)
        if error:
            return error

        entries = discover_both_roots(
            builtin_root=BUILTIN_OPERATIONS_DIR,
            user_root=OPERATIONS_DIR,
            base_class_name="SeriesOperationDialogBase",
        )
        if not any(entry["name"] == class_name for entry in entries):
            return _(
                "The class imported, but series_operation_scanner did not discover "
                "it as an operation - check that it subclasses "
                "SeriesOperationDialogBase directly."
            )
        return ""
