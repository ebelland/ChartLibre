"""Data Operation Builder: scaffold a new data-operation dialog file
(todo.txt P3-6).

A data operation lives entirely in one file under app/data_operations/
(built in) or user/data_operations/ (user-authored, kept out of the
app's own source tree - see app.utils.config.USER_DATA_OPERATIONS_DIR),
as a class that directly subclasses DataOperationDialogBase -
data_operation_scanner.py AST-scans both folders at import time and
picks it up with no registration step. This dialog is a short form for
Name and Description that writes a new, immediately-importable file with
those fields already filled in, a working (if trivial) compute_results
that passes each selected series' data through unchanged, and the rest of
the dialog shell (model selector, parameter form, preview pane) left at
their base-class defaults for the user to override where their operation
actually needs one. See DataOperationDialogBase's own docstring
(app/data_operations/dialog_base/, hooks.py) for the full contract, and
app/data_operations/function_dialog.py for the smallest complete,
hand-written dialog to read alongside it.
"""
from __future__ import annotations

from pathlib import Path

from app.developer_helpers.developer_helper_base import DeveloperDialogBase, slug as _slug
from app.utils.config import USER_DATA_OPERATIONS_DIR

#: Where data_operation_scanner.py itself looks for the built-in
#: operations - used here only to check a new Name against them, never
#: written to.
BUILTIN_OPERATIONS_DIR: Path = (
    Path(__file__).resolve().parent.parent / "data_operations"
)

#: Where this dialog writes - a user-authored operation, kept separate
#: from the app's own shipped source. Created on first use.
OPERATIONS_DIR: Path = USER_DATA_OPERATIONS_DIR


def render_stub_source(*, class_name: str, name: str, description: str, slug: str) -> str:
    """Return the full source text of the scaffolded operation file."""
    return f'''"""{name} data operation.

TODO: describe what this operation computes and when to reach for it, the
way every dialog under app/data_operations/ does at the top of its own
file - app/data_operations/function_dialog.py is the smallest complete,
hand-written example to read alongside this one, and
DataOperationDialogBase's own docstring (app/data_operations/
dialog_base/hooks.py) documents every hook below plus the ones left out here
(build_model_selector, PARAMS/build_parameter_selector, ...) because their
base-class defaults already do something reasonable - override one only
once this operation actually needs it.

This file was scaffolded by the Data Operation Builder (Developer menu).
It is already a valid, discoverable operation - data_operation_scanner.py
picks up any class in app/data_operations/ that directly subclasses
DataOperationDialogBase, with no registration step - and compute_results
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
from app.data_operations.dialog_base import (
    ResultSeriesSpec,
    DataOperationDialogBase,
)
from app.data_operations.results import TableResult


@dataclass(slots=True)
class {class_name}Result(TableResult):
    """One computed result for one source series.

    TableResult saves to_df() as a table and draws it through
    result_series_spec below; model, metadata and source_name are what
    its model_name, parameters and series report.
    """

    source_name: str
    model: str
    x: Any
    y: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({{"x": self.x, "y": self.y}})


class {class_name}(DataOperationDialogBase):
    """TODO: one line describing this operation."""

    Name: str = {name!r}
    Description: str = {description!r}
    #: Words to find it by in the search, beyond its name and description
    #: (synonyms, in English and Italian), and the guided analysis goals it
    #: serves (keys of app.analysis.catalog.GOALS): "describe", "compare", ...
    Keywords: str = ""
    Goals: list[str] = []

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
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
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
            applogger.warning(message)
        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
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


class DataOperationBuilderDialog(DeveloperDialogBase):
    """Scaffold a new DataOperationDialogBase file from a short form."""

    TITLE = "Data Operation Builder"
    ICON = "data_operation_builder"
    CARD_NAME = "dataOperationBuilderCard"
    HINT = (
        "Writes a new file under user/data_operations/ with a class "
        "already filled in from these fields - no registration step, "
        "the scanner picks it up as soon as the file exists. "
        "compute_results already passes each selected series through "
        "unchanged; replace it with the real computation."
    )
    NAME_PLACEHOLDER = "e.g. Envelope Detector"
    FILE_PLACEHOLDER = "e.g. envelope_detector_dialog.py"
    DESCRIPTION_PLACEHOLDER = "One line, shown in the Data Operations panel."
    FILE_SUFFIX = "_dialog.py"
    CLASS_SUFFIX = "Dialog"
    CREATE_ACTION = "create_data_operation"
    BUILTIN_DIR = BUILTIN_OPERATIONS_DIR
    USER_DIR = OPERATIONS_DIR
    BASE_CLASS_NAME = "DataOperationDialogBase"
    CREATED_MESSAGE = "dev.data_operation_created"
    FAILED_MESSAGE = "dev.data_operation_create_failed"
    BAD_FILE_NAME = "File name must be a valid Python file name, e.g. my_operation_dialog.py."
    FILE_EXISTS = "A file named \"{file}\" already exists under user/data_operations/."
    NAME_TAKEN = "\"{name}\" is already used by an existing data operation."
    NOT_DISCOVERED = (
        "The class imported, but data_operation_scanner did not discover "
        "it as an operation - check that it subclasses "
        "DataOperationDialogBase directly."
    )

    def render_source(self, *, class_name: str, name: str, description: str) -> str:
        return render_stub_source(
            class_name=class_name, name=name, description=description, slug=_slug(name) or "operation"
        )
