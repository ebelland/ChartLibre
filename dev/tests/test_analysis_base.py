"""Both kinds of analysis window share one base, and declare the data they pick."""
from __future__ import annotations

from app.plugin_api.analysis_base import AnalysisDialogBase
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase
from app.plugin_api.operation_shell import OperationShell
from app.plugin_api.data_operations.roles_base import RolesOperationDialogBase


def test_both_bases_derive_from_the_shared_analysis_base() -> None:
    assert issubclass(DataOperationDialogBase, AnalysisDialogBase)
    assert issubclass(RolesOperationDialogBase, AnalysisDialogBase)
    assert issubclass(AnalysisDialogBase, OperationShell)


def test_each_base_declares_the_data_it_picks() -> None:
    assert DataOperationDialogBase.DATA_PICKER == "series"
    assert RolesOperationDialogBase.DATA_PICKER == "roles"
