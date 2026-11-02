"""Both kinds of analysis window share one base, and declare the data they pick."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog

from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase
from app.plugin_api.data_operations.roles_base import RolesOperationDialogBase


def test_both_bases_are_one_window() -> None:
    """One base, straight on QDialog: no intermediate class with a single descendant."""
    assert DataOperationDialogBase.__bases__ == (QDialog,)
    assert RolesOperationDialogBase.__bases__ == (DataOperationDialogBase,)


def test_each_base_declares_the_data_it_picks() -> None:
    assert DataOperationDialogBase.DATA_PICKER == "series"
    assert RolesOperationDialogBase.DATA_PICKER == "roles"
