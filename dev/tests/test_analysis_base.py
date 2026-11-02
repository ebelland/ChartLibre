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


def test_the_window_opens_at_its_declared_size(qapp, repo) -> None:
    """WINDOW_SIZE, not the shared dialog size: the layout must not resize the window it lays out."""
    from app.data_operations.regression_dialog import SeriesRegressionDialog

    figure_id = repo.create_figure_descriptor(name="Size")
    dialog = SeriesRegressionDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        assert (dialog.width(), dialog.height()) == SeriesRegressionDialog.WINDOW_SIZE
    finally:
        dialog.close()
