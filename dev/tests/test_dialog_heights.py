"""No dialog needs more height than a laptop screen has.

A 1366 x 768 laptop leaves about 697 px of client area under the title bar and
above the taskbar. screen_fit.fit_to_screen shrinks a dialog to the screen but
never below its own minimum, so a dialog whose layout asks for more than that
simply runs off the bottom - taking its OK and Cancel with it. Import data
stacked four cards and asked for 826.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo

#: The client height a 1366 x 768 laptop leaves a dialog.
LAPTOP_CLIENT_HEIGHT = 690


@pytest.fixture
def repo(tmp_db_path: Path) -> Iterator[SqliteRepo]:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def test_import_data_fits_a_laptop(qapp, repo: SqliteRepo) -> None:
    from app.dialogs.import_data_dialog import ImportDataDialog

    dialog = ImportDataDialog(repo)
    dialog.show()
    qapp.processEvents()
    assert dialog.minimumSizeHint().height() <= LAPTOP_CLIENT_HEIGHT
    dialog.close()


def test_the_settings_column_scrolls_instead_of_stretching_the_dialog(qapp, repo: SqliteRepo) -> None:
    from PySide6.QtWidgets import QScrollArea

    from app.dialogs.import_data_dialog import ImportDataDialog

    dialog = ImportDataDialog(repo)
    areas = dialog.settings_frame.findChildren(QScrollArea)
    assert areas and areas[0].widget() is not None
    dialog.close()


def test_the_other_dialogs_stay_within_a_laptop_too(qapp, repo: SqliteRepo) -> None:
    from app.dialogs.settings_dialog import SettingsDialog
    from app.data_operations.spectral_dialog import SeriesSpectralDialog
    from app.data_operations.filter_dialog import SeriesFilterDialog

    figure_id = repo.create_figure_descriptor(name="F")
    for build in (
        lambda: SettingsDialog(),
        lambda: SeriesSpectralDialog(repo=repo, figure_id=figure_id),
        lambda: SeriesFilterDialog(repo=repo, figure_id=figure_id),
    ):
        dialog = build()
        dialog.show()
        qapp.processEvents()
        assert dialog.minimumSizeHint().height() <= LAPTOP_CLIENT_HEIGHT, type(dialog).__name__
        dialog.close()


def test_every_data_operation_fits_a_laptop(qapp, repo: SqliteRepo) -> None:
    """Their pages scroll, so no parameter form sets the window's minimum height."""
    import importlib
    import pkgutil

    import app.data_operations as operations
    from app.data_operations.dialog_base import DataOperationDialogBase

    figure_id = repo.create_figure_descriptor(name="F")
    for info in pkgutil.iter_modules(operations.__path__):
        if not info.name.endswith("_dialog"):
            continue
        module = importlib.import_module(f"app.data_operations.{info.name}")
        for value in vars(module).values():
            if (isinstance(value, type) and issubclass(value, DataOperationDialogBase)
                    and value.__module__ == module.__name__):
                dialog = value(repo=repo, figure_id=figure_id, parent=None)
                dialog.show()
                qapp.processEvents()
                assert dialog.minimumSizeHint().height() <= LAPTOP_CLIENT_HEIGHT - 150, value.__name__
                dialog.close()
