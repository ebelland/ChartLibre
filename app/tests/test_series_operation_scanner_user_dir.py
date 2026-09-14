"""series_operation_scanner also scans user/series_operations/, alongside
the built-in app/series_operations/ (todo.txt P3-6 / the Series Operation
Builder's write target).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.scanners import series_operation_scanner as scanner

_STUB = '''from __future__ import annotations
class {cls}(SeriesOperationDialogBase):
    Name: str = "{name}"
'''


@pytest.fixture
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(scanner, "USER_SERIES_OPERATIONS_DIR", tmp_path)
    return tmp_path


def test_a_user_operation_is_discovered(user_dir: Path) -> None:
    (user_dir / "mine.py").write_text(
        _STUB.format(cls="MineDialog", name="My Own Operation"), encoding="utf-8"
    )

    discovered = scanner._discover_series_operations()

    assert any(entry["value"] == "My Own Operation" for entry in discovered)


def test_every_built_in_operation_still_appears(user_dir: Path) -> None:
    discovered = scanner._discover_series_operations()
    assert any(entry["value"] == "Outliers" for entry in discovered)


def test_a_missing_user_folder_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scanner, "USER_SERIES_OPERATIONS_DIR", tmp_path / "does_not_exist")

    discovered = scanner._discover_series_operations()

    assert any(entry["value"] == "Outliers" for entry in discovered)
