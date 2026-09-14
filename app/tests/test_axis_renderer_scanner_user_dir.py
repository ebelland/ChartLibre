"""axis_renderer_scanner also scans user/charts/, alongside the built-in
app/charts/ (todo.txt P3-6 / the Renderer Helper's write target) - see
app.utils.config.USER_CHARTS_DIR for why user-authored renderers live
outside the app's own source tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.scanners import axis_renderer_scanner as scanner

_STUB = '''from __future__ import annotations
from app.charts.base import BaseAxisRenderer


class {cls}(BaseAxisRenderer):
    Name: str = "{name}"
    Category: str = "User"
    Description: str = "d"
    RequiredRoles: list[str] = ["x", "y"]
'''


@pytest.fixture
def user_charts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(scanner, "USER_CHARTS_DIR", tmp_path)
    return tmp_path


def test_a_user_renderer_is_discovered(user_charts_dir: Path) -> None:
    (user_charts_dir / "mine.py").write_text(
        _STUB.format(cls="MineAxisRenderer", name="My Own Chart"), encoding="utf-8"
    )

    discovered = scanner._discover_axis_renderers()

    assert any(entry["value"] == "My Own Chart" for entry in discovered)


def test_every_built_in_renderer_still_appears(user_charts_dir: Path) -> None:
    discovered = scanner._discover_axis_renderers()
    assert any(entry["value"] == "Scatter Plot" for entry in discovered)


def test_a_name_collision_prefers_the_built_in_one(user_charts_dir: Path) -> None:
    (user_charts_dir / "fake_scatter.py").write_text(
        _STUB.format(cls="FakeScatterAxisRenderer", name="Scatter Plot"),
        encoding="utf-8",
    )

    discovered = scanner._discover_axis_renderers()
    matches = [entry for entry in discovered if entry["value"] == "Scatter Plot"]

    assert len(matches) == 1
    assert matches[0]["name"] != "FakeScatterAxisRenderer"


def test_a_missing_user_charts_folder_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scanner, "USER_CHARTS_DIR", tmp_path / "does_not_exist")

    discovered = scanner._discover_axis_renderers()

    assert any(entry["value"] == "Scatter Plot" for entry in discovered)
