"""Frame titles keep the case they are written in ("Select Columns", not "SELECT COLUMNS"), as macOS's own (I-09)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STYLES = Path(__file__).resolve().parents[2] / "app" / "styles"


@pytest.mark.parametrize("sheet", ["fluent_win11.qss", "macos_native.qss", "ubuntu_native.qss"])
def test_a_section_title_is_not_forced_to_capitals(sheet: str) -> None:
    text = (STYLES / sheet).read_text(encoding="utf-8")
    for body in re.findall(r'QLabel\[sectionTitle="true"\]\s*\{([^}]*)\}', text):
        assert "text-transform" not in body, f"{sheet}: a section title is still transformed"
