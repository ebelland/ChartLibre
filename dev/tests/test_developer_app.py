"""dev/ChartLibreDeveloper.py: the launcher builds, and its quick jobs answer without running anything long."""
from __future__ import annotations

import runpy
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "ChartLibreDeveloper.py"


def test_the_launcher_shows_every_tool_and_counts_the_project(qapp) -> None:
    from PySide6.QtWidgets import QPushButton

    module = runpy.run_path(str(LAUNCHER), run_name="developer_app_test")
    window = module["DeveloperWindow"]()
    try:
        texts = {button.text() for button in window.findChildren(QPushButton)}
        for wanted in ("Run the tests", "NIST report", "Build the bundle", "Compile the user manual",
                       "Developer manual", "Make chart previews", "Update Python libraries"):
            assert wanted in texts
        window._project_size()
        assert window._log.toPlainText().startswith("Now: app/")
    finally:
        window.close()
