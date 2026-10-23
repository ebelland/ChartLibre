"""Every tracked file can be checked out on Windows: a ':' in a demo's name broke the Windows build and pull."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: What Windows refuses in a file name, and the device names it reserves.
INVALID = re.compile(r'[<>:"|?*\x00-\x1f]')
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def test_every_tracked_name_is_valid_on_windows() -> None:
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    bad = []
    for path in filter(None, listed.decode("utf-8").split("\0")):
        for part in path.split("/"):
            if INVALID.search(part) or part.endswith((" ", ".")) or part.split(".")[0].lower() in RESERVED:
                bad.append(path)
                break
    assert bad == []


def test_demo_titles_make_valid_file_names() -> None:
    from app.data.demos import DEMO_PROJECTS

    assert [demo.file_name for demo in DEMO_PROJECTS if INVALID.search(demo.file_name)] == []
