"""Append the application's size to dev/appsize.csv: one row per commit, run by the post-commit hook.

Counted: the Python files under app/ (the application itself), not dev/ (tools, tests, plans). Lines are all
lines; code lines leave out blank lines and comments. Tests are counted apart, for comparison.

    .python/bin/python3 dev/tools/app_size.py            # a row for HEAD
    .python/bin/python3 dev/tools/app_size.py --install  # write .git/hooks/post-commit

The row a commit writes goes into the next commit: a hook cannot add to the commit it runs after.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "dev" / "appsize.csv"
HEADER = ["date", "commit", "lines", "code_lines", "files", "test_lines", "test_files"]
HOOK = """#!/bin/sh
# dev/tools/app_size.py --install wrote this: the application's size after every commit, in dev/appsize.csv.
python3 dev/tools/app_size.py >/dev/null 2>&1 || true
"""


def measure(folder: Path) -> tuple[int, int, int]:
    """(lines, code lines, files) of the tracked Python files under *folder*."""
    listed = subprocess.run(["git", "ls-files", "--", f"{folder.relative_to(ROOT)}/*.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split("\n")
    lines = code = files = 0
    for name in filter(None, listed):
        path = ROOT / name
        if not path.is_file():
            continue
        files += 1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            lines += 1
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code += 1
    return lines, code, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--install", action="store_true", help="write the post-commit hook")
    if parser.parse_args().install:
        hook = ROOT / ".git" / "hooks" / "post-commit"
        hook.write_text(HOOK, encoding="utf-8")
        hook.chmod(0o755)
        print(f"Wrote {hook}")
        return 0
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    lines, code, files = measure(ROOT / "app")
    test_lines, _, test_files = measure(ROOT / "dev" / "tests")
    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(HEADER)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), commit, lines, code, files, test_lines,
                         test_files])
    print(f"{commit}: {lines} lines ({code} code) in {files} files; tests {test_lines} in {test_files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
