"""Append the application's size to dev/appsize.csv: one row per commit, run by the post-commit hook.

Counted: the Python files under app/ (the application itself), not dev/ (tools, tests, plans). Lines are all
lines; code lines leave out blank lines and comments. Tests are counted apart, for comparison.

    .python/bin/python3 dev/tools/app_size.py            # a row for HEAD
    .python/bin/python3 dev/tools/app_size.py --install  # write .git/hooks/post-commit
    .python/bin/python3 dev/tools/app_size.py --history  # a row for every past commit missing from the file

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


def _count(text: str) -> tuple[int, int]:
    lines = code = 0
    for line in text.splitlines():
        lines += 1
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            code += 1
    return lines, code


def measure_commit(commit: str, folder: str) -> tuple[int, int, int]:
    """(lines, code lines, files) of the Python files under *folder* as *commit* had them - read from git."""
    names = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", commit, "--", folder], cwd=ROOT,
                           capture_output=True, check=True).stdout.decode("utf-8", "replace").split("\0")
    names = [name for name in names if name.endswith(".py")]
    if not names:
        return 0, 0, 0
    batch = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, check=True, capture_output=True,
                           input="".join(f"{commit}:{name}\n" for name in names).encode()).stdout
    lines = code = files = 0
    position = 0
    while position < len(batch):
        header_end = batch.index(b"\n", position)
        header = batch[position:header_end].split()
        position = header_end + 1
        if len(header) < 3 or header[1] != b"blob":
            continue
        size = int(header[2])
        counted = _count(batch[position:position + size].decode("utf-8", "replace"))
        lines, code, files = lines + counted[0], code + counted[1], files + 1
        position += size + 1
    return lines, code, files


def history() -> int:
    """Rows for the commits of HEAD's history missing from the file; the whole file rewritten by date."""
    rows: dict[str, list[str]] = {}
    if CSV.exists():
        with CSV.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["commit"]] = [row[key] for key in HEADER]
    log = subprocess.run(["git", "log", "--reverse", "--format=%h %cd", "--date=format:%Y-%m-%d %H:%M"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    added = 0
    for entry in log:
        commit, date = entry.split(" ", 1)
        if commit in rows:
            continue
        lines, code, files = measure_commit(commit, "app")
        test_lines, _code, test_files = measure_commit(commit, "dev/tests")
        rows[commit] = [date, commit, str(lines), str(code), str(files), str(test_lines), str(test_files)]
        added += 1
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(sorted(rows.values(), key=lambda row: row[0]))
    print(f"{added} commits added; {len(rows)} rows.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--install", action="store_true", help="write the post-commit hook")
    parser.add_argument("--history", action="store_true", help="add a row for every past commit")
    options = parser.parse_args()
    if options.history:
        return history()
    if options.install:
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
