#!/usr/bin/env python3
"""Upgrade every library in requirements.txt to its latest release, then check the result.

    python3 dev/tools/update_dependencies.py               # upgrade ChartLibre's .venv (or .python)
    python3 dev/tools/update_dependencies.py --dry-run     # only say what would change
    python3 dev/tools/update_dependencies.py --write-pins  # and write the new versions into requirements.txt
    python3 dev/tools/update_dependencies.py --test        # and run the test suite afterwards
    python3 dev/tools/update_dependencies.py --python PATH # another interpreter (a .venv-3.15, say)

It is ``pip install --upgrade`` over requirements.txt with the pins taken
off, so each line's own marker (python_version, sys_platform) still decides
whether it applies - with HOLD as constraints, for the releases known not
to work. Before and after, it lists what changed. With --write-pins the
lines that apply to this interpreter get the installed versions; the other
interpreter's lines (the 3.15 pins, say) are left as they are.

Standard library only, like install.py, so any Python can run it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements.txt"

#: Upper bounds kept on purpose, each with its reason. Remove a line once the
#: release it guards against is fixed.
HOLD: dict[str, str] = {
    # 6.12.0 on macOS: Qt's QDir lists the plugin folder as empty, so no
    # platform plugin (cocoa, offscreen) is found and no window can open.
    "PySide6": "<6.12",
    "PySide6_Essentials": "<6.12",
    "PySide6_Addons": "<6.12",
    "shiboken6": "<6.12",
}

_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:[=<>!~]=?[^;#]*)?(;[^#]*)?")


def default_python() -> Path:
    """The interpreter ChartLibre runs on: its .venv, else the launchers' .python, else this one."""
    for candidate in (ROOT / ".venv" / "bin" / "python", ROOT / ".venv" / "Scripts" / "python.exe",
                      ROOT / ".python" / "bin" / "python3", ROOT / ".python" / "python.exe"):
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def requirement_lines(text: str) -> list[tuple[int, str, str]]:
    """(line index, package name, marker or "") of every requirement line."""
    found = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("-"):
            continue
        match = _LINE.match(stripped)
        if match:
            found.append((index, match.group(1), (match.group(2) or "").lstrip(";").strip()))
    return found


def unpinned(text: str) -> str:
    """requirements.txt with every version taken off and every marker kept."""
    return "\n".join(name + (f"; {marker}" if marker else "") for _index, name, marker in requirement_lines(text))


def installed(python: Path) -> dict[str, str]:
    """{lower-case name: version} in *python*'s environment."""
    out = subprocess.run([str(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
                         capture_output=True, text=True, check=True).stdout
    return {item["name"].lower().replace("_", "-"): item["version"] for item in json.loads(out)}


def markers_apply(python: Path, markers: list[str]) -> list[bool]:
    """Whether each marker holds for *python* (evaluated there, with pip's own packaging)."""
    code = ("import json, sys\n"
            "from pip._vendor.packaging.markers import Marker\n"
            "print(json.dumps([not m or Marker(m).evaluate() for m in json.loads(sys.stdin.read())]))")
    out = subprocess.run([str(python), "-c", code], input=json.dumps(markers), capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)


def write_pins(python: Path, versions: dict[str, str]) -> list[str]:
    """Pin the lines that apply to *python* at the installed versions; the lines changed."""
    text = REQUIREMENTS.read_text(encoding="utf-8")
    lines = text.splitlines()
    entries = requirement_lines(text)
    applies = markers_apply(python, [marker for _i, _n, marker in entries])
    changed = []
    for (index, name, marker), here in zip(entries, applies):
        version = versions.get(name.lower().replace("_", "-"))
        line = lines[index]
        requirement, separator, rest = line.partition(";")
        if not here or version is None or "==" not in requirement:
            continue  # another interpreter's pin, a package not installed, or a ">=" kept as it is
        updated = re.sub(r"==\s*[^;#\s]+", f"=={version}", requirement, count=1) + separator + rest
        if updated != line:
            lines[index] = updated
            changed.append(f"{line.strip()}  ->  {updated.strip()}")
    REQUIREMENTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upgrade ChartLibre's libraries to their latest releases.")
    parser.add_argument("--python", type=Path, default=None, help="the interpreter to upgrade (default: .venv)")
    parser.add_argument("--dry-run", action="store_true", help="only report what would be installed")
    parser.add_argument("--write-pins", action="store_true", help="write the new versions into requirements.txt")
    parser.add_argument("--test", action="store_true", help="run the test suite afterwards")
    args = parser.parse_args(argv)

    python = args.python or default_python()
    print(f"Interpreter: {python}")
    before = installed(python)
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / "requirements.txt"
        constraints = Path(scratch) / "constraints.txt"
        requirements.write_text(unpinned(REQUIREMENTS.read_text(encoding="utf-8")), encoding="utf-8")
        constraints.write_text("\n".join(f"{name}{bound}" for name, bound in HOLD.items()), encoding="utf-8")
        command = [str(python), "-m", "pip", "install", "--upgrade", "--disable-pip-version-check",
                   "-r", str(requirements), "-c", str(constraints)]
        if args.dry_run:
            command.append("--dry-run")
        print("Running:", " ".join(command[2:]))
        if subprocess.run(command).returncode:
            print("pip failed: nothing else was done.")
            return 1
    if args.dry_run:
        return 0

    after = installed(python)
    changed = sorted(name for name in after if before.get(name) != after[name])
    print("\nChanged:" if changed else "\nEverything was already at its latest allowed version.")
    for name in changed:
        print(f"  {name:28} {before.get(name, '-'):>12}  ->  {after[name]}")
    for name, bound in HOLD.items():
        print(f"  held: {name}{bound}")

    # The same import check install.py ends with: Qt, the scientific libraries.
    sys.path.insert(0, str(ROOT))
    from install import verify

    problems = verify(python)
    print("\nImport check: " + ("every library loads." if not problems else "PROBLEMS"))
    for problem in problems:
        print("  " + problem)

    if args.write_pins:
        lines = write_pins(python, after)
        print("\nrequirements.txt:" if lines else "\nrequirements.txt already matches.")
        for line in lines:
            print("  " + line)

    if problems:
        return 1
    if args.test:
        print("\nRunning the test suite ...")
        return subprocess.run([str(python), "-m", "pytest", "-q", str(ROOT / "dev" / "tests")], cwd=ROOT,
                              env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
