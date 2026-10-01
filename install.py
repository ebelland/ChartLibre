#!/usr/bin/env python3
"""Install ChartLibre's libraries into a virtual environment (.venv) in its own folder.

    python3 install.py                 # create .venv and install requirements.txt
    python3 install.py --recreate      # throw an existing .venv away and start again
    python3 install.py --dir PATH      # install into PATH instead of this folder

On macOS double-click "Install ChartLibre.command", on Windows
"Install ChartLibre.bat": both find a suitable Python and run this file. It
needs Python 3.11 or newer and an internet connection, and it uses nothing
but the standard library, so any Python that is new enough can run it.

Nothing is written outside the folder. When it has finished it imports Qt
and the scientific libraries once as a check, so a broken install is found
here, with a message, rather than as a crash the first time ChartLibre is
opened.

ChartLibre.app, ChartLibre.exe and ChartLibre.bat do the same thing by
themselves on the first launch; this is the explicit way, and the only one
on Linux.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

#: The oldest Python ChartLibre runs on.
MINIMUM_PYTHON: tuple[int, int] = (3, 11)

HERE = Path(__file__).resolve().parent

#: The import that shows each requirement is really there and working. Qt is
#: imported through QtWidgets, which loads the platform plugins' machinery.
CHECKS: tuple[str, ...] = (
    "PySide6.QtWidgets",
    "matplotlib",
    "numpy",
    "pandas",
    "openpyxl",
    "scipy",
    "sklearn",
    "statsmodels",
    "skimage",
    "pywt",
)


def python_problem(version: Sequence[int]) -> str:
    """Why *version* cannot run ChartLibre, or an empty string when it can."""
    if tuple(version[:2]) < MINIMUM_PYTHON:
        found = ".".join(str(part) for part in version[:3])
        wanted = ".".join(str(part) for part in MINIMUM_PYTHON)
        return f"ChartLibre needs Python {wanted} or newer; this is Python {found}."
    return ""


def venv_python(venv: Path, platform: str = sys.platform) -> Path:
    """The interpreter inside *venv*."""
    if platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python3"


def cloud_sync_warning(folder: Path, home: Path, platform: str = sys.platform) -> str:
    """A warning when *folder* is in a place iCloud syncs, else an empty string.

    With "Desktop & Documents" sync on, iCloud marks everything inside a
    dot-folder as hidden - .venv included - and Qt then cannot find its
    plugins. ChartLibre clears the flag when it starts, but a project folder
    that is being synced (and thinned out to save disk space) is worth a
    warning anyway.
    """
    if platform != "darwin":
        return ""
    for name in ("Desktop", "Documents"):
        if folder == home / name or (home / name) in folder.parents:
            return (
                f"This folder is inside ~/{name}, which iCloud may be syncing. That can "
                "hide files inside .venv and remove them from the disk. If ChartLibre "
                "does not start, move the folder somewhere else (for example ~/Developer) "
                "and run this installer again with --recreate."
            )
    return ""


def create_venv(folder: Path, recreate: bool) -> tuple[Path, bool]:
    """Create ``folder/.venv``; return (its interpreter, whether it was created now)."""
    venv = folder / ".venv"
    python = venv_python(venv)
    if venv.exists() and recreate:
        print(f"Removing the existing {venv} ...")
        shutil.rmtree(venv)
    if python.exists():
        print(f"{venv} already exists; keeping it (use --recreate to start again).")
        return python, False
    print(f"Creating {venv} ...")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    return python, True


def install_requirements(python: Path, requirements: Path) -> None:
    """pip-install *requirements* into the environment of *python*, showing pip's output."""
    print("Installing the libraries. This needs an internet connection and takes a few minutes ...")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)


def clear_hidden_flag(venv: Path, platform: str = sys.platform) -> None:
    """Best effort: clear macOS's "hidden" flag from everything in *venv*."""
    if platform != "darwin":
        return
    subprocess.run(["chflags", "-R", "nohidden", str(venv)], check=False, capture_output=True)


def verify(python: Path) -> list[str]:
    """Import every library ChartLibre needs; return what could not be imported."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    problems: list[str] = []
    for module in CHECKS:
        result = subprocess.run(
            [str(python), "-c", f"import {module}"],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            last = (result.stderr.strip().splitlines() or ["failed"])[-1]
            problems.append(f"{module}: {last}")
    return problems


def start_hint(folder: Path, platform: str = sys.platform) -> str:
    if platform == "darwin":
        return "Open ChartLibre.app (double-click it in Finder)."
    if platform == "win32":
        return "Open ChartLibre.exe (or ChartLibre.bat) by double-clicking it."
    return f"Start it with:  {venv_python(folder / '.venv')} {folder / 'main.py'}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install ChartLibre's libraries into .venv.")
    parser.add_argument("--dir", type=Path, default=HERE, help="the ChartLibre folder (default: where this file is)")
    parser.add_argument("--recreate", action="store_true", help="delete an existing .venv and install again")
    parser.add_argument("--skip-packages", action="store_true", help="only create the .venv (for testing)")
    args = parser.parse_args(argv)
    folder = args.dir.resolve()

    problem = python_problem(tuple(sys.version_info[:3]))
    if problem:
        print(problem)
        print("Install a newer one from https://www.python.org/downloads/ and run this again.")
        return 1
    requirements = folder / "requirements.txt"
    if not args.skip_packages and not requirements.is_file():
        print(f"{requirements} is missing - is {folder} the ChartLibre folder?")
        return 1

    note = cloud_sync_warning(folder, Path.home())
    if note:
        print(f"Note: {note}\n")

    created = False
    try:
        python, created = create_venv(folder, args.recreate)
        if not args.skip_packages:
            install_requirements(python, requirements)
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"\nThe installation did not complete: {error}")
        print("The messages above say why. Nothing outside this folder was changed.")
        if created:
            shutil.rmtree(folder / ".venv", ignore_errors=True)
            print("The unfinished .venv was removed, so a second run starts clean.")
        return 1

    clear_hidden_flag(folder / ".venv")
    if args.skip_packages:
        print("\n.venv created (packages skipped).")
        return 0

    print("\nChecking the installation ...")
    problems = verify(python)
    if problems:
        print("The libraries were installed, but these would not load:")
        for line in problems:
            print(f"  {line}")
        print("Run this again with --recreate; if it keeps failing, please report the lines above.")
        return 1

    print("\nChartLibre is installed. " + start_hint(folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
