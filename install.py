#!/usr/bin/env python3
"""Install ChartLibre's libraries into a virtual environment (.venv) in its own folder.

    python3 install.py                 # create .venv and install requirements.txt
    python3 install.py --recreate      # throw an existing .venv away and start again
    python3 install.py --dir PATH      # install into PATH instead of this folder

The launchers - ChartLibre.app, ChartLibre.exe/.bat, ChartLibre.sh - run
this file on the first launch with the Python they download into .python
(dev/tools/make_launcher.py), so nobody has to install one. Run by hand, it
needs Python 3.11 or newer and an internet connection, and it uses nothing
but the standard library, so any Python that is new enough can run it.

Nothing is written outside the folder. When it has finished it imports Qt
and the scientific libraries once as a check, so a broken install is found
here, with a message, rather than as a crash the first time ChartLibre is
opened.

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
    if python.exists() and not venv_has_pip(python):
        # Left by a run that failed half-way - on Ubuntu, a venv made without
        # python3-venv has no pip - and every later run would keep it.
        print(f"{venv} is incomplete (no pip); making it again ...")
        shutil.rmtree(venv)
    if python.exists():
        print(f"{venv} already exists; keeping it (use --recreate to start again).")
        return python, False
    print(f"Creating {venv} ...")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    return python, True


def venv_has_pip(python: Path) -> bool:
    """True when *python*'s environment can run pip."""
    result = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True, check=False)
    return result.returncode == 0


def venv_problem() -> str:
    """Why this Python cannot make a working .venv, or "" when it can.

    Debian and Ubuntu ship Python without ensurepip, which venv needs to put
    pip in the environment; it comes with the python3-venv package.
    """
    try:
        import ensurepip  # noqa: F401
        import venv  # noqa: F401
    except ImportError:
        return (
            "This Python cannot create a .venv yet. On Ubuntu or Debian install it with\n\n"
            "    sudo apt install python3-venv\n\n"
            "and run this again."
        )
    return ""


def install_requirements(python: Path, requirements: Path) -> None:
    """pip-install *requirements* into the environment of *python*, showing pip's output."""
    print("Installing the libraries. This needs an internet connection and takes a few minutes ...")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)


def clear_hidden_flag(venv: Path, platform: str = sys.platform) -> None:
    """Best effort: clear macOS's "hidden" flag from everything in *venv*,
    and from the Python beside it that the launchers download (.python)."""
    if platform != "darwin":
        return
    for folder in (venv, venv.parent / ".python"):
        if folder.exists():
            subprocess.run(["chflags", "-R", "nohidden", str(folder)], check=False, capture_output=True)


#: Libraries Qt needs on Linux that a desktop install may lack, with the
#: Debian/Ubuntu package that provides each. Without libxcb-cursor Qt 6 cannot
#: open a window at all ("could not load the Qt platform plugin xcb").
LINUX_QT_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("xcb-cursor", "libxcb-cursor0"),
    ("xkbcommon-x11", "libxkbcommon-x11-0"),
    ("EGL", "libegl1"),
)


def missing_linux_packages(find_library=None) -> list[str]:
    """The packages to install for the libraries Qt needs and cannot find."""
    if find_library is None:
        from ctypes.util import find_library
    return [package for library, package in LINUX_QT_LIBRARIES if not find_library(library)]


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
    return f"Start it from the applications menu, or with:  {folder / 'ChartLibre.sh'}"


def linux_menu_entry(folder: Path) -> str:
    """The freedesktop menu entry that starts ChartLibre from *folder*.

    Absolute paths, which is why it is written here, on the machine, rather
    than shipped: it points at ChartLibre.sh and the icon in this folder.
    """
    launcher = folder / "ChartLibre.sh"
    icon = folder / "dev" / "tools" / "launcher" / "chartlibre.png"
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=ChartLibre\n"
        "Comment=Charts and data analysis\n"
        f'Exec="{launcher}"\n'
        f"Path={folder}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Science;Education;Office;\n"
    )


def write_linux_menu_entry(folder: Path, home: Path | None = None) -> Path:
    """Add ChartLibre to the applications menu (GNOME, KDE, Xfce...), for this user."""
    applications = (home or Path.home()) / ".local" / "share" / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    entry = applications / "chartlibre.desktop"
    entry.write_text(linux_menu_entry(folder), encoding="utf-8")
    entry.chmod(0o755)
    launcher = folder / "ChartLibre.sh"
    if launcher.exists():
        launcher.chmod(0o755)  # a copy can lose the bit (a shared folder, a zip)
    return entry


#: Written into the private Python (.python) once its libraries are installed
#: and checked: what the launchers look for to know they can start ChartLibre.
READY_MARKER: str = "CHARTLIBRE_READY"


def runtime_ready(prefix: Path) -> Path:
    """The marker file that says the Python at *prefix* has ChartLibre's libraries."""
    return prefix / READY_MARKER


def linux_notes(folder: Path, menu: bool) -> None:
    """On Linux: name the system libraries Qt lacks, and add the menu entry."""
    if not sys.platform.startswith("linux"):
        return
    packages = missing_linux_packages()
    if packages:
        print("\nQt also needs a few system libraries that are missing here. Install them with\n\n"
              f"    sudo apt install {' '.join(packages)}\n\n"
              "(on Fedora: sudo dnf install xcb-util-cursor libxkbcommon-x11 mesa-libEGL), then start ChartLibre.")
    if menu:
        entry = write_linux_menu_entry(folder)
        print(f"ChartLibre is in the applications menu now ({entry}).")


def install_runtime(folder: Path, requirements: Path, menu: bool,
                    python: Path | None = None, prefix: Path | None = None) -> int:
    """Install the libraries into the Python running this file - the private
    one the launchers download into .python - with no .venv.

    A standalone Python can be moved with its folder, which a .venv cannot:
    this is what lets the ready-made packages ship with the libraries already
    in place (dev/tools/build_bundle.py).
    """
    python = python or Path(sys.executable)
    marker = runtime_ready(prefix or Path(sys.prefix))
    marker.unlink(missing_ok=True)
    try:
        install_requirements(python, requirements)
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"\nThe installation did not complete: {error}")
        print("The messages above say why. Run it again to carry on where it stopped.")
        return 1
    clear_hidden_flag(folder / ".venv")
    print("\nChecking the installation ...")
    problems = verify(python)
    if problems:
        print("The libraries were installed, but these would not load:")
        for line in problems:
            print(f"  {line}")
        print("Delete the .python folder and start ChartLibre again; if it keeps failing, please report the lines above.")
        return 1
    marker.write_text("ChartLibre's libraries are installed in this Python.\n", encoding="utf-8")
    print("\nChartLibre is installed. " + start_hint(folder))
    linux_notes(folder, menu)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install ChartLibre's libraries into .venv.")
    parser.add_argument("--dir", type=Path, default=HERE, help="the ChartLibre folder (default: where this file is)")
    parser.add_argument("--recreate", action="store_true", help="delete an existing .venv and install again")
    parser.add_argument("--skip-packages", action="store_true", help="only create the .venv (for testing)")
    parser.add_argument("--no-menu", action="store_true", help="on Linux, do not add ChartLibre to the applications menu")
    parser.add_argument("--runtime", action="store_true",
                        help="install into this Python itself, not a .venv (what the launchers do with .python)")
    parser.add_argument("--menu-only", action="store_true", help="on Linux, only add ChartLibre to the applications menu")
    args = parser.parse_args(argv)
    folder = args.dir.resolve()

    if args.menu_only:
        if sys.platform.startswith("linux"):
            linux_notes(folder, menu=True)
        return 0

    problem = python_problem(tuple(sys.version_info[:3]))
    if problem:
        print(problem)
        print("Install a newer one from https://www.python.org/downloads/ and run this again.")
        return 1
    requirements = folder / "requirements.txt"
    if not args.skip_packages and not requirements.is_file():
        print(f"{requirements} is missing - is {folder} the ChartLibre folder?")
        return 1

    if args.runtime:
        return install_runtime(folder, requirements, menu=not args.no_menu)

    problem = venv_problem()
    if problem and not venv_python(folder / ".venv").exists():
        print(problem)
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
    linux_notes(folder, menu=not args.no_menu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
