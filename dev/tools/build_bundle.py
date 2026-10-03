"""Build a ready-made ChartLibre package for the system this runs on.

    python dev/tools/build_bundle.py                # build, test, package
    python dev/tools/build_bundle.py --no-package   # build and test only

The package is the ChartLibre folder with its private Python (``.python``)
already holding every library, so it opens the first time with no download
and no internet. It is the same folder the launchers make on a first launch
(dev/tools/make_launcher.py), built ahead of time:

1. the application's own files, as git tracks them - without dev/ (tests,
   sample data) and without the other systems' launchers;
2. the pinned standalone Python for this system, checked against its
   SHA-256 and unpacked into ``.python``;
3. requirements.txt installed into it by ``install.py --runtime``, which
   imports every library as a check and writes the CHARTLIBRE_READY marker;
4. ChartLibre started for a few seconds (after step 5, so it is the slimmed
   package that is tried) without a window, to show it runs;
5. slimmed: on macOS only this processor's half of each universal library,
   and on macOS and Linux identical copies replaced by links;
6. packed for the system: a .dmg on macOS, a .zip (and, with Inno Setup, an
   installer .exe) on Windows, a .tar.gz on Linux.

Built by .github/workflows/bundles.yml on each system. Needs only the
standard library, git, and on macOS hdiutil.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"

_spec = importlib.util.spec_from_file_location("make_launcher", Path(__file__).with_name("make_launcher.py"))
assert _spec is not None and _spec.loader is not None
make_launcher = importlib.util.module_from_spec(_spec)
# make_launcher imports app (for the name and version) - fine: no Qt at import.
sys.path.insert(0, str(ROOT))
_spec.loader.exec_module(make_launcher)

#: Launchers that belong to each system; the others are left out of its package.
LAUNCHERS: dict[str, tuple[str, ...]] = {
    "macos": ("ChartLibre.app",),
    "windows": ("ChartLibre.exe", "ChartLibre.bat"),
    "linux": ("ChartLibre.sh",),
}
ALL_LAUNCHERS = {name for names in LAUNCHERS.values() for name in names}

#: The one file under dev/ the application uses: the Linux menu icon.
KEEP_FROM_DEV = ("dev/tools/launcher/chartlibre.png",)

#: Tracked files that are notes for whoever develops ChartLibre, not for users.
LEAVE_OUT = ("todo.txt", ".gitignore")

#: Tracked folders that are the repository's, not the application's.
LEAVE_OUT_FOLDERS = (".github",)


def platform_key() -> str:
    """This system's key in make_launcher.PYTHON_ARCHIVES."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "macos-arm64" if machine == "arm64" else "macos-x86_64"
    if sys.platform == "win32":
        return "windows-x86_64"
    return "linux-aarch64" if machine in ("aarch64", "arm64") else "linux-x86_64"


def system_of(key: str) -> str:
    return key.split("-")[0]


def wanted_file(path: str, system: str) -> bool:
    """Whether the tracked *path* goes into *system*'s package."""
    top = path.split("/")[0]
    if path in LEAVE_OUT or top in LEAVE_OUT_FOLDERS:
        return False
    if top == "dev":
        return path in KEEP_FROM_DEV
    if top in ALL_LAUNCHERS:
        return top in LAUNCHERS[system]
    return True


def copy_application(folder: Path, system: str) -> int:
    """Copy the tracked files *system* needs into *folder*; return how many."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    count = 0
    for raw in listed.split(b"\0"):
        path = raw.decode("utf-8")
        if not path or not wanted_file(path, system):
            continue
        source = ROOT / path
        if not source.is_file():
            continue  # deleted in the working tree
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def download_python(key: str, folder: Path) -> Path:
    """Download, check and unpack the pinned Python into folder/.python."""
    url, digest = make_launcher._archive(key)
    archive = folder.parent / "python-download.tar.gz"
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    got = hashlib.sha256(archive.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f"The Python download is damaged: sha256 {got}, expected {digest}.")
    runtime = folder / ".python"
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            member.name = str(Path(*parts[1:]))  # drop the top "python/" folder
            tar.extract(member, runtime, filter="tar")
    archive.unlink()
    return runtime / ("python.exe" if sys.platform == "win32" else "bin/python3")


#: Mach-O "fat" (universal) headers, in either byte order.
_FAT_MAGICS = (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf", b"\xbe\xba\xfe\xca", b"\xbf\xba\xfe\xca")


def thin_universal_binaries(runtime: Path, arch: str) -> int:
    """Keep only *arch* in every universal Mach-O file under *runtime*; return bytes saved.

    PySide6's macOS wheels carry Intel and Apple Silicon code in every Qt
    library, and QtWebEngine alone is 450 MB of it. A package is for one of
    the two, so the other half goes. Each thinned file is signed again
    (ad hoc, as the wheels ship them): Apple Silicon refuses unsigned code.
    """
    # codesign refuses files carrying extended attributes ("detritus"),
    # which a folder iCloud syncs is full of.
    subprocess.run(["xattr", "-cr", str(runtime)], check=False)
    saved = 0
    thinned: list[Path] = []
    for path in runtime.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(4) not in _FAT_MAGICS:
                continue
        info = subprocess.run(["lipo", "-archs", str(path)], capture_output=True, text=True)
        archs = info.stdout.split()
        if info.returncode != 0 or arch not in archs or len(archs) < 2:
            continue  # a Java class file shares the magic; or nothing to drop
        before = path.stat().st_size
        thin = path.with_name(path.name + ".thin")
        subprocess.run(["lipo", str(path), "-thin", arch, "-output", str(thin)], check=True)
        mode = path.stat().st_mode
        thin.replace(path)
        path.chmod(mode)
        thinned.append(path)
        saved += before - path.stat().st_size
    # Innermost first: signing a framework's library seals the helper app
    # inside it, which must already carry its own signature.
    for path in sorted(thinned, key=lambda item: len(item.parts), reverse=True):
        result = subprocess.run(["codesign", "--force", "--sign", "-", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"codesign failed on {path}:\n{result.stderr}")
    return saved


def link_duplicates(root: Path, minimum: int = 256 * 1024) -> int:
    """Replace files under *root* identical to an earlier one by a relative
    symbolic link to it; return bytes saved.

    Wheels cannot hold symbolic links, so a Qt framework's Versions/Current
    and a library's version-numbered names arrive as full copies.
    """
    seen: dict[tuple[int, str], Path] = {}
    saved = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.stat().st_size < minimum:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = (path.stat().st_size, digest)
        original = seen.get(key)
        if original is None:
            seen[key] = path
            continue
        size = path.stat().st_size
        path.unlink()
        path.symlink_to(os.path.relpath(original, path.parent))
        saved += size
    return saved


def slim(runtime: Path, key: str) -> None:
    """Drop what this package cannot use: the other processor's code, duplicates."""
    saved = 0
    if system_of(key) == "macos":
        saved += thin_universal_binaries(runtime, "arm64" if key == "macos-arm64" else "x86_64")
    if system_of(key) != "windows":
        saved += link_duplicates(runtime)
    print(f"Slimmed the Python by {saved / 1e6:.0f} MB.")


#: Loads a page in Qt's web engine - the HTML results pane - and prints what
#: it reads back. Slimming touches QtWebEngine more than anything else.
WEB_CHECK = """
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
app = QApplication(sys.argv)
view = QWebEngineView()
def loaded(ok):
    view.page().runJavaScript("document.getElementById('x').textContent", lambda text: (print("WEB", ok, text), app.quit()))
view.loadFinished.connect(loaded)
view.setHtml("<p id='x'>web engine works</p>")
view.show()
QTimer.singleShot(60000, app.quit)
app.exec()
"""


def windowless(command: list[str]) -> tuple[list[str], dict[str, str]]:
    """*command* and its environment, to run where there may be no screen.

    On a Linux server with no display, matplotlib refuses Qt outright
    ("headless"), so there the command runs on a virtual X server - which
    also tries Qt's real Linux platform plugin and the libraries it needs.
    Elsewhere Qt's own offscreen platform is enough.
    """
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", *command], env
    env["QT_QPA_PLATFORM"] = "offscreen"
    return command, env


def web_check(python: Path) -> None:
    """Qt's web engine must load a page and run its JavaScript."""
    command, env = windowless([str(python), "-c", WEB_CHECK])
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    if "WEB True web engine works" not in result.stdout:
        print(result.stdout[-3000:], result.stderr[-6000:])
        raise SystemExit("Qt's web engine did not load a page; its output is above.")
    print("Qt's web engine loads pages: fine.")


def smoke_test(python: Path, folder: Path, seconds: float = 20.0) -> None:
    """Start ChartLibre without a window; it must still be running after *seconds*."""
    command, env = windowless([str(python), str(folder / "main.py")])
    log = folder.parent / "smoke.log"
    with log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(command, cwd=folder, env=env,
                                   stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.5)
        exited = process.poll()
        if exited is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
    text = log.read_text(encoding="utf-8", errors="replace")
    log.unlink()
    if exited is not None:
        print(text[-6000:])
        raise SystemExit(f"ChartLibre stopped after start-up (exit code {exited}); its output is above.")
    print(f"ChartLibre ran for {seconds:g} s without a window: fine.")


def clean_runtime_state(folder: Path) -> None:
    """Remove what the smoke test wrote: settings, logs, caches."""
    for name in ("user.json", "logs", "user"):
        path = folder / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for log in (folder / "app" / "logs").glob("*.log*"):
        log.unlink()
    for cache in folder.rglob("__pycache__"):
        if ".python" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)


def package(folder: Path, key: str, version: str) -> list[Path]:
    """Pack *folder* the way *key*'s system expects; return the files made."""
    stem = f"ChartLibre-{version}-{key}"
    DIST.mkdir(parents=True, exist_ok=True)  # not in a fresh clone
    made: list[Path] = []
    system = system_of(key)
    if system == "macos":
        staging = folder.parent
        applications = staging / "Applications"
        if not applications.exists():
            applications.symlink_to("/Applications")
        dmg = DIST / f"{stem}.dmg"
        dmg.unlink(missing_ok=True)
        subprocess.run(["hdiutil", "create", "-volname", "ChartLibre", "-srcfolder", str(staging),
                        "-ov", "-format", "ULMO", str(dmg)], check=True)
        applications.unlink()
        made.append(dmg)
    elif system == "windows":
        archive = DIST / f"{stem}-portable.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
            for path in sorted(folder.rglob("*")):
                out.write(path, Path("ChartLibre") / path.relative_to(folder))
        made.append(archive)
        iscc = shutil.which("iscc") or next(
            (str(p) for p in (Path(os.environ.get("ProgramFiles(x86)", "")) / "Inno Setup 6" / "ISCC.exe",
                              Path(os.environ.get("ProgramFiles", "")) / "Inno Setup 6" / "ISCC.exe") if p.exists()),
            None,
        )
        if iscc:
            script = Path(__file__).with_name("installer") / "chartlibre.iss"
            subprocess.run([iscc, f"/DAppVersion={version}", f"/DSourceDir={folder}", f"/DIconFile={make_launcher.WINDOWS_ICON}",
                            f"/DOutputDir={DIST}", f"/DOutputName={stem}-setup", str(script)], check=True)
            made.append(DIST / f"{stem}-setup.exe")
        else:
            print("Inno Setup not found: no installer .exe, only the portable .zip.")
    else:
        archive = DIST / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as out:
            out.add(folder, arcname="ChartLibre")
        made.append(archive)
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-package", action="store_true", help="build and test, but do not pack")
    parser.add_argument("--version", default=make_launcher.APP_VERSION, help="the version in the file names")
    parser.add_argument("--work", type=Path, default=DIST,
                        help="where to build (default dist/); on a Mac, somewhere iCloud does not sync")
    args = parser.parse_args()

    key = platform_key()
    system = system_of(key)
    work = args.work.resolve() / f"bundle-{key}"
    folder = work / "ChartLibre"
    if work.exists():
        shutil.rmtree(work)
    folder.mkdir(parents=True)

    print(f"Copied {copy_application(folder, system)} files for {system}.")
    python = download_python(key, folder)
    subprocess.run([str(python), str(folder / "install.py"), "--runtime", "--no-menu", "--dir", str(folder)],
                   check=True)
    if not (folder / ".python" / "CHARTLIBRE_READY").exists():
        raise SystemExit("install.py finished without marking the Python ready.")
    slim(folder / ".python", key)
    smoke_test(python, folder)
    web_check(python)
    clean_runtime_state(folder)
    if args.no_package:
        print(f"Built {folder}")
        return 0
    for made in package(folder, key, args.version):
        print(f"Made {made} ({made.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
