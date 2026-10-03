"""install.py: creates .venv and installs into it, without touching anything else.

The full install (PySide6 and the rest, a gigabyte) is not run here; the
parts around it are: which Python is good enough, where the interpreter of a
venv is, the iCloud warning, that a failed install leaves no half-made
.venv, and that --recreate really starts again.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("chartlibre_install", ROOT / "install.py")
assert _spec is not None and _spec.loader is not None
install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(install)


def test_python_below_3_11_is_refused_and_newer_is_fine() -> None:
    assert "3.11" in install.python_problem((3, 10, 9))
    assert "3.10.9" in install.python_problem((3, 10, 9))
    assert install.python_problem((3, 11, 0)) == ""
    assert install.python_problem((3, 14, 1)) == ""


def test_the_interpreter_sits_where_the_platform_puts_it(tmp_path: Path) -> None:
    assert install.venv_python(tmp_path / ".venv", "darwin") == tmp_path / ".venv" / "bin" / "python3"
    assert install.venv_python(tmp_path / ".venv", "linux") == tmp_path / ".venv" / "bin" / "python3"
    assert install.venv_python(tmp_path / ".venv", "win32") == tmp_path / ".venv" / "Scripts" / "python.exe"


def test_only_icloud_places_on_a_mac_get_the_warning(tmp_path: Path) -> None:
    home = tmp_path / "me"
    assert "~/Desktop" in install.cloud_sync_warning(home / "Desktop" / "ChartLibre", home, "darwin")
    assert "~/Documents" in install.cloud_sync_warning(home / "Documents" / "x" / "ChartLibre", home, "darwin")
    assert install.cloud_sync_warning(home / "Developer" / "ChartLibre", home, "darwin") == ""
    assert install.cloud_sync_warning(home / "Desktop" / "ChartLibre", home, "linux") == ""
    assert install.cloud_sync_warning(home / "Desktop" / "ChartLibre", home, "win32") == ""


def test_the_start_hint_names_what_to_open(tmp_path: Path) -> None:
    assert "ChartLibre.app" in install.start_hint(tmp_path, "darwin")
    assert "ChartLibre.exe" in install.start_hint(tmp_path, "win32")
    assert "ChartLibre.sh" in install.start_hint(tmp_path, "linux")


def test_a_venv_is_created_kept_and_recreated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert install.main(["--dir", str(tmp_path), "--skip-packages"]) == 0
    python = install.venv_python(tmp_path / ".venv")
    assert python.exists()

    marker = tmp_path / ".venv" / "marker.txt"
    marker.write_text("from the first run")
    assert install.main(["--dir", str(tmp_path), "--skip-packages"]) == 0
    assert marker.exists()  # asked again without --recreate: kept
    assert "already exists" in capsys.readouterr().out

    assert install.main(["--dir", str(tmp_path), "--skip-packages", "--recreate"]) == 0
    assert not marker.exists() and python.exists()


def test_a_missing_requirements_file_is_reported_before_anything_is_made(tmp_path: Path) -> None:
    assert install.main(["--dir", str(tmp_path)]) == 1
    assert not (tmp_path / ".venv").exists()


def test_a_failed_install_leaves_no_half_made_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "requirements.txt").write_text("numpy\n")

    def fail(python: Path, requirements: Path) -> None:
        raise subprocess.CalledProcessError(1, "pip")

    monkeypatch.setattr(install, "install_requirements", fail)
    assert install.main(["--dir", str(tmp_path)]) == 1
    assert not (tmp_path / ".venv").exists()
    assert "did not complete" in capsys.readouterr().out


def test_a_venv_that_already_existed_is_not_deleted_by_a_failed_install(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    assert install.main(["--dir", str(tmp_path), "--skip-packages"]) == 0
    (tmp_path / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(install, "install_requirements",
                        lambda python, requirements: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "pip")))
    assert install.main(["--dir", str(tmp_path)]) == 1
    assert (tmp_path / ".venv").exists()


def test_an_old_python_stops_the_installer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install, "python_problem", lambda version: "ChartLibre needs Python 3.11 or newer; this is Python 3.9.0.")
    assert install.main(["--dir", str(tmp_path), "--skip-packages"]) == 1
    assert not (tmp_path / ".venv").exists()


def test_the_check_names_every_library_an_empty_venv_lacks(tmp_path: Path) -> None:
    assert install.main(["--dir", str(tmp_path), "--skip-packages"]) == 0
    problems = install.verify(install.venv_python(tmp_path / ".venv"))
    joined = "\n".join(problems)
    assert len(problems) == len(install.CHECKS)
    assert "PySide6.QtWidgets" in joined and "ModuleNotFoundError" in joined


def test_every_library_the_check_imports_is_in_requirements() -> None:
    text = (ROOT / "requirements.txt").read_text().lower()
    for package in ("pyside6", "matplotlib", "numpy", "pandas", "openpyxl", "scipy",
                    "scikit-learn", "statsmodels", "scikit-image", "pywavelets"):
        assert package in text


def test_linux_gets_a_menu_entry_that_starts_this_folder(tmp_path: Path) -> None:
    folder = tmp_path / "ChartLibre"
    folder.mkdir()
    launcher = folder / "ChartLibre.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o644)  # as a copy through a shared folder can leave it
    entry = install.write_linux_menu_entry(folder, home=tmp_path / "home")
    assert entry == tmp_path / "home" / ".local" / "share" / "applications" / "chartlibre.desktop"
    text = entry.read_text(encoding="utf-8")
    assert text.startswith("[Desktop Entry]\n")
    assert f'Exec="{launcher}"' in text and f"Path={folder}" in text
    assert f"Icon={folder}/dev/tools/launcher/chartlibre.png" in text
    assert launcher.stat().st_mode & 0o111


def test_the_linux_launcher_and_its_icon_ship_with_the_project() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "ChartLibre.sh").read_text(encoding="utf-8")
    assert script.startswith("#!/bin/sh") and "python3 install.py" in script and "main.py" in script
    assert (root / "ChartLibre.sh").stat().st_mode & 0o111
    assert (root / "dev" / "tools" / "launcher" / "chartlibre.png").stat().st_size > 1000


def test_an_incomplete_venv_is_made_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A venv left without pip by a failed run is replaced, not kept."""
    python = install.venv_python(tmp_path / ".venv")
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    made: list[str] = []
    monkeypatch.setattr(install, "venv_has_pip", lambda _python: False)
    monkeypatch.setattr(install.subprocess, "run", lambda args, **_k: made.append(" ".join(map(str, args))))
    _python, created = install.create_venv(tmp_path, recreate=False)
    assert created and any("-m venv" in command for command in made)


def test_a_python_without_ensurepip_is_told_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def without_ensurepip(name, *args, **kwargs):
        if name == "ensurepip":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ensurepip)
    assert "sudo apt install python3-venv" in install.venv_problem()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert install.venv_problem() == ""


def test_linux_names_the_packages_for_the_libraries_qt_cannot_find() -> None:
    assert install.missing_linux_packages(lambda name: "lib.so") == []
    missing = install.missing_linux_packages(lambda name: None if name == "xcb-cursor" else "lib.so")
    assert missing == ["libxcb-cursor0"]


# ----------------------------------------------------------------------
# The launchers: they download their own Python, so nobody installs one
# ----------------------------------------------------------------------
_launcher_spec = importlib.util.spec_from_file_location("make_launcher", ROOT / "dev" / "tools" / "make_launcher.py")
assert _launcher_spec is not None and _launcher_spec.loader is not None
make_launcher = importlib.util.module_from_spec(_launcher_spec)
_launcher_spec.loader.exec_module(make_launcher)

LAUNCHERS = {
    "macos": ROOT / "ChartLibre.app" / "Contents" / "MacOS" / "ChartLibre",
    "windows": ROOT / "ChartLibre.bat",
    "linux": ROOT / "ChartLibre.sh",
}


def test_every_pinned_python_has_a_sha256_and_an_https_url() -> None:
    import re

    for key in make_launcher.PYTHON_ARCHIVES:
        url, digest = make_launcher._archive(key)
        assert url.startswith("https://github.com/astral-sh/python-build-standalone/releases/download/")
        assert url.endswith("-install_only.tar.gz") and make_launcher.PYTHON_VERSION in url
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_the_shipped_launchers_are_the_generated_ones_with_every_archive_filled_in() -> None:
    import re

    generated = {"macos": make_launcher.MACOS_SCRIPT, "windows": make_launcher.WINDOWS_SCRIPT,
                 "linux": make_launcher.LINUX_SCRIPT}
    wanted = {"macos": ("macos-arm64", "macos-x86_64"), "windows": ("windows-x86_64",),
              "linux": ("linux-x86_64", "linux-aarch64")}
    for platform, path in LAUNCHERS.items():
        shipped = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        assert shipped == generated[platform], f"{path.name} is stale: run dev/tools/make_launcher.py"
        assert not re.search(r"@[A-Z0-9_]+@", shipped), "a placeholder was left in"
        for key in wanted[platform]:
            url, digest = make_launcher._archive(key)
            assert url in shipped and digest in shipped
        assert "install.py" in shipped and "main.py" in shipped


def test_the_unix_launchers_are_valid_shell() -> None:
    assert subprocess.run(["bash", "-n", str(LAUNCHERS["macos"])], check=False).returncode == 0
    assert subprocess.run(["sh", "-n", str(LAUNCHERS["linux"])], check=False).returncode == 0


def test_the_linux_launcher_downloads_checks_and_unpacks_python(tmp_path: Path) -> None:
    """The real get_python, fed a local archive: a wrong digest stops it, the right one unpacks."""
    import hashlib
    import shutil
    import tarfile

    if not shutil.which("curl") or not (shutil.which("sha256sum") or shutil.which("shasum")):
        pytest.skip("needs curl and sha256sum")
    staging = tmp_path / "staging" / "python" / "bin"
    staging.mkdir(parents=True)
    (staging / "python3").write_text("#!/bin/sh\necho fake python \"$@\"\n", encoding="utf-8")
    (staging / "python3").chmod(0o755)
    archive = tmp_path / "python.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "staging" / "python", arcname="python")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    folder = tmp_path / "ChartLibre"
    folder.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    found = shutil.which("sha256sum")  # macOS has it as shasum -a 256
    check = f'exec "{found}" "$@"' if found else 'exec shasum -a 256 "$@"'
    (bindir / "sha256sum").write_text(f"#!/bin/sh\n{check}\n", encoding="utf-8")
    (bindir / "sha256sum").chmod(0o755)
    (bindir / "uname").write_text("#!/bin/sh\necho x86_64\n", encoding="utf-8")
    (bindir / "uname").chmod(0o755)
    env = {"PATH": f"{bindir}:/usr/bin:/bin"}

    def launch(sha: str) -> subprocess.CompletedProcess[str]:
        url, real = make_launcher._archive("linux-x86_64")
        script = make_launcher.LINUX_SCRIPT.replace(url, archive.as_uri()).replace(real, sha)
        (folder / "ChartLibre.sh").write_text(script, encoding="utf-8")
        return subprocess.run(["sh", str(folder / "ChartLibre.sh")], capture_output=True, text=True, env=env)

    bad = launch("0" * 64)
    assert bad.returncode != 0 and "damaged" in bad.stdout
    assert not (folder / ".python").exists() and not (folder / ".python-download.tar.gz").exists()

    good = launch(digest)
    assert good.returncode == 0, good.stdout + good.stderr
    assert (folder / ".python" / "bin" / "python3").exists()
    assert "fake python install.py --runtime" in good.stdout and "fake python main.py" in good.stdout
    assert not (folder / ".python-download.tar.gz").exists()

    # Once the libraries are in, the launcher goes straight to ChartLibre.
    (folder / ".python" / "CHARTLIBRE_READY").write_text("ready", encoding="utf-8")
    (tmp_path / "home" / ".local" / "share" / "applications").mkdir(parents=True)
    (tmp_path / "home" / ".local" / "share" / "applications" / "chartlibre.desktop").write_text("", encoding="utf-8")
    env["HOME"] = str(tmp_path / "home")
    again = subprocess.run(["sh", str(folder / "ChartLibre.sh")], capture_output=True, text=True, env=env)
    assert again.stdout.strip() == "fake python main.py" and "Downloading" not in again.stdout


def test_the_runtime_install_marks_the_python_ready_only_when_every_library_loads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "requirements.txt").write_text("numpy\n")
    prefix = tmp_path / ".python"
    prefix.mkdir()
    monkeypatch.setattr(install, "install_requirements", lambda python, requirements: None)
    monkeypatch.setattr(install, "verify", lambda python: ["PySide6.QtWidgets: missing"])
    assert install.install_runtime(tmp_path, tmp_path / "requirements.txt", menu=False, prefix=prefix) == 1
    assert not install.runtime_ready(prefix).exists()
    monkeypatch.setattr(install, "verify", lambda python: [])
    assert install.install_runtime(tmp_path, tmp_path / "requirements.txt", menu=False, prefix=prefix) == 0
    assert install.runtime_ready(prefix).exists()
    assert not (tmp_path / ".venv").exists()  # into the Python itself, no .venv


_bundle_spec = importlib.util.spec_from_file_location("build_bundle", ROOT / "dev" / "tools" / "build_bundle.py")
assert _bundle_spec is not None and _bundle_spec.loader is not None
build_bundle = importlib.util.module_from_spec(_bundle_spec)
_bundle_spec.loader.exec_module(build_bundle)


def test_a_package_holds_the_application_and_only_its_own_system_launchers() -> None:
    wanted = build_bundle.wanted_file
    for system in ("macos", "windows", "linux"):
        assert wanted("main.py", system) and wanted("app/charts/bar.py", system) and wanted("install.py", system)
        assert not wanted("dev/tests/test_installer.py", system) and not wanted("todo.txt", system)
        assert not wanted(".github/workflows/bundles.yml", system) and not wanted(".gitignore", system)
        assert wanted("dev/tools/launcher/chartlibre.png", system)
    assert wanted("ChartLibre.app/Contents/MacOS/ChartLibre", "macos")
    assert not wanted("ChartLibre.app/Contents/MacOS/ChartLibre", "windows")
    assert wanted("ChartLibre.exe", "windows") and wanted("ChartLibre.bat", "windows")
    assert not wanted("ChartLibre.exe", "linux") and wanted("ChartLibre.sh", "linux")


def test_identical_files_become_links_to_the_first(tmp_path: Path) -> None:
    import os

    if os.name == "nt":
        pytest.skip("packages for Windows keep their copies")
    data = os.urandom(300_000)
    (tmp_path / "Versions" / "A").mkdir(parents=True)
    (tmp_path / "Versions" / "A" / "lib").write_bytes(data)
    (tmp_path / "lib").write_bytes(data)
    (tmp_path / "other").write_bytes(os.urandom(300_000))
    saved = build_bundle.link_duplicates(tmp_path)
    assert saved == 300_000
    links = [path for path in tmp_path.rglob("*") if path.is_symlink()]
    assert len(links) == 1 and links[0].read_bytes() == data
