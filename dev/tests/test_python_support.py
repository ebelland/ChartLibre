"""What running on a newer Python needs from the repository (3.15).

No interpreter-specific code here: it checks the things a new Python breaks
without a traceback in the tests - a requirements file pip cannot satisfy, and
text files read or written in the locale's encoding, which 3.15 replaces with
UTF-8 (PEP 686).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _requirements() -> dict[str, list[tuple[str, str]]]:
    """name -> [(spec, marker)] for every requirement line."""
    found: dict[str, list[tuple[str, str]]] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement, _, marker = line.partition(";")
        match = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", requirement.strip())
        assert match, line
        found.setdefault(match.group(1).lower(), []).append((match.group(2).strip(), marker.strip()))
    return found


#: Optional importers with no Python 3.15 wheels yet (2026-10): left out on 3.15,
#: where their format says which package it needs. Drop the marker when they ship.
WAITING_FOR_315_WHEELS = {"pyreadstat", "pyreadr"}


def test_no_requirement_depends_on_the_python_version() -> None:
    """Python 3.15 is the default: one pin per package, with no `python_version` marker."""
    for name, specs in _requirements().items():
        for _spec, marker in specs:
            if name in WAITING_FOR_315_WHEELS:
                assert marker.replace(" ", "") == 'python_version<"3.15"', f"{name}: {marker}"
                continue
            assert "python_version" not in marker, f"{name} still has a python_version marker"


def test_every_pin_is_exact() -> None:
    for name, specs in _requirements().items():
        for spec, marker in specs:
            if "sys_platform" in marker:
                continue  # a platform-only package (pyobjc) may take a floor
            assert spec.startswith("=="), f"{name} is not pinned exactly: {spec}"


def _text_opens_without_encoding() -> list[str]:
    hits: list[str] = []
    for path in sorted((ROOT / "app").rglob("*.py")) + [ROOT / "main.py", ROOT / "install.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            function = call.func
            if isinstance(function, ast.Name) and function.id == "open":
                mode_position = 1  # open(file, mode)
            elif isinstance(function, ast.Attribute) and function.attr == "open":
                # Path.open(mode) - but ZipFile.open(member) and the like take a name first.
                if call.args and not (isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                    continue
                mode_position = 0
            elif isinstance(function, ast.Attribute) and function.attr in ("read_text", "write_text"):
                mode_position = None
            else:
                continue
            if any(keyword.arg == "encoding" for keyword in call.keywords):
                continue
            if mode_position is not None:
                mode = call.args[mode_position] if len(call.args) > mode_position else next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "mode"), None
                )
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value:
                    continue
            hits.append(f"{path.relative_to(ROOT)}:{call.lineno}")
    return hits


def test_every_text_file_the_app_opens_names_its_encoding() -> None:
    """Without encoding= a file is read in the locale's encoding - cp1252 on Windows - until 3.15, UTF-8 after."""
    assert _text_opens_without_encoding() == []
