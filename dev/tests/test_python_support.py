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


def test_a_package_pinned_below_315_is_pinned_for_315_or_left_out_on_purpose() -> None:
    """A pin with `python_version < "3.15"` and no partner for 3.15 is a package 3.15 does not install."""
    left_out_on_purpose: set[str] = set()  # every pin below 3.15 has a partner
    for name, specs in _requirements().items():
        markers = [marker for _spec, marker in specs]
        if any("< \"3.15\"" in marker for marker in markers):
            has_315 = any(">= \"3.15\"" in marker for marker in markers)
            assert has_315 or name in left_out_on_purpose, f"{name} has no pin for Python 3.15"


def test_the_pins_for_315_are_exact_and_not_older_than_the_ones_they_replace() -> None:
    def version(spec: str) -> tuple[int, ...]:
        return tuple(int(part) for part in spec.removeprefix("==").split("."))

    for name, specs in _requirements().items():
        old = [spec for spec, marker in specs if "< \"3.15\"" in marker]
        new = [spec for spec, marker in specs if ">= \"3.15\"" in marker]
        if old and new:
            assert new[0].startswith("=="), name
            assert version(new[0]) >= version(old[0]), name


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
