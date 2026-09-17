"""FunctionScanner also scans user/functions/ by default, alongside the
built-in app/functions/ (todo.txt P3-4 / the Function Creator's write
target).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.scanners.functions_scanner import FunctionScanner

_STUB = '''from __future__ import annotations
import numpy as np
from app.functions.base import base_function


class {cls}(base_function):
    name = "{name}"
    category = "User functions"
    description = "d"
    expression = "y = p[0]"
    p0 = [1.0]
    params = ["a"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.ones_like(x)
'''


def test_a_user_function_is_discovered(tmp_path: Path) -> None:
    (tmp_path / "mine.py").write_text(
        _STUB.format(cls="MineFunction", name="My Own Function"), encoding="utf-8"
    )

    scanner = FunctionScanner(extra_roots=(tmp_path,))

    assert any(spec.name == "My Own Function" for spec in scanner.specs())


def test_every_built_in_function_still_appears(tmp_path: Path) -> None:
    scanner = FunctionScanner(extra_roots=(tmp_path,))
    assert any(spec.name == "Sample user linear" for spec in scanner.specs())


def test_a_missing_extra_root_is_not_an_error(tmp_path: Path) -> None:
    scanner = FunctionScanner(extra_roots=(tmp_path / "does_not_exist",))
    assert any(spec.name == "Sample user linear" for spec in scanner.specs())


def test_defaults_to_user_functions_dir_when_extra_roots_is_not_given() -> None:
    from app.utils.config import USER_FUNCTIONS_DIR

    scanner = FunctionScanner()
    assert scanner.extra_roots == (USER_FUNCTIONS_DIR,)


def test_a_function_creator_file_is_discovered_through_the_real_user_dir() -> None:
    """End-to-end, not extra_roots=(tmp_path,): a file dropped exactly where
    the Function Creator writes (app.utils.config.USER_FUNCTIONS_DIR, the
    real project-root user/functions/) has to be found by a plain,
    default-constructed FunctionScanner() - the one SeriesFitDialog and
    SeriesFunctionDialog actually build - with no extra_roots override to
    paper over a path mismatch between the writer and the reader."""
    from app.utils.config import USER_FUNCTIONS_DIR

    USER_FUNCTIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = USER_FUNCTIONS_DIR / "_probe_end_to_end_function.py"
    target.write_text(
        _STUB.format(cls="ProbeEndToEndFunction", name="Probe End To End"),
        encoding="utf-8",
    )
    try:
        scanner = FunctionScanner()
        names = [spec.name for spec in scanner.specs()]
        assert "Probe End To End" in names
    finally:
        target.unlink(missing_ok=True)
