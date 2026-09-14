"""Shared plumbing behind the Developer menu's scaffolding tools."""
from __future__ import annotations

from pathlib import Path

from app.dialogs.dev_tools_common import (
    class_name,
    discover_both_roots,
    import_check,
    slug,
)


def test_slug_lowercases_and_joins_with_underscores() -> None:
    assert slug("Ridgeline Plot") == "ridgeline_plot"


def test_slug_strips_punctuation() -> None:
    assert slug("3D Bar-ish Chart!") == "3d_bar_ish_chart"


def test_class_name_uses_the_given_suffix() -> None:
    assert class_name("My Thing", suffix="Widget") == "MyThingWidget"


def test_class_name_prefixes_a_leading_digit() -> None:
    name = class_name("3D Thing", suffix="X")
    assert name[0].isalpha()
    assert name.endswith("X")


def test_import_check_succeeds_on_a_real_class(tmp_path: Path) -> None:
    target = tmp_path / "ok.py"
    target.write_text("class Found:\n    pass\n", encoding="utf-8")

    assert import_check(target, "Found") == ""


def test_import_check_reports_a_syntax_error(tmp_path: Path) -> None:
    target = tmp_path / "broken.py"
    target.write_text("this is not python (((", encoding="utf-8")

    error = import_check(target, "Whatever")

    assert error != ""


def test_import_check_reports_a_missing_class(tmp_path: Path) -> None:
    target = tmp_path / "wrong_name.py"
    target.write_text("class SomethingElse:\n    pass\n", encoding="utf-8")

    error = import_check(target, "NotThere")

    assert "NotThere" in error


def test_discover_both_roots_merges_builtin_and_user(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    (builtin / "a.py").write_text(
        "class A(Base):\n    Name: str = 'A'\n", encoding="utf-8"
    )
    (user / "b.py").write_text(
        "class B(Base):\n    Name: str = 'B'\n", encoding="utf-8"
    )

    entries = discover_both_roots(builtin_root=builtin, user_root=user, base_class_name="Base")

    assert {entry["value"] for entry in entries} == {"A", "B"}


def test_discover_both_roots_prefers_the_builtin_one_on_a_name_collision(
    tmp_path: Path,
) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    (builtin / "a.py").write_text(
        "class A(Base):\n    Name: str = 'Same'\n", encoding="utf-8"
    )
    (user / "b.py").write_text(
        "class B(Base):\n    Name: str = 'Same'\n", encoding="utf-8"
    )

    entries = discover_both_roots(builtin_root=builtin, user_root=user, base_class_name="Base")

    matches = [entry for entry in entries if entry["value"] == "Same"]
    assert len(matches) == 1
    assert matches[0]["name"] == "A"


def test_discover_both_roots_tolerates_a_missing_folder(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "a.py").write_text(
        "class A(Base):\n    Name: str = 'A'\n", encoding="utf-8"
    )

    entries = discover_both_roots(
        builtin_root=builtin, user_root=tmp_path / "does_not_exist", base_class_name="Base"
    )

    assert {entry["value"] for entry in entries} == {"A"}
