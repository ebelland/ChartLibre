"""Embed each demo project's preview picture in the project (issue #8).

The shipped demo set keeps ``<name>.preview.png`` beside ``<name>.dhub``. This puts
the picture inside the .dhub (``__project_preview__``), drops the old ``preview_path``
entry, and - with --remove-files - deletes the now unneeded PNGs.

    python3 dev/tools/embed_demo_previews.py            # the demo/ folder
    python3 dev/tools/embed_demo_previews.py PATH...    # other folders or .dhub files
    python3 dev/tools/embed_demo_previews.py --dry-run

Plain sqlite3, no Qt: nothing here opens the project through the application.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = "__project_preview__"


def embed(project: Path, *, remove_file: bool = False, dry_run: bool = False) -> str:
    """Embed *project*'s preview; what was done, as one word."""
    png_file = project.with_name(f"{project.stem}.preview.png")
    with closing(sqlite3.connect(str(project))) as con:
        have = con.execute(f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{TABLE}'").fetchone()
        if have and con.execute(f"SELECT 1 FROM {TABLE} WHERE id = 1").fetchone():
            return "already embedded"
        if not png_file.is_file():
            return "no picture"
        if dry_run:
            return "would embed"
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (id INTEGER PRIMARY KEY CHECK (id = 1), png BLOB NOT NULL)")
        con.execute(f"INSERT OR REPLACE INTO {TABLE} (id, png) VALUES (1, ?)", (sqlite3.Binary(png_file.read_bytes()),))
        try:
            con.execute("DELETE FROM __project_info__ WHERE key = 'preview_path'")
        except sqlite3.Error:
            pass  # a project without information entries
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if remove_file:
        png_file.unlink(missing_ok=True)
    return "embedded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="projects or folders (default: demo/)")
    parser.add_argument("--remove-files", action="store_true", help="delete the PNG beside each project")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    projects: list[Path] = []
    for path in args.paths or [ROOT / "demo"]:
        projects += sorted(path.glob("*.dhub")) if path.is_dir() else [path]
    for project in projects:
        print(f"{embed(project, remove_file=args.remove_files, dry_run=args.dry_run):18s} {project.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
