"""Pictures kept in the project (__images__): markers drawn as pictures, later bars and pictograms.

SVG is kept as its text (sharp at any size); PNG and JPEG as they are, at most MAX_SIDE pixels a side - larger
ones are scaled down when they are added. Each keeps where it came from and its licence, so a chart that uses it
can say so. Adding and removing one are single Undo steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.data.repo._common import ensure_connection_wrapper

MAX_SIDE = 1024
TABLE = "__images__"


@dataclass(frozen=True)
class StoredImage:
    name: str
    kind: str  # "svg" | "png" | "jpeg"
    data: bytes
    source: str = ""
    license: str = ""


def _scaled(data: bytes, kind: str) -> tuple[bytes, str]:
    """A raster picture at most MAX_SIDE a side, as PNG (JPEG kept as JPEG when it needs no scaling)."""
    if kind == "svg":
        return data, kind
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
    from PySide6.QtGui import QImage

    image = QImage.fromData(data)
    if image.isNull():
        raise ValueError("Not a picture that can be read (PNG, JPEG or SVG).")
    if max(image.width(), image.height()) <= MAX_SIDE:
        return data, kind
    image = image.scaled(MAX_SIDE, MAX_SIDE, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    out = QByteArray()
    buffer = QBuffer(out)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(out.data()), "png"


def kind_of(data: bytes) -> str:
    """"svg", "png" or "jpeg" from the bytes themselves; ValueError for anything else."""
    head = data[:512].lstrip()
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpeg"
    if b"<svg" in data[:4096] and (head.startswith(b"<") or head.startswith(b"\xef\xbb\xbf<")):
        return "svg"
    raise ValueError("Only SVG, PNG and JPEG pictures can be added.")


class ImagesMixin:
    """The project's pictures."""

    @ensure_connection_wrapper
    def _ensure_images_table(self) -> None:
        self._con.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                name    TEXT PRIMARY KEY,
                kind    TEXT NOT NULL,
                data    BLOB NOT NULL,
                source  TEXT NOT NULL DEFAULT '',
                license TEXT NOT NULL DEFAULT '',
                added   TEXT NOT NULL
            )"""
        )

    @ensure_connection_wrapper
    def add_image(self, name: str, data: bytes, *, source: str = "", license: str = "") -> str:
        """Keep *data* as the picture *name* (replacing one of that name); returns the name."""
        clean = str(name or "").strip()
        if not clean:
            raise ValueError("A picture needs a name.")
        data, kind = _scaled(bytes(data), kind_of(bytes(data)))
        self.snapshot_for_undo([TABLE], label=f"Add picture '{clean}'")
        self._ensure_images_table()
        self._con.execute(
            f"INSERT OR REPLACE INTO {TABLE} (name, kind, data, source, license, added) VALUES (?, ?, ?, ?, ?, ?)",
            (clean, kind, data, source, license, datetime.now(UTC).isoformat(timespec="seconds")),
        )
        self._commit()
        return clean

    @ensure_connection_wrapper
    def image_names(self) -> list[str]:
        if not self._con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone():
            return []
        return [row[0] for row in self._con.execute(f"SELECT name FROM {TABLE} ORDER BY name COLLATE NOCASE")]

    @ensure_connection_wrapper
    def get_image(self, name: str) -> StoredImage | None:
        if not self.image_names():
            return None
        row = self._con.execute(f"SELECT name, kind, data, source, license FROM {TABLE} WHERE name = ?",
                                (str(name),)).fetchone()
        return StoredImage(row[0], row[1], bytes(row[2]), row[3] or "", row[4] or "") if row else None

    @ensure_connection_wrapper
    def remove_image(self, name: str) -> bool:
        if self.get_image(name) is None:
            return False
        self.snapshot_for_undo([TABLE], label=f"Remove picture '{name}'")
        self._con.execute(f"DELETE FROM {TABLE} WHERE name = ?", (str(name),))
        self._commit()
        return True

    @ensure_connection_wrapper
    def set_image_details(self, name: str, *, source: str, license: str) -> None:
        """Where the picture came from and its licence (not an Undo step: notes, not data)."""
        self._con.execute(f"UPDATE {TABLE} SET source = ?, license = ? WHERE name = ?", (source, license, name))
        self._commit()

    @ensure_connection_wrapper
    def rename_image(self, old: str, new: str) -> None:
        """Rename a picture, and the series that draw it with it (one Undo step)."""
        import json

        new = str(new or "").strip()
        if not new or self.get_image(old) is None or self.get_image(new) is not None:
            raise ValueError("A picture can only take a free, non-empty name.")
        self.snapshot_for_undo([TABLE, "__series_descriptors__"], label=f"Rename picture '{old}'")
        self._con.execute(f"UPDATE {TABLE} SET name = ? WHERE name = ?", (new, old))
        for series_id, text in self._con.execute(
                "SELECT id, style_json FROM __series_descriptors__ WHERE style_json LIKE ?",
                (f"%image:{old}%",)).fetchall():
            try:
                style = json.loads(text or "{}")
            except ValueError:
                continue
            if isinstance(style, dict) and style.get("marker") == f"image:{old}":
                style["marker"] = f"image:{new}"
                self._con.execute("UPDATE __series_descriptors__ SET style_json = ? WHERE id = ?",
                                  (json.dumps(style), series_id))
        self._commit()

