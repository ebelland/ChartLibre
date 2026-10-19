"""Formats no free reader opens: say so, and how to get the data out (todo I-01)."""
from __future__ import annotations

from pathlib import Path

from app.plugin_api.importers.base import FileImporter, ImportedData

_EXPORT = {
    ".jmp": "In JMP: File > Export > CSV (or Excel).",
    ".jrp": "In JMP: open the report's data table, then File > Export > CSV.",
    ".mtw": "In Minitab: File > Save Worksheet As > Text (CSV) or Excel.",
    ".mpx": "In Minitab: open the project, then save each worksheet as CSV or Excel.",
    ".mpj": "In Minitab: open the project, then save each worksheet as CSV or Excel.",
    ".opju": "In Origin: File > Export > ASCII (or Excel) for each worksheet.",
    ".opj": "In Origin: File > Export > ASCII (or Excel) for each worksheet.",
}


class ClosedFormatImporter(FileImporter):
    Name = "Closed formats (JMP, Minitab, Origin)"
    Description = "Files of programs whose format has no free reader: export them as CSV or Excel."
    Suffixes = [".jmp", ".jrp", ".mtw", ".mpx", ".mpj", ".opju", ".opj"]
    Category = "Other"
    Order = 999

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        how = _EXPORT.get(Path(path).suffix.lower(), "Export it from its program as CSV or Excel.")
        raise ValueError(f"{Path(path).name} is in a closed format that no free program reads. {how} "
                         "Then import that file.")
