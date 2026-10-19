"""Instrument and program text formats read without any package: JCAMP-DX, LabVIEW .lvm, GPX, GraphPad .pzfx (todo I-01)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from app.plugin_api.importers.base import ALL_DATASETS, FileImporter, ImportedData, combine

# ----------------------------------------------------------------------
# JCAMP-DX
# ----------------------------------------------------------------------
_JCAMP_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _jcamp_blocks(path: str) -> list[dict[str, object]]:
    """Each spectrum of the file: its labelled records and its data lines."""
    blocks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    key = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("$$", 1)[0].rstrip()
        if line.startswith("##"):
            label, _eq, value = line[2:].partition("=")
            key = re.sub(r"[\s_\-/]", "", label).upper()
            if key == "TITLE" and (current is None or "DATA" in current):
                current = {"DATA": [], "DATAKIND": ""}
                blocks.append(current)
            if current is None:
                continue
            if key in ("XYDATA", "XYPOINTS", "PEAKTABLE"):
                current["DATAKIND"] = key + "=" + value.strip().replace(" ", "")
            elif key == "END":
                key = ""
            else:
                current[key] = value.strip()
        elif current is not None and key in ("XYDATA", "XYPOINTS", "PEAKTABLE") and line.strip():
            current["DATA"].append(line)  # type: ignore[union-attr]
    return [block for block in blocks if block["DATA"]]


def _jcamp_frame(block: dict[str, object]) -> pd.DataFrame:
    kind = str(block["DATAKIND"])
    lines: list[str] = block["DATA"]  # type: ignore[assignment]
    x_factor = float(str(block.get("XFACTOR", "1")) or 1)
    y_factor = float(str(block.get("YFACTOR", "1")) or 1)
    x_name = str(block.get("XUNITS") or "X")
    y_name = str(block.get("YUNITS") or "Y")
    if kind.startswith("XYDATA") and "(X++(Y..Y))" in kind:
        if any(re.search(r"[@A-Za-z%]", line) for line in lines):
            raise ValueError("This JCAMP-DX file is compressed (SQZ/DIF): only plain numbers (AFFN) are read for now.")
        first, last = float(str(block.get("FIRSTX", "nan"))), float(str(block.get("LASTX", "nan")))
        count = int(float(str(block.get("NPOINTS", "0")) or 0))
        xs, ys = [], []
        step = (last - first) / (count - 1) if count > 1 else 0.0
        for line in lines:
            numbers = [float(n) for n in _JCAMP_NUMBER.findall(line)]
            if not numbers:
                continue
            x0 = numbers[0] * x_factor
            for k, y in enumerate(numbers[1:]):
                xs.append(x0 + k * step)
                ys.append(y * y_factor)
        frame = pd.DataFrame({x_name: xs, y_name: ys})
        return frame.drop_duplicates(subset=[x_name], keep="first").reset_index(drop=True)
    numbers = [float(n) for line in lines for n in _JCAMP_NUMBER.findall(line)]
    pairs = np.asarray(numbers[: len(numbers) // 2 * 2]).reshape(-1, 2)
    return pd.DataFrame({x_name: pairs[:, 0] * x_factor, y_name: pairs[:, 1] * y_factor})


class JcampImporter(FileImporter):
    Name = "JCAMP-DX spectrum"
    Description = "A spectrum (IR, UV-Vis, Raman, NMR, MS) in JCAMP-DX: x and y with their units; each block a dataset."
    Suffixes = [".jdx", ".dx", ".jcamp"]
    Category = "Spectroscopy and XRD"
    Order = 70

    def datasets(self, path: str) -> list[str]:
        blocks = _jcamp_blocks(path)
        return [str(block.get("TITLE") or f"Spectrum {k + 1}") for k, block in enumerate(blocks)] \
            if len(blocks) > 1 else []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        blocks = _jcamp_blocks(path)
        if not blocks:
            raise ValueError("No ##XYDATA, ##XYPOINTS or ##PEAKTABLE block: is this a JCAMP-DX file?")
        named = {str(block.get("TITLE") or f"Spectrum {k + 1}"): block for k, block in enumerate(blocks)}
        if dataset == ALL_DATASETS:
            return ImportedData(frame=combine({name: _jcamp_frame(block) for name, block in named.items()}))
        block = named.get(dataset or "", blocks[0])
        metadata = {key.title(): value for key, value in block.items()
                    if key not in ("DATA", "DATAKIND") and isinstance(value, str) and value}
        return ImportedData(frame=_jcamp_frame(block), metadata=metadata)


# ----------------------------------------------------------------------
# LabVIEW .lvm
# ----------------------------------------------------------------------
_LVM_SEGMENT_KEYS = ("Channels", "Samples", "Date", "Time", "Y_Unit_Label", "X_Dimension", "X0", "Delta_X")


class LabviewImporter(FileImporter):
    Name = "LabVIEW measurement (.lvm)"
    Description = "A LabVIEW Write To Measurement File text file: each segment a dataset."
    Suffixes = [".lvm"]
    Category = "Lab instruments"
    Order = 61

    def _segments(self, path: str) -> tuple[dict[str, str], list[pd.DataFrame]]:
        """The file header, then each segment: its own header up to ***End_of_Header***, the names, the rows."""
        header: dict[str, str] = {}
        separator, decimal = "\t", "."
        segments: list[pd.DataFrame] = []
        mode = "file"  # file header -> segment header -> data -> (a new segment header) ...
        table: list[list[str]] = []
        columns: list[str] | None = None

        def close() -> None:
            nonlocal table, columns
            if table:
                width = max(map(len, table))
                names = (columns or [])[:width] + [f"Column {k + 1}" for k in range(len(columns or []), width)]
                frame = pd.DataFrame([row + [""] * (width - len(row)) for row in table], columns=names)
                segments.append(frame.apply(lambda s: pd.to_numeric(s.str.replace(decimal, ".", regex=False),
                                                                    errors="coerce")))
            table, columns = [], None

        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [f.strip() for f in line.split(separator)]
            if line.startswith("***End_of_Header***"):
                mode = "segment" if mode == "file" else "data"
                continue
            if not line.strip():
                continue
            if mode == "data" and fields[0] in _LVM_SEGMENT_KEYS:
                close()
                mode = "segment"
            if mode in ("file", "segment"):
                if fields[0] == "Separator" and len(fields) > 1:
                    separator = {"Tab": "\t", "Comma": ","}.get(fields[1], separator)
                elif fields[0] == "Decimal_Separator" and len(fields) > 1:
                    decimal = fields[1] or "."
                elif len(fields) > 1 and fields[0] and any(fields[1:]):
                    header.setdefault(fields[0], " ".join(f for f in fields[1:] if f))
                continue
            if columns is None and not table and any(f and not re.fullmatch(r"[-+0-9.,eE]+", f) for f in fields):
                columns = [f or f"Column {k + 1}" for k, f in enumerate(fields)]
                if columns[-1] == "Comment":
                    columns.pop()
                continue
            table.append(fields[: len(columns)] if columns else fields)
        close()
        return header, segments

    def datasets(self, path: str) -> list[str]:
        _header, segments = self._segments(path)
        return [f"Segment {k + 1}" for k in range(len(segments))] if len(segments) > 1 else []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        header, segments = self._segments(path)
        if not segments:
            raise ValueError("No data under the headers: is this a LabVIEW .lvm file?")
        named = {f"Segment {k + 1}": frame.dropna(axis=1, how="all") for k, frame in enumerate(segments)}
        if dataset == ALL_DATASETS:
            return ImportedData(frame=combine(named), metadata=header)
        return ImportedData(frame=named.get(dataset or "", next(iter(named.values()))), metadata=header)


# ----------------------------------------------------------------------
# GPX
# ----------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class GpxImporter(FileImporter):
    Name = "GPS track (GPX)"
    Description = "A GPX file's track, route and waypoints: latitude, longitude, elevation, time."
    Suffixes = [".gpx"]
    Category = "Other"
    Order = 90

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        root = ET.parse(path).getroot()
        rows = []
        for element in root.iter():
            kind = _local(element.tag)
            if kind not in ("trkpt", "rtept", "wpt"):
                continue
            row = {"Kind": {"trkpt": "track", "rtept": "route", "wpt": "waypoint"}[kind],
                   "Latitude": float(element.get("lat", "nan")), "Longitude": float(element.get("lon", "nan"))}
            for child in element:
                name = _local(child.tag)
                if name == "ele":
                    row["Elevation (m)"] = float(child.text or "nan")
                elif name == "time":
                    row["Time"] = child.text
                elif name == "name":
                    row["Name"] = child.text
            rows.append(row)
        if not rows:
            raise ValueError("No track, route or waypoint: is this a GPX file?")
        return ImportedData(frame=pd.DataFrame(rows), units={"Elevation (m)": "m"})


# ----------------------------------------------------------------------
# GraphPad Prism .pzfx
# ----------------------------------------------------------------------
def _pzfx_tables(path: str) -> dict[str, pd.DataFrame]:
    root = ET.parse(path).getroot()
    tables: dict[str, pd.DataFrame] = {}
    for table in root.iter():
        if _local(table.tag) not in ("Table", "HugeTable"):
            continue
        title = next((child.text for child in table if _local(child.tag) == "Title"), None) or f"Table {len(tables) + 1}"
        columns: dict[str, list[str]] = {}
        for column in table:
            kind = _local(column.tag)
            if kind not in ("XColumn", "YColumn", "RowTitlesColumn"):
                continue
            name = next((child.text for child in column if _local(child.tag) == "Title"), None) \
                or {"XColumn": "X", "RowTitlesColumn": "Row"}.get(kind, f"Y{len(columns)}")
            subcolumns = [sub for sub in column if _local(sub.tag) == "Subcolumn"]
            for k, sub in enumerate(subcolumns):
                label = name if len(subcolumns) == 1 else f"{name} {k + 1}"
                columns[label] = [(d.text or "") for d in sub if _local(d.tag) == "d"]
        if columns:
            length = max(len(values) for values in columns.values())
            frame = pd.DataFrame({name: values + [""] * (length - len(values)) for name, values in columns.items()})
            tables[title] = frame.apply(lambda s: pd.to_numeric(s, errors="coerce")
                                        if pd.to_numeric(s.replace("", np.nan), errors="coerce").notna().sum()
                                        == s.replace("", np.nan).notna().sum() else s.replace("", np.nan))
    return tables


class PrismImporter(FileImporter):
    Name = "GraphPad Prism (.pzfx)"
    Description = "A Prism XML project: each data table a dataset, replicates as numbered columns."
    Suffixes = [".pzfx"]
    Category = "Statistics software"
    Order = 15

    def datasets(self, path: str) -> list[str]:
        tables = _pzfx_tables(path)
        return list(tables) if len(tables) > 1 else []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        tables = _pzfx_tables(path)
        if not tables:
            raise ValueError("No data table: is this a Prism .pzfx file?")
        if dataset == ALL_DATASETS:
            return ImportedData(frame=combine(tables))
        return ImportedData(frame=tables.get(dataset or "", next(iter(tables.values()))))
