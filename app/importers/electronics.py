"""Microelectronics files read without any package: Touchstone, SPICE .raw, VCD (todo I-01).

* Touchstone .sNp (v1): a VNA's S-, Y- or Z-parameters. The option line
  ("# GHz S MA R 50") says the frequency unit, the parameter, the format (MA
  magnitude-angle, DB dB-angle, RI real-imaginary) and the reference
  impedance; each frequency then has 2 n^2 numbers - in the order S11 S21 S12
  S22 for two ports (the standard's own exception), row by row otherwise.
* SPICE .raw (LTspice, ngspice): ASCII ("Values:") or binary ("Binary:");
  LTspice writes UTF-16 and, for a transient, a double time and single
  values; complex values (AC) become magnitude and phase. A .step run, whose
  time or frequency starts again, gets a Step column.
* VCD (Value Change Dump, a logic analyser's or a digital simulation's): one
  row per time a signal changes, every signal's value at that time; a vector
  as its integer, x or z as empty.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.plugin_api.importers.base import ALL_DATASETS, FileImporter, ImportedData, combine

# ----------------------------------------------------------------------
# Touchstone
# ----------------------------------------------------------------------
_FREQUENCY_UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def _ports(path: str) -> int:
    match = re.search(r"\.s(\d+)p$", path.lower())
    if not match:
        raise ValueError("A Touchstone file's suffix says its ports: .s1p, .s2p, .s4p...")
    return int(match.group(1))


class TouchstoneImporter(FileImporter):
    Name = "Touchstone S-parameters"
    Description = "A VNA's .sNp file: frequency and each S (or Y, Z) parameter as its file gives it."
    Suffixes = [".s1p", ".s2p", ".s3p", ".s4p", ".s5p", ".s6p", ".s8p"]
    Category = "RF"
    Order = 30

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        ports = _ports(path)
        unit, parameter, form, impedance = "GHZ", "S", "MA", "50"
        numbers: list[float] = []
        comments: list[str] = []
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            text, _bang, comment = line.partition("!")
            if comment.strip():
                comments.append(comment.strip())
            text = text.strip()
            if not text:
                continue
            if text.startswith("#"):
                words = text[1:].upper().split()
                for index, word in enumerate(words):
                    if word in _FREQUENCY_UNITS:
                        unit = word
                    elif word in ("S", "Y", "Z", "G", "H"):
                        parameter = word
                    elif word in ("MA", "DB", "RI"):
                        form = word
                    elif word == "R" and index + 1 < len(words):
                        impedance = words[index + 1]
                continue
            if text.startswith("["):  # a v2 keyword line: not read yet
                raise ValueError("This is a Touchstone v2 file ([Version] 2.0): only v1 is read for now.")
            numbers.extend(float(value) for value in text.split())
        per_frequency = 1 + 2 * ports * ports
        if not numbers or len(numbers) % per_frequency:
            raise ValueError(f"The values do not divide into frequencies of {per_frequency} numbers "
                             f"({ports} port(s)): is the suffix right?")
        rows = np.asarray(numbers).reshape(-1, per_frequency)
        if ports == 2:
            order = [(1, 1), (2, 1), (1, 2), (2, 2)]
        else:
            order = [(i, j) for i in range(1, ports + 1) for j in range(1, ports + 1)]
        parts = {"MA": ("mag", "deg"), "DB": ("dB", "deg"), "RI": ("re", "im")}[form]
        frame = pd.DataFrame({"Frequency (Hz)": rows[:, 0] * _FREQUENCY_UNITS[unit]})
        units = {"Frequency (Hz)": "Hz"}
        for k, (i, j) in enumerate(order):
            name = f"{parameter}{i}{j}"
            frame[f"{name} {parts[0]}"] = rows[:, 1 + 2 * k]
            frame[f"{name} {parts[1]}"] = rows[:, 2 + 2 * k]
            if form != "RI":
                units[f"{name} {parts[1]}"] = "deg"
        metadata = {"Ports": ports, "Parameter": parameter, "Format": form, "Reference impedance (ohm)": impedance,
                    "Frequency unit in the file": unit}
        if comments:
            metadata["Comments"] = " / ".join(comments[:20])
        return ImportedData(frame=frame, metadata=metadata, units=units)


# ----------------------------------------------------------------------
# SPICE .raw
# ----------------------------------------------------------------------
def _raw_text(data: bytes) -> tuple[str, str]:
    """The header's encoding - LTspice writes UTF-16 LE - and the header up to the data keyword."""
    encoding = "utf-16-le" if len(data) > 1 and data[1] == 0 else "latin-1"
    text = data.decode(encoding, errors="replace")
    return encoding, text


def _parse_raw(path: str) -> list[tuple[dict[str, str], pd.DataFrame]]:
    """Every plot of a raw file: (its header, its values)."""
    data = Path(path).read_bytes()
    encoding, text = _raw_text(data)
    plots = []
    offset_chars = 0
    while True:
        match = re.search(r"^(Binary|Values):[ \t]*\r?\n", text[offset_chars:], re.M)
        if not match:
            break
        header_text = text[offset_chars:offset_chars + match.start()]
        header: dict[str, str] = {}
        variables: list[tuple[str, str]] = []
        in_variables = False
        for line in header_text.splitlines():
            if in_variables and line[:1] in ("\t", " ") and line.strip():
                fields = line.split()
                if len(fields) >= 3 and fields[0].isdigit():
                    variables.append((fields[1], fields[2]))
                continue
            key, _colon, value = line.partition(":")
            if key.strip():
                header[key.strip()] = value.strip()
                in_variables = key.strip() == "Variables"
        count = int(header.get("No. Variables", len(variables)))
        points = int(header.get("No. Points", "0").split()[0] or 0)
        flags = header.get("Flags", "").lower()
        complex_values = "complex" in flags
        names = [name for name, _kind in variables][:count]
        start = offset_chars + match.end()
        if match.group(1) == "Values":
            numbers: list[complex] = []
            body = text[start:]
            end_match = re.search(r"^Title:", body, re.M)
            body = body[:end_match.start()] if end_match else body
            for token in body.split():
                if token.isdigit() and len(numbers) % count == 0:
                    continue  # the point's index
                if complex_values:
                    real, _comma, imaginary = token.partition(",")
                    numbers.append(complex(float(real), float(imaginary or 0)))
                else:
                    numbers.append(float(token))
            values = np.asarray(numbers).reshape(-1, count)
            consumed = len(text) if not end_match else start + end_match.start()
        else:
            byte_start = len(text[:start].encode(encoding))
            if complex_values:
                width = count * 16
                block = np.frombuffer(data, dtype="<f8", count=points * count * 2, offset=byte_start)
                values = (block[0::2] + 1j * block[1::2]).reshape(points, count)
            elif "double" in flags or "ngspice" in header.get("Command", "").lower() or "fastaccess" in flags:
                width = count * 8
                values = np.frombuffer(data, dtype="<f8", count=points * count, offset=byte_start).reshape(points, count)
            else:  # LTspice transient: the time a double, the rest singles
                width = 8 + 4 * (count - 1)
                dtype = np.dtype([("t", "<f8")] + [(f"v{k}", "<f4") for k in range(count - 1)])
                record = np.frombuffer(data, dtype=dtype, count=points, offset=byte_start)
                values = np.column_stack([record["t"]] + [record[f"v{k}"].astype(float) for k in range(count - 1)])
            consumed_bytes = byte_start + points * width
            consumed = len(data[:consumed_bytes].decode(encoding, errors="replace"))
        plots.append((header, _raw_frame(names, values, complex_values)))
        if consumed >= len(text) or consumed <= offset_chars:
            break
        offset_chars = consumed
    if not plots:
        raise ValueError("No 'Values:' or 'Binary:' section: is this a SPICE raw file?")
    return plots


def _raw_frame(names: list[str], values: np.ndarray, complex_values: bool) -> pd.DataFrame:
    frame = pd.DataFrame()
    for k, name in enumerate(names):
        column = values[:, k]
        if complex_values:
            if k == 0:
                frame[name] = np.abs(column)  # the frequency, written as a complex
            else:
                frame[f"{name} mag"] = np.abs(column)
                frame[f"{name} phase (deg)"] = np.degrees(np.angle(column))
        else:
            frame[name] = np.abs(column) if k == 0 else np.real(column)  # LTspice marks some times negative
    first = frame.iloc[:, 0].to_numpy(float)
    restarts = np.flatnonzero(np.diff(first) < 0) + 1
    if restarts.size:  # a .step run: the sweep starts again for each step
        step = np.zeros(len(first), dtype=int)
        for index in restarts:
            step[index:] += 1
        frame.insert(0, "Step", step + 1)
    return frame


class SpiceRawImporter(FileImporter):
    Name = "SPICE raw (LTspice, ngspice)"
    Description = "A circuit simulation's waveforms: one column per trace; each plot a dataset, .step runs numbered."
    Suffixes = [".raw"]
    Category = "Circuit simulation"
    Order = 40

    def datasets(self, path: str) -> list[str]:
        plots = _parse_raw(path)
        return [header.get("Plotname", f"Plot {k + 1}") for k, (header, _frame) in enumerate(plots)] \
            if len(plots) > 1 else []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        plots = _parse_raw(path)
        named = {header.get("Plotname", f"Plot {k + 1}"): (header, frame) for k, (header, frame) in enumerate(plots)}
        if dataset == ALL_DATASETS:
            return ImportedData(frame=combine({name: frame for name, (_header, frame) in named.items()}))
        header, frame = named.get(dataset or "", plots[0])
        metadata = {key: header[key] for key in ("Title", "Date", "Plotname", "Flags", "Command") if header.get(key)}
        notes = [f"{frame['Step'].max()} steps of a .step run, numbered in the Step column."] if "Step" in frame else []
        return ImportedData(frame=frame, metadata=metadata, notes=notes)


# ----------------------------------------------------------------------
# VCD
# ----------------------------------------------------------------------
def _vcd_value(text: str) -> float:
    digits = text.lower()
    if any(c in digits for c in "xzu-"):
        return np.nan
    try:
        return float(int(digits, 2))
    except ValueError:
        return np.nan


class VcdImporter(FileImporter):
    Name = "VCD (value change dump)"
    Description = "A logic analyser's or a digital simulation's signals: a row per change, every signal's value."
    Suffixes = [".vcd"]
    Category = "Oscilloscopes and logic"
    Order = 50

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        codes: dict[str, str] = {}
        scope: list[str] = []
        metadata: dict[str, str] = {}
        for match in re.finditer(r"\$(\w+)(.*?)\$end", text, re.S):
            keyword, body = match.group(1), match.group(2).split()
            if keyword == "scope" and len(body) >= 2:
                scope.append(body[1])
            elif keyword == "upscope" and scope:
                scope.pop()
            elif keyword == "var" and len(body) >= 4:
                code, name = body[2], body[3]
                codes.setdefault(code, ".".join([*scope[1:], name]) if len(scope) > 1 else name)
            elif keyword in ("timescale", "date", "version") and body:
                metadata[keyword.capitalize()] = " ".join(body)
        start = text.find("$enddefinitions")
        if start < 0 or not codes:
            raise ValueError("No signal definitions ($var ... $enddefinitions): is this a VCD file?")
        body = text[text.index("$end", start) + 4:]
        current = {code: np.nan for code in codes}
        rows: list[list[float]] = []
        time = None
        tokens = body.split()
        k = 0
        while k < len(tokens):
            token = tokens[k]
            if token.startswith("#"):
                if time is not None:
                    rows.append([time, *current.values()])
                time = float(token[1:])
            elif token[0] in "bBrR" and k + 1 < len(tokens):
                value = token[1:]
                current[tokens[k + 1]] = float(value) if token[0] in "rR" else _vcd_value(value)
                k += 1
            elif token[0] in "01xXzZ" and token[1:] in codes:
                current[token[1:]] = _vcd_value(token[0])
            k += 1
        if time is not None:
            rows.append([time, *current.values()])
        frame = pd.DataFrame(rows, columns=["Time", *codes.values()])
        unit = metadata.get("Timescale", "")
        return ImportedData(frame=frame, metadata=metadata, units={"Time": unit} if unit else {})
