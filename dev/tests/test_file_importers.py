"""Other programs' and instruments' files (todo I-01): each built-in importer on a small file written here.

The files are synthetic but laid out as the programs write them; the
packages that are not always installed (pyreadstat, pyarrow...) are only
checked for their message.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.plugin_api.importers.base import ALL_DATASETS, FileImporter
from app.scanners import importer_scanner
from app.utils import data_sources


def _read(path: Path, dataset: str | None = None):
    return data_sources.read_imported(path, dataset)


def test_every_builtin_importer_is_found_with_its_suffixes() -> None:
    names = {entry["value"] for entry in importer_scanner.importers}
    assert {"Stata", "MATLAB", "Touchstone S-parameters", "SPICE raw (LTspice, ngspice)",
            "VCD (value change dump)", "JCAMP-DX spectrum", "GraphPad Prism (.pzfx)"} <= names
    for suffix in (".dta", ".mat", ".s2p", ".raw", ".vcd", ".jdx", ".lvm", ".gpx", ".pzfx", ".sav", ".parquet"):
        assert data_sources.is_importable(f"x{suffix}"), suffix
        assert f"*{suffix}" in data_sources.IMPORT_FILE_FILTER


def test_stata_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "a.dta"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_stata(path, write_index=False)
    frame = data_sources.read_any_file(str(path))
    assert list(frame.columns) == ["a", "b"] and frame["b"].tolist() == ["x", "y"]


def test_matlab_variables_are_datasets(tmp_path: Path) -> None:
    import scipy.io
    path = tmp_path / "a.mat"
    scipy.io.savemat(path, {"x": np.arange(5.0), "y": np.arange(5.0) ** 2, "z": np.arange(3.0)})
    assert data_sources.file_datasets(path) == [ALL_DATASETS, "x", "y", "z"]
    together = _read(path, ALL_DATASETS)
    assert list(together.frame.columns) == ["x", "y"] and together.notes  # z, of another length, said
    assert _read(path, "z").frame.shape == (3, 1)


def test_wav_channels_and_time(tmp_path: Path) -> None:
    import scipy.io.wavfile
    path = tmp_path / "a.wav"
    scipy.io.wavfile.write(path, 100, np.zeros((10, 2), dtype=np.int16))
    result = _read(path)
    assert list(result.frame.columns) == ["Time (s)", "Channel 1", "Channel 2"]
    assert result.frame["Time (s)"].iloc[1] == pytest.approx(0.01)


def test_touchstone_two_ports_in_the_standard_order(tmp_path: Path) -> None:
    path = tmp_path / "a.s2p"
    path.write_text("! dut\n# GHz S MA R 50\n1.0 0.9 -10 0.1 80 0.2 70 0.8 -20\n")
    result = _read(path)
    row = result.frame.iloc[0]
    assert row["Frequency (Hz)"] == 1e9
    assert (row["S21 mag"], row["S12 mag"], row["S22 deg"]) == (0.1, 0.2, -20)
    assert result.metadata["Reference impedance (ohm)"] == "50"


def test_spice_raw_ascii_and_ltspice_binary_steps(tmp_path: Path) -> None:
    ascii_raw = tmp_path / "a.raw"
    ascii_raw.write_text("Title: t\nPlotname: Transient Analysis\nFlags: real\nNo. Variables: 2\nNo. Points: 2\n"
                         "Variables:\n\t0\ttime\ttime\n\t1\tV(out)\tvoltage\nValues:\n0\t0.0\n\t1.0\n1\t1e-3\n\t2.0\n")
    assert _read(ascii_raw).frame["V(out)"].tolist() == [1.0, 2.0]
    binary = tmp_path / "b.raw"
    header = ("Title: t\nPlotname: Transient Analysis\nFlags: real forward\nNo. Variables: 2\nNo. Points: 4\n"
              "Variables:\n\t0\ttime\ttime\n\t1\tV(a)\tvoltage\nBinary:\n")
    body = b"".join(struct.pack("<df", t, v) for t, v in [(0, 1), (1e-3, 2), (0, 3), (1e-3, 4)])
    binary.write_bytes(header.encode("utf-16-le") + body)
    frame = _read(binary).frame
    assert frame["Step"].tolist() == [1, 1, 2, 2] and frame["V(a)"].tolist() == [1, 2, 3, 4]


def test_vcd_a_row_per_change(tmp_path: Path) -> None:
    path = tmp_path / "a.vcd"
    path.write_text("$timescale 1ns $end\n$scope module top $end\n$var wire 1 ! clk $end\n"
                    "$var wire 4 \" bus $end\n$upscope $end\n$enddefinitions $end\n"
                    "#0\n0!\nb0011 \"\n#5\n1!\n#10\nbxxxx \"\n")
    frame = _read(path).frame
    assert frame["clk"].tolist() == [0, 1, 1]
    assert frame["bus"].tolist()[:2] == [3, 3] and np.isnan(frame["bus"].iloc[2])


def test_jcamp_xydata(tmp_path: Path) -> None:
    path = tmp_path / "a.jdx"
    path.write_text("##TITLE=spec\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##YFACTOR=0.001\n##FIRSTX=400\n"
                    "##LASTX=405\n##NPOINTS=6\n##XYDATA=(X++(Y..Y))\n400 100 200 300\n403 400 500 600\n##END=\n")
    frame = _read(path).frame
    assert frame["1/CM"].tolist() == [400, 401, 402, 403, 404, 405]
    assert frame["ABSORBANCE"].iloc[-1] == pytest.approx(0.6)


def test_labview_segments(tmp_path: Path) -> None:
    path = tmp_path / "a.lvm"
    segment = "Channels\t2\n***End_of_Header***\t\nX_Value\tVoltage\tCurrent\tComment\n0\t1.0\t0.1\n1\t2.0\t0.2\n"
    path.write_text("LabVIEW Measurement\t\nSeparator\tTab\n***End_of_Header***\t\n\n" + segment + segment)
    assert data_sources.file_datasets(path) == [ALL_DATASETS, "Segment 1", "Segment 2"]
    frame = _read(path, ALL_DATASETS).frame
    assert list(frame.columns) == ["Dataset", "X_Value", "Voltage", "Current"] and len(frame) == 4


def test_gpx_and_prism(tmp_path: Path) -> None:
    gpx = tmp_path / "a.gpx"
    gpx.write_text('<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg><trkpt lat="45.1" lon="9.2">'
                   "<ele>100</ele></trkpt></trkseg></trk></gpx>")
    assert _read(gpx).frame[["Latitude", "Elevation (m)"]].iloc[0].tolist() == [45.1, 100.0]
    prism = tmp_path / "a.pzfx"
    prism.write_text('<GraphPadPrismFile xmlns="http://graphpad.com/prism/Prism.htm"><Table><Title>Dose</Title>'
                     "<XColumn><Title>Conc</Title><Subcolumn><d>1</d><d>2</d></Subcolumn></XColumn>"
                     "<YColumn><Title>Resp</Title><Subcolumn><d>5</d><d>6</d></Subcolumn>"
                     "<Subcolumn><d>7</d><d/></Subcolumn></YColumn></Table></GraphPadPrismFile>")
    frame = _read(prism).frame
    assert list(frame.columns) == ["Conc", "Resp 1", "Resp 2"] and np.isnan(frame["Resp 2"].iloc[1])


def test_a_closed_format_says_what_to_do(tmp_path: Path) -> None:
    path = tmp_path / "a.jmp"
    path.write_bytes(b"\0\1")
    with pytest.raises(ValueError, match="CSV|Excel"):
        _read(path)


def test_a_missing_package_says_what_to_install(monkeypatch) -> None:
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pyreadstat" else real(name, *a))

    class Needs(FileImporter):
        Name = "SPSS"
        Packages = ["pyreadstat"]

    with pytest.raises(ValueError, match="pip install pyreadstat"):
        Needs().require()


def test_the_import_dialog_reads_with_an_importer(qapp, tmp_db_path: Path, tmp_path: Path) -> None:
    import scipy.io

    from app.data.sqlite_repo import SqliteRepo
    from app.dialogs.import_data_dialog import ImportDataDialog

    path = tmp_path / "scope.mat"
    scipy.io.savemat(path, {"x": np.arange(4.0), "y": np.arange(4.0) * 2})
    repo = SqliteRepo(db_path=tmp_db_path)
    dialog = ImportDataDialog(repo)
    try:
        dialog.load_file(path)
        assert not dialog._read_card.isEnabled()
        assert "MATLAB" in dialog._reader_label.text()
        assert dialog._sheet_label.text() == "Dataset"
        assert [dialog._sheet.itemData(k) for k in range(dialog._sheet.count())] == [ALL_DATASETS, "x", "y"]
        assert list(dialog._df.columns) == ["x", "y"]
        dialog._sheet.setCurrentIndex(2)
        assert list(dialog._df.columns) == ["y"]
        dialog._table.setText("scope")
        dialog._on_accept()
        assert repo.get_table_info("scope")["source_metadata"]["dataset"] == "y"
        assert repo.get_table_link("scope")["settings"]["source"]["importer"] == "MATLAB"
    finally:
        repo.close()
