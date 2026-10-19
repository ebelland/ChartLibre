"""The base of a file importer: one format, its suffixes, the packages it needs, how it reads (todo I-01).

A plugin family like the charts and the operations (app.scanners.importer_scanner):
a class deriving from FileImporter in app/importers - or user/importers, for a
lab's own instrument - declares what it reads in literal class attributes and
reads a file into an ImportedData: the rows, and what the file says about them
(the instrument's metadata, the columns' units).

A file may hold several datasets - an Excel workbook's sheets, the sweeps of a
parameter analyser, a TDMS file's groups, a MATLAB file's variables. datasets()
names them; read() takes one, or ALL_DATASETS for every one in one table with a
Dataset column (combine()), what New chart's Group by and Separate axes by read.

A format whose reader needs a package that is not installed is still listed;
reading says which package to install (require()).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: The dataset that means "every one, in one table with a Dataset column".
ALL_DATASETS = "*"
#: The column combine() adds.
DATASET_COLUMN = "Dataset"
#: A package's import name, where it differs from the name pip installs it by.
_MODULES = {"odfpy": "odf", "scikit-rf": "skrf", "spc-spectra": "spc_spectra"}


@dataclass(slots=True)
class ImportedData:
    """What a file holds: the rows, and what it says about them."""

    frame: pd.DataFrame
    #: The instrument's or the program's own information: setup, date, device...
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Column -> unit, when the file says it.
    units: dict[str, str] = field(default_factory=dict)
    #: What the reader did, said in a line each ("3 sweeps in one table").
    notes: list[str] = field(default_factory=list)


class FileImporter:
    """One file format. Subclass it directly, with literal class attributes (the scanner reads them)."""

    #: The format's name, as the Import dialog says it: "Touchstone S-parameters".
    Name: str = ""
    Description: str = ""
    #: The suffixes it reads, lower case with the dot: [".s2p"].
    Suffixes: list[str] = []
    #: Its group in the file chooser and the manual: "RF", "Statistics software"...
    Category: str = "Other"
    #: The packages it needs, by the name pip installs them by; empty for none.
    Packages: list[str] = []
    Order: int = 100

    def datasets(self, path: str) -> list[str]:
        """The datasets the file holds, when more than one; [] for a file of one table."""
        del path
        return []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        """The rows of *dataset* (None: the only one, or the first; ALL_DATASETS: all, combined)."""
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------

    @classmethod
    def missing_packages(cls) -> list[str]:
        """The packages this reader needs that are not installed."""
        return [name for name in cls.Packages if importlib.util.find_spec(_MODULES.get(name, name)) is None]

    @classmethod
    def require(cls) -> None:
        """Raise, saying what to install, when a package is missing."""
        missing = cls.missing_packages()
        if missing:
            raise ValueError(
                f"Reading {cls.Name} files needs {' and '.join(missing)}: pip install {' '.join(missing)}"
            )


def combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every dataset's rows one under the other, a Dataset column first saying whose they are."""
    parts = [frame.assign(**{DATASET_COLUMN: name})[[DATASET_COLUMN, *frame.columns]]
             for name, frame in frames.items()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
