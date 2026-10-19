"""Other statistics programs' files: Stata, SAS, SPSS, R (todo I-01).

Read through pandas (Stata and SAS with nothing more; SPSS with pyreadstat) and
pyreadr for R. Stata's and SPSS' value labels come in as the labels themselves,
categories rather than codes.
"""
from __future__ import annotations

import pandas as pd

from app.plugin_api.importers.base import ALL_DATASETS, FileImporter, ImportedData, combine


class StataImporter(FileImporter):
    Name = "Stata"
    Description = "A Stata dataset (.dta), value labels as categories."
    Suffixes = [".dta"]
    Category = "Statistics software"
    Order = 10

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        with pd.io.stata.StataReader(path) as reader:
            frame = reader.read(convert_categoricals=True)
            label = reader.data_label
            variables = reader.variable_labels()
        metadata = {"Dataset label": label} if label else {}
        metadata.update({f"Label of {name}": text for name, text in variables.items() if text})
        return ImportedData(frame=frame, metadata=metadata)


class SasImporter(FileImporter):
    Name = "SAS"
    Description = "A SAS dataset (.sas7bdat) or transport file (.xpt)."
    Suffixes = [".sas7bdat", ".xpt"]
    Category = "Statistics software"
    Order = 11

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        frame = pd.read_sas(path, encoding="utf-8" if path.lower().endswith(".sas7bdat") else None)
        return ImportedData(frame=frame)


class SpssImporter(FileImporter):
    Name = "SPSS"
    Description = "An SPSS data file (.sav, .zsav), value labels as categories."
    Suffixes = [".sav", ".zsav"]
    Category = "Statistics software"
    Packages = ["pyreadstat"]
    Order = 12

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        self.require()
        import pyreadstat

        frame, meta = pyreadstat.read_sav(path, apply_value_formats=True)
        labels = dict(zip(meta.column_names, meta.column_labels or []))
        metadata = {f"Label of {name}": text for name, text in labels.items() if text}
        if getattr(meta, "file_label", None):
            metadata["File label"] = meta.file_label
        return ImportedData(frame=frame, metadata=metadata)


class RImporter(FileImporter):
    Name = "R"
    Description = "An R data file (.rds, .RData): each object a dataset."
    Suffixes = [".rds", ".rdata", ".rda"]
    Category = "Statistics software"
    Packages = ["pyreadr"]
    Order = 13

    def _objects(self, path: str) -> dict[str, pd.DataFrame]:
        self.require()
        import pyreadr

        return {str(name or "data"): frame for name, frame in pyreadr.read_r(path).items()}

    def datasets(self, path: str) -> list[str]:
        objects = self._objects(path)
        return list(objects) if len(objects) > 1 else []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        objects = self._objects(path)
        if dataset == ALL_DATASETS:
            return ImportedData(frame=combine(objects))
        return ImportedData(frame=objects.get(dataset or "", next(iter(objects.values()))))
