"""MATLAB and audio files, read with SciPy (todo I-01)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from app.plugin_api.importers.base import ALL_DATASETS, FileImporter, ImportedData


def _mat_arrays(path: str) -> dict[str, np.ndarray]:
    """The numeric arrays of a MATLAB file, by variable name; a v7.3 file (HDF5) is said to need h5py."""
    from scipy.io import loadmat

    try:
        contents = loadmat(path, squeeze_me=True)
    except NotImplementedError:
        raise ValueError("This is a MATLAB v7.3 file (HDF5): save it with save(..., '-v7') in MATLAB, "
                         "or install h5py.") from None
    return {name: np.asarray(value) for name, value in contents.items()
            if not name.startswith("__") and isinstance(value, np.ndarray)
            and value.dtype.kind in "biufc" and 1 <= value.ndim <= 2}


def _columns(name: str, array: np.ndarray) -> dict[str, np.ndarray]:
    """A variable as columns: a vector as itself, a matrix one column per column."""
    if array.ndim == 1:
        return {name: array}
    if array.shape[0] < array.shape[1] and array.shape[0] <= 16:
        array = array.T  # a few rows of many values: the values are the rows
    return {f"{name}_{k + 1}": array[:, k] for k in range(array.shape[1])}


class MatlabImporter(FileImporter):
    Name = "MATLAB"
    Description = "A MATLAB .mat file (v4 to v7): each numeric variable a dataset, or all of one length side by side."
    Suffixes = [".mat"]
    Category = "Statistics software"
    Order = 14

    def datasets(self, path: str) -> list[str]:
        arrays = _mat_arrays(path)
        return list(arrays) if len(arrays) > 1 else []

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        arrays = _mat_arrays(path)
        if not arrays:
            raise ValueError("The file holds no numeric vector or matrix.")
        if dataset and dataset != ALL_DATASETS:
            return ImportedData(frame=pd.DataFrame(_columns(dataset, arrays[dataset])))
        if dataset is None and len(arrays) == 1:
            name, array = next(iter(arrays.items()))
            return ImportedData(frame=pd.DataFrame(_columns(name, array)))
        # All: the variables of the commonest length, side by side - they are columns of one table.
        columns = {}
        for name, array in arrays.items():
            columns.update(_columns(name, array))
        length = Counter(len(values) for values in columns.values()).most_common(1)[0][0]
        kept = {name: values for name, values in columns.items() if len(values) == length}
        notes = []
        if len(kept) < len(columns):
            notes.append(f"Variables of another length than {length} left out: "
                         + ", ".join(sorted(set(columns) - set(kept))) + " - choose one as the dataset.")
        return ImportedData(frame=pd.DataFrame(kept), notes=notes)


class WavImporter(FileImporter):
    Name = "WAV audio"
    Description = "A WAV recording: time and one column per channel."
    Suffixes = [".wav"]
    Category = "Lab instruments"
    Order = 60

    def read(self, path: str, dataset: str | None = None) -> ImportedData:
        from scipy.io import wavfile

        rate, samples = wavfile.read(path)
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples[:, None]
        frame = pd.DataFrame({"Time (s)": np.arange(samples.shape[0]) / float(rate)})
        for k in range(samples.shape[1]):
            frame[f"Channel {k + 1}"] = samples[:, k]
        return ImportedData(frame=frame, metadata={"Sample rate (Hz)": int(rate), "Channels": samples.shape[1],
                                                   "Sample type": str(samples.dtype)},
                            units={"Time (s)": "s"})
