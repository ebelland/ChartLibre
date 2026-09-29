"""The sampling frequency of a series, read off its x role.

Shared by every operation that works in the frequency domain (Spectral
analysis, Filtering): one estimator, one wording for what is wrong with the
spacing. Plain arrays in, no Qt, nothing logged - the caller shows the note.
"""
from __future__ import annotations

import numpy as np


def sampling_frequency(x_values: np.ndarray) -> tuple[float, str]:
    """Return (fs, note) - samples per unit of x read off the median spacing.

    *note* is empty when there is nothing to say, else why fs is doubtful:
    a spectrum of unevenly sampled data is not defined, so a non-uniform x
    is reported rather than silently averaged - the numbers would look fine
    and mean nothing.
    """
    if x_values.size < 2:
        return 1.0, ""

    spacing = np.diff(x_values)
    median_spacing = float(np.median(spacing))
    if median_spacing <= 0.0:
        return 1.0, "the x role is not increasing; assuming fs = 1"

    deviation = float(np.max(np.abs(spacing - median_spacing)) / median_spacing)
    note = ""
    if deviation > 0.01:
        note = (
            f"the series is not uniformly sampled (spacing varies by "
            f"{deviation * 100.0:.1f} %); the frequency axis is approximate"
        )
    return 1.0 / median_spacing, note
