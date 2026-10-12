"""Optional Numba: compile a numeric kernel when Numba is installed, run it as Python otherwise.

Numba is an optional dependency. ``njit`` here is Numba's ``njit(cache=True)``
when it imports, and a decorator that returns the function unchanged when it
does not - so a kernel is ordinary Python everywhere, just faster where Numba
is. A kernel takes and returns only numbers and NumPy arrays: no dataclasses,
dicts, closures or pandas (see todo.txt, PERFORMANCE - NUMBA).

``CHARTLIBRE_NO_NUMBA=1`` turns it off, to compare or to rule it out.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

try:
    if os.environ.get("CHARTLIBRE_NO_NUMBA"):
        raise ImportError("disabled by CHARTLIBRE_NO_NUMBA")
    from numba import njit as _numba_njit
except ImportError:  # not installed, or turned off
    _numba_njit = None

#: Whether kernels are compiled.
NUMBA_AVAILABLE = _numba_njit is not None


def njit(function: Callable[..., Any]) -> Callable[..., Any]:
    """Compile *function* with Numba (cached on disk) when Numba is there; else return it as it is."""
    if _numba_njit is None:
        return function
    return _numba_njit(cache=True)(function)
