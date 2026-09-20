"""Fit-function scanner for SeriesFitDialog.

The fit dialog is intentionally decoupled from concrete model definitions.
Built-in functions live in ``app/functions/functions.py``, a hand-written
example lives in ``app/functions/user_functions.py``, and a function the
Function Creator (Developer menu) scaffolds lives in ``user/functions/``
instead (kept out of the app's own shipped source, see
``app.utils.config.USER_FUNCTIONS_DIR``) - this scanner reads all of
``app/functions/`` plus that folder together.  It uses the generic
``class_discovery`` helpers to find classes that directly inherit from
``base_function`` and then loads their ``execute(x, p)`` method on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np

from app.logs.logger import applogger
from app.scanners.class_discovery import (
    discover_classes_merged,
    import_class_from_discovery_entry,
)
from app.utils.config import USER_FUNCTIONS_DIR


@dataclass(frozen=True, slots=True)
class FitFunctionSpec:
    """Metadata and loader information for one discovered fit function."""

    name: str
    category: str
    description: str
    expression: str
    p0: tuple[float, ...]
    params: tuple[str, ...]
    discovery_entry: dict[str, Any]
    #: Number of independent variables: 1 for y = f(x) (``base_function``),
    #: 2 for z = f(x, y) (``base_surface_function``). Carried on the spec
    #: rather than re-derived from the class each time, because the fit
    #: dialog needs it after the class has already been discovered and
    #: turned into a plain dict payload for the tree widget.
    ndim: int = 1

    @property
    def class_name(self) -> str:
        return str(self.discovery_entry.get("name", ""))

    @property
    def path(self) -> str:
        return str(self.discovery_entry.get("path", ""))

    def as_catalog_payload(self) -> dict[str, Any]:
        """Return the payload stored on a QTreeWidgetItem."""
        return {
            "name": self.name,
            "kind": "function",  # internal dialog dispatch only, not a base_function field
            "category": self.category,
            "description": self.description,
            "expression": self.expression,
            "p0": list(self.p0),
            "params": list(self.params),
            "discovery_entry": dict(self.discovery_entry),
            "function_class": self.class_name,
            "path": self.path,
            "ndim": self.ndim,
        }


class FunctionScanner:
    """Discover and instantiate fit functions using class_discovery helpers.

    Function classes must directly inherit from ``base_function`` and expose:

    - ``name``: label shown in the model tree
    - ``category``: top-level tree category
    - ``description``: short human-readable purpose
    - ``expression``: HTML or plain formula displayed below Parameters/Expression
    - ``params``: list of parameter names
    - ``p0``: list of numeric initial values
    - ``execute(x, p)``: static/class method used by least-squares fitting
    """

    #: Overridden by ``SurfaceFunctionScanner`` rather than duplicated: every
    #: other method here - discovery, caching, the catalog tree, filtering -
    #: is identical between a 1-variable and a 2-variable function library,
    #: so only what actually differs (which base class, which default
    #: category, how many independent variables the model takes) is a class
    #: attribute.
    BASE_CLASS_NAME: Final[str] = "base_function"
    DEFAULT_CATEGORY: Final[str] = "Functions"
    NDIM: Final[int] = 1

    def __init__(
        self, *, root: Path | None = None, extra_roots: tuple[Path, ...] | None = None
    ) -> None:
        self.root = root or Path(__file__).resolve().parent.joinpath("..", "functions")
        #: Scanned alongside ``root`` - defaults to user/functions/, where the
        #: Function Creator writes. Overridable (an empty tuple included) for
        #: tests that want ``root`` scanned on its own.
        self.extra_roots = (USER_FUNCTIONS_DIR,) if extra_roots is None else extra_roots
        self._specs: list[FitFunctionSpec] | None = None

    def refresh(self) -> list[FitFunctionSpec]:
        """Force a rescan and return discovered specs."""
        self._specs = self._discover()
        return list(self._specs)

    def specs(self) -> list[FitFunctionSpec]:
        """Return cached specs, scanning if needed."""
        if self._specs is None:
            self._specs = self._discover()
        return list(self._specs)

    def catalog(self) -> dict[str, list[dict[str, Any]]]:
        """Return discovered functions grouped for the SeriesFitDialog tree."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for spec in self.specs():
            grouped.setdefault(spec.category, []).append(spec.as_catalog_payload())

        for models in grouped.values():
            models.sort(key=lambda item: str(item.get("name", "")).lower())

        return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))

    def load_class(self, payload: dict[str, Any]) -> Any:
        """Return the function class a tree payload describes.

        The callable from :meth:`make_model` is enough to evaluate a function
        and not enough to ask it anything: the starting-point estimator is a
        second staticmethod on the class (``initial_guess``), so the caller
        that wants one needs the class rather than the closure.
        """
        entry = payload.get("discovery_entry")
        if not isinstance(entry, dict):
            raise ValueError("Function payload has no discovery_entry.")

        cls = import_class_from_discovery_entry(entry, module_prefix="_fit_function")
        if cls is None:
            raise ValueError(f"Could not load function class from entry: {entry!r}")
        return cls

    def make_model(self, payload: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Build a fit callable from a tree payload.

        For a 1-variable function (``ndim`` 1, the default) the returned
        model reads ``x_or_xy`` as x alone, discarding a second column when
        one happens to be present - see ``_primary_x``.

        For a 2-variable surface function (``ndim`` 2), the fit dialog's own
        convention applies: ``x_or_xy`` is an ``(N, 2)`` array, column 0 is x
        and column 1 is y (the same shape ``fit_dialog._split_xy`` already
        expects for its own two-input models), and ``execute(x, y, p)`` is
        called with the two columns split apart.
        """
        cls = self.load_class(payload)

        execute = getattr(cls, "execute", None)
        if not callable(execute):
            raise TypeError(f"Discovered function {cls!r} has no callable execute(x, p).")

        if int(payload.get("ndim", 1)) == 2:

            def surface_model(x_or_xy: np.ndarray, p: np.ndarray) -> np.ndarray:
                x, y = self._split_xy(np.asarray(x_or_xy, dtype=float))
                params = np.asarray(p, dtype=float).reshape(-1)
                return np.asarray(execute(x, y, params), dtype=float)

            return surface_model

        def model(x_or_xy: np.ndarray, p: np.ndarray) -> np.ndarray:
            x = self._primary_x(np.asarray(x_or_xy, dtype=float))
            params = np.asarray(p, dtype=float).reshape(-1)
            return np.asarray(execute(x, params), dtype=float)

        return model

    def _discover(self) -> list[FitFunctionSpec]:
        entries = discover_classes_merged(
            roots=(self.root, *self.extra_roots),
            base_class_name=self.BASE_CLASS_NAME,
            value_attr="name",
            string_attrs=("category", "description", "expression"),
            string_list_attrs=("params",),
            require_value_attr=False,
        )

        specs: list[FitFunctionSpec] = []
        seen: set[tuple[str, str]] = set()

        for entry in entries:
            cls_name = str(entry.get("name", ""))
            path = str(entry.get("path", ""))
            key = (path, cls_name)
            if key in seen:
                continue
            seen.add(key)

            cls = import_class_from_discovery_entry(entry, module_prefix="_fit_function_meta")
            if cls is None:
                continue

            execute = getattr(cls, "execute", None)
            if not callable(execute):
                applogger.warning("Fit function %s in %s has no execute(x, p); skipped.", cls_name, path)
                continue

            display_name = str(entry.get("value") or getattr(cls, "name", cls_name) or cls_name)
            category = str(entry.get("category") or getattr(cls, "category", self.DEFAULT_CATEGORY) or self.DEFAULT_CATEGORY)
            description = str(entry.get("description") or getattr(cls, "description", "") or "")
            expression = str(entry.get("expression") or getattr(cls, "expression", "") or "")
            params = self._string_tuple(entry.get("params") or getattr(cls, "params", []))
            p0 = self._float_tuple(getattr(cls, "p0", []), fallback_length=len(params))

            if params and len(params) != len(p0):
                applogger.warning(
                    "Fit function %s has %d params but %d initial values.",
                    display_name,
                    len(params),
                    len(p0),
                )

            specs.append(
                FitFunctionSpec(
                    name=display_name,
                    category=category,
                    description=description,
                    expression=expression,
                    p0=p0,
                    params=params,
                    discovery_entry=dict(entry),
                    ndim=self.NDIM,
                )
            )

        specs.sort(key=lambda item: (item.category.lower(), item.name.lower()))
        return specs

    @staticmethod
    def _primary_x(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2:
            return arr[:, 0]
        return arr

    @staticmethod
    def _split_xy(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split an ``(N, 2)`` array into its x and y columns.

        Raises rather than guessing when the caller handed a 2-variable
        model a plain 1D array: that means whoever built ``x_data`` did not
        actually assemble the second independent variable, which is a bug
        worth surfacing immediately rather than fitting garbage.
        """
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                "This surface model requires an (N, 2) array of (x, y) "
                "independent columns."
            )
        return arr[:, 0], arr[:, 1]

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return ()

    @staticmethod
    def _float_tuple(value: Any, *, fallback_length: int = 0) -> tuple[float, ...]:
        try:
            values = tuple(float(item) for item in value)
        except Exception:
            values = ()
        if values:
            return values
        if fallback_length > 0:
            return tuple(1.0 for _ in range(fallback_length))
        return (1.0, 1.0)


class SurfaceFunctionScanner(FunctionScanner):
    """Discover z = f(x, y) surface fit functions.

    Everything but *which classes count* is shared with ``FunctionScanner`` -
    discovery, caching, the catalog tree, filtering, model building - so this
    overrides only the three class attributes that describe the difference,
    rather than re-implementing ``_discover``/``make_model`` from scratch.
    Built-ins live in ``app/functions/surface_functions.py``, a hand-written
    example in ``app/functions/user_surface_functions.py``, and both are
    scanned from the same ``app/functions/`` root ``FunctionScanner`` uses -
    the base-class filter is what keeps the two libraries apart.
    """

    BASE_CLASS_NAME: Final[str] = "base_surface_function"
    DEFAULT_CATEGORY: Final[str] = "Surfaces"
    NDIM: Final[int] = 2
