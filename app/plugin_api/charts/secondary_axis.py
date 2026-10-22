"""A secondary axis that is the primary one in other units (todo G-17).

``ax.secondary_xaxis`` / ``secondary_yaxis`` draw a second scale along the
top or the right of an axes, as a function of the first: degrees Celsius
under, degrees Fahrenheit above; a wavelength in nm under, the photon energy in
eV above. It carries no data of its own - unlike a twin axis (``twin_of``),
which is a second series on its own y - so nothing can disagree with the first
scale. Each coordinate's option (``secondary_x``, ``secondary_y``) names one
of the conversions below; empty means none.

No Qt here: the properties widget lists CONVERSIONS and the renderer applies them.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from app.logs.logger import applogger

Function = Callable[[Any], Any]


def _linear(scale: float, offset: float = 0.0) -> tuple[Function, Function]:
    return (lambda v: np.asarray(v, dtype=float) * scale + offset), (lambda v: (np.asarray(v, dtype=float) - offset) / scale)


def _reciprocal(constant: float) -> tuple[Function, Function]:
    def convert(values: Any) -> Any:
        with np.errstate(divide="ignore", invalid="ignore"):
            return constant / np.asarray(values, dtype=float)

    return convert, convert  # its own inverse


_C_F = _linear(1.8, 32.0)
_F_C = _linear(1 / 1.8, -32.0 / 1.8)
_C_K = _linear(1.0, 273.15)
_K_C = _linear(1.0, -273.15)

#: key -> (label in the list, function primary -> secondary, its inverse, label of the secondary axis)
CONVERSIONS: dict[str, tuple[str, Function, Function, str]] = {
    "degC_degF": ("°C → °F", *_C_F, "°F"),
    "degF_degC": ("°F → °C", *_F_C, "°C"),
    "degC_K": ("°C → K", *_C_K, "K"),
    "K_degC": ("K → °C", *_K_C, "°C"),
    "m_ft": ("m → ft", *_linear(3.280839895), "ft"),
    "ft_m": ("ft → m", *_linear(0.3048), "m"),
    "km_mi": ("km → mi", *_linear(0.621371192), "mi"),
    "mi_km": ("mi → km", *_linear(1.609344), "km"),
    "kg_lb": ("kg → lb", *_linear(2.204622622), "lb"),
    "lb_kg": ("lb → kg", *_linear(0.45359237), "kg"),
    "rad_deg": ("rad → °", *_linear(180 / np.pi), "degrees"),
    "deg_rad": ("° → rad", *_linear(np.pi / 180), "rad"),
    "s_min": ("s → min", *_linear(1 / 60), "min"),
    "s_h": ("s → h", *_linear(1 / 3600), "h"),
    "day_year": ("days → years", *_linear(1 / 365.25), "years"),
    "nm_eV": ("nm → eV (photon)", *_reciprocal(1239.84198), "energy (eV)"),
    "eV_nm": ("eV → nm (photon)", *_reciprocal(1239.84198), "wavelength (nm)"),
    "Hz_s": ("Hz → period (s)", *_reciprocal(1.0), "period (s)"),
    "s_Hz": ("s → frequency (Hz)", *_reciprocal(1.0), "frequency (Hz)"),
    "nm_cm1": ("nm → wavenumber (cm⁻¹)", *_reciprocal(1e7), "wavenumber (cm⁻¹)"),
    "cm1_nm": ("wavenumber (cm⁻¹) → nm", *_reciprocal(1e7), "wavelength (nm)"),
}


def conversion_choices() -> list[tuple[str, str]]:
    """(stored key, label) for each conversion, after the empty "None"."""
    return [("", "None"), *((key, entry[0]) for key, entry in CONVERSIONS.items())]


def apply_secondary_axes(ax: Any, options: dict[str, Any]) -> None:
    """Draw the secondary x (top) and y (right) axes the options name; a re-draw replaces them."""
    for old in getattr(ax, "_dhub_secondary_axes", ()):
        try:
            old.remove()
        except (ValueError, NotImplementedError, AttributeError):
            pass
    ax._dhub_secondary_axes = []  # noqa: SLF001 - our own attribute
    if getattr(ax, "name", "rectilinear") != "rectilinear":
        return  # polar and 3D axes have no top or right edge to hang a scale on
    for axis, make, location in (("x", "secondary_xaxis", "top"), ("y", "secondary_yaxis", "right")):
        key = str(options.get(f"secondary_{axis}") or "")
        if not key:
            continue
        entry = CONVERSIONS.get(key)
        if entry is None:
            applogger.warning("Unknown secondary axis conversion %r ignored.", key)
            continue
        _label, forward, inverse, caption = entry
        try:
            secondary = getattr(ax, make)(location, functions=(forward, inverse))
            (secondary.set_xlabel if axis == "x" else secondary.set_ylabel)(caption)
        except Exception:  # noqa: BLE001 - an axes that cannot: the setting is stored and ignored there
            applogger.info("The %s axis cannot take a secondary axis (%s).", axis, key)
            continue
        ax._dhub_secondary_axes.append(secondary)  # noqa: SLF001
