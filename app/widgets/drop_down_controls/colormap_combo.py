"""app.widgets.drop_down_controls.colormap_combo

ColormapCombo: Matplotlib's colormaps, each with its gradient as the icon.

The usual ones first - the perceptually uniform five, then the diverging and
the cyclic most charts use - then every other registered colormap in name
order; the reversed variants ("viridis_r") are not listed but can be typed:
the combo is editable, so a colormap a plugin registered is accepted too.
The first entry, empty, leaves the style sheet's own (rcParams image.cmap).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QWidget

from app.widgets.drop_down_controls.icon_combo import IconComboBox

#: Listed first, in this order: what a chart reaches for most.
COMMON_COLORMAPS: tuple[str, ...] = (
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdBu", "RdYlBu", "Spectral", "bwr", "seismic",
    "Greys", "Blues", "Greens", "Reds", "Oranges", "Purples", "YlOrRd", "YlGnBu",
    "twilight", "hsv", "turbo", "jet", "terrain", "gist_earth", "ocean",
    "tab10", "tab20", "Set1", "Set2", "Pastel1", "Paired",
)
_ICON = QSize(48, 12)


def colormap_names() -> list[str]:
    """Every registered colormap but the reversed variants: the common ones first."""
    import matplotlib

    registered = [name for name in matplotlib.colormaps if not name.endswith("_r")]
    common = [name for name in COMMON_COLORMAPS if name in registered]
    return common + sorted((name for name in registered if name not in common), key=str.lower)


@lru_cache(maxsize=512)
def _gradient_icon(name: str) -> QIcon:
    """The colormap's colours left to right, as an icon."""
    if not name:
        return QIcon()
    import matplotlib

    try:
        colormap = matplotlib.colormaps[name]
    except (KeyError, ValueError):
        return QIcon()
    rgba = (colormap(np.linspace(0.0, 1.0, _ICON.width())) * 255).astype(np.uint8)
    strip = np.ascontiguousarray(np.repeat(rgba[np.newaxis, :, :], _ICON.height(), axis=0))
    image = QImage(strip.data, _ICON.width(), _ICON.height(), 4 * _ICON.width(), QImage.Format.Format_RGBA8888)
    return QIcon(QPixmap.fromImage(image.copy()))



def colormap_preview_icon(name: str) -> QIcon:
    """Public gradient preview icon for property grids."""
    return _gradient_icon((name or "").strip())


class ColormapCombo(IconComboBox):
    """Editable combo of Matplotlib colormaps with gradient previews."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, entries=[("", ""), *((name, name) for name in colormap_names())],
                         icon_for=_gradient_icon, icon_size=_ICON)
        self.setEditable(True)

    def current_colormap(self) -> str:
        """The colormap chosen or typed; "" for the style sheet's own."""
        return self.currentText().strip()

    def set_current_colormap(self, name: str | None) -> None:
        text = str(name or "").strip()
        if not self.set_current_value(text):
            self.setEditText(text)
