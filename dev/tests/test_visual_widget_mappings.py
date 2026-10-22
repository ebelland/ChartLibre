"""Regression coverage for artist aliases and dotted Matplotlib rcParams."""
import pytest
from app.widgets.dictionary_editor import _kind_for_key

CASES = {
    "color": {"c", "color", "facecolor", "edgecolor", "markerfacecolor",
              "markeredgecolor", "markerfacecoloralt", "mfc", "mec", "ecolor",
              "labelcolor", "titlecolor", "text.color", "axes.labelcolor",
              "axes.titlecolor", "xtick.color", "ytick.color", "grid.color",
              "legend.facecolor", "legend.edgecolor", "legend.labelcolor",
              "figure.facecolor", "figure.edgecolor", "savefig.facecolor",
              "savefig.edgecolor"},
    "linestyle": {"linestyle", "ls", "negative_linestyle", "grid.linestyle",
                  "contour.negative_linestyle"},
    "marker": {"marker", "sym"},
    "joinstyle": {"joinstyle", "solid_joinstyle", "dash_joinstyle",
                  "lines.solid_joinstyle", "lines.dash_joinstyle"},
    "capstyle": {"capstyle", "solid_capstyle", "dash_capstyle",
                 "lines.solid_capstyle", "lines.dash_capstyle"},
    "hatch": {"hatch", "patch.hatch"},
    "colormap": {"cmap", "image.cmap"},
    "fontlist": {"font.family", "font.serif", "font.sans-serif", "font.monospace",
                 "font.cursive", "font.fantasy", "fontfamily", "font_family"},
}

@pytest.mark.parametrize(("key", "expected"),
                         [(key, kind) for kind, keys in CASES.items() for key in sorted(keys)])
def test_visual_kind_mapping(key: str, expected: str) -> None:
    assert _kind_for_key(key, {}) == expected
