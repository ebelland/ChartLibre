"""Settings > Default chart style: a figure with no style of its own takes it."""
from __future__ import annotations

from app.charts.render_figure import DEFAULT_MPL_STYLE_KEY, default_style_text
from app.utils.config import set_value


def test_the_default_style_is_read_from_settings() -> None:
    set_value(DEFAULT_MPL_STYLE_KEY, "")
    assert default_style_text() == ""
    set_value(DEFAULT_MPL_STYLE_KEY, "jmp-classic")
    assert "JMP Classic" in default_style_text()
    set_value(DEFAULT_MPL_STYLE_KEY, "no-such-style")
    assert default_style_text() == ""
    set_value(DEFAULT_MPL_STYLE_KEY, "")
