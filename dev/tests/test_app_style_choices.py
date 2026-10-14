"""Settings: the app styles offered, and the default chart style's grouping."""
from __future__ import annotations

from app.dialogs.settings_dialog import _mpl_style_choices
from app.styles.style import APP_STYLE_AUTOMATIC, APP_STYLE_QSS, available_app_styles, resolve_app_style
from app.utils.config import MPLSTYLES_DIR


def test_only_automatic_and_the_shipped_sheets_are_offered() -> None:
    keys = [key for key, _label in available_app_styles()]
    assert keys[0] == APP_STYLE_AUTOMATIC
    assert set(keys[1:]) == set(APP_STYLE_QSS)
    assert not any(key.startswith(("qt:", "file:")) for key in keys)
    assert "ubuntu_native" in keys  # a sheet dropped into app/styles/ is a theme


def test_a_stored_qt_style_or_file_reads_as_automatic() -> None:
    assert resolve_app_style("qt:Fusion") == APP_STYLE_AUTOMATIC
    assert resolve_app_style("file:/somewhere/mine.qss") == APP_STYLE_AUTOMATIC
    assert resolve_app_style("ubuntu_native") == "ubuntu_native"


def test_the_default_chart_style_is_grouped_by_folder() -> None:
    choices = _mpl_style_choices()
    assert choices[0] == ("ChartLibreDefault", "ChartLibreDefault")
    assert choices[1] == ("", "Matplotlib default")
    assert choices[2] is None
    values = [choice[0] for choice in choices if choice is not None]
    apps = [value for value in values if value.startswith("Apps/")]
    journals = [value for value in values if value.startswith("journals/")]
    assert apps and journals and values.index(apps[0]) < values.index(journals[0])
    assert choices[choices.index(None, 3)] is None  # a separator between Apps and Journals
    on_disk = {p.relative_to(MPLSTYLES_DIR).with_suffix("").as_posix() for p in MPLSTYLES_DIR.rglob("*.mplstyle")}
    assert set(values) - {""} == on_disk  # nothing lost, nothing invented
