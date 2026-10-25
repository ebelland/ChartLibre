"""The colour swatches and the colour button speak Matplotlib's colours, not Qt's (matplotlib_widget_mappings_patch)."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor

from app.widgets.drop_down_controls.color_combo import color_preview_icon
from app.widgets.drop_down_controls.color_picker import ColorPickerButton, matplotlib_hex, qcolor_from_matplotlib


def _centre(value: str) -> QColor:
    image = color_preview_icon(value).pixmap(QSize(16, 16)).toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


@pytest.mark.parametrize(("value", "hexa"), [
    ("red", "#ff0000"), ("#1f77b4", "#1f77b4"), ("C0", "#1f77b4"), ("tab:blue", "#1f77b4"), ("0.5", "#808080"),
])
def test_the_swatch_shows_the_colour_it_names(qapp, value: str, hexa: str) -> None:
    assert _centre(value).name() == hexa  # red used to come out blue


def test_a_colour_with_alpha_keeps_its_channels(qapp) -> None:
    colour = _centre("#ff000080")
    assert (colour.red(), colour.green(), colour.blue()) == (255, 0, 0)


def test_an_unknown_colour_has_no_swatch(qapp) -> None:
    assert color_preview_icon("not-a-colour").isNull()


@pytest.mark.parametrize("value", ["C0", "tab:blue", "0.5", "red", "#ff000080", "#abc"])
def test_the_button_keeps_what_was_written(qapp, value: str) -> None:
    button = ColorPickerButton()
    button.set_color(value)
    assert button.color() == value  # a name Qt cannot read used to be erased


def test_a_chosen_colour_is_written_the_way_matplotlib_reads_it() -> None:
    assert matplotlib_hex(QColor(255, 0, 0)) == "#ff0000"
    assert matplotlib_hex(QColor(255, 0, 0, 128)) == "#ff000080"  # Qt's own text would be #80ff0000
    assert qcolor_from_matplotlib("nope") is None
    assert qcolor_from_matplotlib("C1").name() == "#ff7f0e"


def test_the_colour_picker_survives_its_editor_closing_while_the_dialog_is_open(qapp, monkeypatch) -> None:
    """As a property editor's cell editor, the button was deleted while its dialog was open (macOS' colour
    panel took the focus, the cell closed its editor): the application crashed when the dialog returned."""
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QColorDialog, QWidget

    from app.widgets.drop_down_controls.color_picker import ColorPickerButton

    holder = QWidget()
    button = ColorPickerButton(holder)
    seen: list[bool] = []

    def exec_and_lose_the_editor(dialog: QColorDialog) -> int:
        seen.append(dialog.testOption(QColorDialog.ColorDialogOption.DontUseNativeDialog))
        button.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        return QColorDialog.DialogCode.Accepted

    monkeypatch.setattr(QColorDialog, "exec", exec_and_lose_the_editor)
    button.click()
    assert seen == [True] and not shiboken6.isValid(button)
    holder.deleteLater()
