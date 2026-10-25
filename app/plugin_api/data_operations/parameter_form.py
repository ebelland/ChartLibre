"""Build a live form from a parameter declaration, and read it back.

The Qt half of ``parameter_spec``.  Kept separate so the declarations stay
importable and testable without a window server, which is most of what makes
them worth having.

One class, three jobs: build the widgets in declaration order, hand back the
current values by name, and show or hide each row as the declared
``visible_for`` rules dictate.  Every widget's change signal is routed to a
single callback, because the thing every existing dialog does with it is call
``mark_results_stale`` - and connecting eight signals by hand is exactly the
step that gets forgotten when a ninth control is added.
"""

from __future__ import annotations

import html
from typing import Any, Callable, Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.plugin_api.data_operations.parameter_spec import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    Param,
    TextParam,
)
from app.styles.style import TitledCard
from app.utils.i18n import _
from app.widgets.drop_down_controls.color_combo import MatplotlibColorCombo


#: The narrowest a number box may become. Without it a box is as wide as
#: its largest value needs - "-1000000000000.000000" for a range of 1e12 -
#: and one such box pushed every field past the panel's right edge. The
#: field still grows to fill the row; it only no longer refuses to shrink.
SPIN_MIN_WIDTH: int = 80
#: The widest a field becomes in a wide window: a number needs a few digits,
#: a choice or a line its words - stretched across the whole settings panel,
#: "Threshold: 2.400" ran a hand's width from its label (Explore Outliers).
SPIN_MAX_WIDTH: int = 140
FIELD_MAX_WIDTH: int = 260
COLOR_MAX_WIDTH: int = 180


class ParameterForm:
    """A QFormLayout built from ``Param`` declarations."""

    def __init__(
        self,
        params: Sequence[Param],
        parent: QWidget,
        *,
        on_change: Callable[[], None] | None = None,
        context: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        """``context`` supplies values this form does not own.

        Visibility routinely depends on the model combo, which belongs to the
        base dialog rather than to the parameter set - so without this, a rule
        like ``visible_for={"model": ...}`` would look up a name that is never
        present and hide the row forever.
        """
        self._params = tuple(params)
        self._parent = parent
        self._on_change = on_change
        self._context = context
        self._widgets: dict[str, QWidget] = {}
        #: Called after every visibility refresh - for a dialog that has moved
        #: some fields out of the form (into a grid, say) and keeps the
        #: labels it drew for them in step.
        self.after_refresh: list[Callable[[], None]] = []

        self.widget = QWidget(parent)
        column = QVBoxLayout(self.widget)
        column.setContentsMargins(0, 0, 0, 0)
        form_widget = QWidget(self.widget)
        #: The main rows; the advanced ones are in their own form (see below).
        self.layout = self._new_form(form_widget)
        column.addWidget(form_widget)
        #: The closed "Advanced" frame, when any parameter is declared advanced.
        self.advanced_card: TitledCard | None = None
        advanced_form: QFormLayout | None = None
        # One advanced parameter alone is no reason for a closed frame: it
        # joins the others (Control Chart's one advanced check box was cut).
        advanced_count = sum(1 for param in self._params if param.advanced)
        if advanced_count > 1:
            self.advanced_card = TitledCard(self.widget, _("Advanced"), "parameterAdvancedCard",
                                            collapsible=True, collapsed=True)
            advanced_form = self._new_form(QWidget(self.advanced_card.card))
            self.advanced_card.card.layout().addWidget(advanced_form.parentWidget())
            column.addWidget(self.advanced_card)
        # The rows at the top, the room left under them - not around them.
        column.addStretch(1)
        self._forms: dict[str, QFormLayout] = {}
        #: Each section's heading, and the rows under it.
        self._sections: dict[str, tuple[QLabel, list[str]]] = {}
        #: The choices a ChoiceParam has now, when set_choices replaced its declared ones.
        self._choices: dict[str, tuple[tuple[str, Any], ...]] = {}

        for param in self._params:
            widget = self._build_widget(param)
            self._widgets[param.name] = widget
            form = advanced_form if param.advanced and advanced_form is not None else self.layout
            if param.section:
                if param.section not in self._sections:
                    heading = QLabel(f"<b>{html.escape(_(param.section))}</b>", self.widget)
                    form.addRow(heading)
                    self._sections[param.section] = (heading, [])
                self._sections[param.section][1].append(param.name)
            if isinstance(param, BoolParam):
                # As on the Mac: the box in the field column, its words on its right.
                form.addRow("", widget)
            else:
                form.addRow(_(param.label), widget)
            self._forms[param.name] = form

        self.refresh_visibility()

    @staticmethod
    def _new_form(owner: QWidget) -> QFormLayout:
        form = QFormLayout(owner)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # Room between rows: the native macOS check box, drawn larger than
        # its size hint, was cut at the bottom by the row under it.
        form.setVerticalSpacing(8)
        return form

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _build_widget(self, param: Param) -> QWidget:
        """Return the control for one parameter, already wired."""
        if isinstance(param, ChoiceParam):
            widget: QWidget = QComboBox(self.widget)
            for label, value in param.labelled_choices():
                widget.addItem(_(str(label)), value)
            index = widget.findData(param.default)
            widget.setCurrentIndex(max(0, index))
            widget.currentIndexChanged.connect(self._changed)

        elif isinstance(param, IntParam):
            widget = QSpinBox(self.widget)
            widget.setRange(param.minimum, param.maximum)
            # An odd-only control stepping by one would let the user land on
            # an even value that coerce() then silently moves. Step by two so
            # the control cannot express what the parameter refuses.
            widget.setSingleStep(param.step * 2 if param.odd_only else param.step)
            widget.setValue(param.default)
            if param.suffix:
                widget.setSuffix(_(param.suffix))
            if param.special_text:
                widget.setSpecialValueText(_(param.special_text))
            widget.valueChanged.connect(self._changed)

        elif isinstance(param, FloatParam):
            widget = QDoubleSpinBox(self.widget)
            widget.setRange(param.minimum, param.maximum)
            widget.setDecimals(param.decimals)
            widget.setSingleStep(param.step)
            widget.setValue(param.default)
            if param.suffix:
                widget.setSuffix(_(param.suffix))
            if param.special_text:
                widget.setSpecialValueText(_(param.special_text))
            widget.valueChanged.connect(self._changed)

        elif isinstance(param, BoolParam):
            widget = QCheckBox(_(param.label).rstrip(":"), self.widget)
            widget.setMinimumHeight(widget.sizeHint().height() + 6)
            # No blue focus square around the box on macOS (the first box of a tab takes the focus).
            widget.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
            widget.setChecked(param.default)
            widget.toggled.connect(self._changed)

        elif isinstance(param, ColorParam):
            widget = MatplotlibColorCombo(self.widget, include_none=False)
            if not widget.set_current_hex(param.default):
                widget.setCurrentIndex(0)
            widget.currentIndexChanged.connect(self._changed)

        elif isinstance(param, TextParam):
            widget = QLineEdit(self.widget)
            widget.setText(param.default)
            if param.placeholder:
                widget.setPlaceholderText(_(param.placeholder))
            widget.editingFinished.connect(self._changed)

        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setMinimumWidth(SPIN_MIN_WIDTH)
            widget.setMaximumWidth(SPIN_MAX_WIDTH)
        elif isinstance(widget, MatplotlibColorCombo):
            # Its longest name ("lightgoldenrodyellow") is no measure of the one shown.
            widget.setMaximumWidth(COLOR_MAX_WIDTH)
        elif isinstance(widget, QComboBox):
            # As wide as its longest choice (the arrow included), never past the limit.
            widget.setMaximumWidth(min(FIELD_MAX_WIDTH, max(widget.sizeHint().width(), SPIN_MAX_WIDTH)))
        elif isinstance(widget, QLineEdit):
            widget.setMaximumWidth(FIELD_MAX_WIDTH)
        if param.tooltip:
            widget.setToolTip(_(param.tooltip))
        return widget

    def _changed(self, *_args: Any) -> None:
        """Re-evaluate visibility, then tell the dialog something moved."""
        self.refresh_visibility()
        if self._on_change is not None:
            self._on_change()

    # ------------------------------------------------------------------
    # Reading and writing
    # ------------------------------------------------------------------

    def values(self) -> dict[str, Any]:
        """Return every parameter's current value, coerced, keyed by name.

        Every parameter, including the hidden ones: a hidden control still
        holds a value, and an operation that reads ``params["window"]``
        regardless of which model is selected must not get a KeyError for it.
        """
        result: dict[str, Any] = {}
        for param in self._params:
            widget = self._widgets[param.name]
            result[param.name] = self._coerce(param, self._read_widget(widget))
        return result

    def _coerce(self, param: Param, value: Any) -> Any:
        """*value* in *param*'s type - among the choices it has now, for one whose choices were set."""
        choices = self._choices.get(param.name)
        if choices is None:
            return param.coerce(value)
        values = [choice for _label, choice in choices]
        return value if value in values else (values[0] if values else None)

    def set_choices(self, name: str, choices: Sequence[tuple[str, Any]]) -> None:
        """Offer *choices* ((label, value) pairs) in the ChoiceParam *name*: for choices that depend on another
        (the charts a model can draw). The current value stays when it is still among them."""
        widget = self._widgets[name]
        assert isinstance(widget, QComboBox)
        previous = widget.currentData()
        self._choices[name] = tuple((str(label), value) for label, value in choices)
        widget.blockSignals(True)
        try:
            widget.clear()
            for label, value in self._choices[name]:
                widget.addItem(_(label), value)
            widget.setCurrentIndex(max(0, widget.findData(previous)))
        finally:
            widget.blockSignals(False)

    @staticmethod
    def _read_widget(widget: QWidget) -> Any:
        if isinstance(widget, MatplotlibColorCombo):
            return widget.current_hex()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def set_values(self, values: Mapping[str, Any]) -> None:
        """Restore saved values, ignoring names this form does not have."""
        for param in self._params:
            if param.name not in values:
                continue
            widget = self._widgets[param.name]
            value = self._coerce(param, values[param.name])

            widget.blockSignals(True)
            try:
                if isinstance(widget, MatplotlibColorCombo):
                    widget.set_current_hex(str(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentIndex(max(0, widget.findData(value)))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            finally:
                widget.blockSignals(False)

        self.refresh_visibility()

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def refresh_visibility(self) -> None:
        """Show or hide each row according to the declared rules."""
        current = dict(self.values())
        if self._context is not None:
            # Context does not override a real parameter of the same name: the
            # form's own control is the authority on its own value.
            for key, value in self._context().items():
                current.setdefault(key, value)

        for param in self._params:
            widget = self._widgets[param.name]
            visible = param.is_visible(current)
            widget.setVisible(visible)
            # The label is a separate widget, and hiding only the field leaves
            # a labelled empty row - which is what makes a hand-written
            # visibility method easy to get subtly wrong.
            label = self._forms.get(param.name, self.layout).labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        for heading, names in self._sections.values():
            heading.setVisible(any(not self._widgets[name].isHidden() for name in names))
        if self.advanced_card is not None:
            # No frame of nothing: shown while one of its rows is.
            self.advanced_card.setVisible(any(
                param.is_visible(current) for param in self._params if param.advanced))
        for callback in self.after_refresh:
            callback()

    def field(self, name: str) -> QWidget:
        """The control built for the parameter *name*."""
        return self._widgets[name]
