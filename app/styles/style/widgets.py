"""Widget factories and helpers: cards, titles, buttons, menus, dialog shells, sizing."""
from __future__ import annotations

import html


from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap, QTextOption
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from app.logs.logger import applogger
from app.styles.style.metrics import (
    COMBO_MIN_CONTENTS_LENGTH,
    DIALOG_SIZES,
    IS_MACOS,
    MARGIN_CARD,
    MARGIN_DIALOG,
    MenuCallback,
    MenuItem,
    SPACING_DEFAULT,
    ShortcutLike,
    _COMBO_CHROME_WIDTH,
)
from app.styles.style.icons import action_presentation, load_icon


# UI helpers
# ----------------------------------------------------------------------

_PRIMARY_BUTTON_TEXTS = {"Apply", "OK"}


def _mark_primary_button(widget: QWidget, text: str) -> None:
    """Mark primary buttons for QSS styling."""
    if text in _PRIMARY_BUTTON_TEXTS:
        widget.setProperty("primary", True)



def create_section_title(text: str, parent: QWidget | None = None) -> QtWidgets.QLabel:
    """Create a standard section title label styled through QSS."""
    label = QtWidgets.QLabel(text, parent)
    label.setProperty("sectionTitle", True)
    label.setContentsMargins(0, 0, 0, 0)
    return label


def create_compact_section_title(
    text: str, parent: QWidget | None = None,
) -> QtWidgets.QLabel:
    """A section title one point smaller and demi-bold.

    ``create_section_title`` alone (the QSS rule's own 11pt / weight 600) is
    right for a dialog with one or two section headers. A panel that lists
    several of them one after another - the Data Operations list's own
    Analysis/Statistics/... headings - reads better with this shorter,
    lighter version, which used to be a copy of this exact font math kept
    inside ``DataOperationWidget``; other panels asking to match it is
    what gave it this one home instead.
    """
    label = create_section_title(text, parent)
    font = QFont(label.font())
    point_size = font.pointSize()
    if point_size > 0:
        font.setPointSize(max(point_size - 1, 9))
    else:
        font.setPixelSize(12)
    font.setWeight(QFont.Weight.DemiBold)
    label.setFont(font)
    return label


class CardFrame(QFrame):
    """A rounded ``[card="true"]`` surface with its own layout pre-margined.

    What ``create_card_widget()`` and ``apply_card_layout()`` did as two
    separate calls - the QSS side (the "card" property, no frame shape, an
    expanding size policy) and the layout side (margins, spacing) - always
    had to travel together, and a call site that built its own layout with
    ``stdSizeAndlayout`` instead of ``apply_card_layout`` by mistake got a
    card with its content flush against the rounded border: cramped, not
    wrong enough to fail a review, and it happened more than once (axis
    kwargs, series properties, an operation dialog's model selector, all
    independently). Baking the layout into construction makes that specific
    mistake impossible rather than something to remember not to make.

    ``self.layout()`` is ready to use immediately - add widgets to it, or
    nest another layout inside one of them. Pass ``orientation`` for a
    card whose top-level content reads left-to-right instead of stacked.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        object_name: str | None = None,
        *,
        orientation: Qt.Orientation = Qt.Orientation.Vertical,
        margins: tuple[int, int, int, int] = MARGIN_CARD,
        spacing: int = SPACING_DEFAULT,
    ) -> None:
        super().__init__(parent)
        if object_name:
            self.setObjectName(object_name)
        self.setProperty("card", True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout_cls = (
            QVBoxLayout if orientation == Qt.Orientation.Vertical else QHBoxLayout
        )
        layout = layout_cls(self)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        self._box_layout: QBoxLayout = layout

    def layout(self) -> QBoxLayout:
        """The card's own box layout, typed as one so addLayout/addStretch resolve."""
        return self._box_layout


class TitledCard(QWidget):
    """A section title *above* a CardFrame, never inside its own border.

    Impostazioni di Sistema puts "macOS" / "Schermi" outside the grouped
    box beneath them, not as a title baked into its frame the way
    QGroupBox - and this app's own cards, before this - do. Drop-in
    replacement for the old ``card = CardFrame(parent, name); card_layout
    = card.layout(); card_layout.addWidget(create_section_title(...))``
    idiom: construct this instead, then use ``self.card`` exactly like
    that ``card`` variable - ``self.card.layout()`` is ready the same way.
    Add *this* widget to your layout, not ``self.card`` - the title has to
    travel with its card as one block. ``self.title_label`` is the title
    QLabel itself, for the rare card whose title changes at runtime
    (series_properties.py's own count-in-the-title, for one).

    *collapsible*: an arrow before the title shows or hides the box - a
    click on the title too - for the options most people leave alone
    ("More options", todo H-10); *collapsed* is how it starts.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        title: str = "",
        object_name: str | None = None,
        *,
        orientation: Qt.Orientation = Qt.Orientation.Vertical,
        margins: tuple[int, int, int, int] = MARGIN_CARD,
        spacing: int = SPACING_DEFAULT,
        collapsible: bool = False,
        collapsed: bool = False,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.title_label = create_section_title(title, self)
        self.disclosure: QtWidgets.QToolButton | None = None
        if collapsible:
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(2)
            self.disclosure = QtWidgets.QToolButton(self)
            self.disclosure.setObjectName("cardDisclosure")
            self.disclosure.setAutoRaise(True)
            self.disclosure.setCheckable(True)
            self.disclosure.setFixedSize(16, 16)
            self.disclosure.setToolTip(title)
            self.disclosure.toggled.connect(lambda shown: self.set_collapsed(not shown))
            self.title_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self.title_label.installEventFilter(self)
            header.addWidget(self.disclosure)
            header.addWidget(self.title_label)
            header.addStretch(1)
            layout.addLayout(header)
        else:
            layout.addWidget(self.title_label)

        self.card = CardFrame(
            self, object_name, orientation=orientation, margins=margins, spacing=spacing,
        )
        layout.addWidget(self.card)
        if collapsible:
            self.set_collapsed(collapsed)

    @property
    def collapsed(self) -> bool:
        return self.disclosure is not None and not self.card.isVisibleTo(self)

    def set_collapsed(self, collapsed: bool) -> None:
        """Hide the box (the title stays, its arrow pointing right) or show it again."""
        if self.disclosure is None:
            return
        self.card.setVisible(not collapsed)
        self.disclosure.blockSignals(True)
        self.disclosure.setChecked(not collapsed)
        self.disclosure.blockSignals(False)
        self.disclosure.setArrowType(Qt.ArrowType.RightArrow if collapsed else Qt.ArrowType.DownArrow)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if (
            watched is self.title_label
            and self.disclosure is not None
            and event.type() == QtCore.QEvent.Type.MouseButtonRelease
        ):
            self.disclosure.toggle()
            return True
        return super().eventFilter(watched, event)


def mark_editor_panel(widget: QWidget) -> QWidget:
    """Mark list/tree/editor widgets for shared panel styling."""
    widget.setProperty("editorPanel", True)
    repolish_widget(widget)
    return widget


def mark_icon_only(button: QtWidgets.QAbstractButton) -> QtWidgets.QAbstractButton:
    """Drop a button's label and style it as a square icon button.

    Both stylesheets have carried an ``[iconOnly="true"]`` rule for a
    while - a 32x32 borderless square with a hover wash, the shape a
    toolbar-ish button takes next to a field rather than in a row of
    labelled buttons - and nothing in Python ever set the property, so no
    button had ever actually looked like one.

    The text is cleared rather than merely hidden: the label is what makes
    a QPushButton ask for a wide sizeHint, and the tooltip (already set
    from the action catalogue's description by create_action_button)
    carries the meaning that the words were carrying.
    """
    button.setText("")
    button.setProperty("iconOnly", True)
    repolish_widget(button)
    return button


def mark_destructive_button(button: QtWidgets.QAbstractButton) -> QtWidgets.QAbstractButton:
    """Style a button as a destructive action - solid red, on both platforms.

    "Clear list" sat in the exact same column as every recent-project row
    above it, one more left-aligned button among several - nothing set it
    apart from an action a misclick could not undo. ``[destructive="true"]``
    is a new rule in both stylesheets (fluent_win11.qss, macos_native.qss),
    the same ``setProperty`` + QSS-attribute-selector idiom ``[primary="true"]``
    already uses for the opposite emphasis (see _mark_primary_button) - not
    a fixed inline color, so it still follows whatever palette the active
    theme resolves disabled/hover state through.
    """
    button.setProperty("destructive", True)
    repolish_widget(button)
    return button


def create_action_button(
    parent: QWidget,
    action_id: str,
    action: MenuCallback | None,
    layout: QBoxLayout | None = None,
    *,
    presentation: tuple[QIcon, str, str] | None = None,
) -> QPushButton:
    """Create a push button from a catalogue action, and nothing else.

    Label, tooltip and icon all come from ``config.json`` and go through the
    translator, so a button is one line at the call site and its wording is
    editable and localisable without touching Python.  A button that wants
    different words wants its own action id.

    ``presentation`` supplies ``(icon, text, tooltip)`` directly, for the
    callers that are deliberately not in the catalogue: a data operation
    carries its own name, description and icon so that the plugin is one
    self-contained file.  Passing it skips the lookup entirely rather than
    overriding it afterwards - an id that is not in config.json is worth a
    warning, and a plugin would trip it on every button.
    """
    icon, text, tooltip = (
        presentation if presentation is not None else action_presentation(action_id)
    )

    button = QPushButton(text, parent)
    button.setToolTip(tooltip)
    button.setStatusTip(tooltip)
    button.setIcon(icon)

    if action is not None:
        button.clicked.connect(action)
    else:
        # Nothing to connect: a button that silently does nothing when pressed
        # is worse than one that looks unavailable.
        button.setEnabled(False)

    _mark_primary_button(button, text)

    if layout is not None:
        layout.addWidget(button)

    return button


def create_toolbar_button(
    parent: QWidget,
    action_id: str,
    action: MenuCallback | None,
    layout: QBoxLayout | None = None,
) -> QToolButton:
    """Create a flat toolbar button from a catalogue action.

    Same contract as ``create_action_button``; only the widget class differs.
    """
    icon, text, tooltip = action_presentation(action_id)

    button = QToolButton(parent)
    # Named after the action, not after its label: the object name is what QSS
    # selects on, and a stylesheet keyed to translated text would stop matching
    # the moment the interface was translated.
    button.setObjectName(f"{action_id}Button")
    button.setToolTip(tooltip)
    button.setStatusTip(tooltip)
    button.setAutoRaise(True)
    button.setText(text)
    button.setIcon(icon)

    if action is not None:
        button.clicked.connect(action)
    else:
        button.setEnabled(False)

    _mark_primary_button(button, text)

    if layout is not None:
        layout.addWidget(button)

    return button

#: Context menus on macOS carry no icons - Finder's, Mail's, every Apple
#: app's - so the menus that follow the platform pass this as ``icons``.
CONTEXT_MENU_ICONS: bool = not IS_MACOS


def create_menu_item(
    parent: QWidget,
    menu: QMenu | None,
    icon: str | QIcon | None,
    checkable: bool,
    text: str,
    tooltip: str,
    key: ShortcutLike,
    action: MenuCallback | None,
    *,
    action_id: str | None = None,
    checked: bool = False,
    enabled: bool = True,
    icons: bool = True,
) -> QAction:
    """Create and optionally add a QAction to a menu.

    With *icons* False the item shows no icon in a menu, and reserves no
    space for one.
    """
    qaction = QAction(parent)
    qaction.setText(text)
    qaction.setToolTip(tooltip)
    qaction.setStatusTip(tooltip)
    qaction.setIconVisibleInMenu(icons)
    qaction.setCheckable(checkable)
    qaction.setChecked(checked)
    qaction.setEnabled(enabled)

    if not icons:
        pass
    elif isinstance(icon, QIcon):
        qaction.setIcon(icon)
    elif icon:
        qaction.setIcon(load_icon(icon))
    elif IS_MACOS:
        # macOS's native menu does not reserve an icon gutter for an item
        # that carries none, the way Fusion/Windows menus always do - so an
        # icon-less item's text sits flush left while its siblings' text
        # starts after their icon, out of line with the rest of the menu. A
        # blank icon the same size as a real one keeps the column reserved.
        blank = QPixmap(20, 20)
        blank.fill(Qt.GlobalColor.transparent)
        qaction.setIcon(QIcon(blank))

    if action_id is not None:
        qaction.setData(action_id)

    # Guarded rather than passed through: PySide6 raises if setShortcut is
    # given None, and "this action has no shortcut" is the common case.
    if key is not None:
        qaction.setShortcut(key)

    if action is not None:
        qaction.triggered.connect(action)

    if menu is not None:
        menu.addAction(qaction)

    return qaction


def create_menu(
    parent: QWidget,
    items: list[MenuItem | None],
    *,
    icons: bool = True,
) -> QMenu:
    """Create a QMenu from typed menu item descriptors.

    An item carrying ``submenu`` becomes a nested menu built from the same
    descriptors, so a caller composes one list and both the popup menu and
    (on macOS) the real menu bar are built from it - which is the whole
    reason the descriptors exist.
    """
    menu = QMenu(parent)

    for item in items:
        if item is None:
            menu.addSeparator()
            continue

        if item.submenu is not None:
            child = create_menu(parent, item.submenu, icons=icons)
            child.setTitle(item.text)
            if icons and item.icon is not None:
                child.setIcon(
                    item.icon if isinstance(item.icon, QIcon) else load_icon(item.icon)
                )
            child.setEnabled(item.enabled)
            menu.addMenu(child)
            continue

        create_menu_item(
            parent=parent,
            menu=menu,
            icon=item.icon,
            checkable=item.checkable,
            text=item.text,
            tooltip=item.tooltip,
            key=item.shortcut,
            action=item.callback,
            action_id=item.action_id,
            checked=item.checked,
            enabled=item.enabled,
            icons=icons,
        )

    return menu


def repolish_widget(widget: QtWidgets.QWidget) -> None:
    """Force Qt to re-apply style rules to a widget."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    # QWidget.update explicitly, not widget.update().  On an item view the name
    # resolves to QAbstractItemView.update(index), which requires an argument
    # and raises TypeError - so this raised on every list, tree and table it was
    # asked to repolish, which is every panel that goes through
    # mark_editor_panel.
    QWidget.update(widget)


def create_doc_link(parent: QWidget | None = None) -> QtWidgets.QLabel:
    """Create the label used for "read the documentation" links.

    Five dialogs each configured their own; four of them forgot to escape the
    title and URL, so a ``&`` in either produced broken markup.  Building the
    label here means the escaping is not something a caller can forget.
    """
    label = QtWidgets.QLabel("", parent)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(True)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    label.setWordWrap(True)
    return label


def set_doc_link(label: QtWidgets.QLabel, title: str, url: str) -> None:
    """Point a documentation label at a URL, or clear it when there is none."""
    clean_title = str(title or "").strip()
    clean_url = str(url or "").strip()

    if not clean_url:
        label.clear()
        label.setToolTip("")
        return

    label.setText(
        f'<a href="{html.escape(clean_url, quote=True)}">'
        f"{html.escape(clean_title or clean_url)}</a>"
    )
    label.setToolTip(clean_url)


def apply_card_layout(
    layout: QBoxLayout,
    *,
    margins: tuple[int, int, int, int] = MARGIN_CARD,
    spacing: int = SPACING_DEFAULT,
) -> None:
    """Give a card's inner layout breathing room.

    ``stdSizeAndlayout`` zeroes the margins, which is right for a *nested*
    layout but wrong for the outermost layout inside a card: the card draws a
    border and a background, and content flush against that border reads as
    cramped.  Use this for the layout directly inside a create_card_widget().
    """
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)


def apply_toolbox_header_metrics(
    toolbox: QtWidgets.QToolBox,
    *,
    extra_height: int = 22,
    minimum_height: int = 40,
) -> None:
    """Give a QToolBox's section headers enough room for their text.

    Why this cannot be done in QSS alone: a QToolBox header is a private
    ``QToolBoxButton`` whose height comes from its own ``sizeHint()``, computed
    from the font and icon *before* the stylesheet's padding is added.  Setting
    ``padding`` in QSS therefore grows the box the text is drawn in without
    growing the button, so the label ends up vertically clipped - which is
    exactly what the properties accordion was doing.

    Setting a minimum height from the real font metrics fixes it for whatever
    font the platform actually resolved, instead of guessing a pixel value.
    Call again after adding items.
    """
    for button in toolbox.findChildren(QtWidgets.QAbstractButton):
        # Only the section headers are direct children of the toolbox itself;
        # anything deeper belongs to a page.
        if button.parent() is not toolbox:
            continue

        line_height = button.fontMetrics().height()
        height = max(minimum_height, line_height + extra_height)
        # Both, not just the minimum: on macOS the style draws the header from
        # its own sizeHint, and a minimum alone still let it collapse back to
        # the default height once the toolbox was laid out.
        button.setMinimumHeight(height)
        button.setFixedHeight(height)
        button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # Headers are section titles, not truncatable captions.
        button.setToolTip(button.text())


#: What a QToolBox page inset from its edges: enough for the white cards
#: inside it to read as floating on the page rather than as filling it,
#: which is the whole point of a grey ground behind them. Top is taller
#: than the other three: the page sits directly under the toolbox's own
#: tab header ("Proprietà della figura" and its underline), and 10px
#: there read as the section title crowding the header rather than as
#: its own block.
MARGIN_TOOLBOX_PAGE: tuple[int, int, int, int] = (10, 16, 10, 12)


def apply_toolbox_page_metrics(
    toolbox: QtWidgets.QToolBox,
    *,
    margins: tuple[int, int, int, int] = MARGIN_TOOLBOX_PAGE,
) -> None:
    """Give a QToolBox's pages a ground of their own, and room inside it.

    Two things, both of which QSS alone cannot do here.

    The *ground*: a page is a plain QWidget, so a stylesheet background on
    it is parsed and never painted without ``WA_StyledBackground`` - the
    long-standing Qt gotcha. The pages are marked ``toolboxPage`` and each
    sheet then says what a page is: the window grey on macOS, where the
    white cards inside it are System Settings' floating grouped boxes, and
    white on Windows, where WinUI3 puts outlined cards on one continuous
    white page. Without it the page inherited whatever was behind it and a
    white card on a white page showed only its corners - which is how it
    was reported.

    The *room*: a page's own layout margins. QSS padding on a widget pads
    the box its background is painted in, not the layout inside it, so the
    inset that makes the cards float has to be set here.

    Call after adding the items, like the header metrics.
    """
    for index in range(toolbox.count()):
        page = toolbox.widget(index)
        if page is None:
            continue
        page.setProperty("toolboxPage", True)
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Only where the page set none of its own: the properties panels
        # already choose MARGIN_PANEL, and a helper called for the ground
        # colour has no business overruling a margin somebody picked. A
        # page with (0, 0, 0, 0) is one that never thought about it, which
        # is exactly the flat-to-the-edges case this is here to fix.
        layout = page.layout()
        if layout is not None and layout.contentsMargins().isNull():
            layout.setContentsMargins(*margins)
        repolish_widget(page)


def apply_dialog_shell(
    dialog: QtWidgets.QDialog,
    root_layout: QBoxLayout,
    *,
    size: str | QSize | None = "medium",
    margins: tuple[int, int, int, int] = MARGIN_DIALOG,
    spacing: int = SPACING_DEFAULT,
) -> None:
    """Give a dialog the shared root margin, spacing, and size.

    Every dialog root goes through here so padding is decided in one place.
    ``size`` accepts a key from DIALOG_SIZES, an explicit QSize, or None to
    leave the current size alone.
    """
    root_layout.setContentsMargins(*margins)
    root_layout.setSpacing(spacing)
    # Every dialog fits the screen it opens on, a laptop's included.
    from app.utils.screen_fit import fit_on_show

    fit_on_show(dialog)

    if size is None:
        return

    resolved = DIALOG_SIZES.get(size) if isinstance(size, str) else size
    if resolved is None:
        applogger.warning(
            "Unknown dialog size %r; leaving the dialog size untouched.",
            size,
        )
        return

    dialog.resize(resolved)

def configure_combo_width(
    combo: QComboBox,
    minimum_contents_length: int = 0,
) -> QComboBox:
    """Make a combo fill the available width without dictating a minimum.

    ``minimum_contents_length`` is the width the caller would *like*, in
    characters; it is capped at :data:`COMBO_MIN_CONTENTS_LENGTH` because the
    combo's minimum propagates all the way up to the window's minimum width.
    Text that no longer fits is elided by Qt, so the full value is put in the
    tooltip and kept in step with the selection.

    Two identical copies of this lived in the axis and figure property panels,
    with different caps - which is why the two panels could not be narrowed to
    the same width.
    """
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    characters = min(minimum_contents_length or COMBO_MIN_CONTENTS_LENGTH,
                     COMBO_MIN_CONTENTS_LENGTH)
    combo.setMinimumContentsLength(characters)

    # And an explicit pixel minimum, which is what actually takes effect.
    # QComboBox caches its minimum size hint and only recomputes it when the
    # item list changes, so setting the policy on a combo that is already
    # populated - which is exactly what relax_minimum_width does - leaves the
    # old, item-derived floor in place.  An explicit minimum size is consulted
    # by the layout ahead of the hint, so it does not depend on that timing.
    combo.setMinimumWidth(
        combo.fontMetrics().horizontalAdvance("x") * characters + _COMBO_CHROME_WIDTH
    )

    combo.setToolTip(combo.currentText())
    combo.currentTextChanged.connect(combo.setToolTip)
    return combo


def fit_spin_width(spin: QtWidgets.QAbstractSpinBox) -> None:
    """Give *spin* the least width that shows its widest value in full.

    ``relax_minimum_width`` lets everything in a panel shrink toward nothing
    so the panel itself can be as narrow as the splitter allows - which
    squeezed a spin box until its number was cut ("0.00" for 0.040, "16.2"
    for "16.26 cm"). A number is not something to clip: this sets the width
    it needs, from the widest of its minimum and maximum as displayed
    (prefix and suffix included) plus the style's own chrome, and marks the
    box so ``relax_minimum_width`` leaves it. A caller that would rather
    reflow than clip (:class:`app.widgets.pair_grid.PairGrid`) reads the
    result.
    """
    spin.ensurePolished()
    metrics = spin.fontMetrics()
    candidates = [spin.text()]
    # A range in the millions is "any number", not a width to reserve.
    if isinstance(spin, QtWidgets.QDoubleSpinBox):
        for bound in (spin.minimum(), spin.maximum()):
            if abs(bound) < 1e6:
                candidates.append(spin.prefix() + spin.textFromValue(bound) + spin.suffix())
    elif isinstance(spin, QtWidgets.QSpinBox):
        for whole in (spin.minimum(), spin.maximum()):
            if abs(whole) < 1e6:
                candidates.append(spin.prefix() + spin.textFromValue(whole) + spin.suffix())
    text_width = max(metrics.horizontalAdvance(text) for text in candidates)

    # The chrome is whatever the style takes out of the box besides the
    # text field: padding, border and the stepper.
    option = QtWidgets.QStyleOptionSpinBox()
    spin.initStyleOption(option)
    option.rect = QRect(0, 0, 200, max(spin.sizeHint().height(), 24))
    field = spin.style().subControlRect(
        QtWidgets.QStyle.ComplexControl.CC_SpinBox,
        option,
        QtWidgets.QStyle.SubControl.SC_SpinBoxEditField,
        spin,
    )
    chrome = 200 - field.width()
    spin.setMinimumWidth(text_width + chrome + 4)
    spin.setProperty(KEEP_MINIMUM_WIDTH, True)


#: Property that makes ``relax_minimum_width`` skip a widget whose minimum
#: was set to what its content needs (see ``fit_spin_width``).
KEEP_MINIMUM_WIDTH = "keepMinimumWidth"


def relax_minimum_width(root: QWidget, *, minimum: int = 0) -> QWidget:
    """Let *root* and everything inside it shrink to *minimum* pixels wide.

    A container is only as narrow as its widest child's minimum, and that
    minimum is rarely set on purpose: a QComboBox reserves room for N
    characters, a QAbstractScrollArea reserves a couple of rows, and the sum
    quietly becomes the floor for the whole panel - and, through the layout,
    for the window.  This walks the tree once and removes those implicit
    floors, so a splitter can be dragged to whatever the caller decided the
    real minimum is.

    Widgets whose width was pinned on purpose - ``setFixedWidth`` leaves the
    minimum and the maximum equal - are left alone: the activity rail and its
    icon buttons are that width because someone decided so, not by accident.

    ``setMinimumWidth(0)`` on its own is a no-op on any widget whose
    horizontal size policy is Expanding rather than Shrink (which is most
    plain container widgets - a property panel, a page, a card): Qt's own
    ``qSmartMinSize`` only lets an explicit minimum override
    ``max(sizeHint, minimumSizeHint)`` when that minimum is greater than
    zero. A combo's or a scroll area's own extra handling below already
    worked around this by changing what ``sizeHint`` itself returns, which
    is why relaxing *those* ever looked like it was working; every other
    widget's implicit floor was silently untouched. At least 1px, always -
    imperceptible, and the one value Qt actually honours as an override.

    Returns *root* so it can be used inline where a widget is expected.
    """
    for widget in [root, *root.findChildren(QWidget)]:
        if widget.minimumWidth() == widget.maximumWidth():
            continue
        if widget.property(KEEP_MINIMUM_WIDTH):
            continue
        widget.setMinimumWidth(max(minimum, 1))
        if isinstance(widget, QComboBox):
            # Through configure_combo_width, not setMinimumContentsLength
            # alone: the contents length is only consulted under an
            # AdjustToMinimumContentsLength* policy, so on a combo built
            # elsewhere setting it by itself changed nothing at all.
            configure_combo_width(widget)
        elif isinstance(widget, QtWidgets.QAbstractScrollArea):
            # Views size themselves from their content; without this a table
            # with one wide column dictates the panel width.
            widget.setSizeAdjustPolicy(
                QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
            )
    return root


def stdSizeAndlayout(
    obj: QWidget | QBoxLayout | QFormLayout | QScrollArea | QPlainTextEdit,
    minimum_contents_length: int = 0,
    visible_lines:int = 5,
    word_wrap:bool=True
) -> None:
    """
    Apply standard compact sizing/layout rules.

    The intent is:
    - widgets expand when there is room;
    - widgets remain shrinkable in narrow side panels;
    - combo boxes do not force the parent wider than necessary;
    - forms grow their field column properly;
    - scroll areas do not introduce horizontal scrolling.
    """
    if isinstance(obj, QPlainTextEdit):
        stdPlainTextEdit(obj,visible_lines=visible_lines,word_wrap=word_wrap)
        return
    if isinstance(obj, QScrollArea):
        obj.setMinimumWidth(0)
        obj.setWidgetResizable(True)
        obj.setFrameShape(QFrame.Shape.NoFrame)
        obj.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        obj.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        return

    if isinstance(obj, QFormLayout):
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setHorizontalSpacing(8)
        obj.setVerticalSpacing(10)
        obj.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        obj.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        obj.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        return

    if isinstance(obj, QBoxLayout):
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSpacing(8)
        return

    if not isinstance(obj, QWidget):
        return

    obj.setMinimumWidth(0)

    if isinstance(obj, QCheckBox):
        obj.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        return

    if isinstance(obj, QComboBox):
        obj.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        obj.setMinimumContentsLength(minimum_contents_length)
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return

    if isinstance(obj, QLineEdit):
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return

    if isinstance(obj, QPlainTextEdit):
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return

    obj.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )


def stdPlainTextEdit(
    editor: QPlainTextEdit,
    *,
    visible_lines: int = 5,
    word_wrap: bool = True,
) -> None:
    """Apply standard sizing and wrapping for compact multiline editors."""
    editor.setMinimumWidth(0)

    if word_wrap:
        editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        editor.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
    else:
        editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.NoWrap
        )
        editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

    editor.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Fixed,
    )

    editor.setFixedHeight(
        editor.fontMetrics().lineSpacing() * visible_lines + 12
    )


# ----------------------------------------------------------------------
