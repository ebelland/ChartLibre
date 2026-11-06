"""New plot's chart types, on two pages: the recent ones and the categories, then one category's charts.

A search lists its matches on the second page, whatever their category. The name kept is the renderer's
English one (the chart_type written to the database); only what is read goes through tr().
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import mark_editor_panel
from app.utils.config import get_constant, get_value, set_value
from app.utils.i18n import _, tr

#: How many charts "Recent" shows.
NUM_RECENT: int = get_constant("num_recent_charts", 3)
_RECENT_KEY = "recent_chart_types"

#: Matplotlib's own families first, in its order (matplotlib.org/stable/plot_types), then the application's.
CATEGORY_ORDER: tuple[str, ...] = (
    "Pairwise data",
    "Statistical distributions",
    "Gridded data",
    "Irregularly gridded data",
    "3D and volumetric data",
    "Diagnostic plots",
    "Quality and process",
    "Specialized charts",
)

_ICON = QSize(44, 30)
_KIND, _VALUE = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1


def recent_charts(known: Sequence[str]) -> list[str]:
    """The charts last created, newest first; one no longer shipped is left out."""
    stored = get_value(_RECENT_KEY, [])
    names = [str(name) for name in stored] if isinstance(stored, list) else []
    return [name for name in names if name in known][:NUM_RECENT]


def remember_chart(name: str) -> None:
    """*name* was just used: the first of the recent ones."""
    stored = get_value(_RECENT_KEY, [])
    names = [str(item) for item in stored] if isinstance(stored, list) else []
    set_value(_RECENT_KEY, [name, *[item for item in names if item != name]][:NUM_RECENT])


def _draw(painter: QPainter, category: str, ink: QColor) -> None:
    """The family's picture, on a 44 x 30 canvas."""
    soft = QColor(ink)
    soft.setAlphaF(0.45)
    painter.setPen(QPen(ink, 1.5))
    painter.setBrush(ink)
    if category == "Pairwise data":
        for x, y in ((6, 24), (13, 17), (20, 19), (27, 11), (34, 9), (39, 4)):
            painter.drawEllipse(QPointF(x, y), 1.8, 1.8)
    elif category == "Statistical distributions":
        painter.setPen(Qt.PenStyle.NoPen)
        for index, height in enumerate((5, 12, 22, 26, 17, 8, 3)):
            painter.drawRect(QRectF(4 + index * 5.3, 28 - height, 4.3, height))
    elif category == "Gridded data":
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(4):
            for column in range(6):
                shade = QColor(ink)
                shade.setAlphaF(0.15 + 0.85 * ((row + column) % 6) / 5)
                painter.setBrush(shade)
                painter.drawRect(QRectF(5 + column * 5.8, 3 + row * 6, 5.3, 5.5))
    elif category == "Irregularly gridded data":
        points = [QPointF(5, 25), QPointF(17, 4), QPointF(22, 18), QPointF(38, 6), QPointF(39, 26)]
        painter.setBrush(soft)
        for triangle in ((0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 2, 4)):
            painter.drawPolygon(QPolygonF([points[index] for index in triangle]))
    elif category == "3D and volumetric data":
        painter.setBrush(soft)
        painter.drawPolygon(QPolygonF([QPointF(8, 11), QPointF(22, 4), QPointF(36, 11), QPointF(22, 18)]))
        painter.drawPolygon(QPolygonF([QPointF(8, 11), QPointF(22, 18), QPointF(22, 28), QPointF(8, 21)]))
        painter.setBrush(ink)
        painter.drawPolygon(QPolygonF([QPointF(36, 11), QPointF(22, 18), QPointF(22, 28), QPointF(36, 21)]))
    elif category == "Diagnostic plots":
        painter.setPen(QPen(soft, 1.5))
        painter.drawLine(QPointF(5, 26), QPointF(39, 4))
        painter.setPen(QPen(ink, 1.5))
        for x, y in ((8, 26), (14, 20), (20, 17), (26, 12), (32, 10), (37, 3)):
            painter.drawEllipse(QPointF(x, y), 1.5, 1.5)
    elif category == "Quality and process":
        painter.setPen(QPen(soft, 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(4, 7), QPointF(40, 7))
        painter.drawLine(QPointF(4, 23), QPointF(40, 23))
        painter.setPen(QPen(ink, 1.5))
        painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in
                                        ((5, 15), (12, 11), (19, 18), (26, 13), (33, 19), (39, 14))]))
    else:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(5, 4, 15, 22))
        painter.setBrush(soft)
        painter.drawRect(QRectF(22, 4, 17, 12))
        painter.drawRect(QRectF(22, 18, 8, 8))
        painter.drawRect(QRectF(32, 18, 7, 8))


def _category_pixmap(category: str, ink: QColor) -> QPixmap:
    ratio = 2
    pixmap = QPixmap(_ICON * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _draw(painter, category, ink)
    painter.end()
    return pixmap


def category_icon(category: str, ink: QColor, selected_ink: QColor) -> QIcon:
    """A small picture of what the family's charts look like; on a selected row, in that row's text colour."""
    icon = QIcon(_category_pixmap(category, ink))
    icon.addPixmap(_category_pixmap(category, selected_ink), QIcon.Mode.Selected)
    return icon


class RendererPicker(QWidget):
    """The search field above the two pages; ``changed`` says the name of the chart chosen."""

    changed = Signal(str)

    def __init__(self, charts: Sequence[Mapping], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._charts = list(charts)
        self._current = ""
        self._category = ""

        self.search = QLineEdit()
        self.search.setPlaceholderText(_("Search chart types..."))
        self.search.setToolTip(_("Filter by name or category. Sections with no match are hidden."))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._searched)

        self.home = QListWidget()
        self.home.setIconSize(_ICON)
        self.home.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.home.setWordWrap(True)
        mark_editor_panel(self.home)
        self.home.itemClicked.connect(self._home_chosen)
        self.home.itemActivated.connect(self._home_chosen)

        self.back = QToolButton()
        self.back.setArrowType(Qt.ArrowType.LeftArrow)
        self.back.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.back.setText(_("Back"))
        self.back.setToolTip(_("Back to the categories"))
        self.back.clicked.connect(self.show_home)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600;")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.back, 0)
        header.addWidget(self.title, 1)

        self.charts = QListWidget()
        self.charts.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mark_editor_panel(self.charts)
        self.charts.currentItemChanged.connect(self._chart_chosen)
        list_page = QWidget()
        list_layout = QVBoxLayout(list_page)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addLayout(header)
        list_layout.addWidget(self.charts, 1)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.home)
        self.pages.addWidget(list_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.search, 0)
        layout.addWidget(self.pages, 1)
        self._fill_home()

    # ------------------------------------------------------------------
    @property
    def current(self) -> str:
        """The chart chosen."""
        return self._current

    def categories(self) -> list[str]:
        """Known families first, in order; one nobody has heard of still appears, at the end."""
        present = {str(chart.get("category") or "") for chart in self._charts}
        return [name for name in CATEGORY_ORDER if name in present] + sorted(present - set(CATEGORY_ORDER))

    def select(self, name: str, *, show: bool = True) -> bool:
        """Choose the chart *name*; whether it exists. *show*: go to its category's list."""
        chart = next((item for item in self._charts if item["value"] == name), None)
        if chart is None:
            return False
        if show:
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
            self.show_category(str(chart.get("category") or ""))
        self._set_current(name)
        return True

    def show_home(self) -> None:
        """The first page: back from a search as well, so the search is cleared."""
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.pages.setCurrentIndex(0)

    def show_category(self, category: str) -> None:
        """The second page, with the family's charts."""
        self._category = category
        self.title.setText(tr(category))
        self._fill_charts([chart for chart in self._charts if str(chart.get("category") or "") == category])
        self.pages.setCurrentIndex(1)

    # ------------------------------------------------------------------
    def _fill_home(self) -> None:
        ink = self.palette().color(QPalette.ColorRole.Highlight)
        selected_ink = self.palette().color(QPalette.ColorRole.HighlightedText)
        known = [str(chart["value"]) for chart in self._charts]
        by_name = {str(chart["value"]): chart for chart in self._charts}

        def heading(text: str) -> None:
            item = QListWidgetItem(text)
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setForeground(self.palette().color(QPalette.ColorRole.WindowText))
            self.home.addItem(item)

        recent = recent_charts(known)
        if recent:
            heading(_("Recent"))
            for name in recent:
                item = QListWidgetItem(category_icon(str(by_name[name].get("category") or ""), ink, selected_ink), tr(name))
                item.setData(_KIND, "chart")
                item.setData(_VALUE, name)
                item.setToolTip(tr(str(by_name[name].get("description") or "")))
                self.home.addItem(item)
            heading(_("Categories"))
        for category in self.categories():
            count = sum(str(chart.get("category") or "") == category for chart in self._charts)
            item = QListWidgetItem(category_icon(category, ink, selected_ink), f"{tr(category)} ({count})")
            item.setData(_KIND, "category")
            item.setData(_VALUE, category)
            self.home.addItem(item)

    def _home_chosen(self, item: QListWidgetItem) -> None:
        if item.data(_KIND) == "chart":
            self.select(str(item.data(_VALUE)))
        elif item.data(_KIND) == "category":
            self.show_category(str(item.data(_VALUE)))

    def _fill_charts(self, charts: Sequence[Mapping]) -> None:
        self.charts.blockSignals(True)
        self.charts.clear()
        for chart in charts:
            item = QListWidgetItem(tr(str(chart["value"])))
            item.setData(_VALUE, chart["value"])
            item.setToolTip(tr(str(chart.get("description") or "")))
            self.charts.addItem(item)
            if chart["value"] == self._current:
                self.charts.setCurrentItem(item)
        self.charts.blockSignals(False)

    def _chart_chosen(self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        if item is not None:
            self._set_current(str(item.data(_VALUE)))

    def _set_current(self, name: str) -> None:
        if name != self._current:
            self._current = name
            self.changed.emit(name)
        for row in range(self.charts.count()):
            item = self.charts.item(row)
            if item.data(_VALUE) == name:
                self.charts.blockSignals(True)
                self.charts.setCurrentItem(item)
                self.charts.blockSignals(False)
                return

    def _searched(self, text: str) -> None:
        """Matches of both spellings, of the chart's name or its category's, on the second page."""
        needle = str(text or "").strip().lower()
        if not needle:
            self.pages.setCurrentIndex(0)
            return

        def matches(chart: Mapping) -> bool:
            category = str(chart.get("category") or "")
            names = (str(chart["value"]), tr(str(chart["value"])), category, tr(category))
            return any(needle in name.lower() for name in names)

        found = [chart for chart in self._charts if matches(chart)]
        self.title.setText(_("Results ({count})").format(count=len(found)))
        self._fill_charts(found)
        self.pages.setCurrentIndex(1)
