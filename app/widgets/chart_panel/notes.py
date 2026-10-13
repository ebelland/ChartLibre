"""The chart panel's notes: the HTML results pane under the chart."""
from __future__ import annotations

from app.widgets.chart_panel.constants import (
    CHART_AREA_MIN_HEIGHT,
    NOTES_FIRST_OPEN_MIN_HEIGHT,
    NOTES_INITIAL_FRACTION,
)


class NotesMixin:
    """The chart panel's notes: the HTML results pane under the chart. A part of ChartPanel (see panel.py)."""

    # ------------------------------------------------------------------
    # Notes / HTML results pane
    # ------------------------------------------------------------------
    def set_notes_html(self, markup: str, *, append: bool = False) -> None:
        """Show HTML results below the chart, revealing the pane if needed."""
        markup = str(markup or "")
        if not markup.strip():
            return

        if append and self._notes_html.strip():
            self._notes_html = f"{self._notes_html}<hr>{markup}"
        else:
            self._notes_html = markup

        self._notes_view.setHtml(self._notes_html)
        self._set_notes_visible(True)
        self._persist_view_state()

    def notes_html(self) -> str:
        """Return the markup currently shown below the chart."""
        return self._notes_html

    def clear_notes(self) -> None:
        """Empty and fully collapse the notes pane."""
        self._notes_html = ""
        self._notes_view.clear()
        self._set_notes_visible(False)
        self._persist_view_state()

    def _restore_notes_state(self) -> None:
        """Re-show persisted notes and splitter position for this figure."""
        markup = getattr(self, "_pending_notes_html", "")
        if not markup:
            return

        self._notes_html = markup
        self._notes_view.setHtml(markup)
        self._notes_view.setVisible(True)

        sizes = getattr(self, "_pending_notes_sizes", None)
        if sizes:
            self._content_split.setSizes(sizes)

    def _set_notes_visible(self, visible: bool) -> None:
        """Show/hide notes and let users collapse the pane completely."""
        was_visible = self._notes_view.isVisible()
        self._notes_view.setVisible(visible)
        if not visible:
            sizes = self._content_split.sizes()
            chart_height = max(CHART_AREA_MIN_HEIGHT, sum(sizes) if sizes else self.height())
            self._content_split.setSizes([chart_height, 0])
            self._schedule_canvas_geometry_sync(redraw=True, late=True)
            return
        if was_visible:
            return

        total = max(self.height(), CHART_AREA_MIN_HEIGHT + NOTES_FIRST_OPEN_MIN_HEIGHT)
        notes_height = max(NOTES_FIRST_OPEN_MIN_HEIGHT, int(total * NOTES_INITIAL_FRACTION))
        self._content_split.setSizes([total - notes_height, notes_height])
        self._schedule_canvas_geometry_sync(redraw=True, late=True)
