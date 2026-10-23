"""What every analysis window shares, whichever kind it is (todo O-01).

The data operations (DataOperationDialogBase: a chart's series) and the table
analyses (TableOperationDialogBase: a table's columns) differ in what they
read; around that, they are the same window and behave the same way. What
they both did, each in its own words, lives here once:

* the Results pane: a report shown as markup, plain text escaped
  (show_results), and a failure said there in red, not only in the log
  (show_failure);
* the Apply in the project's history (record_in_history): what was read,
  written and reported, with the dialog and the version that did it;
* the "Draw on" choices both Charts tabs offer, worded once.

A base derives from it beside QDialog and says where its results pane is
(results_widget). No Qt object of its own: a mixin.
"""
from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from app import APP_VERSION
from app.logs.logger import applogger
from app.utils.i18n import _
from app.widgets.html_results import looks_like_html


class OperationShell:
    """The parts of an analysis window both kinds share."""

    #: The places a result can be drawn, the same words in both Charts tabs.
    DEST_SAME_AXIS: str = "same_axis"
    DEST_NEW_AXIS: str = "new_axis"
    DEST_NEW_FIGURE: str = "new_figure"

    @classmethod
    def destination_label(cls, value: str) -> str:
        """The Charts tab's wording of a "Draw on" choice."""
        return {cls.DEST_SAME_AXIS: _("Current axis"), cls.DEST_NEW_AXIS: _("New axis in the current figure"),
                cls.DEST_NEW_FIGURE: _("New figure")}[value]

    # ------------------------------------------------------------------
    # The Results pane
    # ------------------------------------------------------------------

    def results_widget(self) -> Any:
        """The base's results pane (an HtmlResultsView)."""
        raise NotImplementedError

    def show_results(self, content: str) -> None:
        """Show a report in the Results pane: markup rendered, plain text kept as it is written."""
        widget = self.results_widget()
        text = str(content or "")
        if looks_like_html(text):
            widget.setHtml(text)
        elif hasattr(widget, "setContent"):
            widget.setContent(text)
        else:
            widget.setText(text)

    def show_failure(self, error: BaseException | str, *, headline: str = "") -> None:
        """Say in the Results pane why nothing came of the run: the log alone left the pane unchanged."""
        message = str(error) or type(error).__name__
        self.results_widget().setHtml(
            "<p style='color:#a40000'><b>" + html.escape(headline or _("Nothing was drawn.")) + "</b></p>"
            + "<p>" + html.escape(message) + "</p>")

    # ------------------------------------------------------------------
    # The project's history
    # ------------------------------------------------------------------

    def record_in_history(
        self,
        *,
        operation: str,
        parameters: Mapping[str, Any],
        entries: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        report: str,
    ) -> None:
        """An Apply in the project's ``__operations__``. Logged, never raised: the record must not lose the result."""
        try:
            self._repo.record_operation(  # type: ignore[attr-defined]
                operation=operation,
                dialog=f"{type(self).__module__.rsplit('.', 1)[-1]}:{type(self).__name__}",
                parameters=dict(parameters),
                entries=dict(entries),
                sources=[dict(source) for source in sources],
                results=[dict(item) for item in results],
                app_version=str(APP_VERSION),
                report=report,
            )
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Could not record %s in the project's history", operation)
