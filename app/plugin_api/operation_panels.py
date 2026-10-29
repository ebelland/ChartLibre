"""The pages of an analysis window that hold values the calculation reads: the base they share.

Parameters and Charts are such pages. Each gives the window its values by name, takes them back, and
refreshes what depends on another choice. ParameterForm is the one that builds them from declarations
(parameter_spec); anything else that gives values (a page a plugin builds by hand) derives from this too,
so the window reads every one of them the same way (merged_values).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class OperationPanel:
    """A page of an analysis window that gives the calculation its values.

    ``widget`` is the page itself. The default ``refresh_visibility`` does nothing: a page with rows that
    depend on other choices overrides it.
    """

    widget: Any

    def values(self) -> dict[str, Any]:
        """The page's current values, keyed by name."""
        raise NotImplementedError

    def set_values(self, values: Mapping[str, Any]) -> None:
        """Put back the values *values* names; those it does not own are left alone."""
        raise NotImplementedError

    def refresh_visibility(self) -> None:
        """Show or hide what depends on another choice; nothing by default."""


def merged_values(panels: Iterable[OperationPanel]) -> dict[str, Any]:
    """Every panel's values in one dict, the later panels' names winning a clash."""
    merged: dict[str, Any] = {}
    for panel in panels:
        merged.update(panel.values())
    return merged
