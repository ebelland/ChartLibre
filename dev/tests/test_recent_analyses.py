"""The rail's Recent section: the analyses opened last, newest first, as many as Settings says."""
from __future__ import annotations

from app.utils import recent_analyses
from app.utils.config import set_value

KNOWN = [{"name": f"Op{index}", "value": f"Analysis {index}"} for index in range(6)]


def test_newest_first_without_repeats_and_as_many_as_set():
    set_value(recent_analyses.LIST_KEY, [])
    set_value(recent_analyses.COUNT_KEY, 3)
    for name in ("Op1", "Op2", "Op3", "Op1", "Op4"):
        recent_analyses.remember({"name": name})
    assert [op["name"] for op in recent_analyses.operations(KNOWN)] == ["Op4", "Op1", "Op3"]
    assert not recent_analyses.remember({"name": "Op4"})  # already first: nothing to redraw
    assert not recent_analyses.remember({"name": "NewPlotTabDialog"})
    set_value(recent_analyses.COUNT_KEY, 0)
    assert recent_analyses.operations(KNOWN) == []
    set_value(recent_analyses.COUNT_KEY, 3)
    set_value(recent_analyses.LIST_KEY, ["Gone", "Op2"])  # an analysis no longer installed is left out
    assert [op["name"] for op in recent_analyses.operations(KNOWN)] == ["Op2"]
