"""What to ask next: links at the end of a report that open the next analysis, its columns cast (G-02).

The quick answers lead on to one another as a person's questions do:
are there differences? -> what has the most impact? -> a model on the
strongest factors; and each can be asked again with another column. A
report lists these as links (``https://chartlibre.invalid/next?...`` - a
web address, which every viewer hands to the page's own navigation check,
on a host the .invalid domain guarantees never exists); clicked - in the
dialog's results or in a chart's notes, where the report is kept - the main
window opens that analysis on the same table with the columns already in
their roles. The link carries everything, so it still works in a report
reopened days later.
"""
from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urlsplit

from app.utils import report_html
from app.utils.i18n import _

#: Where a "What next?" link points: intercepted, never fetched (RFC 2606: .invalid never resolves).
HOST = "chartlibre.invalid"
PATH = "/next"
#: Columns no question is about: the selection and hiding flags, the row id.
_NOT_DATA = {"Hide", "Selected", "_id"}


@dataclass(slots=True)
class NextStep:
    """A question to ask next: its label, the analysis (by Name), the table and the columns' roles.

    ``roles`` may hold "_x": a model's explanatory columns, cast as its effects.
    """

    label: str
    operation: str
    table: str
    roles: dict[str, list[str]]
    kinds: dict[str, str] = field(default_factory=dict)


def step_url(step: NextStep) -> str:
    query = urlencode({"op": step.operation, "table": step.table, "roles": json.dumps(step.roles),
                       "kinds": json.dumps(step.kinds)})
    return f"https://{HOST}{PATH}?{query}"


def is_step_url(url: str) -> bool:
    parts = urlsplit(str(url))
    return parts.netloc == HOST and parts.path == PATH


def parse_step(url: str) -> NextStep | None:
    """The step a link stands for, or None for any other link."""
    parts = urlsplit(str(url))
    if parts.netloc != HOST or parts.path != PATH:
        return None
    values = {key: items[0] for key, items in parse_qs(parts.query).items()}
    try:
        roles = {str(k): [str(c) for c in v] for k, v in json.loads(values.get("roles", "{}")).items()}
        kinds = {str(k): str(v) for k, v in json.loads(values.get("kinds", "{}")).items()}
    except (ValueError, AttributeError, TypeError):
        return None
    if not values.get("op") or not values.get("table"):
        return None
    return NextStep(label="", operation=values["op"], table=values["table"], roles=roles, kinds=kinds)


def section_html(steps: Iterable[NextStep]) -> str:
    """The "What next?" section: one link per step; empty when there is none."""
    items = [f"<li><a href=\"{html.escape(step_url(step), quote=True)}\">{html.escape(step.label)}</a></li>"
             for step in steps]
    return report_html.section(_("What next?"), "<ul>" + "".join(items) + "</ul>") if items else ""


def other_columns(columns: Iterable[str], *, leave_out: Iterable[str] = (), limit: int = 8) -> list[str]:
    """The table's columns a question could be about, without *leave_out* and the generated flags."""
    skipped = set(leave_out) | _NOT_DATA
    return [str(c) for c in columns if str(c) not in skipped and not str(c).startswith("__")][:limit]


def is_level(kinds: Mapping[str, str], column: str) -> bool:
    """True when *column* is cast as levels (nominal or ordinal), not as a measurement."""
    return kinds.get(column) in ("nominal", "ordinal")
