"""What a series operation works with: its models, result specs and jobs, and the generated-table names."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from app.analysis import Stopped

_TABLE_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")

# Every table an operation writes starts with this.  One character, and the
# source list can hide the whole class of them: a project with six fits and a
# spectral analysis has more generated tables than imported ones, and the
# imported ones are what the user is looking for.
GENERATED_TABLE_PREFIX: str = "_"

def generated_table_name(raw: str, *, fallback: str = "Result") -> str:
    """Return a safe generated-table name, prefixed and never bare.

    Callers pass whatever they assembled - an axis id, a series name, a model -
    and get back something SQLite accepts and the source list can recognise.
    """
    safe = _TABLE_SAFE_RE.sub("_", str(raw).strip()).strip("_") or fallback
    return f"{GENERATED_TABLE_PREFIX}{safe}"

@dataclass(frozen=True, slots=True, kw_only=True)


class OperationModel:
    """One model an operation offers, as the Model combo lists it.

    The base needs only where its documentation is. An operation that needs
    more per model - which parameters it shows, which function it calls -
    subclasses this and adds the fields (FilterModel, smoothing's ModelSpec),
    so everything about one model sits in one entry of the MODELS dict
    rather than in a tuple of names beside a dict of links beside a set of
    the models that need an extra row.
    """

    doc_title: str = ""
    doc_url: str = ""
    #: Models of different groups are separated by a line in the combo.
    group: str = ""

    @property
    def doc(self) -> tuple[str, str]:
        """(title, url) for the Docs link."""
        return self.doc_title, self.doc_url


@dataclass(slots=True)
class ResultSeriesSpec:
    """Repository descriptor for one generated/updated chart series."""
    name: str
    sql_query: str
    roles: Mapping[str, Any]
    style: Mapping[str, Any]

class OperationJob(Protocol):
    """What prepare_job returns: an operation's inputs, read off the window.

    ``run`` is the calculation alone - no widget, no repository - so it can
    run on a worker thread. It polls *should_stop* where it can and raises
    app.analysis.Stopped when told to.
    """

    def run(self, should_stop: Callable[[], bool] | None = None) -> Any: ...


@dataclass(slots=True)
class CallJob:
    """A job that is one call: for an operation that computes all its series together.

    ``compute`` is prepared on the GUI thread with everything it needs
    already read, and must touch neither a widget nor the repository.
    """

    compute: Callable[[], Any]

    def run(self, should_stop: Callable[[], bool] | None = None) -> Any:
        if should_stop is not None and should_stop():
            raise Stopped()
        return self.compute()


@dataclass(slots=True)
class SeriesOutcome:
    """What a SeriesJob hands back: one outcome per series that worked, and why the others did not."""

    outcomes: list[tuple[str, Any]]
    errors: list[str]


@dataclass(slots=True)
class SeriesJob:
    """The usual job: every selected series read on the GUI thread, each computed on the worker.

    ``inputs`` holds (series name, data) pairs, read off the repository by
    ``read_series``; ``compute`` is the dialog's ``compute_series``, which
    must touch neither a widget nor the repository. ``progress`` is written
    here and read by the dialog's timer - ``[done, total]`` - which is how
    the bar moves without a signal crossing threads for every series.
    """

    inputs: list[tuple[str, Any]]
    settings: Any
    compute: Callable[[str, Any, Any], Any]
    errors: list[str] = field(default_factory=list)
    progress: list[int] = field(default_factory=lambda: [0, 0])

    def run(self, should_stop: Callable[[], bool] | None = None) -> SeriesOutcome:
        total = len(self.inputs)
        self.progress[:] = [0, total]
        outcomes: list[tuple[str, Any]] = []
        errors = list(self.errors)
        for done, (name, data) in enumerate(self.inputs):
            if should_stop is not None and should_stop():
                raise Stopped()
            try:
                outcomes.append((name, self.compute(name, data, self.settings)))
            except Stopped:
                raise
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")
            self.progress[:] = [done + 1, total]
        return SeriesOutcome(outcomes, errors)
