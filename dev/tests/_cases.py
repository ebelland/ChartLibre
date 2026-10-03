"""Many cases in one test: every case runs, every failure is reported together."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def check_all(check: Callable[..., Any], cases: Iterable[Any]) -> None:
    """Run *check* on each case (a tuple of arguments, or one argument).

    Instead of one parametrised test per case: the suite stays short, and a
    failure still names each case that failed and why.
    """
    failures = []
    for case in cases:
        arguments = case if isinstance(case, tuple) else (case,)
        try:
            check(*arguments)
        except AssertionError as exc:
            failures.append(f"{arguments!r}: {exc}")
    assert not failures, f"{len(failures)} case(s) failed:\n" + "\n".join(failures)
