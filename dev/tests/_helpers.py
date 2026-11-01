"""Small helpers more than one test module needs."""
from __future__ import annotations

import time
from typing import Any


def wait_for_job(qapp: Any, dialog: Any, timeout: float = 60.0) -> None:
    """Until the dialog's background computation is done (the first statsmodels import takes seconds), then the
    events its result posts; fails the test if it never finishes."""
    deadline = time.monotonic() + timeout
    while dialog.job_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not dialog.job_running, "the computation did not finish"
    qapp.processEvents()
