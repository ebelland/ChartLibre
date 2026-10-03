"""The credits name every work by others that ships with the application."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.utils import credits
from dev.tests._cases import check_all

ROOT = Path(__file__).resolve().parents[2]


_CASES_EVERY_WORK_SAYS_WHOSE_IT_IS_AND_WHERE_TO_FIND_IT = [(case,) for case in credits.BUNDLED + credits.ICONS + credits.DATA]


def _every_work_says_whose_it_is_and_where_to_find_it(work: credits.Work) -> None:
    assert work.author and work.use and work.license
    assert work.url.startswith("https://")


def test_every_work_says_whose_it_is_and_where_to_find_it() -> None:
    check_all(_every_work_says_whose_it_is_and_where_to_find_it, _CASES_EVERY_WORK_SAYS_WHOSE_IT_IS_AND_WHERE_TO_FIND_IT)


_CASES_MIT_NOTICES_TRAVEL_WITH_THE_COPIED_FILES = [(case,) for case in ["mplstyles/LICENSE-SciencePlots"]]


def _mit_notices_travel_with_the_copied_files(notice: str) -> None:
    text = (ROOT / notice).read_text(encoding="utf-8")
    assert "MIT License" in text and "Permission is hereby granted" in text


def test_mit_notices_travel_with_the_copied_files() -> None:
    check_all(_mit_notices_travel_with_the_copied_files, _CASES_MIT_NOTICES_TRAVEL_WITH_THE_COPIED_FILES)


def test_the_page_links_each_work(qapp) -> None:
    from app.dialogs.credits_dialog import credits_html

    page = credits_html()
    for work in credits.BUNDLED + credits.ICONS + credits.DATA:
        assert f"href='{work.url}'" in page
