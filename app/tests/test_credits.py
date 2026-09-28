"""The credits name every work by others that ships with the application."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.utils import credits

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("work", credits.BUNDLED + credits.ICONS + credits.DATA, ids=lambda w: w.name)
def test_every_work_says_whose_it_is_and_where_to_find_it(work: credits.Work) -> None:
    assert work.author and work.use and work.license
    assert work.url.startswith("https://")


@pytest.mark.parametrize(
    "notice",
    ["mplstyles/LICENSE-SciencePlots", "app/styles/LICENSE-GTRONICK-QSS"],
)
def test_mit_notices_travel_with_the_copied_files(notice: str) -> None:
    text = (ROOT / notice).read_text(encoding="utf-8")
    assert "MIT License" in text and "Permission is hereby granted" in text


def test_the_page_links_each_work(qapp) -> None:
    from app.dialogs.credits_dialog import credits_html

    page = credits_html()
    for work in credits.BUNDLED + credits.ICONS + credits.DATA:
        assert f"href='{work.url}'" in page
