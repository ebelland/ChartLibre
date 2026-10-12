"""Figure properties: Apply to all figures gives every figure the current style, undoably."""
from __future__ import annotations

from app.data.sqlite_repo import SqliteRepo
from app.dialogs import main_chart_properties
from app.main_window.main_window import MainWindow


def test_every_figure_takes_the_style_and_undo_brings_theirs_back(qapp, tmp_path, monkeypatch) -> None:
    repo = SqliteRepo(db_path=tmp_path / "s.dhub")
    first = int(repo.create_figure_descriptor(name="a", nrows=1, ncols=1, options={"mpl_style": "lines.linewidth: 3\n"}))
    second = int(repo.create_figure_descriptor(name="b", nrows=1, ncols=1, options={}))
    window = MainWindow(repo, tmp_path / "s.dhub")
    try:
        monkeypatch.setattr(main_chart_properties, "ask", lambda *_a, **_k: True)
        window._on_style_applied_to_all("axes.grid: True\n")
        assert all(repo.get_figure_options(f).get("mpl_style") == "axes.grid: True\n" for f in (first, second))
        repo.undo_last()
        assert repo.get_figure_options(first).get("mpl_style") == "lines.linewidth: 3\n"
        assert "mpl_style" not in repo.get_figure_options(second)
    finally:
        window.close()
        repo.close()
