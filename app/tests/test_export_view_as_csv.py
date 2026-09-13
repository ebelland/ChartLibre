""""Export the current view as data" (todo.txt N-20).

The rows on screen, not the whole series: _current_view_rows reads each
axes' own xlim/ylim rather than the source table, so a zoomed-in panel next
to an untouched one exports only the zoomed rows from the first - the same
"what you see" scope save-as-picture and copy-to-clipboard already have,
just as a CSV instead of an image.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets import chart_panel as chart_panel_module
from app.widgets.chart_panel import ChartPanel


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    built = ChartPanel(repo, figure_id)
    built.resize(600, 400)
    built.show()
    qapp.processEvents()
    yield built
    built.close()
    repo.undo_store.discard_file()


def test_the_full_view_lists_every_point(panel) -> None:
    rows = panel._current_view_rows()
    assert sorted(row["x"] for row in rows) == list(range(10))
    assert all(row["series"] == "s" for row in rows)
    assert all(row["axis"] == rows[0]["axis"] for row in rows)


def test_zooming_in_shrinks_the_exported_view(panel) -> None:
    axes = panel._figure.axes[0]
    axes.set_xlim(2.5, 6.5)
    axes.set_ylim(2.5, 6.5)

    rows = panel._current_view_rows()

    assert sorted(row["x"] for row in rows) == [3.0, 4.0, 5.0, 6.0]


def test_exporting_writes_a_csv_with_a_header(panel, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    destination = tmp_path / "view.csv"
    monkeypatch.setattr(
        chart_panel_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(destination), "")),
    )

    panel.export_view_as_csv()

    with open(destination, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert set(rows[0].keys()) == {"axis", "series", "x", "y"}


def test_cancelling_the_dialog_writes_nothing(panel, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        chart_panel_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )

    panel.export_view_as_csv()  # must not raise

    assert list(tmp_path.iterdir()) == []


def test_an_empty_view_reports_instead_of_writing_an_empty_file(
    panel, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    axes = panel._figure.axes[0]
    axes.set_xlim(50.0, 60.0)
    axes.set_ylim(50.0, 60.0)
    destination = tmp_path / "view.csv"
    monkeypatch.setattr(
        chart_panel_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(destination), "")),
    )
    reported: list[str] = []
    monkeypatch.setattr(
        chart_panel_module, "show_message",
        lambda parent, message_id, **fields: reported.append(message_id),
    )

    panel.export_view_as_csv()

    assert reported == ["chart.export_view_empty"]
    assert not destination.exists()
