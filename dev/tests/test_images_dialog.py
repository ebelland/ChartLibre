"""The Pictures dialog: list, details, rename (series follow), remove."""
import json

from app.dialogs.images_dialog import ImagesDialog

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'


def test_rename_follows_series_and_details(qapp, repo, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    repo.add_image("sq", SVG)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="t", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query="SELECT 1 AS x, 1 AS y",
                                  roles={"x": "x", "y": "y"}, style={"marker": "image:sq"})
    dialog = ImagesDialog(repo)
    assert dialog._list.count() == 1
    dialog._license.setText("MIT")
    dialog._save_details()
    assert repo.get_image("sq").license == "MIT"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("square", True))
    dialog._rename()
    assert repo.image_names() == ["square"]
    style = json.loads(repo._con.execute("SELECT style_json FROM __series_descriptors__").fetchone()[0])
    assert style["marker"] == "image:square"
    repo.undo_last()
    assert repo.image_names() == ["sq"]
