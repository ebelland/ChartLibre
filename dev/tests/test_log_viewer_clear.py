"""The log viewer's Clear: asks, then empties the log file and the view."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

import app.dialogs.log_viewer_dialog as viewer_module
import app.logs.logger as logger_module


@pytest.fixture
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "datahub.log"
    handler = RotatingFileHandler(path, encoding="utf-8")
    logger = logging.getLogger(logger_module._LOGGER_NAME)
    logger.addHandler(handler)
    monkeypatch.setattr(logger_module, "LOG_FILE", path)
    monkeypatch.setattr(viewer_module, "LOG_FILE", path)
    handler.stream.write("old line\n")
    handler.flush()
    yield path, handler
    logger.removeHandler(handler)
    handler.close()


def test_clear_empties_the_file_through_its_handler(qapp, log_file, monkeypatch: pytest.MonkeyPatch) -> None:
    path, handler = log_file
    others = [h for h in logging.getLogger(logger_module._LOGGER_NAME).handlers if isinstance(h, RotatingFileHandler) and h is not handler]
    for other in others:  # only the test's own file handler, for this test
        logging.getLogger(logger_module._LOGGER_NAME).removeHandler(other)
    try:
        dialog = viewer_module.LogViewerDialog(None)  # pyright: ignore[reportArgumentType]
        assert "old line" in dialog._viewer.toPlainText()
        monkeypatch.setattr(viewer_module, "ask", lambda *_a, **_k: False)
        dialog._clear_log()
        assert path.read_text(encoding="utf-8") == "old line\n"
        monkeypatch.setattr(viewer_module, "ask", lambda *_a, **_k: True)
        dialog._clear_log()
        assert path.read_text(encoding="utf-8") == "" and dialog._viewer.toPlainText() == ""
        handler.stream.write("new line\n")
        handler.flush()
        assert path.read_text(encoding="utf-8") == "new line\n"  # still writing to the same file
        dialog.close()
    finally:
        for other in others:
            logging.getLogger(logger_module._LOGGER_NAME).addHandler(other)


def test_the_clear_button_is_red(qapp, log_file) -> None:
    dialog = viewer_module.LogViewerDialog(None)  # pyright: ignore[reportArgumentType]
    from PySide6.QtWidgets import QPushButton

    clear = [b for b in dialog.findChildren(QPushButton) if b.property("destructive")]
    assert len(clear) == 1
    dialog.close()
