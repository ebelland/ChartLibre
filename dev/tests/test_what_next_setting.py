"""The "What next?" section of a table analysis is on by default and switched off by its setting."""
from __future__ import annotations

from app.plugin_api.table_operations import next_steps


def test_what_next_is_on_by_default(monkeypatch) -> None:
    monkeypatch.setattr(next_steps, "get_value", lambda name, default=None: default)
    assert next_steps.enabled() is True


def test_what_next_follows_its_setting(monkeypatch) -> None:
    monkeypatch.setattr(next_steps, "get_value", lambda name, default=None: False)
    assert next_steps.enabled() is False


def test_what_next_section_is_empty_without_steps() -> None:
    assert next_steps.section_html([]) == ""
