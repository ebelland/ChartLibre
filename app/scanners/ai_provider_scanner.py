"""Discovery of the assistant's providers (app.plugin_api.ai.provider).

Classes directly subclassing AIProvider in app/ai_providers (and
user/ai_providers, for the user's own) are found by an AST scan - the
service's package is not imported to find them - and loaded when the
Assistant is used. How a family is found is app.scanners.plugin_family's.
"""
from __future__ import annotations

from pathlib import Path

from app.scanners.plugin_family import PluginFamily
from app.utils.config import USER_CONTENT_DIR

#: Where the user's own providers live, beside their operations.
USER_AI_PROVIDERS_DIR: Path = USER_CONTENT_DIR / "ai_providers"

FAMILY = PluginFamily(
    label="assistant providers",
    base_class="AIProvider",
    builtin_root=Path(__file__).resolve().parent.parent / "ai_providers",
    user_root=USER_AI_PROVIDERS_DIR,
    module_prefix="_dynamic_ai_provider",
    string_attrs=("Description",),
)


def provider_class(entry: dict):
    """The provider class a discovery entry describes."""
    return FAMILY.load(entry)


# Scanned once at import time.
ai_providers: list[dict] = FAMILY.discover()
