"""The assistant's settings: what it may do, what it may send, which provider and model - and the key.

The settings are in user.json; the key is not. It is kept by the system - the
macOS Keychain, the Windows Credential Manager - through the keyring package,
when it is installed; without it the key is read from the provider's
environment variable (ANTHROPIC_API_KEY for Claude) and Settings says so.
"""
from __future__ import annotations

import os

from app.logs.logger import applogger
from app.utils.config import get_value, set_value

#: What the assistant may do (Settings > Assistant): read and answer in the panel; that and run analyses;
#: everything, the project's tables and database included - each change confirmed.
READ_ONLY, ANALYSES, EVERYTHING = "read_only", "analyses", "everything"
PERMISSIONS: tuple[tuple[str, str], ...] = (
    (READ_ONLY, "Read only - answers in the panel"),
    (ANALYSES, "Analyses - runs them, answers in the panel"),
    (EVERYTHING, "Everything - the database included, each change confirmed"),
)
#: Auto: the fast model for simple requests, the accurate one for the rest; or always one of them.
AUTO = "auto"
MODES: tuple[tuple[str, str], ...] = ((AUTO, "Automatic"), ("fast", "Fast"), ("accurate", "Accurate"))

PROVIDER_KEY = "assistant_provider"
PERMISSION_KEY = "assistant_permission"
CONSENT_KEY = "assistant_consent"
SAMPLE_ROWS_KEY = "assistant_sample_rows"
MODE_KEY = "assistant_mode"
#: Whether the Assistant has its row in the navigation bar (Settings > Assistant); shown by default.
NAVIGATION_KEY = "assistant_in_navigation"
#: Where the keychain files the keys.
KEYRING_SERVICE = "ChartLibre"
#: The provider's own variable, read when no key is kept: {key name: environment variable}.
ENVIRONMENT_KEYS = {"anthropic_api_key": "ANTHROPIC_API_KEY"}


def permission() -> str:
    value = str(get_value(PERMISSION_KEY, READ_ONLY) or READ_ONLY)
    return value if value in dict(PERMISSIONS) else READ_ONLY


def consented() -> bool:
    """Whether the user allowed data to be sent to the provider (Settings > Assistant)."""
    return bool(get_value(CONSENT_KEY, False))


def sample_rows() -> int:
    """How many rows of a table the assistant may send as an example; 0 for none."""
    try:
        return max(0, min(200, int(get_value(SAMPLE_ROWS_KEY, 20))))
    except (TypeError, ValueError):
        return 20


def mode() -> str:
    value = str(get_value(MODE_KEY, AUTO) or AUTO)
    return value if value in dict(MODES) else AUTO


def in_navigation() -> bool:
    return bool(get_value(NAVIGATION_KEY, True))


def set_in_navigation(shown: bool) -> None:
    set_value(NAVIGATION_KEY, bool(shown))


def provider_name() -> str:
    return str(get_value(PROVIDER_KEY, "Claude") or "Claude")


def save(*, provider: str, permission_level: str, consent: bool, rows: int, assistant_mode: str) -> None:
    set_value(PROVIDER_KEY, provider)
    set_value(PERMISSION_KEY, permission_level)
    set_value(CONSENT_KEY, bool(consent))
    set_value(SAMPLE_ROWS_KEY, int(rows))
    set_value(MODE_KEY, assistant_mode)


def _keyring():
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def keychain_available() -> bool:
    """Whether a key can be kept by the system (the keyring package is installed)."""
    return _keyring() is not None


def api_key(key_name: str) -> str:
    """The key kept for *key_name*, else its environment variable; "" when there is neither."""
    keyring = _keyring()
    if keyring is not None:
        try:
            stored = keyring.get_password(KEYRING_SERVICE, key_name)
        except Exception:  # noqa: BLE001 - a keychain that cannot be read is no key
            applogger.exception("Could not read the assistant's key from the system keychain")
            stored = None
        if stored:
            return stored
    return os.environ.get(ENVIRONMENT_KEYS.get(key_name, ""), "")


def set_api_key(key_name: str, key: str) -> bool:
    """Keep *key* in the system keychain (or forget it when empty); False when there is no keychain."""
    keyring = _keyring()
    if keyring is None:
        return False
    try:
        if key:
            keyring.set_password(KEYRING_SERVICE, key_name, key)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, key_name)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception:  # noqa: BLE001 - reported, the setting is not kept
        applogger.exception("Could not keep the assistant's key in the system keychain")
        return False
    return True
