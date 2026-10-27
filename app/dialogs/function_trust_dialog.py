"""Ask before a project's stored functions run on this computer (todo ID47).

A stored function - Function's Expression or Code, kept in the project's __sympy__ table - runs its text with
sympify or exec whenever it is drawn or fitted, and the import guard is no sandbox: a project from someone else
could run anything. So its functions run only once trusted here - their code shown first, the answer remembered
on this computer (sympy_store.trust) until the code changes. Those saved on this computer are trusted already.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QWidget

from app.data_operations import sympy_store
from app.dialogs.two_panels_dialog_base import PanelDialog
from app.utils.i18n import _


class FunctionTrustDialog(PanelDialog):
    """The functions' code, and whether to run it."""

    def __init__(self, functions: Sequence[Mapping[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent, title=_("Run the project's functions?"), size="small")
        intro = QLabel(_("This project keeps functions that run code on this computer whenever they are drawn "
                         "or fitted. Run them only if you trust where the project comes from."), self)
        intro.setWordWrap(True)
        self.body_layout.addWidget(intro)
        card = self.add_card(_("Code"), "functionTrustCard", stretch=1)
        self._code = QPlainTextEdit(card)
        self._code.setReadOnly(True)
        self._code.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._code.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._code.setPlainText(code_text(functions))
        card.layout().addWidget(self._code, 1)
        note = QLabel(_("Trusted, they are remembered on this computer and asked about again only if their "
                        "code changes. Not run, what they draw stays empty."), self)
        note.setWordWrap(True)
        self.body_layout.addWidget(note)
        # Not running is the default: Return must not be the fast path to running a stranger's code.
        self.add_action("dont_run", self.reject, default=True)
        self.add_action("trust_functions", self.accept)


def code_text(functions: Sequence[Mapping[str, Any]]) -> str:
    """Each function's name, kind and code, one after the other."""
    parts = []
    for data in functions:
        kind = str(data.get("source_type", "")).lower()
        parts.append(f"# {data.get('function_name') or data.get('name', '')} ({kind})\n"
                     f"{str(data.get('code_string', '')).strip()}\n")
    return "\n".join(parts)


def ask_to_trust(parent: QWidget | None, repo: Any) -> bool:
    """The project's functions not trusted here, asked about now; True once trusted, or with none to ask about."""
    functions = sympy_store.untrusted(repo)
    if not functions:
        return True
    return _ask(parent, repo, functions)


def _ask(parent: QWidget | None, repo: Any, functions: Sequence[Mapping[str, Any]]) -> bool:
    if FunctionTrustDialog(functions, parent).exec() != PanelDialog.DialogCode.Accepted:
        return False
    sympy_store.trust(functions)
    # The charts of them were read empty: the database did not change, so the series cache would not know.
    repo.invalidate_series_cache()
    return True


def allowed_to_run(parent: QWidget | None, repo: Any, function: Mapping[str, Any]) -> bool:
    """Whether *function* may run here: trusted already, or trusted now with the project's other functions.

    Asked once for the whole project - every function of it not trusted yet, *function* first - as a project
    is trusted or not as a whole.
    """
    if sympy_store.is_trusted(function):
        return True
    wanted = sympy_store.fingerprint(function)
    others = [row for row in sympy_store.untrusted(repo) if sympy_store.fingerprint(row) != wanted]
    return _ask(parent, repo, [function, *others])
