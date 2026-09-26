"""Tests for the fit dialog's two verbs and its report.

Fitting and drawing used to be the same button: Preview called ``on_fit``, so
the optimiser replaced the parameters before anything reached the chart and a
hand-typed starting guess could never be seen.  They are now separate, with the
parameter table as the single source of truth for what gets drawn.

The dialog needs a QDialog to exist, so the behaviour that can be checked
without one - the report markup, and the shape of the split - is checked here.
"""
from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
FIT_SOURCE = (APP_DIR / "series_operations" / "fit_dialog.py").read_text(
    encoding="utf-8"
)


def _body(source: str, name: str) -> str:
    """Return one method's source, up to the next method at class level."""
    start = source.index(f"def {name}")
    end = source.index("\n    def ", start + 10)
    return source[start:end]


# ----------------------------------------------------------------------
# Fit and Preview are different acts
# ----------------------------------------------------------------------


def test_fit_optimises() -> None:
    body = _body(FIT_SOURCE, "on_fit")
    assert "optimise=True" in body


def test_fit_writes_the_optimum_into_the_parameter_table() -> None:
    """Otherwise Preview would immediately draw the old guess again."""
    body = _body(FIT_SOURCE, "on_fit")
    assert "_set_initial_params" in body


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def test_the_report_has_a_row_per_parameter_with_its_error() -> None:
    body = _body(FIT_SOURCE, "_results_html")

    assert "Std. error" in body
    # The row label carries the parameter's meaning, not just its index.
    assert "_param_label(row)" in body
    assert "report_html.table(" in body


# ----------------------------------------------------------------------
# What the parameters mean
# ----------------------------------------------------------------------
def _library() -> list[dict]:
    """Return every discovered fit function, the way the dialog gets them.

    Through the scanner, not through the dialog: ``_catalog_data`` is a
    one-line forward to it and calling it unbound needed a fake self, which
    stopped working the moment the method touched an attribute. What these
    tests are about is the library's own metadata, so they read it at
    the source.
    """
    from app.scanners.functions_scanner import FunctionScanner

    return [
        payload
        for payloads in FunctionScanner().catalog().values()
        for payload in payloads
    ]


def test_every_function_declares_a_formula() -> None:
    """``helpers.gaussian(x, p)`` said nothing about what p[2] was.

    The expression is the only thing in the dialog that says what the
    parameters mean before the fit runs, so a function without one is a row of
    spin boxes labelled p[0]..p[4].
    """
    missing = [
        payload["name"] for payload in _library() if not str(payload.get("expression", "")).strip()
    ]
    assert missing == []


# ----------------------------------------------------------------------
# Layout: the fit-options row does not overflow, and the cards have padding
# ----------------------------------------------------------------------


