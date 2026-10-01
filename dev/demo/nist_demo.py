"""The "NIST reference datasets" demo: certified problems, ready to check ChartLibre on.

Ten of the NIST Statistical Reference Datasets
(https://www.itl.nist.gov/div898/strd/), read from the original files in
dev/tests/data/nist: five nonlinear regressions for Fit, three linear ones
(a line, a parabola, Filip's tenth-degree polynomial) for Fit and
Interpolation, a sample for the summary statistics and five instruments for
a one-way ANOVA. Under every chart its certificate: what to run, where to
start, and the values the answer has to match - the check a reviewer of a
statistics package makes.

    python3 -m dev.demo.nist_demo
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.demos import DEMO_DIR, DEMO_PROJECTS
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils import report_html
from dev.demo.build_demos import FigureSpec, SeriesSpec, _create_figure
from dev.tests import _nist_strd as nist

NIST_DEMO = next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_nist_demo"))

_VIEW_KEY = "view"

#: Nonlinear problems and the Fit model that carries each one (app/functions/nist_functions.py).
NONLINEAR: tuple[tuple[str, str], ...] = (
    ("Misra1a", "Misra1a (NIST, lower difficulty)"),
    ("Gauss3", "Gauss3 (NIST, average difficulty)"),
    ("Thurber", "Thurber (NIST, higher difficulty)"),
    ("Eckerle4", "Eckerle4 (NIST, higher difficulty)"),
    ("Rat43", "Rat43 (NIST, higher difficulty)"),
)

#: Linear problems: name, what to run.
LINEAR: tuple[tuple[str, str], ...] = (
    ("Norris", "Fit: <b>Linear</b> (or Interpolation: <b>Linear</b>)."),
    ("Pontius", "Fit: <b>Quadratic</b> (or Interpolation: <b>Polynomial</b>, degree 2)."),
    (
        "Filip",
        "Interpolation: <b>Polynomial</b>, degree 10. The hardest of NIST's linear problems: "
        "many packages report a wrong answer, or none.",
    ),
)


def _description(name: str) -> str:
    """The "Description:" paragraph of a NIST file, as one line of text."""
    text = (nist.DATA / f"{name}.dat").read_text(encoding="latin-1")
    match = re.search(r"Description:\s*(.*?)\n\s*\n", text, flags=re.DOTALL)
    if match is None:
        return ""
    return " ".join(line.strip() for line in match.group(1).splitlines() if line.strip())


def _number(value: float) -> str:
    return f"{value:.10g}"


def _report(title: str, subtitle: str, description: str, steps: str, *sections: str) -> str:
    return report_html.document(
        title,
        subtitle,
        report_html.section("The data", report_html.note(description)),
        report_html.section("Try it", report_html.raw_note(steps)),
        *sections,
    )


def _store_report(repo: SqliteRepo, figure_id: int, markup: str) -> None:
    options = repo.get_figure_options(figure_id)
    view = dict(options.get(_VIEW_KEY) or {})
    view.update({"resize_mode": "FIT", "notes_html": markup, "notes_split": [480, 340]})
    options[_VIEW_KEY] = view
    repo.set_figure_options(figure_id, options)


def _scatter(index: int, name: str, table: str, title: str) -> FigureSpec:
    return FigureSpec(
        name=f"{index} · NIST {name}",
        key=f"nist_{name.lower()}",
        tables=(table,),
        chart_type="Scatter Plot",
        title=title,
        x_label="x",
        y_label="y",
        axis_options={"grid": True},
        series=[SeriesSpec(name=name, sql=f'SELECT x, y FROM "{table}" ORDER BY x', roles={"x": "x", "y": "y"}, style={})],
    )


def build_nist_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the NIST demo project into *directory* and return its path."""
    path = Path(directory) / NIST_DEMO.path_name
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", ".undo.db"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    repo = SqliteRepo(db_path=path)
    certified_rows: list[dict[str, object]] = []
    index = 0

    for name, model in NONLINEAR:
        data = nist.nonlinear(name)
        table = f"nist_{name.lower()}"
        repo.import_dataframe(pd.DataFrame({"x": data.x, "y": data.y}), table_name=table, normalize_columns=False)
        index += 1
        figure_id = _create_figure(repo, _scatter(index, name, table, f"{name} - {data.difficulty.lower()} difficulty"))
        rows = [
            (f"b{i + 1}", _number(start1), _number(start2), _number(value), _number(std))
            for i, (start1, start2, value, std) in enumerate(zip(data.start1, data.start2, data.params, data.std))
        ]
        certified_rows += [
            {"dataset": name, "quantity": row[0], "certified": float(value), "standard_error": float(std)}
            for row, value, std in zip(rows, data.params, data.std)
        ]
        certified_rows.append({"dataset": name, "quantity": "residual sum of squares", "certified": data.rss, "standard_error": None})
        _store_report(repo, figure_id, _report(
            f"NIST {name}",
            f"Nonlinear regression, {data.difficulty.lower()} difficulty - {data.x.size} points",
            _description(name),
            f"Fit, category <b>NIST reference models</b>, model <b>{html.escape(model)}</b>. It opens at NIST's "
            "start 1 (the far one); <b>Estimate</b> gives start 2. Press <b>Fit</b> and compare with the "
            "certified values below - ChartLibre agrees to 7 or more significant digits.",
            report_html.section("Certified values", report_html.table(
                ["Parameter", "Start 1", "Start 2", "Certified value", "Standard error"], rows,
            )),
            report_html.section("Residuals", report_html.summary_table([
                ("Residual sum of squares", _number(data.rss)),
                ("Residual standard deviation", _number(data.residual_sd)),
                ("Degrees of freedom", str(data.dof)),
            ])),
        ))

    for name, steps in LINEAR:
        data = nist.linear(name)
        table = f"nist_{name.lower()}"
        repo.import_dataframe(pd.DataFrame({"x": data.x, "y": data.y}), table_name=table, normalize_columns=False)
        index += 1
        degree = data.params.size - 1
        figure_id = _create_figure(repo, _scatter(index, name, table, f"{name} - polynomial of degree {degree}"))
        rows = [(f"B{i}", _number(value), _number(std)) for i, (value, std) in enumerate(zip(data.params, data.std))]
        certified_rows += [
            {"dataset": name, "quantity": row[0], "certified": float(value), "standard_error": float(std)}
            for row, value, std in zip(rows, data.params, data.std)
        ]
        _store_report(repo, figure_id, _report(
            f"NIST {name}",
            f"Linear least squares, y = B0 + B1 x + ... + B{degree} x^{degree} - {data.x.size} points",
            _description(name),
            steps + " Interpolation lists the coefficients highest power first (c0 is B"
            f"{degree}).",
            report_html.section("Certified values", report_html.table(
                ["Coefficient", "Certified value", "Standard error"], rows,
            )),
            report_html.section("Fit quality", report_html.summary_table([
                ("Residual standard deviation", _number(data.residual_sd)),
                ("R²", _number(data.r_squared)),
            ])),
        ))

    # Summary statistics.
    sample = nist.univariate("Michelso")
    repo.import_dataframe(
        pd.DataFrame({"run": np.arange(1, sample.values.size + 1), "speed": sample.values}),
        table_name="nist_michelso", normalize_columns=False,
    )
    index += 1
    figure_id = _create_figure(repo, FigureSpec(
        name=f"{index} · NIST Michelso", key="nist_michelso", tables=("nist_michelso",),
        chart_type="Histogram", title="Michelson's speed of light measurements",
        x_label="speed of light (1000 km/s)", y_label="",
        axis_options={"bins": 20, "grid": True, "grid_axis": "y"},
        series=[SeriesSpec(name="Michelso", sql="SELECT speed AS value FROM nist_michelso", roles={"value": "value"}, style={})],
    ))
    certified_rows += [
        {"dataset": "Michelso", "quantity": "mean", "certified": sample.mean, "standard_error": None},
        {"dataset": "Michelso", "quantity": "standard deviation", "certified": sample.std, "standard_error": None},
    ]
    _store_report(repo, figure_id, _report(
        "NIST Michelso", f"Univariate summary statistics - {sample.values.size} values",
        _description("Michelso"),
        "Statistics, on this series: the <b>Descriptive</b> table's mean and standard deviation.",
        report_html.section("Certified values", report_html.summary_table([
            ("Mean", _number(sample.mean)),
            ("Standard deviation (n - 1)", _number(sample.std)),
            ("Lag-1 autocorrelation", _number(sample.autocorrelation)),
        ])),
    ))

    # One-way ANOVA.
    anova = nist.anova("SiRstv")
    frame = pd.DataFrame(
        [(f"Instrument {group}", value) for group, values in anova.groups.items() for value in values],
        columns=["instrument", "resistance"],
    )
    repo.import_dataframe(frame, table_name="nist_sirstv", normalize_columns=False)
    index += 1
    figure_id = _create_figure(repo, FigureSpec(
        name=f"{index} · NIST SiRstv", key="nist_sirstv", tables=("nist_sirstv",),
        chart_type="Box Plot", title="Silicon resistivity, five instruments",
        x_label="instrument", y_label="resistance (ohm cm)",
        axis_options={"grid": True, "grid_axis": "y", "showmeans": True},
        series=[
            SeriesSpec(
                name=group,
                sql=f"SELECT instrument AS \"group\", resistance AS value FROM nist_sirstv WHERE instrument = '{group}'",
                roles={"value": "value", "group": "group"},
                style={},
            )
            for group in sorted(frame["instrument"].unique())
        ],
    ))
    certified_rows += [
        {"dataset": "SiRstv", "quantity": "F statistic", "certified": anova.f_statistic, "standard_error": None},
        {"dataset": "SiRstv", "quantity": "R-squared", "certified": anova.r_squared, "standard_error": None},
    ]
    _store_report(repo, figure_id, _report(
        "NIST SiRstv", f"One-way analysis of variance - {len(anova.groups)} groups, {len(frame)} values",
        _description("SiRstv"),
        "Statistics, with all five series selected: the <b>One-way ANOVA</b> row's F (its note gives the "
        "degrees of freedom and eta², which is the R² below).",
        report_html.section("Certified values", report_html.summary_table([
            ("Between instruments", f"df {anova.between_df}, sum of squares {_number(anova.between_ss)}"),
            ("Within instruments", f"df {anova.within_df}, sum of squares {_number(anova.within_ss)}"),
            ("F statistic", _number(anova.f_statistic)),
            ("R²", _number(anova.r_squared)),
        ])),
    ))

    repo.import_dataframe(pd.DataFrame(certified_rows), table_name="nist_certified", normalize_columns=False)
    repo.set_table_notes(
        "nist_certified",
        "Every certified value of this project's NIST datasets, from the original files "
        "(https://www.itl.nist.gov/div898/strd/). Public domain, NIST/ITL.",
    )
    report = repo.optimize_db()
    applogger.info("NIST demo check: %s", report.summary())
    repo.close()
    Path(f"{path}.undo.db").unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    print(build_nist_demo())
