"""The "Diagnostic plots" demo: the seven diagnostic charts on data they were made for.

The Palmer penguins for the pair plot, the interaction plot, the mosaic and
a normal Q-Q plot; Michelson's speed of light (NIST's Michelso) for a P-P
plot; the 6-MP leukaemia trial (Freireich 1963), the textbook Kaplan-Meier
example; and the BCG vaccine trials (Colditz 1994), the textbook forest
plot, pooled by DerSimonian and Laird. Every figure carries a note on what
to look for (todo R-06).

    python3 -m dev.demo.diagnostics_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.data.demos import DEMO_DIR, DEMO_PROJECTS
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils import report_html
from dev.demo.build_demos import FigureSpec, SeriesSpec, _create_figure, _penguins
from dev.tests import _nist_strd as nist

DIAGNOSTICS_DEMO = next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_diagnostics_demo"))

#: Weeks in remission, and whether the relapse was seen (0: still in remission when last seen).
LEUKAEMIA: dict[str, list[tuple[int, int]]] = {
    "6-MP": [(6, 1), (6, 1), (6, 1), (6, 0), (7, 1), (9, 0), (10, 1), (10, 0), (11, 0), (13, 1), (16, 1),
             (17, 0), (19, 0), (20, 0), (22, 1), (23, 1), (25, 0), (32, 0), (32, 0), (34, 0), (35, 0)],
    "Placebo": [(t, 1) for t in (1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23)],
}

#: The BCG trials: author, year, vaccinated with TB, without; unvaccinated with TB, without.
BCG: tuple[tuple[str, int, int, int, int, int], ...] = (
    ("Aronson", 1948, 4, 119, 11, 128),
    ("Ferguson & Simes", 1949, 6, 300, 29, 274),
    ("Rosenthal et al.", 1960, 3, 228, 11, 209),
    ("Hart & Sutherland", 1977, 62, 13536, 248, 12619),
    ("Frimodt-Moller et al.", 1973, 33, 5036, 47, 5761),
    ("Stein & Aronson", 1953, 180, 1361, 372, 1079),
    ("Vandiviere et al.", 1973, 8, 2537, 10, 619),
    ("TPT Madras", 1980, 505, 87886, 499, 87892),
    ("Coetzee & Berjak", 1968, 29, 7470, 45, 7232),
    ("Rosenthal et al.", 1961, 17, 1699, 65, 1600),
    ("Comstock et al.", 1974, 186, 50448, 141, 27197),
    ("Comstock & Webster", 1969, 5, 2493, 3, 2338),
    ("Comstock et al.", 1976, 27, 16886, 29, 17825),
)


def bcg_table() -> pd.DataFrame:
    """Risk ratio and 95% interval per trial, and the random-effects pooled one."""
    rows = []
    a, b, c, d = (np.array([trial[i] for trial in BCG], float) for i in (2, 3, 4, 5))
    log_rr = np.log((a / (a + b)) / (c / (c + d)))
    variance = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)
    weight = 1 / variance
    fixed = np.sum(weight * log_rr) / weight.sum()
    q = np.sum(weight * (log_rr - fixed) ** 2)
    tau2 = max(0.0, (q - (len(BCG) - 1)) / (weight.sum() - np.sum(weight**2) / weight.sum()))
    random_weight = 1 / (variance + tau2)
    pooled = np.sum(random_weight * log_rr) / random_weight.sum()
    pooled_se = np.sqrt(1 / random_weight.sum())
    for (author, year, *_counts), y, v, w in zip(BCG, log_rr, variance, random_weight):
        se = np.sqrt(v)
        rows.append({"trial": f"{author} {year}", "risk_ratio": np.exp(y), "lower": np.exp(y - 1.96 * se),
                     "upper": np.exp(y + 1.96 * se), "weight": w / random_weight.sum() * 100, "summary": 0})
    rows.append({"trial": "Random effects (DerSimonian-Laird)", "risk_ratio": np.exp(pooled),
                 "lower": np.exp(pooled - 1.96 * pooled_se), "upper": np.exp(pooled + 1.96 * pooled_se),
                 "weight": None, "summary": 1})
    return pd.DataFrame(rows)


def _note(repo: SqliteRepo, figure_id: int, title: str, subtitle: str, data: str, read: str) -> None:
    markup = report_html.document(
        title, subtitle,
        report_html.section("The data", report_html.note(data)),
        report_html.section("What to look for", report_html.raw_note(read)),
    )
    options = repo.get_figure_options(figure_id)
    view = dict(options.get("view") or {})
    view.update({"resize_mode": "FIT", "notes_html": markup, "notes_split": [480, 340]})
    options["view"] = view
    repo.set_figure_options(figure_id, options)


def build_diagnostics_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the diagnostic plots demo into *directory* and return its path."""
    path = Path(directory) / DIAGNOSTICS_DEMO.path_name
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", ".undo.db"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    repo = SqliteRepo(db_path=path)

    penguins = _penguins()
    penguins["sex"] = penguins["sex"].fillna("unknown")
    repo.import_dataframe(penguins, table_name="penguins", normalize_columns=False)
    repo.import_dataframe(
        pd.DataFrame([(arm, weeks, event) for arm, rows in LEUKAEMIA.items() for weeks, event in rows],
                     columns=["arm", "weeks", "relapse"]),
        table_name="leukaemia_6mp", normalize_columns=False,
    )
    repo.import_dataframe(bcg_table(), table_name="bcg_trials", normalize_columns=False)
    michelso = nist.univariate("Michelso")
    repo.import_dataframe(pd.DataFrame({"speed": michelso.values}), table_name="michelson_light", normalize_columns=False)

    figure_id = _create_figure(repo, FigureSpec(
        name="1 · Pair plot", chart_type="Pair Plot", title="Penguin measurements by species",
        x_label="", y_label="", axis_options={"upper": "correlation"},
        series=[SeriesSpec("penguins", 'SELECT bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, '
                                        'species AS "group" FROM penguins', {}, {})],
    ))
    _note(repo, figure_id, "Pair plot", "Four measurements of 342 penguins, three species",
          "Palmer Station penguins (Gorman, Williams and Fraser 2014).",
          "Pooled, bill length and depth fall together (all: r = -0.24); within each species they "
          "rise together (r = 0.39 to 0.65). The grid shows both at once - <b>Simpson's paradox</b> "
          "in one picture.")

    figure_id = _create_figure(repo, FigureSpec(
        name="2 · Interaction plot", chart_type="Interaction Plot", title="Body mass: species × sex",
        x_label="species", y_label="body mass (g)", axis_options={"error_bars": "ci"},
        series=[SeriesSpec("body mass", "SELECT species AS x, body_mass_g AS y, sex AS trace FROM penguins "
                                        "WHERE sex <> 'unknown'", {}, {})],
    ))
    _note(repo, figure_id, "Interaction plot", "Mean body mass with 95% intervals",
          "Palmer penguins with a recorded sex.",
          "Males are heavier in every species, but by 800 g in Gentoo and only 400 g in Chinstrap: "
          "the lines are not parallel - a <b>species × sex interaction</b>, which a two-way ANOVA "
          "would test.")

    figure_id = _create_figure(repo, FigureSpec(
        name="3 · Mosaic plot", chart_type="Mosaic Plot", title="Which species lives on which island",
        x_label="island", y_label="", axis_options={"colour_by": "residual"},
        series=[SeriesSpec("islands", "SELECT island AS x, species AS y FROM penguins", {}, {})],
    ))
    _note(repo, figure_id, "Mosaic plot", "Island by species, shaded by Pearson residual",
          "Palmer penguins: the island each was sampled on.",
          "Columns are as wide as each island's share; tiles as tall as each species' share there. "
          "Gentoo only on Biscoe, Chinstrap only on Dream: the blue tiles hold far more birds than "
          "independence would put there. Switch <b>colour by</b> to category for the plain mosaic.")

    figure_id = _create_figure(repo, FigureSpec(
        name="4 · Normal Q-Q plot", chart_type="Q-Q Plot", title="Is flipper length normal within each species?",
        x_label="", y_label="flipper length (mm)", axis_options={"grid": True},
        series=[
            SeriesSpec(species, f"SELECT flipper_length_mm AS value FROM penguins WHERE species = '{species}'",
                       {}, {"marker": "."})
            for species in ("Adelie", "Chinstrap", "Gentoo")
        ],
    ))
    _note(repo, figure_id, "Normal Q-Q plot", "Sample quantiles against normal quantiles, per species",
          "Palmer penguins' flipper lengths.",
          "Adelie and Chinstrap follow their lines inside the 95% bands. Gentoo's shortest flippers "
          "leave the band at the lower left: a heavier lower tail than a normal - mild, and an ANOVA "
          "copes with it. The steps are the measurements' 1 mm resolution, not a departure.")

    figure_id = _create_figure(repo, FigureSpec(
        name="5 · P-P plot", chart_type="P-P Plot", title="Michelson's speed of light against a normal",
        x_label="", y_label="", axis_options={"grid": True},
        series=[SeriesSpec("Michelson 1879", "SELECT speed AS value FROM michelson_light", {}, {"marker": "."})],
    ))
    _note(repo, figure_id, "P-P plot", "Fitted normal probabilities against the sample's",
          "Michelson's 100 measurements of the speed of light, 1879 (NIST StRD Michelso).",
          "The points hug the diagonal: the normal describes the bulk of the sample well. A P-P plot "
          "is sharpest in the middle; the Q-Q plot of the same data shows the tails.")

    figure_id = _create_figure(repo, FigureSpec(
        name="6 · Kaplan-Meier", chart_type="Kaplan-Meier", title="Remission in acute leukaemia: 6-MP against placebo",
        x_label="weeks", y_label="", axis_options={"median_line": True},
        series=[SeriesSpec("trial", 'SELECT weeks AS time, relapse AS event, arm AS "group" FROM leukaemia_6mp', {}, {})],
    ))
    _note(repo, figure_id, "Kaplan-Meier", "Time to relapse, 21 patients per arm",
          "Freireich et al. (1963), the 6-mercaptopurine maintenance trial: weeks in remission; "
          "relapse = 0 marks a patient still in remission when last seen.",
          "Median remission 23 weeks on 6-MP, 8 on placebo; the log-rank test (χ² = 16.8, "
          "p &lt; 0.0001) says the curves differ by far more than chance. The ticks are the censored "
          "patients - used for as long as they were followed, never counted as relapses.")

    figure_id = _create_figure(repo, FigureSpec(
        name="7 · Forest plot", chart_type="Forest Plot", title="BCG vaccine against tuberculosis: 13 trials",
        x_label="risk ratio (log scale)", y_label="",
        axis_options={"log_scale": True, "null_value": 1.0, "value_format": ".2f"},
        series=[SeriesSpec("trials", "SELECT trial AS label, risk_ratio AS estimate, lower, upper, weight, summary "
                                     "FROM bcg_trials", {}, {})],
    ))
    _note(repo, figure_id, "Forest plot", "Risk of tuberculosis, vaccinated against not",
          "Colditz et al. (1994): thirteen trials of the BCG vaccine; risk ratios with 95% intervals, "
          "square areas by random-effects weight.",
          "Most trials sit left of 1 - the vaccine protects - but by very different amounts. The "
          "pooled risk ratio is 0.49 (0.34 to 0.70): about half the risk.")

    report = repo.optimize_db()
    applogger.info("Diagnostics demo check: %s", report.summary())
    repo.close()
    Path(f"{path}.undo.db").unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    print(build_diagnostics_demo())
