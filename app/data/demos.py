"""The demo projects "Load demo" offers, and copying one out to open it.

Only the catalogue and the copy live in the application. The demo files
themselves are built ahead of time, from the datasets under
dev/demo/sample data, by ``python -m dev.demo.build_demos`` - a developer's
step that nothing an end user runs ever needs.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from app.data.sqlite_repo import SqliteRepo

#: The repository root: three levels above this file (app/data/demos.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Where the built demo set lives, and where "Load demo" reads it from.
DEMO_DIR: Path = REPO_ROOT / "demo"

#: Where "Load demo" drops its working copy of a demo before opening it. A
#: folder of its own beside the application rather than the home directory:
#: the copies are throwaway (Load demo overwrites its own last copy in
#: place), and one predictable place to find - or delete - all of them at
#: once beats them scattered loose in ``~``. Not version-controlled.
PROJECTS_DIR: Path = REPO_ROOT / "projects"


#: A bibliographic reference: the citation, and optionally its DOI and a
#: link. The same shape the project's own ``references`` entry holds.
Reference = dict[str, str]


def reference(citation: str, *, doi: str = "", url: str = "") -> Reference:
    """One reference, with only the parts it has."""
    entry = {"citation": citation}
    if doi:
        entry["doi"] = doi
    if url:
        entry["url"] = url
    return entry


#: The data sources the demos draw on (see also app/utils/credits.py).
REF_PENGUINS = reference(
    "Horst A. M., Hill A. P., Gorman K. B. (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data.",
    doi="10.5281/zenodo.3960218", url="https://allisonhorst.github.io/palmerpenguins/")
REF_ANTIBIOTICS = reference("Burtin W. (1951). Antibiotic effectiveness against sixteen bacteria. Scope (Upjohn).")
REF_YEAST = reference("Nakai K. (1991). Yeast. UCI Machine Learning Repository.",
                      doi="10.24432/C5KG68", url="https://archive.ics.uci.edu/dataset/110/yeast")
REF_ANSCOMBE = reference("Anscombe F. J. (1973). Graphs in statistical analysis. The American Statistician 27(1), 17-21.",
                         doi="10.1080/00031305.1973.10478966")
REF_CO2 = reference("Keeling C. D. et al.; NOAA Global Monitoring Laboratory. Mauna Loa CO2 monthly mean data.",
                    url="https://gml.noaa.gov/ccgg/trends/")
REF_SUNSPOTS = reference("WDC-SILSO, Royal Observatory of Belgium, Brussels. Monthly mean total sunspot number.",
                         url="https://www.sidc.be/SILSO/")
REF_GISTEMP = reference("GISTEMP Team (2024). GISS Surface Temperature Analysis (GISTEMP), version 4. NASA GISS.",
                        url="https://data.giss.nasa.gov/gistemp/")
REF_USGS = reference("U.S. Geological Survey. Earthquake Hazards Program, real-time feeds.",
                     url="https://earthquake.usgs.gov/earthquakes/feed/")
REF_NIST = reference("NIST Statistical Reference Datasets (StRD): nonlinear regression and univariate summary statistics.",
                     url="https://www.itl.nist.gov/div898/strd/")
REF_FREIREICH = reference("Freireich E. J. et al. (1963). The effect of 6-mercaptopurine on the duration of steroid-induced "
                          "remissions in acute leukemia. Blood 21(6), 699-716.", doi="10.1182/blood.V21.6.699.699")
REF_COLDITZ = reference("Colditz G. A. et al. (1994). Efficacy of BCG vaccine in the prevention of tuberculosis: "
                        "meta-analysis of the published literature. JAMA 271(9), 698-702.", doi="10.1001/jama.1994.03510330076038")
REF_MICHELSON = reference("Michelson A. A. (1882). Experimental determination of the velocity of light. "
                          "Astronomical Papers 1, 109-145. (NIST StRD dataset Michelso.)")


@dataclass(frozen=True, slots=True)
class DemoProject:
    """One demo file: what it is called, and what it contains.

    The file name is the documentation.  Someone with a dozen .dhub files in a
    folder should be able to open the one that answers their question without
    opening the other eleven, which means the name has to say both the
    subject and what it demonstrates.

    ``summary``, ``author`` and ``references`` are what the build writes into
    the project's own information (``__project_info__``: notes, author,
    references), along with its preview; once built, the Load demo dialog
    reads them back from the file through :meth:`info`, the way any project
    describes itself.
    """

    file_name: str
    summary: str
    figures: tuple[str, ...]
    #: "module:function" that writes this file itself, for a demo that is
    #: not a selection of figures - the Series Operations one runs the
    #: operations' dialogs. Empty: built from ``figures`` as usual.
    builder: str = ""
    #: The figure the preview pictures, by (part of) its name; the first
    #: figure when empty.
    preview: str = ""
    #: Where its data comes from.
    references: tuple[Reference, ...] = ()
    author: str = "ChartLibre"

    @property
    def path_name(self) -> str:
        """Return the file name with its extension."""
        return f"{self.file_name}.dhub"

    @property
    def source_path(self) -> Path:
        """Return where the pre-built copy of this project lives.

        Built by dev/demo/build_demos.py into :data:`DEMO_DIR`, not at
        the point this is read - see :func:`copy_demo_project`.
        """
        return DEMO_DIR / self.path_name

    def info(self) -> dict[str, str]:
        """The built file's project information, read without opening it for
        writing; empty when it is not built or predates the information."""
        from app.data.repo.project_info import read_project_info

        return read_project_info(self.source_path)

    @property
    def description(self) -> str:
        """What the demo is, as the built project's notes say."""
        return self.info().get("notes") or self.summary

    @property
    def preview_path(self) -> Path | None:
        """The picture the built project's ``preview_path`` names, if any."""
        entry = self.info().get("preview_path")
        if not entry:
            return None
        path = Path(entry) if Path(entry).is_absolute() else DEMO_DIR / entry
        return path if path.is_file() else None


#: The demo set.  The first is the complete project - every chart type over
#: every table - and the rest are one real dataset each.
DEMO_PROJECTS: tuple[DemoProject, ...] = (
    DemoProject(
        "Getting started - real data across every chart type",
        "Twenty datasets, twenty-nine figures across every chart type "
        "and six multi-axis layouts, and one saved query.",
        (),
        preview="Penguin flipper length - spread",
        references=(REF_PENGUINS, REF_ANTIBIOTICS, REF_YEAST, REF_ANSCOMBE,),
    ),
    DemoProject(
        "Antibiotics - grouped bars from a classic dataset",
        "1930s antibiotic potency data for sixteen bacteria, as a horizontal "
        "bar chart sorted by effectiveness.",
        ("antibiotics",),
        references=(REF_ANTIBIOTICS,),
    ),
    DemoProject(
        "Penguins - species compared across four chart types",
        "The Palmer penguins: the same three species read as a scatter plot, "
        "a histogram, a violin plot and a box plot.",
        ("penguin_scatter", "penguin_hist", "penguin_violin", "penguin_box"),
        references=(REF_PENGUINS,),
    ),
    DemoProject(
        "Yeast proteins - histogram of three localisation classes",
        "Cytoplasm, nucleus and mitochondria: three overlapping distributions "
        "of one measured feature.",
        ("yeast_hist",),
        references=(REF_YEAST,),
    ),
    DemoProject(
        "DLVO force curve - ready for Fit and Calculus",
        "A real colloidal force-distance curve, repulsive at short range and "
        "attractive beyond it, with a measurable crossover - the Roots "
        "operation's own question, answered exactly where the curve changes "
        "sign - and its curvature is also a fair test for Regression's "
        "Random Forest/Gradient Boosting models and for GP Regression's "
        "uncertainty band.",
        ("dlvo_force",),
    ),
    DemoProject(
        "Stock prices - three tickers, ready for Smoothing",
        "Three years of daily closing prices for three stocks, noisy enough "
        "that a moving average earns its keep - the three series also share "
        "a date axis, which is what Decomposition needs to run PCA/ICA "
        "across them together.",
        ("stock_timeseries",),
    ),
    DemoProject(
        "Employee compensation - ready for the Outlier operation",
        "Salaries across five departments, with one real outlier the box "
        "plot already shows and the operation can confirm - a right-skewed "
        "distribution Transform's Power/Quantile models can reshape, too - "
        "plus a pie chart of the same five departments' headcount.",
        ("employee_box", "employee_dept_pie"),
    ),
    DemoProject(
        "Driver behaviour - ready for the Cluster operation",
        "Four thousand drivers by distance and speeding: the classic "
        "two-feature clustering dataset, and shape-aware enough a "
        "population for Outliers' Isolation Forest/Local Outlier Factor "
        "models to earn their keep over a plain Z-score.",
        ("driver_scatter",),
    ),
    DemoProject(
        "Transactions - histogram and a saved query",
        "Fraud against legitimate transactions by amount, and a saved query "
        "averaging amount by channel.",
        ("transactions_hist", "transactions_query"),
    ),
    DemoProject(
        "Pairwise sample - four variables on two axes",
        "Two coordinates, a colour and a marker size: four variables read "
        "off one scatter plot.",
        ("pairwise_scatter",),
    ),
    DemoProject(
        "3D surfaces - gridded, scattered, contoured and meshed",
        "The same kind of surface in the two layouts the surface renderers "
        "each need - a regular grid, and scattered points triangulated into "
        "one - plus both from directly above as a contour map, and the "
        "scattered one's own triangulation shown bare, then coloured by "
        "height.",
        (
            "surface_3d", "scattered_3d", "surface_contour",
            "scattered_contour", "scattered_tri_mesh", "scattered_tri_color_mesh",
        ),
    ),
    DemoProject(
        "Parametric curve - a helix in projection and in 3D",
        "A helix seen end-on: a circle, coloured by how far along the curve "
        "each point is - and the same helix as discrete points in three "
        "dimensions rather than a projection.",
        ("parametric_scatter", "parametric_scatter3d"),
    ),
    DemoProject(
        "Figure layouts - shared scale, main+secondary, overlapping axes",
        "Three real datasets, each arranged with a different multi-axis "
        "layout preset: three price series sharing one main chart, three "
        "histograms panning and zooming together, and force paired with "
        "potential energy on a second y-axis.",
        ("stock_main_secondary", "penguin_shared_grid", "dlvo_overlapping"),
    ),
    DemoProject(
        "Classic demos - Anscombe, Lissajous, and a signal spectrum",
        "Three teaching examples from mathematics, physics and signal "
        "processing: Anscombe's quartet on a shared-scale grid (the "
        "matplotlib gallery's own multi-axis example), four Lissajous "
        "figures at different frequency ratios, and a noisy signal beside "
        "the frequency spectrum that recovers its three true tones, also "
        "read as a stem plot of discrete lines - the same signal the "
        "Filtering operation's IIR/FIR/Hilbert models can run on directly "
        "(its 5, 20 and 50 Hz tones are exact targets for a lowpass, a "
        "bandpass, or an envelope) and Spectral Analysis's own subject.",
        (
            "anscombe_quartet", "lissajous_grid", "signal_time_and_frequency",
            "signal_spectrum_stem",
        ),
        references=(REF_ANSCOMBE,),
    ),
    DemoProject(
        "Quality and spectroscopy - three quality-control operations",
        "Textbook defective-unit counts for the Control Chart operation's "
        "attribute charts (p/np/c/u), a synthetic spectrum with a "
        "drifting, wandering background for Baseline Correction (AsLS or "
        "rubber band) and, once corrected, for the Peaks operation's "
        "prominence filter to find its two peaks - both figures already "
        "carry an annotation and a reference line with an explicit colour, "
        "so opening Overlay properties shows its colour/line/font/size "
        "editors populated rather than empty - and seven rejected-board "
        "causes as a Pareto chart, the tool this kind of table exists for.",
        ("attribute_counts", "baseline_spectrum", "defect_causes_pareto"),
    ),
    DemoProject(
        "New chart types - heatmap, hexbin, event plot, and true 3D",
        "Chart types added after the first release, each on data suited to "
        "what makes it different from a plain scatter or line: a smooth "
        "surface as a per-cell heatmap, four thousand drivers binned into a "
        "hexbin, three servers' error timestamps as an event raster (and, "
        "on the same table, a running total as a stairs plot), a helix and "
        "a small sales table as a true 3D line and 3D bar chart rather than "
        "a scattered or projected stand-in, and that same sales table again "
        "stacked by region instead of grouped.",
        (
            "heatmap_grid",
            "driver_hexbin",
            "server_events",
            "server_events_stairs",
            "helix_3d_line",
            "regional_sales_3d_bar",
            "regional_sales_stack",
        ),
    ),
    DemoProject(
        "Machine comparison - four gauges measuring the same standard",
        "Thirty repeated readings of a 25.000 mm gauge block from four "
        "coordinate measuring machines: a box plot (which machine reads "
        "high, which is noisier), the same distributions again as an ECDF, "
        "a run-order scatter (whether either drifts over the session), a "
        "summary table, and each machine's own scheduled maintenance as a "
        "Gantt-style broken bar, horizontal and vertical - a shape-aware "
        "population for the Outlier operation, and aligned closely enough "
        "by run number for Statistics's paired t-test to confirm Machine "
        "C's bias against Machine A.",
        (
            "machine_box", "machine_ecdf", "machine_scatter", "machine_table",
            "machine_downtime_hbar", "machine_downtime_vbar",
        ),
    ),
    DemoProject(
        "Release history and regional sales - timeline and text",
        "Ten software releases on a dated timeline, coloured by whether "
        "each was a major or minor version, and the four sales regions "
        "read as plain labelled points rather than bars.",
        ("release_timeline", "regional_text"),
    ),
    DemoProject(
        "A rotating field - quiver, streamlines and wind barbs",
        "One synthetic vector field - solid-body rotation on a regular "
        "grid - drawn the three ways a vector field can be: an arrow per "
        "point, traced streamlines, and meteorology's own wind-barb "
        "notation - plus an unrelated month of daily temperature range, "
        "shaded as a filled band.",
        ("vector_field_quiver", "vector_field_stream", "vector_field_barbs",
         "temperature_fill_between"),
    ),
    DemoProject(
        "Sparse calibration curve - ready for Interpolation",
        "A logistic response curve sampled at only nine irregular points - "
        "exact by construction, so Interpolation's generated curve can be "
        "checked against the true shape rather than against noise.",
        ("sparse_calibration_scatter",),
    ),
    DemoProject(
        "3D Series Operations - Fit, Peaks, Roots and Calculus on a surface",
        "A Gaussian bump and a saddle, run through the four Series "
        "Operations that now understand a series with a z role: a "
        "Gaussian2D surface fitted to 500 scattered points, the bump's one "
        "peak found on its exact grid, a saddle's z=0 level curve traced "
        "as its two true diagonals, and the bump's gradient magnitude "
        "with the volume under it reported alongside.",
        ("ops3d_showcase",),
    ),
    DemoProject(
        "Real case studies - CO2, sunspots, earthquakes and warming",
        "Four published results, each on the institution's own published "
        "data rather than on anything invented here (downloaded by "
        "dev/demo/fetch_case_study_data.py): the Keeling curve breathing "
        "once a year on top of its rise from 315 to 428 ppm; 277 years of "
        "sunspot counts whose spectrum - computed by the Spectral "
        "operation's own FFT - peaks at 11.1 years, the cycle Schwabe "
        "read off by eye in 1843; a month of earthquakes obeying "
        "Gutenberg-Richter with b = 1.06 above the magnitude the "
        "catalogue is complete at, and visibly departing from it below; "
        "and 146 years of global temperature anomaly.",
        (
            "keeling_curve",
            "sunspot_record",
            "sunspot_spectrum",
            "gutenberg_richter",
            "global_temperature",
        ),
        references=(REF_CO2, REF_SUNSPOTS, REF_USGS, REF_GISTEMP,),
    ),
    DemoProject(
        "Root cause analysis - an Ishikawa diagram and its Pareto chart",
        "Why circuit boards are rejected, twice over: the possible causes "
        "brainstormed on a fishbone by the six Ms (method, machine, "
        "material, manpower, measurement, environment), with finer causes "
        "under some of them, and the defects actually counted as a Pareto "
        "chart - the two quality tools that are used together.",
        ("defect_root_causes_fishbone", "defect_causes_pareto"),
    ),
    DemoProject(
        "NIST reference datasets - certified answers for Fit and Statistics",
        "Ten of the NIST Statistical Reference Datasets, the yardstick "
        "statistics packages are measured against: five nonlinear "
        "regressions ready for Fit (their models are in its NIST category), "
        "a line, a parabola and Filip's tenth-degree polynomial, Michelson's "
        "speed of light for the summary statistics and five instruments for "
        "a one-way ANOVA - each with its certified values under the chart.",
        (),
        builder="dev.demo.nist_demo:build_nist_demo",
        preview="NIST Gauss3",
        references=(REF_NIST,),
    ),
    DemoProject(
        "Diagnostic plots - Q-Q, survival, forest, mosaic and more",
        "The seven diagnostic charts on the data they were made for: a "
        "pair plot, an interaction plot, a mosaic and a normal Q-Q plot of "
        "the Palmer penguins, a P-P plot of Michelson's speed of light, "
        "Kaplan-Meier curves of the 6-MP leukaemia trial and a forest plot "
        "of the thirteen BCG vaccine trials - each with what to look for and "
        "the Statistics model that tests it.",
        (),
        builder="dev.demo.diagnostics_demo:build_diagnostics_demo",
        references=(REF_PENGUINS, REF_FREIREICH, REF_COLDITZ, REF_MICHELSON,),
    ),
    DemoProject(
        "Series operations - fifteen operations, each with its report",
        "Fifteen Series Operations run for real, one per chart, each on a "
        "dataset suited to it - smoothing, spectrum, filtering, baseline, "
        "peaks, roots, calculus, fit, interpolation, statistics, outliers, "
        "clustering, a control chart, a robust regression and a transform - "
        "with the result drawn on the chart and the operation's own report in "
        "the results pane below it, exactly as the operation leaves them.",
        (),
        builder="dev.demo.demo_operations:build_operations_demo",
    ),
)


def copy_demo_project(demo: DemoProject, target: Path) -> Path:
    """Copy *demo*'s pre-built file to *target*, and return the path written.

    The application does not build a demo on the spot any more: several of
    the source datasets (four thousand drivers, three years of daily prices)
    are large enough that rebuilding one on every click would make "Create
    demo" feel like it had hung. Run ``python -m dev.demo.build_demos``
    once to populate :data:`DEMO_DIR`; after that, "Load demo" is just a
    file copy.

    *target* must not be open: close its repository first (see
    MainWindow._on_load_demo).
    """
    source = demo.source_path
    if not source.is_file():
        raise FileNotFoundError(
            f"Demo project not built: {source}. "
            "Run 'python -m dev.demo.build_demos' to build the demo set "
            f"into {DEMO_DIR}."
        )

    target = SqliteRepo.ensure_dhub_extension(Path(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    # The files beside the previous copy go with it. SQLite replays a
    # leftover -wal onto whatever file now has its name, so the edits made
    # to the last copy came back on top of the pristine one; and its undo
    # history would "restore" tables from a project that is no longer there.
    for path in (target, *project_side_files(target)):
        path.unlink(missing_ok=True)
    shutil.copy2(source, target)
    # Its preview goes with it, under the name the project's preview_path
    # gives it (the same stem, so the same name).
    preview = demo.preview_path
    if preview is not None:
        shutil.copy2(preview, target.parent / preview.name)
    return target


def project_side_files(project: Path) -> tuple[Path, ...]:
    """The files SQLite and the undo store keep beside *project*."""
    undo = project.with_name(project.name + ".undo.db")
    return tuple(
        base.with_name(base.name + suffix)
        for base in (project, undo)
        for suffix in ("-wal", "-shm")
    ) + (undo,)
