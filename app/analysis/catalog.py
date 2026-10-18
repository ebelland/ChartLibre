"""What the application can do, in one list: for search, and for the guided analysis.

Every analysis (table operations), data operation, chart type and table
tool is an entry with its name and sentence, and the words someone might
look for it by (synonyms, in English and Italian, so a search finds "ANOVA"
or "anomalie" whichever language the window is in) and the goals it serves.
For the plugins all of it is read from the class - Name, Description,
Keywords and Goals, as the panels read them - so a new plugin joins search
and the guided analysis by declaring them. Only the table tools, which are
not classes, are listed here.

The goals are the guided analysis's first question - "what do you want to
do?" - in everyday words; each entry says which of them it answers.

Nothing here opens anything: an entry carries its ``kind`` and a
``payload`` (the scanner's entry, a chart type's name, a tool's key) and the
main window opens it (MainWindow.open_catalog_entry).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any

#: The kinds of entry, in the order search results list them.
ANALYSIS = "analysis"
SERIES = "series"
CHART = "chart"
TOOL = "tool"

#: Each kind as the results label it.
KIND_LABELS: dict[str, str] = {
    ANALYSIS: "Analysis",
    SERIES: "Data operation",
    CHART: "Chart",
    TOOL: "Table",
}

#: The guided analysis's goals: (key, title, sentence).
GOALS: tuple[tuple[str, str, str], ...] = (
    ("describe", "Describe a variable", "Its distribution, centre and spread."),
    ("compare", "Compare groups", "Is a measure different from one group to another?"),
    ("relate", "Relate two variables", "How one measure changes with another."),
    ("explain", "Find what affects a response", "Which factors matter, and by how much."),
    ("predict", "Predict a value", "A model that gives Y from the X."),
    ("classify", "Predict a category", "Pass or fail, a class, a grade."),
    ("signal", "Analyse a signal or a time series", "Noise, trends, frequencies, peaks."),
    ("anomalies", "Find anomalies", "The points that do not belong."),
    ("groups", "Find groups in the data", "Rows or curves that are alike."),
    ("experiment", "Plan an experiment", "Which runs to make, and how to analyse them."),
    ("process", "Monitor a process", "Is it stable? Is it in its limits?"),
    ("prepare", "Prepare the data", "Import, combine, compute and reshape tables."),
)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One thing the application can do."""

    kind: str
    #: The English name, as the class has it; shown through _().
    name: str
    description: str = ""
    #: Words to find it by, beyond its name and sentence.
    keywords: tuple[str, ...] = ()
    #: The GOALS keys it answers.
    goals: tuple[str, ...] = ()
    #: What opening it needs: the scanner entry, the chart type, the tool key.
    payload: Any = field(default=None, compare=False)

    @property
    def key(self) -> str:
        """"analysis:Fit Model", "series:Smoothing", "chart:Box Plot", "tool:doe"."""
        return f"{self.kind}:{self.payload if self.kind == TOOL else self.name}"


#: The table tools: key -> (name, description, keywords, goals).
TOOLS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "doe": (
        "Create a table from a DOE",
        "Generate an experiment matrix with factors and responses.",
        "doe design of experiments disegno degli esperimenti piano sperimentale factorial "
        "fattoriale box-behnken central composite composito centrale latin hypercube runs prove",
        ("experiment",),
    ),
    "import": (
        "Import from file",
        "Create a table from a file: CSV, Excel, JSON.",
        "importa import csv excel xlsx json file",
        ("prepare",),
    ),
    "import_web": (
        "Import from the web",
        "Create a table from a web address or a listed open-data source.",
        "importa import web internet url http download scarica open data dati aperti",
        ("prepare",),
    ),
    "import_clipboard": (
        "Import from clipboard",
        "Create a table from the table copied to the clipboard.",
        "importa import clipboard appunti incolla paste copy copia transposed trasposta",
        ("prepare",),
    ),
    "import_database": (
        "Import Database Table",
        "Create a table from a database table.",
        "importa database sqlite tabella table",
        ("prepare",),
    ),
    "query": (
        "Create Query Table",
        "Create a table from a database query.",
        "query interrogazione sql join select filtro filter union unione concatenate concatena stack impila "
        "split dividi transpose trasponi recode ricodifica bin classi intervalli reshape wide long",
        ("prepare",),
    ),
    "new_table": (
        "New Blank Table",
        "Create an empty table with named columns.",
        "nuova tabella vuota new empty table",
        ("prepare",),
    ),
    "edit_table": (
        "Edit table",
        "Edit cells, rows and columns; group and aggregate; add computed columns.",
        "modifica tabella editor edit cells celle rows righe columns colonne group aggregate "
        "raggruppa aggrega somma sum computed column colonna calcolata formula sort ordina "
        "find replace trova sostituisci fill riempi",
        ("prepare",),
    ),
}


# ----------------------------------------------------------------------
# The guided analysis: per goal, what to use and why
# ----------------------------------------------------------------------

#: The response kinds the guided analysis asks about.
ANY, NUMERIC, CATEGORICAL = "any", "numeric", "categorical"

#: goal -> (catalogue key, why, the response it suits: ANY, NUMERIC or CATEGORICAL), best first.
GUIDE: dict[str, tuple[tuple[str, str, str], ...]] = {
    "describe": (
        ("analysis:Distribution", "Histogram, quantiles, moments and normality, one report per column.", NUMERIC),
        ("series:Statistics", "Mean, median, spread, quartiles and normality tests, in one report.", ANY),
        ("chart:Histogram", "The shape of the distribution: centre, spread, skew, several peaks.", NUMERIC),
        ("chart:Box Plot", "Median, quartiles and outliers at a glance - several columns side by side.", NUMERIC),
        ("chart:Q-Q Plot", "Is it normal? Points on the line say yes.", NUMERIC),
        ("chart:Pareto Chart", "How many rows per category, largest first.", CATEGORICAL),
    ),
    "compare": (
        ("analysis:Are there differences?", "Choose the measure and the groups: the test is chosen for you, "
                                             "and the answer is given in words.", ANY),
        ("analysis:Fit Y by X", "The measure by the groups: ANOVA, then every pair compared "
                                "(Tukey, Games-Howell, Dunn).", NUMERIC),
        ("series:Statistics", "t-test, ANOVA, Kruskal-Wallis and their effect sizes between groups.", NUMERIC),
        ("chart:Box Plot", "The groups side by side: medians, spread, outliers.", NUMERIC),
        ("analysis:Fit Model", "Groups and other factors together: effect tests and estimates.", NUMERIC),
        ("chart:Mosaic Plot", "Two categorical variables: shares, then chi-squared in Statistics.", CATEGORICAL),
        ("chart:Kaplan-Meier", "Time until an event, by group, with the log-rank test.", ANY),
    ),
    "relate": (
        ("analysis:Fit Y by X", "Y against X, the analysis chosen by their types: line, ANOVA, "
                                "contingency table or logistic fit.", ANY),
        ("chart:Scatter Plot", "One measure against the other: trend, curvature, clusters.", NUMERIC),
        ("series:Fit Curve", "A curve through the points: line, polynomial, exponential, your own formula.", NUMERIC),
        ("analysis:Multivariate", "Every pair of columns at once: correlations (Pearson, Spearman, Kendall) "
                                  "with their p-values, and the scatterplot matrix.", NUMERIC),
        ("chart:Pair Plot", "Every variable against every other, to choose which to look at.", ANY),
        ("chart:Mosaic Plot", "Two categorical variables: does one depend on the other?", CATEGORICAL),
    ),
    "explain": (
        ("analysis:What has the most impact?", "Choose the result and the factors: they are ranked for you, "
                                               "and the answer is given in words.", ANY),
        ("analysis:Fit Model", "Which factors matter and by how much: effect tests, estimates, profiler.", NUMERIC),
        ("analysis:Partition", "A tree finds the factors and the cut points that split the response best.", ANY),
        ("analysis:Fit Model", "Logistic regression: how each factor moves the odds of each level.", CATEGORICAL),
        ("chart:Interaction Plot", "Does the effect of one factor depend on another?", NUMERIC),
    ),
    "predict": (
        ("analysis:Time Series", "The next values of a series in time: ARIMA or exponential smoothing, "
                                 "with a prediction interval.", NUMERIC),
        ("analysis:Partial Least Squares", "Many X that move together, like a spectrum: a few components, "
                                           "chosen by cross-validation.", NUMERIC),
        ("analysis:Fit Model", "A model you can read and write down, with confidence intervals.", NUMERIC),
        ("analysis:Partition", "Bootstrap Forest or Boosted Tree: accurate, no formula needed.", ANY),
        ("series:Regression", "Robust or machine-learning regression on one X.", NUMERIC),
        ("series:Gaussian Process", "A smooth prediction with its uncertainty band.", NUMERIC),
    ),
    "classify": (
        ("analysis:Partition", "Classification tree or forest: rules and the probability of each class.", CATEGORICAL),
        ("analysis:Fit Model", "Nominal or ordinal logistic regression.", CATEGORICAL),
        ("analysis:Discriminant", "Groups told apart by continuous measures: linear or quadratic, "
                                  "with the canonical plot.", CATEGORICAL),
    ),
    "signal": (
        ("analysis:Has it changed over time?", "A steady trend or a sudden change? The tests are chosen for you, "
                                               "and the answer is given in words.", NUMERIC),
        ("analysis:Time Series", "Its memory (autocorrelation), its trend and season, and a forecast with "
                                 "its prediction interval.", NUMERIC),
        ("series:Smoothing", "Reduce the noise and keep the shape.", NUMERIC),
        ("series:Spectral Analysis", "Which frequencies the signal holds.", NUMERIC),
        ("series:Peaks", "Find peaks and measure their height, width and area.", NUMERIC),
        ("series:Filtering", "Low-, high- or band-pass filters; remove a trend.", NUMERIC),
        ("series:Baseline Correction", "Subtract a spectrum's background.", NUMERIC),
        ("series:Calculus", "Derivative or integral.", NUMERIC),
        ("chart:Time Series", "The values over time.", NUMERIC),
    ),
    "anomalies": (
        ("series:Explore Outliers", "Mark the points that do not belong, by several methods.", NUMERIC),
        ("series:Control Chart", "Points outside the process limits, and runs that break the rules.", NUMERIC),
        ("chart:Box Plot", "Points beyond the whiskers.", NUMERIC),
    ),
    "groups": (
        ("series:Clustering", "k-means, DBSCAN, hierarchical: rows that are alike, coloured.", ANY),
        ("analysis:Principal Components", "A few directions that carry most of many columns' variation: "
                                          "the score plot shows the groups.", NUMERIC),
        ("series:Decomposition", "PCA, t-SNE on a chart's series: see groups among many variables.", ANY),
        ("analysis:Partition", "Segments described by rules on the factors.", ANY),
    ),
    "experiment": (
        ("tool:doe", "Factorial, Box-Behnken, central composite or space-filling runs.", ANY),
        ("analysis:Fit Model", "Analyse the runs: the design's model is proposed for you.", ANY),
        ("chart:Interaction Plot", "See the interactions between factors.", NUMERIC),
    ),
    "process": (
        ("analysis:Has it changed over time?", "A steady trend or a sudden change? The tests are chosen for you, "
                                               "and the answer is given in words.", NUMERIC),
        ("series:Control Chart", "Is the process stable? I-MR, X-bar, p, c, u charts; with specification "
                                 "limits, is it capable? Cp, Cpk, Pp, Ppk.", ANY),
        ("chart:Pareto Chart", "The causes that account for most defects.", CATEGORICAL),
        ("chart:Fishbone Diagram", "Cause and effect, by category.", ANY),
    ),
    "prepare": (
        ("tool:import", "Bring a file, the clipboard or a web table in.", ANY),
        ("tool:edit_table", "Edit cells; group and aggregate; computed columns.", ANY),
        ("tool:query", "Join and filter tables with a query.", ANY),
        ("series:Transform", "Log, Box-Cox, standardise.", NUMERIC),
        ("series:Interpolation", "Fill missing values.", NUMERIC),
    ),
}


def suggestions(entries: list[CatalogEntry], goal: str, response: str = ANY) -> list[tuple[CatalogEntry, str]]:
    """What the guided analysis suggests for *goal*: (entry, why), for a *response* of that kind."""
    by_key = {entry.key: entry for entry in entries}
    return [
        (by_key[key], why) for key, why, suits in GUIDE.get(goal, ())
        if key in by_key and (response == ANY or suits in (ANY, response))
    ]


# ----------------------------------------------------------------------
# The same task in other programs: the manual's table, and search words
# ----------------------------------------------------------------------

#: The programs the equivalents name, in the manual's column order.
PROGRAMS: tuple[str, ...] = ("JMP", "SPSS", "Minitab", "Origin / Excel")

#: (task, where it is in ChartLibre, the catalogue key that opens it,
#: {program: its menu path}). A program missing from a row has no direct
#: equivalent. Menu paths as of JMP 18, SPSS 29, Minitab 22, Origin 2024 and
#: Excel 365; indicative - versions move things.
EQUIVALENTS: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    ("Descriptive statistics", "Analysis > Statistics > Descriptive", "series:Statistics", {
        "JMP": "Analyze > Distribution",
        "SPSS": "Analyze > Descriptive Statistics > Descriptives / Explore",
        "Minitab": "Stat > Basic Statistics > Display Descriptive Statistics",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics; Excel: Data Analysis > Descriptive Statistics",
    }),
    ("Distribution: quantiles, moments, normality", "Analysis > Distribution", "analysis:Distribution", {
        "JMP": "Analyze > Distribution",
        "SPSS": "Analyze > Descriptive Statistics > Explore / Frequencies",
        "Minitab": "Stat > Basic Statistics > Graphical Summary",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics > Statistics on Columns",
    }),
    ("Process capability (Cp, Cpk, Pp, Ppk)", "Analysis > Control Chart (spec limits)",
     "series:Control Chart", {
        "JMP": "Distribution > Process Capability; Analyze > Quality and Process > Process Capability",
        "SPSS": "Analyze > Quality Control > Control Charts (Process Capability statistics)",
        "Minitab": "Stat > Quality Tools > Capability Analysis > Normal",
        "Origin / Excel": "Origin: Statistics > Quality > Process Capability Analysis",
    }),
    ("Normality test", "Analysis > Statistics > Normality / shape", "series:Statistics", {
        "JMP": "Distribution > Continuous Fit > Normal (Goodness of Fit)",
        "SPSS": "Analyze > Descriptive Statistics > Explore > Normality plots with tests",
        "Minitab": "Stat > Basic Statistics > Normality Test",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics > Normality Test",
    }),
    ("One-sample t-test", "Analysis > Statistics > One-sample tests", "series:Statistics", {
        "JMP": "Distribution > Test Mean",
        "SPSS": "Analyze > Compare Means > One-Sample T Test",
        "Minitab": "Stat > Basic Statistics > 1-Sample t",
        "Origin / Excel": "Origin: Statistics > Hypothesis Testing > One-Sample t-Test",
    }),
    ("Two-sample and paired tests", "Analysis > Statistics > Independent / Paired-sample tests",
     "series:Statistics", {
        "JMP": "Analyze > Fit Y by X > Oneway > t Test; Analyze > Specialized Modeling > Matched Pairs",
        "SPSS": "Analyze > Compare Means > Independent-Samples / Paired-Samples T Test",
        "Minitab": "Stat > Basic Statistics > 2-Sample t / Paired t",
        "Origin / Excel": "Origin: Statistics > Hypothesis Testing > Two-Sample t-Test; Excel: Data Analysis > t-Test",
    }),
    ("One-way ANOVA", "Analysis > Statistics > Group comparison (ANOVA)", "series:Statistics", {
        "JMP": "Analyze > Fit Y by X > Oneway > Means/Anova",
        "SPSS": "Analyze > Compare Means > One-Way ANOVA",
        "Minitab": "Stat > ANOVA > One-Way",
        "Origin / Excel": "Origin: Statistics > ANOVA > One-Way ANOVA; Excel: Data Analysis > Anova: Single Factor",
    }),
    ("Y by X: the analysis chosen by the types", "Analysis > Fit Y by X", "analysis:Fit Y by X", {
        "JMP": "Analyze > Fit Y by X",
        "SPSS": "Analyze > Compare Means / Correlate / Crosstabs / Regression (one per type)",
        "Minitab": "Stat > ANOVA > One-Way; Stat > Regression > Fitted Line Plot; Stat > Tables",
    }),
    ("Pairwise comparisons after ANOVA", "Analysis > Fit Y by X (Oneway)", "analysis:Fit Y by X", {
        "JMP": "Fit Y by X > Oneway > Compare Means (All Pairs, Tukey HSD); Nonparametric > Steel-Dwass",
        "SPSS": "Analyze > Compare Means > One-Way ANOVA > Post Hoc (Tukey, Games-Howell)",
        "Minitab": "Stat > ANOVA > One-Way > Comparisons (Tukey, Games-Howell)",
        "Origin / Excel": "Origin: Statistics > ANOVA > One-Way ANOVA > Means Comparison",
    }),
    ("Two-way ANOVA", "Analysis > Statistics > Two-way ANOVA", "series:Statistics", {
        "JMP": "Analyze > Fit Model (two factors and their interaction)",
        "SPSS": "Analyze > General Linear Model > Univariate",
        "Minitab": "Stat > ANOVA > General Linear Model",
        "Origin / Excel": "Origin: Statistics > ANOVA > Two-Way ANOVA; Excel: Data Analysis > Anova: Two-Factor",
    }),
    ("Correlation", "Analysis > Multivariate Methods > Multivariate", "analysis:Multivariate", {
        "JMP": "Analyze > Multivariate Methods > Multivariate",
        "SPSS": "Analyze > Correlate > Bivariate",
        "Minitab": "Stat > Basic Statistics > Correlation",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics > Correlation Coefficient; Excel: Data Analysis > Correlation",
    }),
    ("Contingency table, chi-squared", "Analysis > Statistics > Contingency table", "series:Statistics", {
        "JMP": "Analyze > Fit Y by X > Contingency",
        "SPSS": "Analyze > Descriptive Statistics > Crosstabs",
        "Minitab": "Stat > Tables > Cross Tabulation and Chi-Square",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics > Cross Tabulation",
    }),
    ("Distribution fit", "Analysis > Statistics > Distribution fit", "series:Statistics", {
        "JMP": "Distribution > Continuous Fit",
        "SPSS": "Analyze > Nonparametric Tests > One Sample (Kolmogorov-Smirnov)",
        "Minitab": "Stat > Quality Tools > Individual Distribution Identification",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics > Distribution Fit",
    }),
    ("Linear model, ANOVA, response surface", "Analysis > Fit Model", "analysis:Fit Model", {
        "JMP": "Analyze > Fit Model (Standard Least Squares)",
        "SPSS": "Analyze > General Linear Model > Univariate; Analyze > Regression > Linear",
        "Minitab": "Stat > Regression > Fit Regression Model; Stat > DOE > Analyze Response Surface Design",
        "Origin / Excel": "Origin: Statistics > Fitting > Multiple Linear Regression; Excel: Data Analysis > Regression",
    }),
    ("Stepwise regression", "Analysis > Fit Model > Stepwise", "analysis:Fit Model", {
        "JMP": "Analyze > Fit Model (Stepwise)",
        "SPSS": "Analyze > Regression > Linear (Method: Stepwise)",
        "Minitab": "Stat > Regression > Fit Regression Model > Stepwise",
    }),
    ("Logistic regression", "Analysis > Fit Model > Nominal / Ordinal Logistic", "analysis:Fit Model", {
        "JMP": "Analyze > Fit Model (Nominal / Ordinal Logistic); Fit Y by X > Logistic",
        "SPSS": "Analyze > Regression > Binary Logistic / Multinomial Logistic / Ordinal",
        "Minitab": "Stat > Regression > Binary / Ordinal / Nominal Logistic Regression",
        "Origin / Excel": "Origin: Statistics > Fitting > Logistic Regression",
    }),
    ("Generalized linear model", "Analysis > Fit Model > Generalized Linear Model", "analysis:Fit Model", {
        "JMP": "Analyze > Fit Model (Generalized Linear Model)",
        "SPSS": "Analyze > Generalized Linear Models > Generalized Linear Models",
        "Minitab": "Stat > Regression > Poisson Regression",
    }),
    ("Decision tree", "Analysis > Partition", "analysis:Partition", {
        "JMP": "Analyze > Predictive Modeling > Partition",
        "SPSS": "Analyze > Classify > Tree",
        "Minitab": "Stat > Predictive Analytics > CART Classification / Regression",
    }),
    ("Curve fitting, nonlinear regression", "Analysis > Fit Curve", "series:Fit Curve", {
        "JMP": "Analyze > Fit Y by X > Bivariate > Fit Polynomial; Analyze > Specialized Modeling > Nonlinear",
        "SPSS": "Analyze > Regression > Curve Estimation / Nonlinear",
        "Minitab": "Stat > Regression > Fitted Line Plot / Nonlinear Regression",
        "Origin / Excel": "Origin: Analysis > Fitting > Nonlinear Curve Fit; Excel: chart Trendline",
    }),
    ("Robust and machine-learning regression", "Analysis > Regression", "series:Regression", {
        "JMP": "Analyze > Predictive Modeling > Bootstrap Forest / Boosted Tree",
        "Minitab": "Stat > Predictive Analytics > Random Forests / TreeNet",
    }),
    ("Gaussian process", "Analysis > Gaussian Process", "series:Gaussian Process", {
        "JMP": "Analyze > Specialized Modeling > Gaussian Process",
    }),
    ("Outliers", "Analysis > Explore Outliers", "series:Explore Outliers", {
        "JMP": "Analyze > Screening > Explore Outliers",
        "SPSS": "Analyze > Descriptive Statistics > Explore (Outliers)",
        "Minitab": "Stat > Basic Statistics > Outlier Test",
        "Origin / Excel": "Origin: Statistics > Descriptive Statistics > Grubbs Test",
    }),
    ("Cluster analysis", "Analysis > Clustering", "series:Clustering", {
        "JMP": "Analyze > Clustering > K Means Cluster / Hierarchical Cluster",
        "SPSS": "Analyze > Classify > K-Means Cluster / Hierarchical Cluster",
        "Minitab": "Stat > Multivariate > Cluster K-Means / Cluster Observations",
        "Origin / Excel": "Origin: Statistics > Multivariate Analysis > K-Means Cluster Analysis",
    }),
    ("Principal components", "Analysis > Multivariate Methods > Principal Components", "analysis:Principal Components", {
        "JMP": "Analyze > Multivariate Methods > Principal Components",
        "SPSS": "Analyze > Dimension Reduction > Factor",
        "Minitab": "Stat > Multivariate > Principal Components",
        "Origin / Excel": "Origin: Statistics > Multivariate Analysis > Principal Component Analysis",
    }),
    ("Control charts", "Analysis > Control Chart", "series:Control Chart", {
        "JMP": "Analyze > Quality and Process > Control Chart Builder",
        "SPSS": "Analyze > Quality Control > Control Charts",
        "Minitab": "Stat > Control Charts",
        "Origin / Excel": "Origin: Plot > Specialized > QC (X bar R) Chart",
    }),
    ("Pareto chart", "Chart: Pareto Chart", "chart:Pareto Chart", {
        "JMP": "Analyze > Quality and Process > Pareto Plot",
        "SPSS": "Analyze > Quality Control > Pareto Charts",
        "Minitab": "Stat > Quality Tools > Pareto Chart",
        "Origin / Excel": "Excel: Insert > Statistic Chart > Pareto",
    }),
    ("Cause-and-effect diagram", "Chart: Fishbone Diagram", "chart:Fishbone Diagram", {
        "JMP": "Analyze > Quality and Process > Diagram",
        "Minitab": "Stat > Quality Tools > Cause-and-Effect",
    }),
    ("Survival analysis", "Analysis > Statistics > Survival; Chart: Kaplan-Meier", "chart:Kaplan-Meier", {
        "JMP": "Analyze > Reliability and Survival > Survival",
        "SPSS": "Analyze > Survival > Kaplan-Meier",
        "Minitab": "Stat > Reliability/Survival > Distribution Analysis (Right Censoring) > Nonparametric",
        "Origin / Excel": "Origin: Statistics > Survival Analysis > Kaplan-Meier Estimator",
    }),
    ("Smoothing, moving average", "Analysis > Smoothing", "series:Smoothing", {
        "JMP": "Analyze > Fit Y by X > Bivariate > Flexible > Kernel Smoother / Fit Spline",
        "SPSS": "Transform > Create Time Series (moving averages, smoothing)",
        "Minitab": "Stat > Time Series > Moving Average / Single Exp Smoothing",
        "Origin / Excel": "Origin: Analysis > Signal Processing > Smooth; Excel: Data Analysis > Moving Average",
    }),
    ("Time series forecast, ARIMA, ACF", "Analysis > Predictive and Specialized Modeling > Time Series",
     "analysis:Time Series", {
        "JMP": "Analyze > Specialized Modeling > Time Series",
        "SPSS": "Analyze > Forecasting > Create Traditional Models / Autocorrelations",
        "Minitab": "Stat > Time Series > ARIMA / Autocorrelation / Decomposition",
        "Origin / Excel": "Origin: Statistics > Time Series Analysis; Excel: Forecast Sheet (ETS)",
    }),
    ("Spectrum, FFT", "Analysis > Spectral Analysis", "series:Spectral Analysis", {
        "JMP": "Analyze > Specialized Modeling > Time Series > Spectral Density",
        "SPSS": "Analyze > Forecasting > Spectral Analysis",
        "Origin / Excel": "Origin: Analysis > Signal Processing > FFT; Excel: Data Analysis > Fourier Analysis",
    }),
    ("Digital filters, detrending", "Analysis > Filtering", "series:Filtering", {
        "Origin / Excel": "Origin: Analysis > Signal Processing > FFT Filters / IIR Filter",
    }),
    ("Peaks", "Analysis > Peaks", "series:Peaks", {
        "Origin / Excel": "Origin: Analysis > Peaks and Baseline > Peak Analyzer",
    }),
    ("Baseline subtraction", "Analysis > Baseline Correction", "series:Baseline Correction", {
        "Origin / Excel": "Origin: Analysis > Peaks and Baseline > Peak Analyzer (Subtract Baseline)",
    }),
    ("Derivative, integral", "Analysis > Calculus", "series:Calculus", {
        "Origin / Excel": "Origin: Analysis > Mathematics > Differentiate / Integrate",
    }),
    ("Interpolation, missing values", "Analysis > Interpolation", "series:Interpolation", {
        "SPSS": "Transform > Replace Missing Values",
        "Origin / Excel": "Origin: Analysis > Mathematics > Interpolate/Extrapolate",
    }),
    ("Transformations (log, Box-Cox, standardize)", "Analysis > Transform", "series:Transform", {
        "JMP": "Cols > New Formula Column > Transform",
        "SPSS": "Transform > Compute Variable",
        "Minitab": "Calc > Standardize; Stat > Control Charts > Box-Cox Transformation",
        "Origin / Excel": "Origin: Analysis > Mathematics > Normalize",
    }),
    ("Design of experiments", "Analysis > Design of Experiments", "tool:doe", {
        "JMP": "DOE > Custom Design / Classical",
        "SPSS": "Data > Orthogonal Design > Generate",
        "Minitab": "Stat > DOE > Create Factorial / Response Surface Design",
    }),
    ("Join tables", "Create Query Table > Build > Join", "tool:query", {
        "JMP": "Tables > Join",
        "SPSS": "Data > Merge Files > Add Variables",
        "Minitab": "Data > Merge Worksheets",
        "Origin / Excel": "Excel: XLOOKUP / Power Query > Merge Queries",
    }),
    ("Concatenate tables", "Create Query Table > Build > Union", "tool:query", {
        "JMP": "Tables > Concatenate",
        "SPSS": "Data > Merge Files > Add Cases",
        "Minitab": "Data > Stack Worksheets",
        "Origin / Excel": "Excel: Power Query > Append Queries",
    }),
    ("Stack, split, transpose", "Create Query Table > Build > Stack / Split / Transpose", "tool:query", {
        "JMP": "Tables > Stack / Split / Transpose",
        "SPSS": "Data > Restructure",
        "Minitab": "Data > Stack / Unstack Columns; Data > Transpose Columns",
        "Origin / Excel": "Excel: Power Query > Unpivot / Pivot Column; Paste Special > Transpose",
    }),
    ("Recode and bin", "Create Query Table > Build > Recode / Bin", "tool:query", {
        "JMP": "Cols > Recode; Cols > Utilities > Make Binning Formula",
        "SPSS": "Transform > Recode into Different Variables; Transform > Visual Binning",
        "Minitab": "Data > Recode; Data > Code",
        "Origin / Excel": "Excel: IFS / XLOOKUP; Data Analysis > Histogram bins",
    }),
    ("Summary table (group and aggregate)", "Edit table > Group and aggregate", "tool:edit_table", {
        "JMP": "Tables > Summary",
        "SPSS": "Data > Aggregate",
        "Minitab": "Stat > Tables > Descriptive Statistics",
        "Origin / Excel": "Excel: Insert > PivotTable",
    }),
    ("Charts", "Analysis > New Chart", "series:New Chart", {
        "JMP": "Graph > Graph Builder",
        "SPSS": "Graphs > Chart Builder",
        "Minitab": "Graph",
        "Origin / Excel": "Origin: Plot; Excel: Insert > Charts",
    }),
)


def _equivalent_words() -> dict[str, tuple[str, ...]]:
    """Catalogue key -> the other programs' menu words, to search by ("Fit Y by X", "Crosstabs")."""
    words: dict[str, list[str]] = {}
    for _task, _where, key, programs in EQUIVALENTS:
        for path in programs.values():
            words.setdefault(key, []).extend(path.replace(">", " ").replace(";", " ").replace("/", " ").split())
    return {key: tuple(dict.fromkeys(found)) for key, found in words.items()}


# ----------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


#: The New plot "operation", as the Data operations panel lists it: the one
#: entry that is not a plugin class, so it carries its own words and goals.
_PLOT: dict[str, Any] = {"name": "NewPlotTabDialog", "value": "New Chart", "description": "Create a new plot",
                         "builtin": True, "keywords": "nuovo grafico new plot chart grafico draw disegna", "goals": []}


def build_catalog(
    table_operations: list[dict] | None = None,
    data_operations: list[dict] | None = None,
    renderers: list[dict] | None = None,
) -> list[CatalogEntry]:
    """Every entry: analyses, data operations, chart types, table tools - in that order.

    The lists default to what the scanners found; tests pass their own.
    """
    if table_operations is None:
        from app.scanners.table_operation_scanner import table_operations
    if data_operations is None:
        from app.scanners.data_operation_scanner import data_operations
    if renderers is None:
        from app.scanners.axis_renderer_scanner import renderers

    entries: list[CatalogEntry] = []
    for kind, found in (
        (ANALYSIS, table_operations),
        (SERIES, [_PLOT, *data_operations]),
        (CHART, renderers),
    ):
        for item in found:
            name = str(item.get("value") or item.get("name") or "")
            if not name:
                continue
            # The words and goals are the plugin's own (its Keywords and Goals
            # class attributes), as its name and sentence are.
            words = str(item.get("keywords") or "")
            goals = tuple(item.get("goals") or ())
            entries.append(CatalogEntry(
                kind=kind, name=name, description=str(item.get("description") or ""),
                keywords=_words(words), goals=goals, payload=item if kind != CHART else name,
            ))
    for key, (name, description, words, goals) in TOOLS.items():
        entries.append(CatalogEntry(TOOL, name, description, _words(words), goals, payload=key))
    # The other programs' names for the same thing find it too - as the last
    # keywords, so its own words still rank first.
    other = _equivalent_words()
    return [replace(entry, keywords=entry.keywords + other.get(entry.key, ())) if entry.key in other else entry
            for entry in entries]


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def fold(text: str) -> str:
    """*text* for matching: lower case, accents off ("Stabilità" finds "stabilita")."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def search(entries: list[CatalogEntry], text: str, translate=lambda s: s, limit: int | None = None) -> list[CatalogEntry]:
    """The entries matching every word of *text*, best first.

    A word matches the name (English or as *translate* gives it), a keyword
    or the description. The name counts most - the start of it most of all -
    then the keywords, then the description; ties keep catalogue order.
    """
    words = fold(text).split()
    if not words:
        return []
    scored = []
    for order, entry in enumerate(entries):
        names = fold(f"{entry.name} {translate(entry.name)}")
        keywords = fold(" ".join(entry.keywords))
        description = fold(f"{entry.description} {translate(entry.description)}")
        score = 0
        for word in words:
            if any(name.startswith(word) for name in (fold(entry.name), fold(translate(entry.name)))):
                score += 8
            elif word in names:
                score += 5
            elif any(keyword.startswith(word) for keyword in keywords.split()):
                score += 3
            elif word in keywords:
                score += 2
            elif word in description:
                score += 1
            else:
                break
        else:
            scored.append((-score, order, entry))
    scored.sort(key=lambda item: (item[0], item[1]))
    found = [entry for _score, _order, entry in scored]
    return found[:limit] if limit else found
