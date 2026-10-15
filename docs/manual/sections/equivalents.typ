#import "helpers.typ": *

#page(flipped: true)[
= Equivalents in other programs <equivalents>

Coming from JMP, SPSS, Minitab, Origin or Excel? Find the task you know in the left column: the second says where
it is in ChartLibre, the others where it is in each program. A dash means the program has no direct equivalent.
The search (the *Search* frame of the Analysis panel, or #emph[Ctrl+K] - #emph[Cmd+K] on macOS) knows these names
too: typing "Fit Y by X", "Crosstabs" or "Control Chart Builder" finds the ChartLibre function.

#note[
  Menu paths are those of JMP 18, SPSS 29, Minitab 22, Origin 2024 and Excel 365 with the Analysis ToolPak. They are
  indicative: versions move things, and an equivalent does not always compute exactly the same statistics or offer
  the same options.
]

#set text(size: 7.6pt)
#set par(justify: false)
#table(
  columns: (1.1fr, 1.25fr, 1.4fr, 1.4fr, 1.4fr, 1.4fr),
  stroke: 0.4pt + rgb("#d0d0d6"),
  inset: 4pt,
  fill: (_, row) => if row == 0 { rgb("#eef3fb") },
  table.header([*Task*], [*ChartLibre*], [*JMP*], [*SPSS*], [*Minitab*], [*Origin / Excel*]),
  [*Descriptive statistics*], [Data operations > Statistics > Descriptive], [Analyze > Distribution], [Analyze > Descriptive Statistics > Descriptives / Explore], [Stat > Basic Statistics > Display Descriptive Statistics], [Origin: Statistics > Descriptive Statistics; Excel: Data Analysis > Descriptive Statistics],
  [*Distribution: quantiles, moments, normality*], [Analysis > Distribution], [Analyze > Distribution], [Analyze > Descriptive Statistics > Explore / Frequencies], [Stat > Basic Statistics > Graphical Summary], [Origin: Statistics > Descriptive Statistics > Statistics on Columns],
  [*Process capability (Cp, Cpk, Pp, Ppk)*], [Analysis > Distribution (spec limits)], [Distribution > Process Capability; Analyze > Quality and Process > Process Capability], [Analyze > Quality Control > Control Charts (Process Capability statistics)], [Stat > Quality Tools > Capability Analysis > Normal], [Origin: Statistics > Quality > Process Capability Analysis],
  [*Normality test*], [Data operations > Statistics > Normality / shape], [Distribution > Continuous Fit > Normal (Goodness of Fit)], [Analyze > Descriptive Statistics > Explore > Normality plots with tests], [Stat > Basic Statistics > Normality Test], [Origin: Statistics > Descriptive Statistics > Normality Test],
  [*One-sample t-test*], [Data operations > Statistics > One-sample tests], [Distribution > Test Mean], [Analyze > Compare Means > One-Sample T Test], [Stat > Basic Statistics > 1-Sample t], [Origin: Statistics > Hypothesis Testing > One-Sample t-Test],
  [*Two-sample and paired tests*], [Data operations > Statistics > Independent / Paired-sample tests], [Analyze > Fit Y by X > Oneway > t Test; Analyze > Specialized Modeling > Matched Pairs], [Analyze > Compare Means > Independent-Samples / Paired-Samples T Test], [Stat > Basic Statistics > 2-Sample t / Paired t], [Origin: Statistics > Hypothesis Testing > Two-Sample t-Test; Excel: Data Analysis > t-Test],
  [*One-way ANOVA*], [Data operations > Statistics > Group comparison (ANOVA)], [Analyze > Fit Y by X > Oneway > Means/Anova], [Analyze > Compare Means > One-Way ANOVA], [Stat > ANOVA > One-Way], [Origin: Statistics > ANOVA > One-Way ANOVA; Excel: Data Analysis > Anova: Single Factor],
  [*Y by X: the analysis chosen by the types*], [Analysis > Fit Y by X], [Analyze > Fit Y by X], [Analyze > Compare Means / Correlate / Crosstabs / Regression (one per type)], [Stat > ANOVA > One-Way; Stat > Regression > Fitted Line Plot; Stat > Tables], [—],
  [*Pairwise comparisons after ANOVA*], [Analysis > Fit Y by X (Oneway)], [Fit Y by X > Oneway > Compare Means (All Pairs, Tukey HSD); Nonparametric > Steel-Dwass], [Analyze > Compare Means > One-Way ANOVA > Post Hoc (Tukey, Games-Howell)], [Stat > ANOVA > One-Way > Comparisons (Tukey, Games-Howell)], [Origin: Statistics > ANOVA > One-Way ANOVA > Means Comparison],
  [*Two-way ANOVA*], [Data operations > Statistics > Two-way ANOVA], [Analyze > Fit Model (two factors and their interaction)], [Analyze > General Linear Model > Univariate], [Stat > ANOVA > General Linear Model], [Origin: Statistics > ANOVA > Two-Way ANOVA; Excel: Data Analysis > Anova: Two-Factor],
  [*Correlation*], [Data operations > Statistics > Correlation / association], [Analyze > Multivariate Methods > Multivariate], [Analyze > Correlate > Bivariate], [Stat > Basic Statistics > Correlation], [Origin: Statistics > Descriptive Statistics > Correlation Coefficient; Excel: Data Analysis > Correlation],
  [*Contingency table, chi-squared*], [Data operations > Statistics > Contingency table], [Analyze > Fit Y by X > Contingency], [Analyze > Descriptive Statistics > Crosstabs], [Stat > Tables > Cross Tabulation and Chi-Square], [Origin: Statistics > Descriptive Statistics > Cross Tabulation],
  [*Distribution fit*], [Data operations > Statistics > Distribution fit], [Distribution > Continuous Fit], [Analyze > Nonparametric Tests > One Sample (Kolmogorov-Smirnov)], [Stat > Quality Tools > Individual Distribution Identification], [Origin: Statistics > Descriptive Statistics > Distribution Fit],
  [*Linear model, ANOVA, response surface*], [Analysis > Fit Model], [Analyze > Fit Model (Standard Least Squares)], [Analyze > General Linear Model > Univariate; Analyze > Regression > Linear], [Stat > Regression > Fit Regression Model; Stat > DOE > Analyze Response Surface Design], [Origin: Statistics > Fitting > Multiple Linear Regression; Excel: Data Analysis > Regression],
  [*Stepwise regression*], [Analysis > Fit Model > Stepwise], [Analyze > Fit Model (Stepwise)], [Analyze > Regression > Linear (Method: Stepwise)], [Stat > Regression > Fit Regression Model > Stepwise], [—],
  [*Logistic regression*], [Analysis > Fit Model > Nominal / Ordinal Logistic], [Analyze > Fit Model (Nominal / Ordinal Logistic); Fit Y by X > Logistic], [Analyze > Regression > Binary Logistic / Multinomial Logistic / Ordinal], [Stat > Regression > Binary / Ordinal / Nominal Logistic Regression], [Origin: Statistics > Fitting > Logistic Regression],
  [*Generalized linear model*], [Analysis > Fit Model > Generalized Linear Model], [Analyze > Fit Model (Generalized Linear Model)], [Analyze > Generalized Linear Models > Generalized Linear Models], [Stat > Regression > Poisson Regression], [—],
  [*Decision tree*], [Analysis > Partition], [Analyze > Predictive Modeling > Partition], [Analyze > Classify > Tree], [Stat > Predictive Analytics > CART Classification / Regression], [—],
  [*Curve fitting, nonlinear regression*], [Data operations > Fit], [Analyze > Fit Y by X > Bivariate > Fit Polynomial; Analyze > Specialized Modeling > Nonlinear], [Analyze > Regression > Curve Estimation / Nonlinear], [Stat > Regression > Fitted Line Plot / Nonlinear Regression], [Origin: Analysis > Fitting > Nonlinear Curve Fit; Excel: chart Trendline],
  [*Robust and machine-learning regression*], [Data operations > Regression], [Analyze > Predictive Modeling > Bootstrap Forest / Boosted Tree], [—], [Stat > Predictive Analytics > Random Forests / TreeNet], [—],
  [*Gaussian process*], [Data operations > GP Regression], [Analyze > Specialized Modeling > Gaussian Process], [—], [—], [—],
  [*Outliers*], [Data operations > Outliers], [Analyze > Screening > Explore Outliers], [Analyze > Descriptive Statistics > Explore (Outliers)], [Stat > Basic Statistics > Outlier Test], [Origin: Statistics > Descriptive Statistics > Grubbs Test],
  [*Cluster analysis*], [Data operations > Clustering], [Analyze > Clustering > K Means Cluster / Hierarchical Cluster], [Analyze > Classify > K-Means Cluster / Hierarchical Cluster], [Stat > Multivariate > Cluster K-Means / Cluster Observations], [Origin: Statistics > Multivariate Analysis > K-Means Cluster Analysis],
  [*Principal components*], [Data operations > Decomposition], [Analyze > Multivariate Methods > Principal Components], [Analyze > Dimension Reduction > Factor], [Stat > Multivariate > Principal Components], [Origin: Statistics > Multivariate Analysis > Principal Component Analysis],
  [*Control charts*], [Data operations > Control Chart], [Analyze > Quality and Process > Control Chart Builder], [Analyze > Quality Control > Control Charts], [Stat > Control Charts], [Origin: Plot > Specialized > QC (X bar R) Chart],
  [*Pareto chart*], [Chart: Pareto Chart], [Analyze > Quality and Process > Pareto Plot], [Analyze > Quality Control > Pareto Charts], [Stat > Quality Tools > Pareto Chart], [Excel: Insert > Statistic Chart > Pareto],
  [*Cause-and-effect diagram*], [Chart: Fishbone Diagram], [Analyze > Quality and Process > Diagram], [—], [Stat > Quality Tools > Cause-and-Effect], [—],
  [*Survival analysis*], [Data operations > Statistics > Survival; Chart: Kaplan-Meier], [Analyze > Reliability and Survival > Survival], [Analyze > Survival > Kaplan-Meier], [Stat > Reliability/Survival > Distribution Analysis (Right Censoring) > Nonparametric], [Origin: Statistics > Survival Analysis > Kaplan-Meier Estimator],
  [*Smoothing, moving average*], [Data operations > Smoothing], [Analyze > Fit Y by X > Bivariate > Flexible > Kernel Smoother / Fit Spline], [Transform > Create Time Series (moving averages, smoothing)], [Stat > Time Series > Moving Average / Single Exp Smoothing], [Origin: Analysis > Signal Processing > Smooth; Excel: Data Analysis > Moving Average],
  [*Spectrum, FFT*], [Data operations > Spectral Analysis], [Analyze > Specialized Modeling > Time Series > Spectral Density], [Analyze > Forecasting > Spectral Analysis], [—], [Origin: Analysis > Signal Processing > FFT; Excel: Data Analysis > Fourier Analysis],
  [*Digital filters, detrending*], [Data operations > Filtering], [—], [—], [—], [Origin: Analysis > Signal Processing > FFT Filters / IIR Filter],
  [*Peaks*], [Data operations > Peaks], [—], [—], [—], [Origin: Analysis > Peaks and Baseline > Peak Analyzer],
  [*Baseline subtraction*], [Data operations > Baseline Correction], [—], [—], [—], [Origin: Analysis > Peaks and Baseline > Peak Analyzer (Subtract Baseline)],
  [*Derivative, integral*], [Data operations > Calculus], [—], [—], [—], [Origin: Analysis > Mathematics > Differentiate / Integrate],
  [*Interpolation, missing values*], [Data operations > Interpolation], [—], [Transform > Replace Missing Values], [—], [Origin: Analysis > Mathematics > Interpolate/Extrapolate],
  [*Transformations (log, Box-Cox, standardize)*], [Data operations > Transform], [Cols > New Formula Column > Transform], [Transform > Compute Variable], [Calc > Standardize; Stat > Control Charts > Box-Cox Transformation], [Origin: Analysis > Mathematics > Normalize],
  [*Design of experiments*], [Tables > Create a table from a DOE], [DOE > Custom Design / Classical], [Data > Orthogonal Design > Generate], [Stat > DOE > Create Factorial / Response Surface Design], [—],
  [*Join tables*], [Create Query Table > Build > Join], [Tables > Join], [Data > Merge Files > Add Variables], [Data > Merge Worksheets], [Excel: XLOOKUP / Power Query > Merge Queries],
  [*Concatenate tables*], [Create Query Table > Build > Union], [Tables > Concatenate], [Data > Merge Files > Add Cases], [Data > Stack Worksheets], [Excel: Power Query > Append Queries],
  [*Stack, split, transpose*], [Create Query Table > Build > Stack / Split / Transpose], [Tables > Stack / Split / Transpose], [Data > Restructure], [Data > Stack / Unstack Columns; Data > Transpose Columns], [Excel: Power Query > Unpivot / Pivot Column; Paste Special > Transpose],
  [*Recode and bin*], [Create Query Table > Build > Recode / Bin], [Cols > Recode; Cols > Utilities > Make Binning Formula], [Transform > Recode into Different Variables; Transform > Visual Binning], [Data > Recode; Data > Code], [Excel: IFS / XLOOKUP; Data Analysis > Histogram bins],
  [*Summary table (group and aggregate)*], [Edit table > Group and aggregate], [Tables > Summary], [Data > Aggregate], [Stat > Tables > Descriptive Statistics], [Excel: Insert > PivotTable],
  [*Charts*], [Data operations > Plot], [Graph > Graph Builder], [Graphs > Chart Builder], [Graph], [Origin: Plot; Excel: Insert > Charts],
)
]
