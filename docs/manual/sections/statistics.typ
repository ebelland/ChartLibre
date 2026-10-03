#import "helpers.typ": *

= Statistics <statistics>

*Statistics* (in the *Statistics* section of Series Operations) answers the questions people ask of their data before or instead of drawing a conclusion: what is the centre and spread, is it normal, do these groups differ, are these variables associated. It reads the series ticked in its list and writes a formatted report in the Results pane; it changes no data and writes no tables.

== Using the window

+ Tick the series to analyse. One series is enough for descriptive, one-sample and normality statistics; tests that compare need two or more.
+ Choose the *Model* — the family of statistics to report, or *All statistics* for every family that applies.
+ Set the options the model uses (reference value, alternative hypothesis, ...).
+ *Preview* computes the report. *Copy* copies it, with its formatting, for a lab notebook or an e-mail.
+ To also draw a chart, tick *Add a chart* and pick one. *Match the model* chooses the chart that shows what the model measures; *Preview* draws it on a new axis of the current figure, *OK* keeps it, and *Close* removes it again.

#defs(
  [*Reference value*], [The value one-sample tests compare the centre with (H#sub[0]: mean = reference).],
  [*Alternative*], [*Two-sided* (different), *greater* or *less* — the direction of the alternative hypothesis, for the tests that support it.],
  [*Trimmed mean*], [The percentage cut from each end for the trimmed mean.],
  [*Anderson method*], [How the Anderson–Darling p-value is obtained: interpolated from tables, or by Monte Carlo (with a number of *resamples* and a *batch* size).],
  [*Rank by*], [For distribution fitting: AIC (default), BIC or the KS statistic.],
  [*Try every SciPy distribution*], [Fit all ~100 continuous distributions instead of the 15 common ones. Slow, and many of the extra candidates fit no real measurement.],
)

#note[
  How to read a p-value: it is the probability of seeing a difference at least this large *if there were no real difference*. A small p-value (conventionally below 0.05) says the data are hard to explain by chance alone. It does not say how large or important the difference is — read the effect sizes (Cohen's d, η², Cramér's V) for that — and a large p-value does not prove there is no difference, only that these data do not show one.
]

== Models on values

These models read the numeric values of each ticked series (its `y`, or its `value` for distribution charts).

=== Descriptive statistics
For each series: count, mean, trimmed mean, geometric and harmonic means (for positive data), median, mode and how often it occurs, standard deviation, variance, standard error of the mean, minimum, quartiles, maximum, range, interquartile range, median absolute deviation, skewness, kurtosis, entropy, and the 95% confidence interval of the mean.

*Charts:* Histogram, ECDF.

=== One-sample tests
Is the centre of the series equal to the reference value?
#defs(
  [*One-sample t-test*], [Compares the mean; assumes roughly normal data (or a large sample).],
  [*Wilcoxon signed-rank*], [Compares the median without assuming normality; assumes a symmetric distribution.],
  [*Sign test*], [Counts values above and below the reference. Assumes almost nothing; least powerful.],
)
*Charts:* Histogram, ECDF.

=== Normality / shape tests
Does the series look like a sample from a normal distribution?
#defs(
  [*Shapiro-Wilk*], [The most powerful general test for small and medium samples.],
  [*D'Agostino-Pearson*], [Combines skewness and kurtosis; needs at least about 20 values.],
  [*Skewness test* / *Kurtosis test*], [Each part separately: is the distribution lopsided, are its tails heavier or lighter than normal?],
  [*Anderson-Darling*], [Gives extra weight to the tails.],
  [*Jarque-Bera*], [Skewness and kurtosis for large samples.],
  [*Kolmogorov-Smirnov vs normal*], [The largest gap between the sample's ECDF and the fitted normal CDF.],
)
With thousands of values, every test rejects normality for tiny departures that do not matter in practice; with ten values, none of them can detect much. Look at the Q-Q plot as well.

*Charts:* Histogram, ECDF, *Q-Q Plot*, *P-P Plot*.

=== Paired-sample tests
For two series measured on the *same units* — before and after on the same parts, two instruments on the same samples. Values are paired by row (or by matching x). For every pair of ticked series:
#defs(
  [*Paired t-test*], [Is the mean difference zero?],
  [*Wilcoxon signed-rank paired test*], [The same without assuming normal differences.],
  [*Sign test*], [Counts which of the two is larger in each pair.],
)
*Chart:* Scatter Plot of one series against the other.

=== Independent-sample tests
For two groups of *different units* — parts from two machines, patients in two arms. For every pair of ticked series:
#defs(
  [*Welch t-test*], [The default reading: compares means without assuming equal variances.],
  [*Student t-test*], [Assumes equal variances.],
  [*Mann-Whitney U*], [Compares the distributions by rank; no normality needed.],
  [*Levene's test* and *Brown-Forsythe*], [Are the variances equal? (Brown-Forsythe uses the median, and is less affected by non-normal data.)],
  [*Difference of means* and *Cohen's d*], [The size of the difference in standard deviations (about 0.2 small, 0.5 medium, 0.8 large).],
)
*Chart:* Box Plot.

=== Group comparison (ANOVA)
All ticked series together, as groups:
#defs(
  [*One-way ANOVA*], [Do the group means differ? With η², the share of variance explained by the groups.],
  [*Welch's ANOVA*], [The same without assuming equal variances — read this one when Levene's test says the variances differ.],
  [*Kruskal-Wallis*], [The rank-based version, with ε² as effect size.],
  [*Levene's test*], [Equal variances across all groups.],
  [*Tukey's HSD*], [Which pairs differ: every pair with its difference, adjusted p-value and 95% interval, so that testing many pairs does not inflate false positives.],
)
*Chart:* Box Plot.

=== Correlation / association
For every pair of ticked series, paired by row: *Pearson* r (linear association), *Spearman* ρ (monotone association, by rank), *Kendall* τ (rank agreement, robust for small samples) and the least-squares *regression slope*. Correlation measures association, not cause.

*Chart:* Scatter Plot.

=== Distribution fit
Fits candidate distributions (normal, lognormal, gamma, Weibull, exponential, ...) to each series by maximum likelihood and ranks them by AIC, BIC or the KS statistic, with their fitted parameters. AIC is the default because the KS statistic alone always favours distributions with more parameters.

*Chart:* Histogram with the chosen fitted distribution — *best fit* follows whichever candidate ranks first.

== Models on tables

The last three models read several *columns* of each series at once rather than its values alone. They find the columns by the series' roles, so a series drawn as the matching chart already has them: a Mosaic Plot series for a contingency table, an Interaction Plot series for a two-way ANOVA, a Kaplan-Meier series for survival. A series without the columns a model needs is listed as *Not analysed*, with the columns it is missing.

=== Contingency table (chi-squared, Fisher)
Are two categorical variables associated?
#roles[a row variable (`x` or `group`), a column variable (`y` or `trace`)][`weight` — a count per row, for a table already counted]

The report shows the table of counts with its totals, then:
#defs(
  [*Pearson chi-squared*], [The standard test of independence, with its degrees of freedom. When some expected counts are below 5, the report says so and recommends Fisher's test.],
  [*Chi-squared, Yates' correction*], [For 2 × 2 tables: a more conservative version.],
  [*Likelihood-ratio G*], [An alternative to chi-squared, with the same reading.],
  [*Fisher exact*], [Exact for 2 × 2 tables (with the odds ratio); for larger tables, a Monte Carlo estimate with a fixed seed, so the same data always give the same p-value.],
  [*Cramér's V*], [The strength of the association, from 0 (none) to 1 (complete).],
)
*Chart:* Mosaic Plot — shade by Pearson residual to see which cells drive the result.

=== Two-way ANOVA
Do two factors affect a response, and does the effect of one depend on the other?
#roles[the first factor (`x`), the second factor (`trace` or `group`), the response (`y` or `value`)]

The ANOVA table has one row for each factor, one for their *interaction* and one for the residual, with sums of squares, degrees of freedom, F, p-value and partial η². Sums of squares are *Type II*, so the result does not depend on which factor is listed first; for a balanced design (the same number of values in every cell) they equal the classic sequential ones. The cell means and counts are listed too. Every combination of levels must have at least one value.

Read the interaction first: if it is significant, the effect of one factor depends on the level of the other, and the main effects alone do not tell the whole story.

*Chart:* Interaction Plot.

=== Survival (Kaplan-Meier, log-rank)
How long until an event, when some cases have not had it yet?
#roles[`time` (or `x`), `event` (1 = happened, 0 = censored)][`group`]

For each group: number of cases, events and censored cases, and the *median survival time* with its 95% confidence interval. The median is *not reached* when the curve never falls to 50%, and the upper end of the interval is ∞ when it never closes. With two or more groups, the *log-rank test* says whether the curves differ.

*Chart:* Kaplan-Meier.

== Reliability of the results

The statistics are computed by SciPy and statsmodels and are checked by ChartLibre's test suite against published reference values — among them SciPy's documented ANOVA example, Agresti's party-identification table for the contingency tests, Fisher's tea-tasting experiment for the exact test, the Freireich 6-MP leukaemia trial for Kaplan–Meier and the log-rank test, and the certified NIST StRD data sets for curve fitting.
