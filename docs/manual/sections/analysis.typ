#import "helpers.typ": *

= Analysis <analysis-chapter>

Where a data operation works on the series of a chart, an *analysis* works on the columns of a table: you say which column is the response and which ones might explain it, and it fits a model of the whole table. The *Analysis* page of the navigation rail lists them with the data operations; each opens on the table selected in the Tables page. Each names, under *Computed with*, the library functions that fit it, linked to their documentation.

== How every analysis window works

- *Select columns*, on the left: the table, its columns with their modelling type — ▲ continuous, ● nominal, ◐ ordinal — and the *Type* to change it. Under them, *Select role variables*: drag or click columns into the roles the analysis asks for (Y, X, groups, …). A role that is short of columns, or a column of the wrong type, is said before anything runs.
- *Parameters*, in the middle, where the analysis has any.
- *Results*, on the right: the report.
- *Preview* computes and shows the report; *OK* writes the results into the project — each row's prediction beside its data, the report's tables as tables — and draws a figure with the report in its notes. The calculation runs in the background, with *Stop*; OK is undoable.

== The analyses

#defs(
  [*Distribution*], [One report per Y column: the quantile table (0% to 100%, at the (n + 1)p position), the moments and the normality tests. OK draws the histogram with its normal curve beside the normal quantile plot. Process capability is in the Control Chart operation (see @data-operations), which knows whether the process is stable. #refs("numpy.quantile", "scipy.stats.shapiro")],
  [*Fit Y by X*], [One Y against each X, the analysis chosen by their types: *Bivariate* (continuous by continuous: the least-squares line, correlations), *Oneway* (continuous by groups: means, ANOVA, Welch and Kruskal-Wallis, and every pair compared — Tukey HSD, Games-Howell for unequal variances, Dunn on the ranks with Holm's adjustment), *Contingency* (groups by groups: counts, row shares, chi-squared, Fisher) or *Logistic* (groups by continuous: the probability of each level along X, odds ratios, AUC). An ordinal column is a measure as Y and groups as X. #refs("statsmodels.regression.linear_model.OLS", "scipy.stats.tukey_hsd", "scipy.stats.studentized_range", "scipy.stats.chi2_contingency", "statsmodels.discrete.discrete_model.Logit", "statsmodels.discrete.discrete_model.MNLogit")],
  [*Fit Model*], [A linear model of one or more responses on the effects you build from the columns (main effects, factorial, response surface). Its *personality* says how: *Standard Least Squares*, *Stepwise* (backward or forward), *Generalized Linear Model* (normal, binomial, Poisson, gamma), *Nominal* or *Ordinal Logistic* for a categorical Y, and *Generalized Regression* — Lasso, Ridge or Elastic Net, the penalty chosen by cross-validation, for many factors or factors that move together. Effect tests, estimates, the prediction profile; for a logistic fit the ROC curve, its AUC and the confusion matrix. #refs("statsmodels.formula.api.ols", "statsmodels.formula.api.glm", "statsmodels.discrete.discrete_model.Logit", "statsmodels.discrete.discrete_model.MNLogit", "statsmodels.miscmodels.ordinal_model.OrderedModel", "sklearn.linear_model.LassoCV", "sklearn.linear_model.RidgeCV", "sklearn.linear_model.ElasticNetCV", "sklearn.metrics.roc_curve")],
  [*Partition*], [A decision tree of one response on many factors, as JMP's: *Split* and *Prune* one step at a time, or *Automatic* on validation rows; or a *Bootstrap Forest* or a *Boosted Tree*. Leaves and their rules, column contributions, and for any of the three the *permutation importance* (how much the fit drops when a column is shuffled) and *partial dependence* (the predicted response as one factor varies); for a nominal Y the ROC curve. #refs("sklearn.tree.DecisionTreeRegressor", "sklearn.tree.DecisionTreeClassifier", "sklearn.ensemble.RandomForestRegressor", "sklearn.ensemble.GradientBoostingRegressor", "sklearn.metrics.roc_auc_score")],
  [*Multivariate*], [How continuous columns move together, two by two: the correlation matrix and a row per pair with its correlation, 95% confidence interval (Fisher's z), p-value and count — each pair on the rows where both columns have a value. *Pearson* for a straight-line relation, *Spearman* or *Kendall* for any relation that keeps going one way. OK draws the scatterplot matrix. #refs("scipy.stats.pearsonr", "scipy.stats.spearmanr", "scipy.stats.kendalltau")],
  [*Principal Components*], [A few directions that carry most of the variation of many columns: the eigenvalues with their share and the cumulative share, the loadings (each column's correlation with each component) and the eigenvectors, *On* the correlations (columns standardized, the usual choice) or the covariances. OK writes every row's scores (Prin1, Prin2, ...) beside its data and draws the scree, score and loading plots. #refs("numpy.linalg.eigh", "numpy.cov")],
  [*Discriminant*], [Which group a row belongs to, from continuous measures: linear or quadratic, with Wilks' lambda, the canonical analysis and plot, and each row's predicted group and probabilities. #refs("scipy.linalg.eigh", "scipy.linalg.cholesky", "scipy.stats.f")],
  [*Partial Least Squares*], [A response from many X that move together — the wavelengths of a spectrum — through a few components, their number chosen by cross-validation; the variables' importance (VIP), loadings, coefficients and scores. #refs("sklearn.cross_decomposition.PLSRegression", "sklearn.model_selection.KFold")],
)

#figure(
  example("analysis_fit_model", width: 100%),
  caption: [Fit Model on a Box-Behnken design: the effects ranked by LogWorth, and the prediction profile.],
)
#figure(
  example("analysis_partition", width: 100%),
  caption: [Partition of the penguins' body mass: the partition graph, each leaf as wide as its rows, and the permutation importance.],
)
#figure(
  example("analysis_discriminant"),
  caption: [Discriminant analysis of the three penguin species from four measures: the canonical plot.],
)
#figure(
  example("analysis_pls", width: 100%),
  caption: [Partial Least Squares on a set of spectra: RMSECV by number of components, VIP along the wavelengths, scores, actual by predicted.],
)

== Guided analysis <guided-analysis>

*Guided analysis* starts from what you want to find out rather than from the name of a function. It follows the table selected in the Tables page and says what it holds — how many numeric columns, categories and dates.

#figure(
  image("../screenshots/screenshot_guided_analysis.png", width: 100%),
  caption: [Guided analysis on the Palmer penguins table: what it holds, its own questions, the goals.],
)

- *Search* finds any function by what it does, in English or Italian, and by the names other programs give it ("anova", "crosstabs", "control chart"). The same search opens from anywhere with *Cmd/Ctrl+K*.
- *Quick answers* ask a question in everyday words and answer it in words, the right test chosen for you:
  - *Are there differences?* — is a measure different from one group to another (t-test or ANOVA, Mann-Whitney or Kruskal-Wallis, chi-squared or Fisher, as the data call for);
  - *What has the most impact?* — the factors ranked by how much they matter for a result (a random forest on rows it had not seen, with a linear model's p-values);
  - *Has it changed over time?* — a steady trend (Mann-Kendall, Sen's slope) or a sudden change (Pettitt), whichever the values show.
  #refs("scipy.stats.ttest_ind", "scipy.stats.mannwhitneyu", "scipy.stats.kruskal", "scipy.stats.tukey_hsd", "scipy.stats.chi2_contingency", "scipy.stats.fisher_exact", "sklearn.ensemble.RandomForestRegressor", "statsmodels.stats.anova.anova_lm", "scipy.stats.theilslopes")

  Above them, questions asked of the selected table itself — "Does “weight” change between the groups of “line”?" — open the answer with its columns already chosen.
- *What do you want to do?* lists goals — describe, compare, relate, explain, predict, monitor… With *Y* and *X* chosen from the table, *Open* starts the suggested analysis or chart with them cast. Suggestions the table cannot feed come last, greyed, saying what they lack.

#figure(
  example("answer_differences", "answer_impact", "answer_change", width: 100%),
  caption: [The three quick answers on real data: penguin body mass by species, what has the most impact on it, and NASA's global temperature anomaly since 1880.],
)

#note[
  The demo *Quick answers – three questions asked of real data* has each quick answer already asked, its answer in the chart's notes.
]
