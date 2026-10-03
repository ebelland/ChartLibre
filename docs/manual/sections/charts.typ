#import "helpers.typ": *

= Creating a chart <creating-a-chart>

A chart in ChartLibre is never a picture of a spreadsheet range: it is a *query*. Every series on every chart is a SQL `SELECT` whose result columns are named after the *roles* the chart type reads — `x` and `y` for a scatter plot, `value` for a histogram, `time` and `event` for a survival curve. The chart is redrawn from that query every time it is shown, so it always reflects the data as it is now.

== The New plot window

*Plot*, the first entry of the *Series Operations* list (see @series-operations), opens the chart-creation window. Building a chart takes three decisions:

+ *The chart type.* The picker lists all the chart types grouped by family, with a search box that matches a name or a whole family ("stat" brings up the statistical charts at once). Selecting one shows its description, the roles it needs and a link to the corresponding Matplotlib documentation.
+ *The data.* Pick a data source — a table or a saved query — then pick which of its columns plays each role. The window writes the SQL for you (`SELECT pressure AS x, flow AS y FROM operating_points`) and shows it; you can edit it directly for a filter, a join or a computed column. Only reading queries (`SELECT`, `WITH`) are accepted.
+ *Group by* (optional). Pick a column the chart does not already use — a species, a batch, a site — and the series becomes one series per value of that column, each with its own colour and legend entry. Each one is the same query with `WHERE "column" = value` added, and rows marked hidden stay out. Up to 50 groups.
+ *Where it goes.* *Create new figure* adds a new figure to the project; *Use current figure* puts the chart in the figure on screen, either as a new axis (*Add new axis*) or as one more series on an axis it already has (*Use existing axis*). Figure, axis and series can be named here or later.

When the window closes, the chart panel shows the new figure, so what you just made is what you see.

== How a chart reads its data

A chart type never sees your raw table — it sees the rows the series' query returns, with columns named after its roles. A Scatter Plot needs `x` and `y`, so its series query is written:

#code[```
SELECT pressure AS x, flow AS y FROM operating_points
```]

*Optional* roles unlock extra behaviour when the query provides them: `color` and `size` on a Scatter Plot drive a colour map and marker sizing; `yerr` adds symmetric error bars and `yerr_low`/`yerr_high` asymmetric ones; a `group` column on a Box Plot splits one series into several boxes. A role the query does not return is simply not used.

Because the source is SQL, everything SQLite can compute is available without touching the data: `WHERE` to filter, `GROUP BY` to aggregate, `strftime` to bin dates, arithmetic to convert units (`SELECT t, v * 1000 AS y`), `CASE` to recode categories. The Query Builder (@query-builder) is the place to write and test longer queries.

#note[
  Rows a chart cannot use are reported, not hidden. A value that is not a number where a number is needed, a missing grid cell for a chart that needs a complete grid, a negative value on a logarithmic axis: each is logged with the series it came from, and the chart draws what it can. Open the log (see @log) when a chart looks emptier than expected.
]

== Figures, axes and series

A *figure* is one page of the chart panel, listed under *Charts* in the rail. It holds one or more *axes* — individual charts — laid out in a grid, and each axis holds one or more *series*. Every axis has its own chart type, so a figure can put a histogram, its Q-Q plot and a box plot side by side.

Most chart types accept any number of series per axis and draw them together (several lines on one Time Series, several samples in one Histogram). A few accept only one — a Pie Chart, a Heatmap, a Surface Plot, a Mosaic Plot — because two of them on the same axis would hide each other; for these the application offers a new axis instead of a second series that would be silently dropped.

*Add series* in Chart Options adds a series to the selected axis. To add an axis to a figure, open *Plot* with *Use current figure* and *Add new axis*.

== Customizing a chart <customizing>

*Chart Options* edits the selected figure, axis and series, each in its own collapsible section:

/ Figure: name (also the tab title), the grid of axes (rows and columns), the *figure layout* (tight, constrained, compressed, or manual spacing), size and DPI, frame, a Matplotlib style for this figure alone, and *downsample* (draw at most N points per series, for very large tables).
/ Axis: chart type, title, axis labels, ranges, scales (linear, log with its base, symlog with its threshold) and direction (invert), shared x/y/z with neighbouring axes, grid, ticks, spines, the axis position and span in the grid, and the chart type's own *options* (number of bins, which distribution to compare with, whether to draw a colour bar, ...).
/ Series: name, SQL query, colour, line style, marker, legend label, visibility, sort by x, and the *plot arguments* passed to Matplotlib (line width, transparency, hatch, ...).

Every setting has a description shown in its tooltip and beside it, so Matplotlib's own documentation is rarely needed. Settings are split into two families:

- *Options* are read by ChartLibre's own chart code: whether to draw a legend, how many bins, which reference line, how to colour tiles.
- *Plot arguments* are handed to the Matplotlib drawing call unchanged: colours, widths, markers, transparency, drawing order.

#figure(
  image("../screenshot_chart_options.png", width: 100%),
  caption: [Chart Options: the figure, axis, series and overlay property
  panels, one collapsible section each, over the chart they apply to.],
)

*Overlay properties*, also in Chart Options, lists the annotations, reference lines and measurements of an axis — one tab each — as editable tables, so they can be positioned by exact value rather than only by dragging.

=== Features shared by several chart types

Some settings appear on many charts and work the same way everywhere:

#defs(
  [*Error bars*], [The roles `xerr`/`yerr` draw symmetric error bars; `xerr_low`/`xerr_high` and `yerr_low`/`yerr_high` draw asymmetric ones. *Cap size*, *cap thickness*, *error line width*, *error colour* and *error every N* (draw a bar on every Nth point only, to keep a dense series readable) style them.],
  [*Trend line*], [*Trend degree* fits a least-squares polynomial through the points (1 is a straight line, 0 draws none). *Trend band* shades either the *confidence* band — the uncertainty of the fitted line itself, narrow — or the *prediction* band — where the next observation would fall, much wider. *Trend sigma* sets the band's half-width in standard deviations.],
  [*Confidence ellipse*], [The covariance ellipse of the point cloud, *N* standard deviations across (2 covers about 95% of a bivariate normal sample). It shows correlation as a tilt, which a cloud of points alone does not make obvious.],
  [*Distribution fit*], [On Histogram and ECDF: fit continuous distributions to each sample and draw their curves over the data. *best* draws the top-ranked candidate, *top3* or *top5* that many, or name one (`norm`, `lognorm`, `gamma`, `weibull_min`, ...).],
  [*Colour mapping*], [When a `color` role holds numbers, *colormap*, *norm* (linear, log, symlog, logit), *vmin* and *vmax* control how values become colours. Integer colour columns are treated as category ids rather than a continuous scale.],
  [*Colour bar*], [On contour, heatmap, hexbin and mesh charts: add a colour bar with an optional label. It takes space from its own axis, so a *constrained* or *compressed* figure layout places it best.],
  [*3D view*], [On 3D charts: *elevation*, *azimuth* and *roll* fix the camera angle in degrees; leave them empty to use the angle you set by dragging.],
  [*Rasterized*], [Draw a dense series as pixels inside an otherwise vector export (SVG, PDF), so a million points do not make the file unopenable.],
)

= Chart types <renderers>

ChartLibre ships *42 chart types*, grouped into six families. The first five follow the taxonomy of Matplotlib's own documentation; *Diagnostic plots* gathers the charts statisticians use to check assumptions and report models. For each one below: what it shows, when to use it, the roles it reads, and the settings that matter most.

Role names are written as the query must return them. Where a role is shown in capitals (`Y`, `X`, `YError`), that is its exact spelling.

// ======================================================================
== Pairwise data

Charts of one quantity against another: the everyday x/y family.

=== Scatter Plot
One mark per row at (x, y). Optional columns colour each point (`color`: numbers through a colormap, integers as categories) and size it (`size`). The basic chart for two measured quantities, and the one most series operations add their results to.
#roles[`x`, `y`][`color`, `size`, `xerr`, `yerr`, `xerr_low`, `xerr_high`, `yerr_low`, `yerr_high`]
#use[Looking for a relationship between two measurements, spotting clusters and outliers, comparing data with a fitted curve.]
Key settings: marker shape and size, the shared *trend line*, *confidence ellipse* and *error bar* settings described above.

=== Time Series
A line through (x, y) in x order, where x is a number or a timestamp. A real date/time column gets date ticks automatically. A *rolling average* line (window *N* samples) can be drawn over the raw line, and *gap threshold* breaks the line wherever two samples are further apart than a given time (`30min`, `2h`, `1D`), so an outage reads as a gap instead of a straight line across it.
#roles[`x`, `y`][`color`, error-bar roles as Scatter Plot]
#use[Measurements over time: sensor logs, prices, monitoring data.]
Key settings: *rolling window*, *show raw* / *show rolling*, *rolling line style*, *gap threshold*.

=== Bar Chart and Horizontal Bar Chart
One bar per row, of height `Y`, at category `X` (or at the row number when there is no `X`). Several series are drawn as groups of bars side by side within each category. `YError`/`XError` add error bars, and `Bottom`/`Left` let bars start somewhere other than zero (for floating or stacked bars). The horizontal version is the same chart turned on its side, better when category names are long.
#roles[`Y`][`X`, `color`, `YError`, `XError`, `Bottom`, `Left`]
#use[Comparing a quantity across categories: sales per region, counts per class, means per group.]
Key settings: *width* (or *height*) of each group, *align*, *hatch*, *log* value axis, and three settings for crowded category axes: *max tick labels*, *tick label rotation*, *tick label font size* (each with an automatic mode).

=== Broken Bar and Broken Bar (Vertical)
Gantt-style interval bars: each row is an interval from `start` lasting `duration`, drawn on the band of its `category`. Several intervals in the same category share one band. The vertical version stacks the bands in columns.
#roles[`category`, `start`, `duration`][`color`]
#use[Schedules, machine states over time, shifts, any set of intervals grouped by resource.]
Key setting: *band height* (thickness of each band, as a fraction of the space between categories).

=== Fill Between
Shades the region between two curves sharing an x: `y` and `y2`, or `y` and a fixed *baseline* when there is no `y2`. The upper curve is drawn as a line over the fill (*edge line*), so a band reads as an estimate with its uncertainty.
#roles[`x`, `y`][`y2`]
#use[Confidence or tolerance bands, min–max envelopes, the area between two scenarios.]

=== Stack Plot
Every series on the axis is stacked into filled bands over a shared x, each band starting where the one below ends, so the top edge is the total.
#roles[`x`, `y`]
#use[How a total divides between its parts over time: energy mix, market shares, traffic by source.]
Key setting: *baseline* — `zero` (usual), `sym` (centred on zero) or the two `wiggle` forms that make a streamgraph.

=== Stem Plot
A vertical stem from a baseline to each value, topped by a marker.
#roles[`x`, `y`]
#use[Values defined only at discrete x: impulse responses, discrete spectra, sampled signals, probability mass functions.]
Key settings: *line*, *marker* and *base format* strings (`C0-`, `C0o`), *bottom* (the baseline value), *orientation*.

=== Table
Draws a table of text on the axis instead of a plot: up to eight columns, chosen with the roles `column_1` … `column_8`, in that order.
#roles[none][`column_1` … `column_8`]
#use[Putting a few key numbers beside a chart in the same figure, for a report or a slide.]
Key settings: *max rows* (default 200 — a large table is unreadable long before it is slow), *font size*, *row labels from index*, cell and header alignment, placement in the axis.

=== Text
Writes the text in `text` at each (x, y). When the optional *adjustText* library is installed, overlapping labels are pushed apart and joined to their point by a thin leader line.
#roles[`x`, `y`, `text`][`color`]
#use[Labelling points by name: countries on a scatter, sample ids, landmarks on a map-like plot.]
Key settings: *font size*, horizontal and vertical alignment, *show markers* at the original point, *leader colour*.

=== Timeline
Dated events along a horizontal baseline, each on its own vertical stem with its label at the top. Stems alternate between several heights so neighbouring labels do not collide. `y` overrides the automatic stem height, `color` and `size` work as on a scatter.
#roles[`x` (the date)][`label`, `y`, `color`, `size`]
#use[Release histories, project milestones, incident lists — anything that is a list of moments.]
Key settings: *levels* (how many stem heights to cycle through), *annotate*, *annotation font size*, *stem colour*.

// ======================================================================
== Statistical distributions

Charts of how values are distributed, rather than of one value against another.

=== Histogram
Counts values into equal-width bins. Several series — or one series split by a `dataset` column — share the same bins, so they can be compared directly; `weight` counts each row that many times.
#roles[`value`][`dataset`, `weight`, `color`]
#use[The first look at the shape of a sample: centre, spread, skew, several peaks.]
Key settings: *bins*, *range min/max*, *histogram type* (side-by-side bars, stacked bars, step outline, filled step), *density* (normalise to compare samples of different sizes), *cumulative*, *log* count axis, and *distribution fit* to overlay fitted densities.

=== Box Plot
The five-number summary of each sample: a box from the first to the third quartile, a line at the median, whiskers reaching 1.5 × IQR (adjustable) and individual points beyond them. A `group` column turns one series into one box per group; without it, each series is a box.
#roles[`value`][`group`, `color`]
#use[Comparing the distribution of several groups at a glance; spotting outliers.]
Key settings: *whisker reach*, *notch* (a confidence interval around the median: notches that do not overlap suggest different medians), *show means*, *show fliers*, *direction*, and *statistical annotations* (median, mean, n, or a custom format string, placed outside or on the box).

=== Violin Plot
Like a box plot, but each sample is drawn as its estimated density (a kernel density estimate), mirrored — so two peaks show as two bulges where a box would hide them. Same roles and grouping as Box Plot, so an axis can be switched between the two.
#roles[`value`][`group`, `color`]
#use[Comparing distributions that may be skewed or have more than one peak.]
Key settings: *bandwidth* (smaller follows the data more closely), *show medians/means/extrema*, *quantiles* to mark (e.g. `0.25, 0.5, 0.75`).

=== ECDF
The empirical cumulative distribution function: for every value, the fraction of the sample at or below it. No bins to choose, every point is drawn, and samples of different sizes compare directly.
#roles[`value`][`weight`]
#use[Comparing distributions honestly, reading percentiles directly ("90% of parts are below 4.2 mm"), judging a distribution fit.]
Key settings: *complementary* (draw 1 − F, the fraction above each value — better for upper tails), *as percent*, *distribution fit* (the gap between the steps and the fitted curve is what the Kolmogorov–Smirnov test measures).

=== Stairs
A step outline: each value is held constant between two edges. With an `x` column, x holds the *edges*, so it has one more row than the values.
#roles[`y`][`x` (bin edges)]
#use[Data that is already binned (a histogram computed elsewhere) or a rate that holds constant between readings, where a line would suggest a slope that is not there.]
Key settings: *fill*, *orientation*.

=== Pareto Chart
A bar chart with categories sorted from largest to smallest, plus a line of the cumulative percentage on a second axis and a reference line (80% by default).
#roles[`Y`][`X`, `color`, error and offset roles as Bar Chart]
#use[Quality and root-cause work — "which few causes account for most of the defects?" Read how far along the axis the cumulative line crosses 80%.]
Key settings: *cumulative line*, *reference percent*, *max categories* (lump the rest into "Other"), *ascending*.

=== Pie Chart
Wedges proportional to `value`, labelled by `label`, with the percentage inside each. `explode` pushes chosen wedges outward. A pie draws only one series.
#roles[`value`][`label`, `explode`]
#use[Showing parts of a whole when there are few parts (up to five or six). For comparisons between parts, a bar chart reads better.]
Key settings: *wedge width* (below 1 makes a donut), *colormap*, *start angle*, *percentage format*, *show legend* instead of labels.

=== Hexbin
Bins the (x, y) plane into hexagons and colours each by how many points fell into it — or, with a `value` column, by the mean, sum, max, min or count of those values.
#roles[`x`, `y`][`value`]
#use[Scatter plots with thousands to millions of points, where markers pile on top of each other and hide where the data really is.]
Key settings: *grid size*, *minimum count* (blank out sparse hexagons), *count scale* linear or log, *reduce function*, *colour bar*.

=== Event Plot
One row of short tick marks per series, one tick at each x where an event happened.
#roles[`x`]
#use[Discrete events with no value attached: a digital signal's edges, error log lines, neuron spikes, door openings.]
Key settings: *orientation*, row spacing (*line offsets*), tick length (*line lengths*), *row labels*.

=== Fishbone Diagram
An Ishikawa cause-and-effect diagram: a spine leading to the problem (the head) with one bone per `category` (Method, Machine, Material, ...), each listing its `cause` rows and, optionally, a finer `subcause` under each.
#roles[`category`, `cause`][`subcause`]
#use[Structuring a root-cause analysis in a quality or engineering review.]
Key settings: *problem* text (defaults to the series name), *font size*, bone, head and category colours.

// ======================================================================
== Diagnostic plots

Charts that check a statistical assumption or report a model. Several are drawn for you by *Statistics* (see @statistics) when *Add a chart* is ticked.

=== Q-Q Plot
Quantile–quantile plot: the sample's sorted values against the quantiles the chosen distribution predicts, after fitting that distribution to the sample. If the distribution fits, the points lie on a straight line. A shaded band shows where a sample truly drawn from that distribution would stay with 95% probability, so points outside it are real departures, not noise. A Q-Q plot is most sensitive in the *tails*.
#roles[`value`]
#use[Checking normality (or any other distribution) before a t-test, an ANOVA or a control chart; seeing *how* a sample departs — an S-shape means heavy or light tails, a curve means skew.]
Key settings: *distribution* (normal, lognormal, exponential, gamma, Weibull, logistic, Laplace, Gumbel, Student's t, uniform), *standardized* (theoretical quantiles as z-scores, like the classic normal plot, or in data units where a perfect fit lies on y = x), *reference line* (through the quartiles — not pulled by outliers — least squares, the fitted distribution, or none), *confidence* of the band.

=== P-P Plot
Probability–probability plot: the sample's cumulative probabilities against those of the fitted distribution. A perfect fit lies on the diagonal. Where the Q-Q plot is sharpest in the tails, the P-P plot is sharpest in the *middle* of the distribution.
#roles[`value`]
#use[Together with the Q-Q plot, when the centre of the distribution matters most.]
Key settings: *distribution* and *confidence* band, as for the Q-Q plot.

=== Pair Plot
A grid with one row and one column per numeric variable (`column_1` … `column_6`): a scatter for each pair below the diagonal, each variable's own histogram or density on the diagonal, and Pearson's correlation coefficient (or the mirrored scatter) above it. A `group` column colours points by group and reports the correlation within each group as well.
#roles[at least two of `column_1` … `column_6`][`group`]
#use[The first look at a table with several measurements: which variables move together, which are skewed, which group separates where — before deciding what to model.]
Key settings: *diagonal* (histogram, density, none), *upper* (correlation, scatter, none), *bins*, *max points* (scatters use a random sample of this many rows; histograms and correlations always use every row).

=== Interaction Plot
The mean of `y` at each level of a factor `x`, with one line per level of a second factor `trace`, and an error bar on every mean. Parallel lines mean the two factors act independently; lines that converge, diverge or cross are an *interaction* — the effect of one factor depends on the other. This is exactly what a two-way ANOVA tests.
#roles[`x`, `y`][`trace`]
#use[Designed experiments and any two-factor comparison: dose × sex, temperature × material, treatment × site.]
Key settings: *error bars* (standard error, confidence interval, standard deviation, or none), *confidence*, *dodge* (shift the lines sideways so error bars do not overlap).

=== Mosaic Plot
Two categorical variables as one square cut into tiles. Each column is as wide as its `x` category's share of the rows, and is split into tiles as tall as each `y` category's share within it — so each tile's *area* is the share of rows in that pair. If the variables are independent, every column is split the same way; uneven splits are the association. `weight` counts each row that many times (for a table already counted).
#roles[`x`, `y`][`weight`]
#use[Showing a contingency table; seeing *which* cells drive a significant chi-squared test.]
Key settings: *colour by* — the `y` category, or the *Pearson residual* (blue where there are more rows than independence predicts, red where fewer) — *labels* (percent, count, none), *gap* between tiles.

=== Forest Plot
One row per study, subgroup or coefficient: a square at the `estimate` and a horizontal line from `lower` to `upper` (its confidence interval), against a vertical line of no effect. `weight` scales each square's area so precise studies stand out; rows with a non-zero `summary` are drawn as diamonds (the pooled estimate).
#roles[`label`, `estimate`, `lower`, `upper`][`weight`, `summary`]
#use[Meta-analyses, subgroup analyses, tables of regression coefficients or odds ratios.]
Key settings: *null value* (0 for differences, 1 for ratios), *log scale* (for ratios, so 0.5 and 2 are symmetric about 1), *marker size*, *show values* and their *format*.

=== Kaplan-Meier
Survival curves for time-to-event data: the estimated probability that the event (failure, relapse, churn) has *not* happened yet, as a step function of time. Rows whose event has not happened by the end of observation (*censored*, `event` = 0) are used for exactly as long as they were observed, which a plain average cannot do. A `group` column draws one curve per group and the log-rank test compares them.
#roles[`time`, `event` (1 = event happened, 0 = censored)][`group`]
#use[Reliability (time to failure), clinical follow-up, customer retention — any "how long until" question where some cases have not reached the event yet.]
Key settings: *confidence* band (Greenwood, log-log scale), *censor marks* (a tick at each censored time), *median line* (dotted guides where each curve crosses 50%: the median survival time), *log-rank* p-value on the chart, *as percent*.

// ======================================================================
== Gridded data

Values sampled on a complete x/y grid: every combination of a set of x values and a set of y values has one row. The grid need not be evenly spaced, but it must be complete.

#note[
  A chart that needs a complete grid draws nothing when the rows are not one, and says so in the log. It never fills missing cells by interpolation and presents the result as data. Use the *scattered* counterpart (Contour Plot (Scattered), Triangular Color Mesh, Surface Plot (Scattered)) for data measured wherever it could be measured.
]

=== Contour Plot
Lines of equal `z` (contours) over the x/y plane, or filled bands between them — a topographic map of a function of two variables.
#roles[`x`, `y`, `z`]
#use[Fields and response surfaces: temperature maps, design-of-experiments responses, potential energy surfaces.]
Key settings: *levels* (a number of levels, or an explicit list like `0, 5, 10, 20`), *filled*, *line overlay* on top of the bands, *label lines* with their values (with *format*, *font size*, *inline*), *extend* (colour values beyond the outer levels), *colour bar*, *show points* (mark where the samples are).

=== Heatmap
Each grid cell painted with the colour of its own value, with no smoothing between cells.
#roles[`x`, `y`, `z`]
#use[Reading the exact value in each cell: correlation matrices, plate readers, confusion matrices, calendars.]
Key settings: *shading* (`flat`/`nearest` keep each cell one colour — the honest reading of measured data; `gouraud` interpolates), *annotate* each cell with its value (with *format*, *font size*, *colour*; skipped automatically on very large grids), *colour bar*.

=== Quiver
An arrow at each (x, y) pointing in the direction of the vector (`u`, `v`), with length proportional to its magnitude. Works on scattered samples too.
#roles[`x`, `y`, `u`, `v`][`color`]
#use[Vector fields: wind, currents, gradients, displacement measurements.]
Key settings: *scale* (data units per arrow length — fix it when comparing two axes), shaft *width*, *head width/length*, *pivot* (which part of the arrow sits on the point).

=== Stream Plot
Streamlines traced through the vector field (`u`, `v`): where a particle released in the flow would travel.
#roles[`x`, `y`, `u`, `v`] (on an evenly spaced grid)
#use[Seeing the overall pattern of a flow — vortices, sources, separation — rather than the vector at each sample.]
Key settings: *density* of the lines, *arrow size*, *broken streamlines*.

=== Wind Barbs
The meteorological symbol for a vector: a shaft pointing along (`u`, `v`) with flags and half-flags counting the speed.
#roles[`x`, `y`, `u`, `v`]
#use[Wind data, where meteorologists read speed from the flags more precisely than from arrow length.]
Key settings: *length*, *barb colour*, *flip barb* (the flags' side, which depends on the hemisphere).

// ======================================================================
== Irregularly gridded data

The same quantities as the gridded family, measured at arbitrary points. The points are joined into triangles (a Delaunay triangulation) and the chart is drawn over the triangles.

=== Contour Plot (Scattered)
Contours of `z` over scattered (x, y) samples, computed on their triangulation. Same settings as the gridded Contour Plot.
#roles[`x`, `y`, `z`]
#use[Field measurements at irregular positions: boreholes, weather stations, probe points on a part.]

=== Triangular Color Mesh
The triangulated samples, each triangle filled by value — the scattered counterpart of a heatmap.
#roles[`x`, `y`, `z`]
#use[Showing a scattered field without the smoothing that contour levels add.]
Key settings: *shading* (`flat` gives each triangle the mean of its corners and shows the triangles; `gouraud` interpolates into a smooth field), *colour bar*.

=== Triangular Mesh
The triangulation itself, drawn as edges and vertices.
#roles[`x`, `y`]
#use[Checking where the samples are, and what an interpolating chart will assume between them, before trusting a scattered contour or surface.]

// ======================================================================
== 3D and volumetric data

Charts drawn in a three-dimensional box. Drag on the chart to rotate it; the *3D view* settings fix the angle exactly.

=== Surface Plot
A surface over a complete x/y grid, coloured by height.
#roles[`x`, `y`, `z`] (complete grid)
#use[Response surfaces and functions of two variables, when the shape in depth matters more than exact values (for exact values, use a Contour Plot or Heatmap).]
Key settings: *circular mask* and *mask radius* (blank the grid outside a circle — a wafer-shaped surface), *row/column stride* (draw every Nth row or column of a large grid), *antialiased*.

=== Surface Plot (Scattered)
A surface over scattered x/y/z points, built from their triangulation.
#roles[`x`, `y`, `z`]
#use[As Surface Plot, for measurements not on a grid.]

=== Scatter Plot (3D)
Points in three dimensions, optionally coloured and sized by further columns. Distant points fade (*depth shade*), the main depth cue a still image has.
#roles[`x`, `y`, `z`][`color`, `size`]
#use[Point clouds whose shape is the message: clusters, a plane, an outlier in three variables.]

=== 3D Line Plot
Points connected in row order through three dimensions.
#roles[`x`, `y`, `z`]
#use[Trajectories and parametric curves: a particle path, a robot's motion, a phase portrait.]
Key settings: line style and width, optional *marker* at each sample.

=== 3D Bar Chart
A bar at each (x, y) rising to height `z`.
#roles[`x`, `y`, `z`]
#use[A small table of values read both ways at once — a value per pair of categories — where a heatmap's colour alone is not enough to compare heights.]
Key settings: bar *width* and *depth*, *bottom*, *shade*.
