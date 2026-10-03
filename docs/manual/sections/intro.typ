#import "helpers.typ": *

= Introduction

ChartLibre is a desktop application for Windows, macOS and Linux for importing tabular data, querying it with SQL, analysing it and turning it into scientific charts — from histograms to survival curves and 3D surfaces — without writing code. It is free and open source (MIT licence). It was written mostly by Claude, Anthropic's AI model, in conversation with its author, who designed it, set its direction and tested it. It is young software: test it on your own data before relying on it.

== What it does

- *Import* data from files (CSV, Excel, JSON, XML), the clipboard, other databases (SQLite, PostgreSQL, MySQL) and web addresses.
- *Query* it with SQL to filter, join and aggregate, without copying it.
- *Chart* it with 42 chart types, from bar charts to Kaplan-Meier curves, each fully customizable.
- *Analyse* it with 19 series operations: curve fitting, smoothing, filtering, spectral analysis, peak finding, statistics and hypothesis tests, control charts, clustering, outlier detection and more.
- *Report* it: save figures for publication, export a chart or the whole project as an HTML or PDF report, and reuse a chart's look as a template.

== Key concepts

#defs(
  [*Project (`.dhub`)*], [The file you work in: a SQLite database that holds the imported data *and* every chart built on it. It is self-contained — copy or e-mail the file and everything travels with it.],
  [*Table*], [A set of rows and columns: imported, produced by an operation (their names start with `_`), or edited by hand.],
  [*Saved query*], [A named SQL `SELECT`, usable anywhere a table is. Its rows are computed when read, so it always reflects the current data.],
  [*Figure*], [One page of the chart panel, listed under *Charts* in the navigation rail. It holds one or more axes.],
  [*Axis*], [One chart inside a figure, with a chart type and one or more series.],
  [*Series*], [The data drawn on an axis: a SQL query whose columns are named after the *roles* the chart type reads — `x` and `y`, `value`, `time` and `event`, ...],
  [*Chart type* (renderer)], [The code that turns a series' rows into a drawing. See @renderers.],
  [*Series operation*], [A calculation on the data of one or more series, whose result is added to the chart. See @series-operations.],
)

== How this manual is organised

@getting-started covers installation, projects and demos. @main-window tours the window. @data-chapter explains importing, editing and querying data. @creating-a-chart and @renderers cover charts: how to make one, then every chart type in detail. @series-operations describes every analysis operation, and @statistics the statistical tests. @output covers saving and reporting, and @settings-chapter the preferences, the log and the credits. The last chapter, @advanced, is for those who want to extend ChartLibre with their own chart types, operations or functions.
