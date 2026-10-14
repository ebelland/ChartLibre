#import "helpers.typ": *

= Introduction

ChartLibre is a desktop application for Windows and macOS for importing tabular data, querying it with SQL, analysing it and turning it into scientific charts — from histograms to survival curves and 3D surfaces — without writing code. It is free and open source (MIT licence). It was written mostly by Claude, Anthropic's AI model, in conversation with its author, who designed it, set its direction and tested it. It is young software: test it on your own data before relying on it.
It should also work under Linux but it has never been tested.
It uses the power of SQLite to store the data, the Qt6 GUI multiplatform library, the charting capability of Matplotlib and powerful mathematical and statistical analysis tools such as SciPy, NumPy, scikit-learn and statsmodels.

#let tile(label, colour, detail: none) = block(
  width: 100%, inset: (x: 7pt, y: 6pt), radius: 4pt, fill: colour.lighten(88%), stroke: 0.8pt + colour,
)[
  #set par(justify: false)
  #text(weight: "bold", fill: colour.darken(25%), size: 9.5pt, label)
  #if detail != none { linebreak(); text(size: 8pt, fill: rgb("#555"), detail) }
]
#let badge(label, colour) = box(
  inset: (x: 6pt, y: 3pt), radius: 3pt, fill: colour, text(fill: white, weight: "bold", size: 8.5pt, label),
)
#let flow-arrow(label) = align(center + horizon)[
  #text(size: 7.5pt, fill: rgb("#666"), label) \
  #text(size: 20pt, fill: rgb("#888"), sym.arrow.r)
]
#let column-title(body) = align(center, text(size: 8.5pt, weight: "bold", fill: rgb("#444"), upper(body)))

#figure(
  block(width: 100%, inset: 10pt, radius: 6pt, fill: rgb("#fafafa"), stroke: 0.5pt + rgb("#ddd"))[
    #grid(
      columns: (1fr, 0.45fr, 1fr, 0.45fr, 1.25fr),
      align: horizon,
      gutter: 4pt,
      // Your data
      stack(spacing: 5pt,
        column-title[Your data],
        tile("Excel", rgb("#1d6f42"), detail: ".xlsx · .xlsm · .xls"),
        tile("Text files", rgb("#6b7280"), detail: "CSV · TSV · TXT · JSON · XML"),
        tile("Databases", rgb("#336791"), detail: "PostgreSQL · MySQL · SQLite"),
        tile("Clipboard · Web", rgb("#8a5a00"), detail: "a pasted table, a URL"),
      ),
      flow-arrow[import],
      // The project
      stack(spacing: 5pt,
        column-title[One project file],
        block(width: 100%, inset: 9pt, radius: 5pt, fill: rgb("#003b57"))[
          #set par(justify: false)
          #align(center)[
            #text(fill: white, weight: "bold", size: 15pt, "SQLite") \
            #v(2pt)
            #text(fill: rgb("#cfe3ee"), size: 8.5pt)[one `.dhub` file: \ the data *and* every chart]
          ]
        ],
        tile("SQL", rgb("#003b57"), detail: "query, join, aggregate — saved queries work like tables"),
      ),
      flow-arrow[read],
      // Charts and analysis
      stack(spacing: 5pt,
        column-title[Charts and analysis],
        block(width: 100%, inset: 7pt, radius: 5pt, fill: white, stroke: 0.8pt + rgb("#11557c"))[
          #align(center)[
            #image("../logos/matplotlib.png", width: 85%)
            #v(-2pt)
            #text(size: 8pt, fill: rgb("#555"))[42 chart types, your styles]
          ]
        ],
        block(width: 100%, inset: 7pt, radius: 5pt, fill: white, stroke: 0.8pt + rgb("#888"))[
          #set par(justify: false)
          #align(center)[
            #grid(columns: 2, gutter: 4pt,
              badge("NumPy", rgb("#4dabcf")), badge("SciPy", rgb("#3b5fa0")),
              badge("pandas", rgb("#150458")), badge("scikit-learn", rgb("#f7931e")),
              grid.cell(colspan: 2, badge("statsmodels", rgb("#6c3c97"))),
            )
            #v(2pt)
            #text(size: 8pt, fill: rgb("#555"))[fits, tests, models, signal processing]
          ]
        ],
      ),
    )
    #v(4pt)
    #align(center, text(size: 8pt, fill: rgb("#555"))[
      In a #text(weight: "bold", fill: rgb("#41cd52"))[Qt 6] window (PySide6) on Windows, macOS and Linux
      #h(6pt) #sym.arrow.r #h(6pt)
      out as PNG, SVG, PDF, and HTML or PDF reports
    ])
  ],
  kind: image,
  caption: [From your data to charts and results: what ChartLibre is built on.],
)

== What it does

- *Import* data from files (CSV, Excel, JSON, XML), the clipboard, other databases (SQLite, PostgreSQL, MySQL) and web addresses.
- *Query* it with SQL to filter, join and aggregate, without copying it.
- *Chart* it with 42 chart types, from bar charts to Kaplan-Meier curves, each fully customizable.
- *Analyse* it with 19 data operations on a chart's series — curve fitting, smoothing, filtering, spectral analysis, peak finding, statistics and hypothesis tests, control charts, clustering, outlier detection and more — and with models of a table's columns: Fit Model, Partition, Discriminant, Partial Least Squares.
- *Ask* it questions in everyday words: *Guided analysis* suggests what to use, and the quick answers reply in a sentence.
- *Report* it: save figures for publication, export a chart or the whole project as an HTML or PDF report, and reuse a chart's look as a template.

== Key concepts

#defs(
  [*Project (`.dhub`)*], [The file you work in: a SQLite database that holds the imported data *and* every chart built on it. It is self-contained — copy or e-mail the file and everything travels with it.],
  [*Table*], [A set of rows and columns: imported, produced by an operation (their names start with `_`), or edited by hand.],
  [*Saved query*], [A named SQL `SELECT`, usable anywhere a table is. Its rows are computed when read, so it always reflects the current data.], 
  [*Figure*], [Matplotlib object: a page of the chart panel, listed under *Charts* in the navigation rail. It holds one or more axes.],
  [*Axis*], [Matplotlib object: a chart inside a figure, with a chart type and one or more series.],
  [*Series*], [Matplotlib object: the data drawn on an axis: a SQL query whose columns are named after the *roles* the chart type reads — `x` and `y`, `value`, `time` and `event`, ...],
  [*Chart type* (renderer)], [The code that turns a series' rows into a drawing. See @renderers.],
  [*Data operation*], [A calculation on the data of one or more series, whose result is added to the chart. See @series-operations.],
  [*Analysis*], [A model of a table's columns — a response and what might explain it. See @analysis-chapter.],
)

== How this manual is organised

@getting-started covers installation, projects and demos. @main-window tours the window. @data-chapter explains importing, editing and querying data. @creating-a-chart and @renderers cover charts: how to make one, then every chart type in detail. @series-operations describes every data operation, @analysis-chapter the analyses of a table and Guided analysis, and @statistics the statistical tests. @output covers saving and reporting, and @settings-chapter the preferences, the log and the credits. The last chapter, @advanced, is for those who want to extend ChartLibre with their own chart types, operations or functions.
