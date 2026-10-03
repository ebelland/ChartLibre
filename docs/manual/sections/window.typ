#import "helpers.typ": *

= The main window <main-window>

#figure(
  image("../screenshot_main_window.png", width: 100%),
  caption: [The main window: the navigation rail on the left, the page it
  selects beside it, and the chart panel on the right.],
)

#note[
  The figures in this manual are taken in the *macOS native* style. On Windows the same screens wear the Fluent style: the panels, the pages and every control are the same and in the same place, but the navigation rail is a column of square tiles and the window draws its own title bar. On Linux the window follows the desktop's Qt style.
]

The window has three parts: the *navigation rail* at the far left — the pages under *Tools*, then the list of *Charts* — the *page* it selects, and the *chart panel*, which takes the rest of the window. The page and the chart panel are separated by a splitter you can drag.

== Navigation rail

#defs(
  [*Workspace*], [Collapses the page to give the chart panel the whole window; click again to bring the last page back.],
  [*File*], [Projects, demos, import, save and export (see below).],
  [*Tables*], [The tables and saved queries of the project, with a preview of the selected one (see @data-chapter).],
  [*Chart Options*], [The settings of the selected figure, axis and series (see @customizing).],
  [*Series Operations*], [New plot, the Query Builder and every analysis operation (see @series-operations).],
  [*Developer*], [The project's history and database information, and the tools for extending ChartLibre (see below).],
)

On Windows and Linux the rail ends with *Settings* and a *Menu* button (Undo, Help, Credits, ...); on macOS these live in the menu bar.

== The File page

#defs(
  [*Workspace*], [*New* and *Open* a project.],
  [*Demo*], [*Load demo* (see @demos).],
  [*Import*], [*Import* data into the project (see @importing).],
  [*Save*], [*Save* and *Save As…* (see @getting-started).],
  [*Export*], [*Project report* — the whole project as one HTML or PDF document (see @output).],
  [*Open recent*], [The projects opened recently; click one to open it, or *Clear* the list.],
)

== The Developer page

Despite its name, the first part of this page is for everyone:

#defs(
  [*History*], [*Project history…* lists every operation applied in the project, oldest first: what was run, on which series, with which settings (see @history).],
  [*Database information*], [The project file's path and size, everything SQLite reports about it (tables and rows, page usage and how much *Optimize DB* would reclaim, journal mode, SQLite version, file dates), and every table with its row count and import link. From here a table can be exported to CSV or Excel or refreshed from its link.],
  [*Optimize DB*], [Checks the project file for problems, reports them, and compacts it — it can shrink a project noticeably after large tables were deleted.],
)

The rest of the page holds the tools for extending ChartLibre — *Edit Localization*, *Series Operation Builder*, *Function Creator*, *Renderer Helper* (see @advanced) — and the *Log viewer* (see @log).

== The chart panel

The chart panel shows one *figure* at a time. Below the pages, the rail's *Charts* list names every figure of the project, numbered, with an icon for its chart type: click one to show it. The strip above the chart repeats the current figure's name — click it for the same list — with ‹ and › arrows to step to the previous or next figure. Creating a chart, or running an operation that adds a figure, shows the new figure. Above the chart, a toolbar offers Matplotlib's navigation tools and shows the pointer's coordinates:

#defs(
  [*Home*], [Back to the initial view.],
  [*Back / Forward*], [Step through the zoom and pan history.],
  [*Pan*], [Drag to move the view; right-drag to zoom each axis separately.],
  [*Zoom*], [Drag a rectangle to zoom in.],
  [*Zoom to fit*], [Bring the whole chart back within the panel.],
)

Resting the pointer on a point shows a label with its series name and its x and y values. Very large series are *downsampled* for drawing (see the Figure options); the data itself is never changed.

=== The chart menu

Right-clicking the chart opens its menu:

#defs(
  [*Reload*], [Redraw the chart from the project, discarding zoom and pan.],
  [*Copy*], [Copy the chart as a picture to the clipboard.],
  [*Save figure*], [Save the chart as PNG, JPEG, SVG or PDF, at the figure's own DPI (see @output).],
  [*Export report…*], [This chart alone, with its notes, as an HTML or PDF report (see @output).],
  [*Export view as CSV…*], [The rows actually on screen now — after zoom and downsampling — as a CSV file, each row tagged with its axis and series.],
  [*Templates*], [*Save this look as a template…*, *Apply a template…*, *Delete a template…* (see @templates).],
  [*Crosshair*], [A guide line under the pointer, shared by every axis of the figure.],
  [*Link zoom/pan across axes*], [Zooming or panning the x range of one axis applies it to every axis of the figure — for stacked time series.],
  [*Select points*], [Turns the pointer into a rectangle selection tool (see below).],
  [*Live updates*], [Redraws the chart on a timer, for a table that an external process keeps filling.],
  [*Delete*], [Deletes the figure, after confirmation.],
)

Right-clicking *inside an axis* adds actions for the exact point clicked: *Add vertical line at x = …*, *Add horizontal line at y = …*, *Add annotation here…*, and *Measure from here* / *Measure to here*.

=== Measuring, annotating and selecting points

*Measure from here* and then *Measure to here* record a two-click measurement — distance, Δx, Δy and slope — kept on the chart like a reference line. Annotations, reference lines and measurements each have a tab in *Overlay properties* (in Chart Options) where they can be edited by exact values or deleted. An annotation can also be dragged on the chart.

With *Select points* on, drag a rectangle over the chart: every point it covers, from every series of that axis, is selected. The menu then offers:

#defs(
  [*Hide N selected point(s)*], [Removes the points from the chart without touching the table: the series' query is narrowed to exclude them. Undo brings them back.],
  [*New series from selection*], [Adds a series holding only the selected points — to isolate a peak or a cluster for its own analysis.],
)

== Window size

Every window of ChartLibre opens fully visible on the screen it appears on, shrinking if needed to fit a small laptop display. Each dialog remembers its size and position between sessions.
