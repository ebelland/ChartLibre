#import "helpers.typ": *

= Advanced: extending ChartLibre <advanced>

ChartLibre discovers chart types and series operations the same way: by scanning a folder for Python classes that directly subclass a known base class, at import time — no registration list to edit and keep in sync. Dropping a well-formed file into the right folder is enough for it to appear in the application.

The *Developer* page offers a form for each of the sections below, so writing the file by hand is the fallback, not the only way:

/ *Renderer Helper*: a short form (name, category, description, link, required/optional roles) that scaffolds a new chart-type file.
/ *Series Operation Builder*: the same, for a new series-operation dialog - the scaffolded file already runs, passing each selected series through unchanged, ready to have the real computation dropped in.
/ *Function Creator*: the same, for a new fit function - the scaffolded `execute(x, p)` already evaluates a real polynomial in the declared parameters.
/ *Edit Localization*: a table editor for a language's translation catalogue, filterable to untranslated strings only, with a button to create a new language from scratch and an optional machine-translation pass to draft from — with Google, MyMemory, DeepL when a DeepL key is set in Settings, or Argos Translate, which runs on this computer once `ctranslate2` and `sentencepiece` are installed (`pip install ctranslate2 sentencepiece`) and downloads each language's model (85–150 MB) the first time it is used.

Every scaffolded file from the first three goes under `user/` at the project root (`user/charts/`, `user/series_operations/`, `user/functions/`) rather than into `app/`'s own shipped source - kept separate on purpose, so a packaged build's own files are never touched and a person's own additions are never confused with what shipped.

== Writing a custom chart renderer <advanced-renderer>

A chart type lives entirely in one file under `app/charts/`, as a class that subclasses `BaseAxisRenderer` (from `app.charts.base`) *directly* — the discovery scanner in `app/scanners/axis_renderer_scanner.py` matches that exact base-class name in the source, even when the class also inherits behaviour from another renderer.

The class is described entirely through its own attributes:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [`Name`], [Display name, and the `chart_type` value stored in the database. Renaming it orphans existing axes unless an alias is added to `CHART_TYPE_ALIASES` in the scanner.],
  [`Category`], [Which family of plot this is, following Matplotlib's own taxonomy (e.g. "Pairwise data", "Statistical distributions").],
  [`Description` / `Link`], [Shown in the chart picker and in the axis properties panel.],
  [`RequiredRoles` / `OptionalRoles`], [Column names the series' SQL query must (or may) produce — see @renderers.],
  [`Kwargs`], [Matplotlib keyword arguments forwarded verbatim to the plot call, as `{name: metadata}`. The metadata (`default`, `type`, `min`/`max`, `kind`, `group`, `description`) drives the generated editor UI automatically.],
  [`Options`], [Settings the renderer consumes itself and never forwards to Matplotlib — same metadata shape, opposite destination.],
  [`MaxSeries`], [How many series this renderer can draw on one axis, or `None` for any number.],
)

The class then implements one method:

#code[```
def render_axis(self, ax, series: list[SeriesData], options: dict | None = None) -> None:
    ...
```]

which receives one `SeriesData` per series — already queried, with its columns in `sd.df` — and draws them onto `ax` (a Matplotlib `Axes`).

`sd.df` is a `SeriesFrame` (`app.data.series_frame`): columns of NumPy arrays read straight from SQLite, read the way a DataFrame is (`sd.df["x"]`, `"x" in sd.df.columns`, `sd.df.loc[mask, "color"]`), with `sd.df.to_pandas()` available for the rare operation that genuinely needs a DataFrame. The developer guide's §4.1 covers it.

*Practical steps:*

+ Create `app/charts/my_chart.py`.
+ Import `BaseAxisRenderer` and, if useful, `SeriesData` from `app.charts.base`.
+ Define a class, e.g. `MyChartAxisRenderer(BaseAxisRenderer)`, setting `Name`, `Category`, `Description`, `RequiredRoles`/`OptionalRoles`, and optionally `Kwargs`/`Options`.
+ Implement `render_axis` using ordinary Matplotlib calls on `ax`.
+ Restart the application (or reopen the chart-creation window) — the new type appears in the picker with no further wiring.

`app/charts/text.py` is a good, compact, complete example to read or copy as a starting point: a handful of roles, a short `Kwargs` block, and a straightforward `render_axis`.

== Writing a custom series operation <advanced-operation>

A series operation lives in `app/series_operations/`, as a class that subclasses `SeriesOperationDialogBase` (from `app.series_operations.dialog_base`) *directly* — discovered the same way, by `app/scanners/series_operation_scanner.py`.

The base class supplies the whole dialog shell (series picker, parameter form, preview pane, action buttons) and the plumbing that turns a result into a new series or table on the chart. A subclass sets a few class attributes —

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [`Name`], [Display name shown in the Series Operations list.],
  [`Description`], [One-line summary shown next to the name.],
  [`Icon`], [An inline SVG source string for the operation's icon.],
)

— and overrides the hooks the base class calls at the right moments. The ones every operation implements are:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [`build_parameter_selector()`], [Build the parameter form. Simple operations describe their parameters declaratively with the helpers in `app.series_operations.parameter_spec` (`FloatParam`, `IntParam`, `ChoiceParam`, ...) rather than laying out widgets by hand.],
  [`prepare_job()` / `finish_job()`], [Read the widgets into a *job* whose `run()` does the calculation without touching the window, and turn its outcome into results. With these two, *Preview* and *OK* work unchanged, and `evaluate(..., background=True)` runs the job on a worker thread with the shared *Stop* button — see the developer guide. An operation that does not need that can override `compute_results()` instead.],
  [`result_series_spec(s)()`], [The series metadata (name, roles, style) the base class writes to the chart. The DataFrame comes from the result's own `to_df()`, and the table name from the `RESULT_TABLE_PREFIX` class attribute, unless you override them.],
  [`format_results()`], [Render the results as the text/HTML shown in the preview pane.],
  [`refresh_results()`], [Recompute and redraw the preview after a parameter changes.],
)

*Practical steps:*

+ Create `app/series_operations/my_operation_dialog.py`.
+ Subclass `SeriesOperationDialogBase`, set `Name`, `Description`, `Icon`.
+ Declare parameters (e.g. with `FloatParam`/`IntParam`/`ChoiceParam`) and implement `build_parameter_selector`.
+ Implement `prepare_job`/`finish_job` (or `compute_results`), `result_series_spec`, and `format_results`. Put the arithmetic in a Qt-free module under `app/analysis/`, so it can be tested on its own.
+ Restart the application — the operation appears in the Series Operations list, under *Other* until its name is added to a section.

`app/series_operations/function_dialog.py` is the smallest complete dialog and a reasonable starting template; `app/series_operations/calculus_dialog.py` is a good second read for an operation that consumes an existing series rather than generating one from scratch.
