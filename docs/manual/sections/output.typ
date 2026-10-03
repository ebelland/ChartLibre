#import "helpers.typ": *

= Saving, reports and templates <output>

== Saving a figure

*Save figure* (chart menu, or Ctrl+Shift+S / Cmd+Shift+S) saves the active figure as an image: *PNG* or *JPEG* (pixels), or *SVG* or *PDF* (vector — sharp at any size, the format journals ask for). The dialog opens on the default format chosen in Settings. The resolution is the figure's own *DPI*, set in Figure options (300 is usual for print), and the picture is trimmed to its content.

For a figure that must match a journal's column width, set the figure *size* in Figure options before saving; fonts keep their point size, so the text in the saved file is the size it will be printed at.

*Copy* puts the figure on the clipboard, to paste into a slide or a document.

== Reports

A *report* is a document for people who read results rather than queries: each figure drawn again, with its notes (the reports of the operations applied to it), and optionally the project's tables and history. The SQL behind the charts is deliberately left out.

#defs(
  [*Project report*], [*File* page → *Export* → *Project report*: every figure of the project, then — if ticked — the tables with their size and notes, and the operation history.],
  [*Export report…*], [Chart menu: the same report for *this figure alone*, titled after it.],
)

The options are the same for both:

#defs(
  [*Format*], [*HTML* — one self-contained file with the pictures inside it, which opens in any browser and can be e-mailed as it is. *PDF* — the same page laid out on A4.],
  [*Pictures*], [*PNG*, or *SVG* for pictures that stay sharp at any zoom (HTML only; the PDF always uses PNG).],
  [*Resolution*], [The DPI the figures are drawn at (72–600).],
  [*Contents*], [For the project report: include the tables, and the operation history.],
)

*Export…* asks where to save the file; the window shows its progress while the figures are drawn, and closes when the report is written. A figure that cannot be drawn is reported in the log and skipped, so one broken chart does not stop the report.

== Templates <templates>

A *template* keeps the *look* of a figure and none of its content: the figure's style sheet and layout, each axis' options except its title, labels, limits and annotations, and each series' colour, marker and line style. Applied to another figure, it changes how that figure looks and nothing it says.

#defs(
  [*Save this look as a template…*], [Chart menu → *Templates*. Give it a name.],
  [*Apply a template…*], [Gives the active figure the look of a saved template, axis by axis and series by series, in order. Undo takes it back.],
  [*Delete a template…*], [Forgets a template.],
)

Templates are kept in the `user/templates` folder of the ChartLibre installation, not in a project, so one template serves every project.

#use[Making every figure of a paper or a thesis look the same — the same fonts, grid, tick style and marker for each series — without setting it by hand on each one.]

== History <history>

Every series operation applied with *OK* is recorded in the project: when it ran, which operation and model, on which series, with which parameters, which tables it wrote, and its report. The record lives in the `.dhub` file, so it travels with the project.

- *Project history…* (on the *Developer* page) lists every operation in the project, oldest first.
- *History…* (right-click a table on the Tables page) lists the operations that read from or wrote to that table.

The history makes an analysis reproducible: anyone opening the project can see exactly how each result table and curve was obtained. It is also included in the project report.
