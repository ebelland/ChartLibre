#let note(body) = block(
  fill: rgb("#eef3fb"),
  stroke: (left: 2.5pt + rgb("#3a6ea5")),
  inset: (x: 10pt, y: 8pt),
  radius: 2pt,
  width: 100%,
)[#body]

#let code(body) = block(
  fill: rgb("#f5f5f7"),
  stroke: 0.5pt + rgb("#dcdce0"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
)[#text(font: "Menlo", size: 8.8pt, body)]


// One line naming the columns a chart type reads.
#let roles(required, ..optional) = block(above: 0.5em, below: 0.7em)[
  #text(size: 9.5pt, fill: rgb("#444"))[
    *Required:* #required
    #for extra in optional.pos() [ #h(0.8em) *Optional:* #extra ]
  ]
]

// A two-column table of names and explanations, used throughout.
#let defs(..rows) = table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 5pt,
  ..rows,
)

// "When to use it" box beside a chart or operation.
#let use(body) = block(
  fill: rgb("#f3f8f1"),
  stroke: (left: 2.5pt + rgb("#5b8c4a")),
  inset: (x: 10pt, y: 6pt),
  radius: 2pt,
  width: 100%,
)[#text(size: 9.8pt)[*When to use it.* #body]]

// The Matplotlib functions a chart type draws with, each linked to its API page.
#let mpl-url(name) = if name.starts-with("Axes3D.") {
  "https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d." + name + ".html"
} else if name.starts-with("patches.") {
  "https://matplotlib.org/stable/api/_as_gen/matplotlib." + name + ".html"
} else {
  "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes." + name + ".html"
}
#let mpl(..names) = block(above: 0.3em, below: 0.7em)[
  #set par(justify: false)
  #text(size: 9.5pt, fill: rgb("#444"))[
    *Matplotlib:* #names.pos().map(name => link(mpl-url(name), text(fill: rgb("#1a5fb4"), raw(name)))).join(", ")
  ]
]

// The library functions an operation or an analysis computes with (SciPy,
// scikit-learn, statsmodels, NumPy), each linked to its documentation page.
#let ref-url(name) = if name.starts-with("scipy.") {
  "https://docs.scipy.org/doc/scipy/reference/generated/" + name + ".html"
} else if name.starts-with("sklearn.") {
  "https://scikit-learn.org/stable/modules/generated/" + name + ".html"
} else if name.starts-with("statsmodels.") {
  "https://www.statsmodels.org/stable/generated/" + name + ".html"
} else {
  "https://numpy.org/doc/stable/reference/generated/" + name + ".html"
}
#let refs(..names) = block(above: 0.4em, below: 0.7em)[
  #set par(justify: false)
  #text(size: 9.5pt, fill: rgb("#444"))[
    *Computed with:* #names.pos().map(name => link(ref-url(name), text(fill: rgb("#1a5fb4"), raw(name)))).join(", ")
  ]
]

// An example picture (or several side by side), drawn by ChartLibre itself:
// docs/manual/examples/, made by dev/manual/make_examples.py.
#let example(..names, width: auto) = {
  let pictures = names.pos()
  let w = if width != auto { width } else if pictures.len() == 1 { 66% } else { 100% }
  align(center, block(width: w, above: 0.6em, below: 0.9em, grid(
    columns: pictures.len(),
    gutter: 10pt,
    ..pictures.map(name => image("../examples/" + name + ".png", width: 100%)),
  )))
}
