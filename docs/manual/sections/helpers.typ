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
